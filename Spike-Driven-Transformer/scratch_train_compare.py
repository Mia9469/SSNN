"""
scratch_train_compare.py
========================
Train Spike-Driven Transformer from scratch (random init) under two conditions:

  Baseline : TET loss only  (CE-based temporal ensemble)
  CVmax    : TET loss + weight_cv_loss (weight heterogeneity regularisation)

  NOTE — why not firing_rate_cv_loss in training?
  CV = std(fr)/mean(fr).  Its gradient pushes below-average LIF neurons toward
  fr=0.  Once inactive, the surrogate gradient ≈ 0 → dead-neuron trap →
  catastrophic collapse (empirically: mean_fr→0, CE→2.30, acc→10%).
  weight_cv_loss regularises the weight distribution directly (no LIF gate)
  and is numerically stable.  Firing-rate CV is *measured* at eval as an
  emergent consequence of weight diversity.

Training follows the original paper config:
  AdamW, lr=3e-4, cosine LR, mixup=0.5, AutoAug, 100 epochs

Metrics logged every EVAL_EVERY epochs:
  cv_fr    – CV(firing_rate) across neurons  [higher = more heterogeneous]
  mean_fr  – mean firing rate                [lower = more efficient]
  ce_loss  – cross-entropy on val set        [lower = better]
  acc      – top-1 accuracy

Outputs:
  scratch_compare_results.json   – full history
  scratch_compare.png            – 5-panel figure
"""

import copy
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from timm.data.mixup import Mixup
from timm.data.auto_augment import rand_augment_transform
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from model import sdt
from timm.models import create_model
from spikingjelly.clock_driven import functional
import criterion_v2

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()         else
    "cpu"
)

N_EPOCHS        = 300       # paper's original setting
N_TRAIN_SAMPLES = 50000     # full CIFAR-10 training set
BATCH_TRAIN     = 64
BATCH_VAL       = 128
EVAL_EVERY      = 10        # evaluate every N epochs
LR              = 3e-4
MIN_LR          = 1e-5
WARMUP_EP       = 20
WEIGHT_DECAY    = 0.06

LAMBDA_WEIGHT_CV = 0.001   # weight diversity — safe gradient term
LAMBDA_FR_CV     = 0.005   # used only for monitoring (detached), not gradient

TET_MEANS = 1.0
TET_LAMB  = 0.0

RESULTS_FILE = os.path.join(REPO, "scratch_compare_results.json")
FIGURE_FILE  = os.path.join(REPO, "scratch_compare.png")

print(f"Device  : {DEVICE}")
print(f"Epochs  : {N_EPOCHS}  |  eval every {EVAL_EVERY}  |  LR={LR}")


# ── Model factory (random init) ───────────────────────────────────────────────
def make_model() -> nn.Module:
    return create_model(
        "sdt",
        T=4,
        num_classes=10,
        img_size_h=32, img_size_w=32,
        patch_size=4,
        embed_dims=256,
        num_heads=8,
        mlp_ratios=4,
        depths=2,
        sr_ratios=1,
        pooling_stat="0011",
        spike_mode="lif",
        dvs_mode=False,
        TET=True,           # TET mode: model returns (T, B, C) logits
        in_channels=3,
        qkv_bias=False,
        drop_rate=0.0,
        drop_path_rate=0.2,
        drop_block_rate=None,
    ).to(DEVICE)


# ── Data ──────────────────────────────────────────────────────────────────────
MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2470, 0.2435, 0.2616)

# AutoAug (rand-m9-n1) + random crop/flip
aa_params = dict(translate_const=int(32 * 0.45), img_mean=tuple([int(c * 255) for c in MEAN]))
train_tf = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    rand_augment_transform("rand-m9-n1-mstd0.4-inc1", aa_params),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])
val_tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])

ds_train_full = torchvision.datasets.CIFAR10(os.path.join(REPO, "data"),
                                             train=True,  download=True, transform=train_tf)
ds_val        = torchvision.datasets.CIFAR10(os.path.join(REPO, "data"),
                                             train=False, download=True, transform=val_tf)

# Use full dataset when N_TRAIN_SAMPLES >= total size, else fixed random subset
if N_TRAIN_SAMPLES >= len(ds_train_full):
    ds_train = ds_train_full
else:
    _g   = torch.Generator().manual_seed(0)
    _idx = torch.randperm(len(ds_train_full), generator=_g)[:N_TRAIN_SAMPLES].tolist()
    ds_train = torch.utils.data.Subset(ds_train_full, _idx)

# On CUDA machines use more workers and pinned memory for speed.
_gpu = torch.cuda.is_available()
_nw  = 4 if _gpu else 2
train_loader = torch.utils.data.DataLoader(
    ds_train, batch_size=BATCH_TRAIN, shuffle=True,  num_workers=_nw, pin_memory=_gpu)
val_loader   = torch.utils.data.DataLoader(
    ds_val,   batch_size=BATCH_VAL,   shuffle=False, num_workers=_nw, pin_memory=_gpu)

print(f"Train: {len(ds_train)}  |  Val: {len(ds_val)}  |  workers={_nw}  pin_memory={_gpu}")

# Mixup
mixup_fn = Mixup(
    mixup_alpha=0.5, cutmix_alpha=0.0, prob=1.0,
    switch_prob=0.5, mode="batch",
    label_smoothing=0.1, num_classes=10,
)


# ── LR schedule: cosine with linear warmup ────────────────────────────────────
def get_lr(epoch: int) -> float:
    if epoch < WARMUP_EP:
        return LR * (epoch + 1) / WARMUP_EP
    t = (epoch - WARMUP_EP) / max(1, N_EPOCHS - WARMUP_EP)
    return MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * t))


# ── TET loss helper ───────────────────────────────────────────────────────────
def tet_loss(outputs, labels, ce_fn):
    """outputs: (T, B, C); returns scalar."""
    T_ = outputs.size(0)
    loss = sum(ce_fn(outputs[t], labels) for t in range(T_)) / T_
    return loss


# ── Metric collection ─────────────────────────────────────────────────────────
def evaluate(model: nn.Module) -> dict:
    model.eval()
    ce_fn = nn.CrossEntropyLoss().to(DEVICE)
    total_loss, total_correct, total_n = 0.0, 0, 0
    layer_fr_acc:  dict[str, list] = defaultdict(list)
    layer_cv_acc:  dict[str, list] = defaultdict(list)

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            hook = {}
            outputs, hook = model(imgs, hook=hook)
            functional.reset_net(model)

            # TET CE on logits averaged across time
            logits_mean = outputs.mean(0)        # (B, C)
            loss = ce_fn(logits_mean, labels)
            total_loss    += loss.item() * imgs.size(0)
            total_correct += (logits_mean.argmax(1) == labels).sum().item()
            total_n       += imgs.size(0)

            for k, spikes in hook.items():
                T_  = spikes.shape[0]
                B   = spikes.shape[1]
                fr  = spikes.float().sum(0).view(B, -1) / T_   # (B, N)
                layer_fr_acc[k].append(fr.mean().item())
                mean_n = fr.mean(dim=1, keepdim=True)
                std_n  = fr.std(dim=1, keepdim=True, unbiased=False) + 1e-8
                cv     = (std_n / (mean_n + 1e-8)).mean().item()
                layer_cv_acc[k].append(cv)

    ce_loss = total_loss / total_n
    acc     = total_correct / total_n * 100.0
    mean_fr = float(np.mean([np.mean(v) for v in layer_fr_acc.values()]))
    cv_fr   = float(np.mean([np.mean(v) for v in layer_cv_acc.values()]))
    layer_cv  = {k: float(np.mean(v)) for k, v in layer_cv_acc.items()}
    layer_mfr = {k: float(np.mean(v)) for k, v in layer_fr_acc.items()}

    return {"cv_fr": cv_fr, "mean_fr": mean_fr,
            "ce_loss": ce_loss, "acc": acc,
            "layer_cv": layer_cv, "layer_mfr": layer_mfr}


# ── Training loop ─────────────────────────────────────────────────────────────
def train_one_epoch(model, optimizer, use_cvmax: bool) -> float:
    model.train()
    ce_fn  = nn.CrossEntropyLoss(label_smoothing=0.1).to(DEVICE)
    total, count = 0.0, 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        imgs, labels = mixup_fn(imgs, labels)
        optimizer.zero_grad()

        if use_cvmax:
            hook = {}
            outputs, hook = model(imgs, hook=hook)
            loss = tet_loss(outputs, labels, ce_fn)
            # Only weight_cv_loss is used as a gradient term.
            # firing_rate_cv_loss is NOT added to the training loss:
            #   CV = std(fr)/mean(fr) — its gradient pushes below-average LIF neurons
            #   toward fr=0. Once a LIF neuron stops firing, the surrogate gradient ≈ 0
            #   and it can never recover (dead-neuron trap) → catastrophic collapse.
            #   Safe approach: regularise the *weights* for diversity; firing-rate CV
            #   is measured at eval time as an emergent property.
            loss = loss + criterion_v2.weight_cv_loss(
                model, lambda_weight_cv=LAMBDA_WEIGHT_CV)
        else:
            outputs, _ = model(imgs, hook=None)
            loss = tet_loss(outputs, labels, ce_fn)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        functional.reset_net(model)

        total += loss.item() * imgs.size(0)
        count += imgs.size(0)

    return total / count


# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    torch.manual_seed(42)
    model_base = make_model()
    # CVmax starts from the same random init
    model_cvm  = copy.deepcopy(model_base)

    opt_base = torch.optim.AdamW(model_base.parameters(),
                                 lr=LR, weight_decay=WEIGHT_DECAY)
    opt_cvm  = torch.optim.AdamW(model_cvm.parameters(),
                                  lr=LR, weight_decay=WEIGHT_DECAY)

    hist_base: dict[str, list] = {"cv_fr": [], "mean_fr": [], "ce_loss": [], "acc": [], "epoch": []}
    hist_cvm:  dict[str, list] = {"cv_fr": [], "mean_fr": [], "ce_loss": [], "acc": [], "epoch": []}

    print("\n=== Starting from-scratch training ===")
    total_t0 = time.time()

    for epoch in range(N_EPOCHS):
        lr_now = get_lr(epoch)
        for pg in opt_base.param_groups: pg["lr"] = lr_now
        for pg in opt_cvm.param_groups:  pg["lr"] = lr_now

        t0 = time.time()
        tr_base = train_one_epoch(model_base, opt_base, use_cvmax=False)
        tr_cvm  = train_one_epoch(model_cvm,  opt_cvm,  use_cvmax=True)
        elapsed = time.time() - t0

        if (epoch + 1) % EVAL_EVERY == 0 or epoch == 0:
            m_base = evaluate(model_base)
            m_cvm  = evaluate(model_cvm)
            ep_num = epoch + 1
            for hist, m in [(hist_base, m_base), (hist_cvm, m_cvm)]:
                hist["epoch"].append(ep_num)
                for k in ("cv_fr", "mean_fr", "ce_loss", "acc"):
                    hist[k].append(m[k])

            print(f"Ep {ep_num:>3}/{N_EPOCHS}  lr={lr_now:.2e}  [{elapsed:.0f}s]  "
                  f"Base: CV={m_base['cv_fr']:.4f} MFR={m_base['mean_fr']:.4f} "
                  f"CE={m_base['ce_loss']:.4f} Acc={m_base['acc']:.2f}%  |  "
                  f"CVmax: CV={m_cvm['cv_fr']:.4f}  MFR={m_cvm['mean_fr']:.4f}  "
                  f"CE={m_cvm['ce_loss']:.4f}  Acc={m_cvm['acc']:.2f}%")
        else:
            print(f"Ep {epoch+1:>3}/{N_EPOCHS}  lr={lr_now:.2e}  [{elapsed:.0f}s]  "
                  f"tr_base={tr_base:.4f}  tr_cvm={tr_cvm:.4f}")

    total_elapsed = (time.time() - total_t0) / 60
    print(f"\nTotal training time: {total_elapsed:.1f} min")

    # Final per-layer metrics
    fin_base = evaluate(model_base)
    fin_cvm  = evaluate(model_cvm)

    data = {
        "hist_base": hist_base,
        "hist_cvm":  hist_cvm,
        "final_base_layer_cv":  fin_base["layer_cv"],
        "final_cvm_layer_cv":   fin_cvm["layer_cv"],
        "final_base_layer_mfr": fin_base["layer_mfr"],
        "final_cvm_layer_mfr":  fin_cvm["layer_mfr"],
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Results saved → {RESULTS_FILE}")

    plot(data)


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot(data: dict):
    hb = data["hist_base"]
    hc = data["hist_cvm"]
    epochs = hb["epoch"]
    n_ep   = len(epochs) - 1

    BLUE  = "#1565C0"
    RED   = "#C62828"
    ALPHA = 0.9

    fig = plt.figure(figsize=(18, 13))
    fig.suptitle(
        "Spike-Driven Transformer: Baseline vs CVmax  —  Trained from Scratch\n"
        f"(CIFAR-10, {N_EPOCHS} epochs, {N_TRAIN_SAMPLES} train / 10000 val, AdamW lr={LR})",
        fontsize=13, fontweight="bold", y=0.99)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    kw_b = dict(color=BLUE, lw=2, marker="o", ms=4, label="Baseline", alpha=ALPHA)
    kw_c = dict(color=RED,  lw=2, marker="s", ms=4, label="CVmax",    alpha=ALPHA)

    # Panel 1: CV(FR)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, hb["cv_fr"], **kw_b)
    ax1.plot(epochs, hc["cv_fr"], **kw_c)
    ax1.set_title("CV(Firing Rate)\n(higher = more heterogeneous coding)", fontsize=11)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("CV = std/mean  (across neurons)")
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

    # Panel 2: Mean Firing Rate
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, hb["mean_fr"], **kw_b)
    ax2.plot(epochs, hc["mean_fr"], **kw_c)
    ax2.set_title("Mean Firing Rate  (ICE proxy)\n(lower = more efficient coding)", fontsize=11)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Mean firing rate")
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

    # Panel 3: CE Loss
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(epochs, hb["ce_loss"], **kw_b)
    ax3.plot(epochs, hc["ce_loss"], **kw_c)
    ax3.set_title("CE Loss  (reconstruction error proxy)\n(lower = better)", fontsize=11)
    ax3.set_xlabel("Epoch"); ax3.set_ylabel("CE Loss")
    ax3.legend(fontsize=9); ax3.grid(alpha=0.3)

    # Panel 4: Accuracy
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(epochs, hb["acc"], **kw_b)
    ax4.plot(epochs, hc["acc"], **kw_c)
    ax4.set_title("Top-1 Accuracy (%)", fontsize=12)
    ax4.set_xlabel("Epoch"); ax4.set_ylabel("Accuracy (%)")
    ax4.legend(fontsize=9); ax4.grid(alpha=0.3)

    # Panel 5: Per-layer CV bar chart
    ax5 = fig.add_subplot(gs[1, 1:])
    bl = data.get("final_base_layer_cv", {})
    cl = data.get("final_cvm_layer_cv",  {})
    all_layers = sorted(set(bl) | set(cl))
    short = [k.replace("MS_SSA_Conv", "SSA").replace("MS_MLP_Conv", "MLP")
              .replace("MS_SPS", "SPS").replace("_lif", "").replace("_first", "")
             for k in all_layers]
    x = np.arange(len(all_layers)); w = 0.35
    vb = [bl.get(k, 0) for k in all_layers]
    vc = [cl.get(k, 0) for k in all_layers]
    bars1 = ax5.bar(x - w/2, vb, w, label="Baseline", color=BLUE, alpha=ALPHA)
    bars2 = ax5.bar(x + w/2, vc, w, label="CVmax",    color=RED,  alpha=ALPHA)
    ax5.set_xticks(x); ax5.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
    ax5.set_title("Per-layer CV(Firing Rate)  (epoch 100)\n"
                  "CVmax directly optimises this quantity", fontsize=11)
    ax5.set_ylabel("CV"); ax5.legend(fontsize=9); ax5.grid(alpha=0.3, axis="y")
    for b1, b2 in zip(bars1, bars2):
        d = b2.get_height() - b1.get_height()
        if abs(d) > 5e-4:
            ax5.text(b2.get_x() + b2.get_width()/2,
                     b2.get_height() + ax5.get_ylim()[1]*0.01,
                     f"{d:+.3f}", ha="center", va="bottom",
                     fontsize=6, color="green" if d > 0 else "darkred", fontweight="bold")

    # Summary box
    d_cv  = hc["cv_fr"][n_ep]   - hb["cv_fr"][n_ep]
    d_fr  = hc["mean_fr"][n_ep] - hb["mean_fr"][n_ep]
    d_ce  = hc["ce_loss"][n_ep] - hb["ce_loss"][n_ep]
    d_acc = hc["acc"][n_ep]     - hb["acc"][n_ep]
    def mk(cond): return "CVmax better" if cond else "baseline better"
    summary = (
        f"Epoch {epochs[n_ep]} results:\n"
        f"  CV(FR)   Δ = {d_cv:+.4f}  ({mk(d_cv>0)})\n"
        f"  Mean FR  Δ = {d_fr:+.4f}  ({mk(d_fr<0)})\n"
        f"  CE Loss  Δ = {d_ce:+.4f}  ({mk(d_ce<0)})\n"
        f"  Accuracy Δ = {d_acc:+.2f}%  ({mk(d_acc>0)})\n\n"
        f"CVmax: lambda_weight={LAMBDA_WEIGHT_CV}  lambda_fr={LAMBDA_FR_CV}"
    )
    fig.text(0.01, 0.01, summary, fontsize=9, family="monospace",
             verticalalignment="bottom",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85))

    plt.savefig(FIGURE_FILE, dpi=150, bbox_inches="tight")
    print(f"Figure saved → {FIGURE_FILE}")
    plt.close()

    # Console table
    print("\n" + "=" * 62)
    print(f"  {'Metric':<26} {'Baseline':>10} {'CVmax':>10} {'Delta':>8}")
    print("-" * 62)
    for key, label, fmt, hi in [
        ("cv_fr",   "CV(firing_rate) (↑)", ".4f", True),
        ("mean_fr", "Mean Firing Rate  (↓)", ".4f", False),
        ("ce_loss", "CE Loss  (↓)",         ".4f", False),
        ("acc",     "Top-1 Acc %  (↑)",     ".2f", True),
    ]:
        vb = hb[key][n_ep]; vc = hc[key][n_ep]; d = vc - vb
        ok = (hi and d > 0) or (not hi and d < 0)
        print(f"  {label:<26} {vb:{fmt}}     {vc:{fmt}}  {d:+{fmt}}  {'✓' if ok else ''}")
    print("=" * 62)


def run_with_cache():
    if os.path.exists(RESULTS_FILE) and "--force" not in sys.argv:
        print(f"Loading cached results from {RESULTS_FILE}")
        with open(RESULTS_FILE) as f:
            data = json.load(f)
        plot(data)
    else:
        run()


if __name__ == "__main__":
    run_with_cache()
