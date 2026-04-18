"""
compare_cvmax.py
================
Compares original Spike-Driven Transformer vs CVmax-regularised version
across three metrics:

  1. 放电稀疏性  (Firing Sparsity)  – fraction of silent neurons
  2. ICE proxy   (Mean Firing Rate)  – lower = more efficient coding
  3. 重构误差    (Reconstruction Error, CE Loss on test set)

Strategy
--------
Both models start from the same pre-trained checkpoint and are
fine-tuned for N_FT epochs on a fixed CIFAR-10 training subset:
  - Baseline : CE loss only
  - CVmax    : CE loss + weight-CV + firing-rate-CV  (from criterion_v2.py)

All metrics are measured on the test set after each epoch.
A three-panel matplotlib figure is saved to compare_cvmax.png.
"""

import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

# ── Add repo root to path ───────────────────────────────────────────────────
REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

import argparse
from model import sdt                          # registers the model
from timm.models import create_model
from spikingjelly.clock_driven import functional
import criterion_v2

# ── Config ───────────────────────────────────────────────────────────────────
CHECKPOINT = os.path.join(
    REPO,
    "output_cv_experiments/baseline_20260405-123537/"
    "sdt-cifar10-baseline/model_best.pth.tar",
)
DATA_DIR   = os.path.join(REPO, "data")
DEVICE     = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()         else
    "cpu"
)
N_FT         = 15       # fine-tuning epochs
N_TRAIN      = 1000     # training samples per epoch (subset for speed)
N_TEST       = 500      # test samples for evaluation
BATCH_TRAIN  = 64
BATCH_TEST   = 64
FT_LR        = 5e-5     # low LR to not destroy pre-trained weights

# CVmax hyper-params (must be small enough to not overwhelm CE)
LAMBDA_WEIGHT_CV = 0.001
LAMBDA_FR_CV     = 0.005

print(f"Device: {DEVICE}")
print(f"Checkpoint: {CHECKPOINT}")

# ── Model factory ────────────────────────────────────────────────────────────
def make_model(ckpt_path: str) -> nn.Module:
    m = create_model(
        "sdt",
        T=4,
        num_classes=10,
        img_size_h=32,
        img_size_w=32,
        patch_size=4,
        embed_dims=256,
        num_heads=8,
        mlp_ratios=4,
        depths=2,
        sr_ratios=1,
        pooling_stat="0011",
        spike_mode="lif",
        dvs_mode=False,
        TET=False,
        in_channels=3,
        qkv_bias=False,
        drop_rate=0.0,
        drop_path_rate=0.2,
        drop_block_rate=None,
    )
    # torch >= 2.6 requires weights_only=False for checkpoints containing
    # non-tensor objects (e.g. argparse.Namespace saved in older checkpoints).
    torch.serialization.add_safe_globals([argparse.Namespace])
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    # Strip DDP/DataParallel prefix if present
    state = {k.replace("module.", ""): v for k, v in state.items()}
    m.load_state_dict(state, strict=False)
    return m.to(DEVICE)


# ── Data ─────────────────────────────────────────────────────────────────────
MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2470, 0.2435, 0.2616)

train_tf = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])
test_tf = T.Compose([
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])

full_train = torchvision.datasets.CIFAR10(
    DATA_DIR, train=True, download=True, transform=train_tf)
full_test  = torchvision.datasets.CIFAR10(
    DATA_DIR, train=False, download=True, transform=test_tf)

# Fixed subsets for reproducibility
g = torch.Generator().manual_seed(42)
train_idx = torch.randperm(len(full_train), generator=g)[:N_TRAIN].tolist()
test_idx  = torch.randperm(len(full_test),  generator=g)[:N_TEST].tolist()

train_sub  = torch.utils.data.Subset(full_train, train_idx)
test_sub   = torch.utils.data.Subset(full_test,  test_idx)

train_loader = torch.utils.data.DataLoader(
    train_sub, batch_size=BATCH_TRAIN, shuffle=True,  num_workers=0)
test_loader  = torch.utils.data.DataLoader(
    test_sub,  batch_size=BATCH_TEST,  shuffle=False, num_workers=0)

print(f"Train subset: {N_TRAIN}  |  Test subset: {N_TEST}")


# ── Metric collection ─────────────────────────────────────────────────────────
def collect_spike_metrics(model: nn.Module, loader) -> dict:
    """
    Run inference and return three main metrics:

      cv_fr    – mean CV(firing_rate) across neurons and layers
                 CV = std(fr) / mean(fr) per layer per sample, then averaged.
                 Measures heterogeneity of neural coding: higher = more
                 differentiated neuron roles = more efficient representation.

      mean_fr  – global mean firing rate across all neurons and layers.
                 ICE proxy: lower = fewer spikes = more energy-efficient coding.

      ce_loss  – cross-entropy loss on test set.
                 Reconstruction error proxy: lower = better information retention.

    Also returns per-layer cv and fr dicts for the bar chart.
    """
    model.eval()
    ce_fn = nn.CrossEntropyLoss().to(DEVICE)

    total_loss    = 0.0
    total_correct = 0
    total_samples = 0

    # Accumulate per-layer: list of (mean_fr, cv_fr) per batch
    layer_fr_batches:  dict[str, list] = defaultdict(list)
    layer_cv_batches:  dict[str, list] = defaultdict(list)

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            hook_dict = {}
            output, hook_dict = model(imgs, hook=hook_dict)
            functional.reset_net(model)

            loss = ce_fn(output, labels)
            total_loss    += loss.item() * imgs.size(0)
            total_correct += (output.argmax(1) == labels).sum().item()
            total_samples += imgs.size(0)

            for k, spikes in hook_dict.items():
                T_  = spikes.shape[0]
                B   = spikes.shape[1]
                # firing rate per neuron: (B, N)
                fr = spikes.float().sum(dim=0).view(B, -1) / T_

                # mean firing rate (ICE proxy)
                layer_fr_batches[k].append(fr.mean().item())

                # CV across neurons, averaged over samples in batch
                mean_n = fr.mean(dim=1, keepdim=True)          # (B,1)
                std_n  = fr.std(dim=1, keepdim=True, unbiased=False) + 1e-8
                cv     = (std_n / (mean_n + 1e-8)).mean().item()
                layer_cv_batches[k].append(cv)

    ce_loss = total_loss / total_samples
    acc     = total_correct / total_samples * 100.0

    layer_mean_fr = {k: float(np.mean(v)) for k, v in layer_fr_batches.items()}
    layer_cv_fr   = {k: float(np.mean(v)) for k, v in layer_cv_batches.items()}

    mean_fr = float(np.mean(list(layer_mean_fr.values())))
    cv_fr   = float(np.mean(list(layer_cv_fr.values())))

    return {
        "cv_fr":        cv_fr,
        "mean_fr":      mean_fr,
        "ce_loss":      ce_loss,
        "acc":          acc,
        "layer_cv_fr":  layer_cv_fr,
        "layer_mean_fr": layer_mean_fr,
    }


# ── Fine-tuning loop ──────────────────────────────────────────────────────────
def fine_tune_epoch(model: nn.Module,
                    loader,
                    optimizer,
                    use_cvmax: bool) -> float:
    """Train for one epoch; return mean training loss."""
    model.train()
    ce_fn  = nn.CrossEntropyLoss().to(DEVICE)
    total  = 0.0
    count  = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        if use_cvmax:
            hook_dict = {}
            output, hook_dict = model(imgs, hook=hook_dict)
            # CE loss
            loss = ce_fn(output, labels)
            # Weight CV loss
            loss = loss + criterion_v2.weight_cv_loss(
                model, lambda_weight_cv=LAMBDA_WEIGHT_CV)
            # Firing-rate CV loss (from head_lif spikes)
            head_spikes = [v.float() for k, v in hook_dict.items()
                           if k == "head_lif"]
            fr_cv = criterion_v2.firing_rate_cv_loss(
                head_spikes, lambda_cv=LAMBDA_FR_CV)
            if isinstance(fr_cv, torch.Tensor):
                loss = loss + fr_cv
        else:
            output, _ = model(imgs, hook=None)
            loss = ce_fn(output, labels)

        loss.backward()
        optimizer.step()
        functional.reset_net(model)

        total += loss.item() * imgs.size(0)
        count += imgs.size(0)

    return total / count


# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    print("\n=== Loading checkpoint ===")
    model_base = make_model(CHECKPOINT)
    model_cvm  = copy.deepcopy(model_base)

    opt_base = torch.optim.Adam(model_base.parameters(), lr=FT_LR)
    opt_cvm  = torch.optim.Adam(model_cvm.parameters(),  lr=FT_LR)

    # Evaluate before fine-tuning (epoch 0)
    print("\n=== Epoch 0 (pre-finetune baseline) ===")
    m0_base = collect_spike_metrics(model_base, test_loader)
    m0_cvm  = collect_spike_metrics(model_cvm,  test_loader)
    print(f"  Baseline  – CV(fr)={m0_base['cv_fr']:.4f} "
          f"MeanFR={m0_base['mean_fr']:.4f} CE={m0_base['ce_loss']:.4f} "
          f"Acc={m0_base['acc']:.2f}%")
    print(f"  CVmax     – CV(fr)={m0_cvm['cv_fr']:.4f}  "
          f"MeanFR={m0_cvm['mean_fr']:.4f}  CE={m0_cvm['ce_loss']:.4f}  "
          f"Acc={m0_cvm['acc']:.2f}%")

    # History
    hist_base = {"cv_fr":   [m0_base["cv_fr"]],
                 "mean_fr": [m0_base["mean_fr"]],
                 "ce_loss": [m0_base["ce_loss"]],
                 "acc":     [m0_base["acc"]]}
    hist_cvm  = {"cv_fr":   [m0_cvm["cv_fr"]],
                 "mean_fr": [m0_cvm["mean_fr"]],
                 "ce_loss": [m0_cvm["ce_loss"]],
                 "acc":     [m0_cvm["acc"]]}

    for epoch in range(1, N_FT + 1):
        t0 = time.time()
        tr_base = fine_tune_epoch(model_base, train_loader, opt_base, use_cvmax=False)
        tr_cvm  = fine_tune_epoch(model_cvm,  train_loader, opt_cvm,  use_cvmax=True)

        m_base = collect_spike_metrics(model_base, test_loader)
        m_cvm  = collect_spike_metrics(model_cvm,  test_loader)

        for hist, m in [(hist_base, m_base), (hist_cvm, m_cvm)]:
            for k in hist:
                hist[k].append(m[k])

        elapsed = time.time() - t0
        print(f"Epoch {epoch:02d}/{N_FT}  [{elapsed:.0f}s]  "
              f"Base: CV={m_base['cv_fr']:.4f} MFR={m_base['mean_fr']:.4f} "
              f"CE={m_base['ce_loss']:.4f} Acc={m_base['acc']:.2f}%  |  "
              f"CVmax: CV={m_cvm['cv_fr']:.4f}  MFR={m_cvm['mean_fr']:.4f}  "
              f"CE={m_cvm['ce_loss']:.4f}  Acc={m_cvm['acc']:.2f}%")

    # ── Layer-wise final metrics ───────────────────────────────────────────────
    final_base = collect_spike_metrics(model_base, test_loader)
    final_cvm  = collect_spike_metrics(model_cvm,  test_loader)
    final_base_layer = final_base["layer_cv_fr"]
    final_cvm_layer  = final_cvm["layer_cv_fr"]
    final_base_fr    = final_base["layer_mean_fr"]
    final_cvm_fr     = final_cvm["layer_mean_fr"]

    # Save results cache
    with open(RESULTS_CACHE, "w") as f:
        json.dump({"hist_base": hist_base, "hist_cvm": hist_cvm,
                   "final_base_layer": final_base_layer,
                   "final_cvm_layer":  final_cvm_layer,
                   "final_base_fr":    final_base_fr,
                   "final_cvm_fr":     final_cvm_fr}, f, indent=2)
    print(f"Results cached → {RESULTS_CACHE}")

    plot_results(hist_base, hist_cvm, final_base_layer, final_cvm_layer,
                 final_base_fr, final_cvm_fr)


def plot_results(hist_base, hist_cvm, final_base_layer, final_cvm_layer,
                 final_base_fr=None, final_cvm_fr=None):
    epochs = list(range(len(hist_base["cv_fr"])))
    n_ep   = len(epochs) - 1   # index of final epoch

    fig = plt.figure(figsize=(18, 13))
    fig.suptitle("Spike-Driven Transformer: Original vs CVmax\n"
                 f"(15-epoch fine-tune, 1000 train / 500 test samples, CIFAR-10)",
                 fontsize=14, fontweight="bold", y=0.99)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    BLUE  = "#1976D2"
    RED   = "#D32F2F"
    ALPHA = 0.9

    kw_base = dict(color=BLUE, lw=2, marker="o", ms=4, label="Baseline", alpha=ALPHA)
    kw_cvm  = dict(color=RED,  lw=2, marker="s", ms=4, label="CVmax",    alpha=ALPHA)

    # ── Panel 1: CV(firing_rate) ───────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, hist_base["cv_fr"], **kw_base)
    ax1.plot(epochs, hist_cvm["cv_fr"],  **kw_cvm)
    ax1.set_title("CV(Firing Rate)\n(higher = more heterogeneous coding)", fontsize=11)
    ax1.set_xlabel("Fine-tuning Epoch")
    ax1.set_ylabel("CV = std / mean  (across neurons)")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # ── Panel 2: Mean Firing Rate (ICE proxy) ─────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, hist_base["mean_fr"], **kw_base)
    ax2.plot(epochs, hist_cvm["mean_fr"],  **kw_cvm)
    ax2.set_title("Mean Firing Rate  (ICE proxy)\n(lower = more energy-efficient coding)",
                  fontsize=11)
    ax2.set_xlabel("Fine-tuning Epoch")
    ax2.set_ylabel("Mean firing rate")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    # ── Panel 3: CE Loss (reconstruction error proxy) ─────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(epochs, hist_base["ce_loss"], **kw_base)
    ax3.plot(epochs, hist_cvm["ce_loss"],  **kw_cvm)
    ax3.set_title("Cross-Entropy Loss  (reconstruction error proxy)\n"
                  "(lower = better information retention)", fontsize=11)
    ax3.set_xlabel("Fine-tuning Epoch")
    ax3.set_ylabel("CE Loss")
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)

    # ── Panel 4: Accuracy ─────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(epochs, hist_base["acc"], **kw_base)
    ax4.plot(epochs, hist_cvm["acc"],  **kw_cvm)
    ax4.set_title("Top-1 Accuracy (%)", fontsize=12)
    ax4.set_xlabel("Fine-tuning Epoch")
    ax4.set_ylabel("Accuracy (%)")
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.3)

    # ── Panel 5: Per-layer CV bar chart ───────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1:])
    all_layers = sorted(set(final_base_layer) | set(final_cvm_layer))
    short_names = [k.replace("MS_SSA_Conv", "SSA")
                    .replace("MS_MLP_Conv", "MLP")
                    .replace("MS_SPS", "SPS")
                    .replace("_lif", "").replace("_first", "")
                   for k in all_layers]

    n = len(all_layers)
    x = np.arange(n)
    w = 0.35
    v_base = [final_base_layer.get(k, 0) for k in all_layers]
    v_cvm  = [final_cvm_layer.get(k,  0) for k in all_layers]

    bars1 = ax5.bar(x - w/2, v_base, w, label="Baseline", color=BLUE, alpha=ALPHA)
    bars2 = ax5.bar(x + w/2, v_cvm,  w, label="CVmax",    color=RED,  alpha=ALPHA)
    ax5.set_xticks(x)
    ax5.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)
    ax5.set_title("Per-layer CV(Firing Rate)  (after fine-tuning)\n"
                  "CVmax directly optimises this quantity", fontsize=11)
    ax5.set_ylabel("CV = std / mean  (across neurons)")
    ax5.legend(fontsize=9)
    ax5.grid(alpha=0.3, axis="y")

    # Annotate deltas
    for b1, b2 in zip(bars1, bars2):
        delta = b2.get_height() - b1.get_height()
        if abs(delta) > 5e-4:
            clr = "green" if delta > 0 else "darkred"
            ax5.text(b2.get_x() + b2.get_width() / 2,
                     b2.get_height() + max(ax5.get_ylim()[1] * 0.01, 0.002),
                     f"{delta:+.3f}", ha="center", va="bottom",
                     fontsize=6, color=clr, fontweight="bold")

    # ── Summary text box ──────────────────────────────────────────────────────
    d_cv  = hist_cvm["cv_fr"][n_ep]   - hist_base["cv_fr"][n_ep]
    d_fr  = hist_cvm["mean_fr"][n_ep] - hist_base["mean_fr"][n_ep]
    d_ce  = hist_cvm["ce_loss"][n_ep] - hist_base["ce_loss"][n_ep]
    d_acc = hist_cvm["acc"][n_ep]     - hist_base["acc"][n_ep]

    def mark(cond): return "CVmax better ✓" if cond else "baseline better"
    summary = (
        f"After {n_ep} fine-tuning epochs  (LR={FT_LR}, "
        f"lambda_weight={LAMBDA_WEIGHT_CV}, lambda_fr={LAMBDA_FR_CV}):\n"
        f"  CV(FR)   Δ = {d_cv:+.4f}  ({mark(d_cv > 0)})\n"
        f"  Mean FR  Δ = {d_fr:+.4f}  ({mark(d_fr < 0)})\n"
        f"  CE Loss  Δ = {d_ce:+.4f}  ({mark(d_ce < 0)})\n"
        f"  Accuracy Δ = {d_acc:+.2f}%"
    )
    fig.text(0.01, 0.01, summary, fontsize=9, family="monospace",
             verticalalignment="bottom",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85))

    out = os.path.join(REPO, "compare_cvmax.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved → {out}")
    plt.close()

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print(f"  {'Metric':<26} {'Baseline':>10} {'CVmax':>10} {'Delta':>8}")
    print("-" * 62)
    rows = [
        ("cv_fr",   "CV(firing_rate) (↑)",  ".4f", True),
        ("mean_fr", "Mean Firing Rate  (↓)", ".4f", False),
        ("ce_loss", "CE Loss  (↓)",          ".4f", False),
        ("acc",     "Top-1 Acc %  (↑)",      ".2f", True),
    ]
    for key, label, fmt, higher_better in rows:
        vb = hist_base[key][n_ep]
        vc = hist_cvm[key][n_ep]
        d  = vc - vb
        ok = (higher_better and d > 0) or (not higher_better and d < 0)
        print(f"  {label:<26} {vb:{fmt}}     {vc:{fmt}}  {d:+{fmt}}  {'✓' if ok else ''}")
    print("=" * 62)


RESULTS_CACHE = os.path.join(REPO, "compare_cvmax_results.json")


def run_with_cache():
    """Run training if no cache exists, else load cache and re-plot."""
    if os.path.exists(RESULTS_CACHE) and "--force" not in sys.argv:
        print(f"Loading cached results from {RESULTS_CACHE}")
        with open(RESULTS_CACHE) as f:
            data = json.load(f)
        plot_results(data["hist_base"], data["hist_cvm"],
                     data["final_base_layer"], data["final_cvm_layer"],
                     data.get("final_base_fr"), data.get("final_cvm_fr"))
    else:
        run()


if __name__ == "__main__":
    run_with_cache()
