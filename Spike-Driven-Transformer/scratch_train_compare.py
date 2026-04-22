"""
scratch_train_compare.py
========================
Ablation study: train Spike-Driven Transformer from scratch under four conditions.

  A. Baseline      : TET loss only
  B. WeightCV      : TET + weight_cv_loss  (weight heterogeneity)
  C. PopFR-CV      : TET + population firing-rate CV loss  (applied after warmup)
  D. Weight+PopCV  : TET + both CV terms
  E. LifetimeCV    : TET + lifetime firing-rate CV loss  (P1.2 control: CV_L
                     tests whether the dead-neuron trap or the biophysical
                     ceiling predicted by Huang et al. 2025 dominates).

Metrics logged every EVAL_EVERY epochs:
  cv_fr      – population CV(firing_rate) across neurons    [↑ = more heterogeneous]
  cv_life    – lifetime CV per neuron across the batch      [↑ = more specialised]
  temp_cv    – temporal CV per neuron across T timesteps    [↑ = more irregular]
  weight_cv  – mean CV of weight magnitudes                 [↑ = more diverse filters]
  mean_fr    – mean firing rate                             [↓ = more efficient]
  ce_loss    – cross-entropy on val set                     [↓ = better]
  acc        – top-1 accuracy

Notes on PopFR-CV safety:
  gradient of -CV(fr) pushes below-average neurons toward fr=0 → dead-neuron trap.
  Mitigation: small lambda (0.0005) + only applied after warmup (epoch >= WARMUP_EP).

Outputs:
  scratch_compare_results.json   – full history for all 4 configs
  scratch_compare.png            – multi-panel figure
"""

import argparse
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

# ── CLI ───────────────────────────────────────────────────────────────────────
# Uses parse_known_args so legacy flags like `--force` still pass through.
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
_parser.add_argument("--epochs",  type=int, default=None,
                     help="override default epoch count (defaults: 300 for cifar10, 300 for cifar100)")
_parser.add_argument("--tag",     type=str, default="",
                     help="suffix appended to output filenames so CIFAR-10 / CIFAR-100 runs don't overwrite each other")
CLI_ARGS, _ = _parser.parse_known_args()

DATASET  = CLI_ARGS.dataset
TAG      = CLI_ARGS.tag or ("" if DATASET == "cifar10" else f"_{DATASET}")
N_CLASSES = 100 if DATASET == "cifar100" else 10

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()         else
    "cpu"
)

# ── Training hyper-params ─────────────────────────────────────────────────────
N_EPOCHS        = CLI_ARGS.epochs if CLI_ARGS.epochs else 300
N_TRAIN_SAMPLES = 50000
BATCH_TRAIN     = 64
BATCH_VAL       = 128
EVAL_EVERY      = 10
LR              = 3e-4
MIN_LR          = 1e-5
WARMUP_EP       = 20
WEIGHT_DECAY    = 0.06

LAMBDA_WEIGHT_CV   = 0.001
LAMBDA_POP_CV      = 0.0005   # small to avoid dead-neuron collapse
LAMBDA_LIFETIME_CV = 0.0005   # same magnitude as Pop-CV for fair comparison
POP_CV_START       = WARMUP_EP  # only apply population CV loss after warmup
LIFETIME_CV_START  = WARMUP_EP  # same warmup gate for lifetime CV

RESULTS_FILE = os.path.join(REPO, f"scratch_compare_results{TAG}.json")
FIGURE_FILE  = os.path.join(REPO, f"scratch_compare{TAG}.png")

# ── Ablation configs ──────────────────────────────────────────────────────────
def _cfg(name, label, w=False, p=False, lf=False):
    return {"name": name, "label": label,
            "use_weight_cv": w, "use_pop_cv": p, "use_lifetime_cv": lf}

CONFIGS = [
    _cfg("baseline",     "Baseline"),
    _cfg("weight_cv",    "WeightCV",     w=True),
    _cfg("pop_cv",       "PopFR-CV",     p=True),
    _cfg("weight_pop",   "Weight+PopCV", w=True, p=True),
    _cfg("lifetime_cv",  "LifetimeCV",   lf=True),       # P1.2 control
]

COLORS = ["#1565C0", "#C62828", "#2E7D32", "#6A1B9A", "#EF6C00"]  # + orange for lifetime

print(f"Device  : {DEVICE}")
print(f"Epochs  : {N_EPOCHS}  |  eval every {EVAL_EVERY}  |  LR={LR}")
print(f"Configs : {[c['label'] for c in CONFIGS]}")


# ── Model factory ─────────────────────────────────────────────────────────────
def make_model() -> nn.Module:
    return create_model(
        "sdt",
        T=4, num_classes=N_CLASSES,
        img_size_h=32, img_size_w=32,
        patch_size=4, embed_dims=256, num_heads=8,
        mlp_ratios=4, depths=2, sr_ratios=1,
        pooling_stat="0011", spike_mode="lif",
        dvs_mode=False, TET=True, in_channels=3,
        qkv_bias=False, drop_rate=0.0,
        drop_path_rate=0.2, drop_block_rate=None,
    ).to(DEVICE)


# ── Data ──────────────────────────────────────────────────────────────────────
# CIFAR-10 and CIFAR-100 share the same (near-identical) image statistics; using
# the CIFAR-10 values for both is a well-established shortcut in the benchmark
# literature.  See e.g. the Spikformer / SDT training recipes.
MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2470, 0.2435, 0.2616)

aa_params = dict(translate_const=int(32 * 0.45), img_mean=tuple(int(c * 255) for c in MEAN))
train_tf = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    rand_augment_transform("rand-m9-n1-mstd0.4-inc1", aa_params),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])
val_tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])

_DSCLS = torchvision.datasets.CIFAR100 if DATASET == "cifar100" else torchvision.datasets.CIFAR10
ds_train_full = _DSCLS(os.path.join(REPO, "data"), train=True,  download=True, transform=train_tf)
ds_val        = _DSCLS(os.path.join(REPO, "data"), train=False, download=True, transform=val_tf)
print(f"Dataset : {DATASET.upper()}  ({N_CLASSES} classes)")

if N_TRAIN_SAMPLES >= len(ds_train_full):
    ds_train = ds_train_full
else:
    _g   = torch.Generator().manual_seed(0)
    _idx = torch.randperm(len(ds_train_full), generator=_g)[:N_TRAIN_SAMPLES].tolist()
    ds_train = torch.utils.data.Subset(ds_train_full, _idx)

_gpu = torch.cuda.is_available()
_nw  = 4 if _gpu else 2
train_loader = torch.utils.data.DataLoader(
    ds_train, batch_size=BATCH_TRAIN, shuffle=True,  num_workers=_nw, pin_memory=_gpu)
val_loader   = torch.utils.data.DataLoader(
    ds_val,   batch_size=BATCH_VAL,   shuffle=False, num_workers=_nw, pin_memory=_gpu)

print(f"Train: {len(ds_train)}  |  Val: {len(ds_val)}  |  workers={_nw}  pin_memory={_gpu}")

mixup_fn = Mixup(
    mixup_alpha=0.5, cutmix_alpha=0.0, prob=1.0,
    switch_prob=0.5, mode="batch",
    label_smoothing=0.1, num_classes=N_CLASSES,
)


# ── LR schedule ───────────────────────────────────────────────────────────────
def get_lr(epoch: int) -> float:
    if epoch < WARMUP_EP:
        return LR * (epoch + 1) / WARMUP_EP
    t = (epoch - WARMUP_EP) / max(1, N_EPOCHS - WARMUP_EP)
    return MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * t))


# ── TET loss ──────────────────────────────────────────────────────────────────
def tet_loss(outputs, labels, ce_fn):
    T_ = outputs.size(0)
    return sum(ce_fn(outputs[t], labels) for t in range(T_)) / T_


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(model: nn.Module) -> dict:
    model.eval()
    ce_fn = nn.CrossEntropyLoss().to(DEVICE)
    total_loss, total_correct, total_n = 0.0, 0, 0
    layer_fr_acc:    dict[str, list] = defaultdict(list)
    layer_pop_cv:    dict[str, list] = defaultdict(list)
    layer_life_cv:   dict[str, list] = defaultdict(list)
    layer_temp_cv:   dict[str, list] = defaultdict(list)

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            hook = {}
            outputs, hook = model(imgs, hook=hook)
            functional.reset_net(model)

            logits_mean = outputs.mean(0)
            loss = ce_fn(logits_mean, labels)
            total_loss    += loss.item() * imgs.size(0)
            total_correct += (logits_mean.argmax(1) == labels).sum().item()
            total_n       += imgs.size(0)

            for k, spikes in hook.items():
                T_   = spikes.shape[0]
                B    = spikes.shape[1]
                flat = spikes.float().view(T_, B, -1)   # (T, B, N)

                # Mean firing rate
                fr = flat.mean(dim=0)                   # (B, N)
                layer_fr_acc[k].append(fr.mean().item())

                # Population CV: std_n / mean_n of firing rates (across neurons, per sample)
                mean_n = fr.mean(dim=1, keepdim=True)
                std_n  = fr.std(dim=1, keepdim=True, unbiased=False) + 1e-8
                layer_pop_cv[k].append((std_n / (mean_n + 1e-8)).mean().item())

                # Lifetime CV: std_b / mean_b of each neuron's firing rate across batch
                fr_flat = fr.view(B, -1)
                mean_b = fr_flat.mean(dim=0) + 1e-8
                std_b  = fr_flat.std(dim=0, unbiased=False) + 1e-8
                layer_life_cv[k].append((std_b / mean_b).mean().item())

                # Temporal CV: std_t / mean_t per neuron
                if T_ >= 2:
                    mean_t = flat.mean(dim=0)
                    std_t  = flat.std(dim=0, unbiased=False) + 1e-8
                    layer_temp_cv[k].append((std_t / (mean_t + 1e-8)).mean().item())

        weight_cv = criterion_v2.compute_weight_cv(model)

    ce_loss  = total_loss / total_n
    acc      = total_correct / total_n * 100.0
    mean_fr  = float(np.mean([np.mean(v) for v in layer_fr_acc.values()]))
    cv_fr    = float(np.mean([np.mean(v) for v in layer_pop_cv.values()]))
    cv_life  = float(np.mean([np.mean(v) for v in layer_life_cv.values()])) if layer_life_cv else 0.0
    temp_cv  = float(np.mean([np.mean(v) for v in layer_temp_cv.values()])) if layer_temp_cv else 0.0

    return {
        "cv_fr":    cv_fr,
        "cv_life":  cv_life,
        "temp_cv":  temp_cv,
        "weight_cv": weight_cv,
        "mean_fr":  mean_fr,
        "ce_loss":  ce_loss,
        "acc":      acc,
        "layer_cv":   {k: float(np.mean(v)) for k, v in layer_pop_cv.items()},
        "layer_lcv":  {k: float(np.mean(v)) for k, v in layer_life_cv.items()},
        "layer_mfr":  {k: float(np.mean(v)) for k, v in layer_fr_acc.items()},
        "layer_tcv":  {k: float(np.mean(v)) for k, v in layer_temp_cv.items()},
    }


# ── Training loop (one config, one epoch) ────────────────────────────────────
def train_one_epoch(model, optimizer, config: dict, epoch: int) -> float:
    model.train()
    ce_fn = nn.CrossEntropyLoss(label_smoothing=0.1).to(DEVICE)
    total, count = 0.0, 0

    apply_pop_cv  = config["use_pop_cv"]      and epoch >= POP_CV_START
    apply_life_cv = config.get("use_lifetime_cv", False) and epoch >= LIFETIME_CV_START
    need_hook     = apply_pop_cv or apply_life_cv

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        imgs, labels = mixup_fn(imgs, labels)
        optimizer.zero_grad()

        hook = {} if need_hook else None
        outputs, hook = model(imgs, hook=hook)

        loss = tet_loss(outputs, labels, ce_fn)

        if config["use_weight_cv"]:
            loss = loss + criterion_v2.weight_cv_loss(model, lambda_weight_cv=LAMBDA_WEIGHT_CV)

        if apply_pop_cv and hook:
            all_spikes = list(hook.values())
            loss = loss + criterion_v2.firing_rate_cv_loss(all_spikes, lambda_cv=LAMBDA_POP_CV)

        if apply_life_cv and hook:
            all_spikes = list(hook.values())
            loss = loss + criterion_v2.lifetime_fr_cv_loss(all_spikes,
                                                           lambda_cv=LAMBDA_LIFETIME_CV)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        functional.reset_net(model)

        total += loss.item() * imgs.size(0)
        count += imgs.size(0)

    return total / count


# ── Main ──────────────────────────────────────────────────────────────────────
METRIC_KEYS = ("cv_fr", "cv_life", "temp_cv", "weight_cv", "mean_fr", "ce_loss", "acc")


def run():
    torch.manual_seed(42)
    base_model = make_model()

    models     = {c["name"]: copy.deepcopy(base_model) for c in CONFIGS}
    optimizers = {c["name"]: torch.optim.AdamW(models[c["name"]].parameters(),
                                                lr=LR, weight_decay=WEIGHT_DECAY)
                  for c in CONFIGS}
    hists = {c["name"]: {k: [] for k in (*METRIC_KEYS, "epoch")} for c in CONFIGS}

    print(f"\n=== Starting ablation: {[c['label'] for c in CONFIGS]} ===")
    total_t0 = time.time()

    for epoch in range(N_EPOCHS):
        lr_now = get_lr(epoch)
        for opt in optimizers.values():
            for pg in opt.param_groups:
                pg["lr"] = lr_now

        t0 = time.time()
        tr_losses = {}
        for c in CONFIGS:
            tr_losses[c["name"]] = train_one_epoch(
                models[c["name"]], optimizers[c["name"]], c, epoch)
        elapsed = time.time() - t0

        do_eval = (epoch + 1) % EVAL_EVERY == 0 or epoch == 0
        if do_eval:
            metrics = {c["name"]: evaluate(models[c["name"]]) for c in CONFIGS}
            ep_num  = epoch + 1
            for c in CONFIGS:
                m = metrics[c["name"]]
                hists[c["name"]]["epoch"].append(ep_num)
                for k in METRIC_KEYS:
                    hists[c["name"]][k].append(m[k])

            header = f"Ep {ep_num:>3}/{N_EPOCHS}  lr={lr_now:.2e}  [{elapsed:.0f}s]"
            for c in CONFIGS:
                m = metrics[c["name"]]
                print(f"  {c['label']:14s} "
                      f"CV_pop={m['cv_fr']:.4f}  CV_L={m['cv_life']:.4f}  "
                      f"CV_t={m['temp_cv']:.4f}  CV_w={m['weight_cv']:.4f}  "
                      f"MFR={m['mean_fr']:.4f}  CE={m['ce_loss']:.4f}  Acc={m['acc']:.2f}%")
            print(header)
        else:
            loss_str = "  ".join(f"{c['label']}={tr_losses[c['name']]:.4f}" for c in CONFIGS)
            print(f"Ep {epoch+1:>3}/{N_EPOCHS}  lr={lr_now:.2e}  [{elapsed:.0f}s]  {loss_str}")

    total_elapsed = (time.time() - total_t0) / 60
    print(f"\nTotal training time: {total_elapsed:.1f} min")

    fin = {c["name"]: evaluate(models[c["name"]]) for c in CONFIGS}
    data = {"hists": hists, "configs": [c["name"] for c in CONFIGS],
            "labels": {c["name"]: c["label"] for c in CONFIGS}}
    for c in CONFIGS:
        data[f"final_{c['name']}_layer_cv"]  = fin[c["name"]]["layer_cv"]
        data[f"final_{c['name']}_layer_lcv"] = fin[c["name"]]["layer_lcv"]
        data[f"final_{c['name']}_layer_mfr"] = fin[c["name"]]["layer_mfr"]
        data[f"final_{c['name']}_layer_tcv"] = fin[c["name"]]["layer_tcv"]

    with open(RESULTS_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Results saved → {RESULTS_FILE}")

    plot(data)


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot(data: dict):
    hists   = data["hists"]
    names   = data["configs"]
    labels  = data["labels"]
    epochs  = hists[names[0]]["epoch"]

    fig = plt.figure(figsize=(22, 16))
    fig.suptitle(
        "Spike-Driven Transformer Ablation: Baseline / WeightCV / PopFR-CV / Weight+PopCV / LifetimeCV\n"
        f"({DATASET.upper()}, {N_EPOCHS} epochs, {N_TRAIN_SAMPLES} train / 10000 val, AdamW lr={LR})",
        fontsize=13, fontweight="bold", y=1.00)

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.35)

    def _kw(i):
        return dict(color=COLORS[i], lw=2, marker="o", ms=3, label=labels[names[i]], alpha=0.9)

    def _panel(ax, key, title, ylabel, hi=True):
        for i, nm in enumerate(names):
            ax.plot(epochs, hists[nm][key], **_kw(i))
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        return ax

    _panel(fig.add_subplot(gs[0, 0]), "cv_fr",
           "Population CV(Firing Rate)\n(std_n / mean_n across neurons)", "CV", hi=True)
    _panel(fig.add_subplot(gs[0, 1]), "cv_life",
           "Lifetime CV per Neuron\n(std_b / mean_b across batch)", "CV", hi=True)
    _panel(fig.add_subplot(gs[0, 2]), "weight_cv",
           "Weight CV\n(std|w| / mean|w| per filter)", "CV", hi=True)
    _panel(fig.add_subplot(gs[1, 0]), "mean_fr",
           "Mean Firing Rate (↓=efficient)", "MFR", hi=False)
    _panel(fig.add_subplot(gs[1, 1]), "ce_loss",
           "CE Loss (↓=better)", "CE Loss", hi=False)
    _panel(fig.add_subplot(gs[1, 2]), "acc",
           "Top-1 Accuracy (%)", "Acc %", hi=True)

    # Per-layer population CV bar chart (final epoch)
    ax_bar = fig.add_subplot(gs[2, :])
    all_layers = sorted(set().union(*[
        set(data.get(f"final_{nm}_layer_cv", {}).keys()) for nm in names]))
    short = [k.replace("MS_SSA_Conv", "SSA").replace("MS_MLP_Conv", "MLP")
              .replace("MS_SPS", "SPS").replace("_lif", "").replace("_first", "")
             for k in all_layers]
    x = np.arange(len(all_layers))
    w = 0.8 / len(names)
    offsets = np.linspace(-(0.8 - w) / 2, (0.8 - w) / 2, len(names))
    for i, nm in enumerate(names):
        vals = [data.get(f"final_{nm}_layer_cv", {}).get(k, 0) for k in all_layers]
        ax_bar.bar(x + offsets[i], vals, w, label=labels[nm], color=COLORS[i], alpha=0.85)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
    ax_bar.set_title(f"Per-layer Population CV(Firing Rate) — epoch {epochs[-1]}", fontsize=11)
    ax_bar.set_ylabel("CV"); ax_bar.legend(fontsize=9); ax_bar.grid(alpha=0.3, axis="y")

    # Summary box
    n_ep = len(epochs) - 1
    lines = [f"Epoch {epochs[n_ep]} results  (Δ vs Baseline):"]
    for key, label, hi in [
        ("cv_fr",    "Pop CV(FR)   ", True),
        ("cv_life",  "Lifetime CV  ", True),
        ("temp_cv",  "Temporal CV  ", True),
        ("weight_cv","Weight CV    ", True),
        ("mean_fr",  "Mean FR      ", False),
        ("ce_loss",  "CE Loss      ", False),
        ("acc",      "Acc %        ", True),
    ]:
        vb = hists[names[0]][key][n_ep]
        row = f"  {label}  Base={vb:.4f}"
        for nm in names[1:]:
            vc = hists[nm][key][n_ep]
            d  = vc - vb
            ok = (hi and d > 0) or (not hi and d < 0)
            row += f"  {labels[nm]}={vc:.4f}(Δ{d:+.4f}{'✓' if ok else ''})"
        lines.append(row)
    lines.append(f"\nlambda_weight={LAMBDA_WEIGHT_CV}  lambda_pop={LAMBDA_POP_CV}  "
                 f"lambda_life={LAMBDA_LIFETIME_CV}  pop_start=life_start=ep{POP_CV_START}")
    fig.text(0.01, -0.02, "\n".join(lines), fontsize=8, family="monospace",
             verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85))

    plt.savefig(FIGURE_FILE, dpi=150, bbox_inches="tight")
    print(f"Figure saved → {FIGURE_FILE}")
    plt.close()

    # Console table
    print("\n" + "=" * 90)
    print(f"  {'Metric':<18}" + "".join(f"  {labels[nm]:>14}" for nm in names))
    print("-" * 90)
    for key, label, hi in [
        ("cv_fr",    "Pop CV(FR)(↑)",  True),
        ("cv_life",  "Lifetime CV(↑)", True),
        ("temp_cv",  "Temporal CV(↑)", True),
        ("weight_cv","Weight CV(↑)",   True),
        ("mean_fr",  "Mean FR(↓)",     False),
        ("ce_loss",  "CE Loss(↓)",     False),
        ("acc",      "Top-1 Acc%(↑)",  True),
    ]:
        row = f"  {label:<18}"
        vb  = hists[names[0]][key][n_ep]
        for nm in names:
            vc = hists[nm][key][n_ep]
            d  = vc - vb
            ok = (hi and d > 0) or (not hi and d < 0)
            mark = "✓" if (nm != names[0] and ok) else " "
            row += f"  {vc:>12.4f}{mark}"
        print(row)
    print("=" * 90)


def _cache_is_current(data: dict) -> bool:
    """Cached JSON must contain all 5 configs and the new cv_life metric."""
    names = data.get("configs", [])
    if set(names) != {c["name"] for c in CONFIGS}:
        return False
    hists = data.get("hists", {})
    for n in names:
        if "cv_life" not in hists.get(n, {}):
            return False
    return True


def run_with_cache():
    if os.path.exists(RESULTS_FILE) and "--force" not in sys.argv:
        with open(RESULTS_FILE) as f:
            data = json.load(f)
        if _cache_is_current(data):
            print(f"Loading cached results from {RESULTS_FILE}")
            plot(data); return
        print(f"Cached {RESULTS_FILE} is stale (pre-LifetimeCV schema) — retraining.")
    run()


if __name__ == "__main__":
    run_with_cache()
