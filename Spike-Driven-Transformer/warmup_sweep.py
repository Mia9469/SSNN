"""
warmup_sweep.py
===============
P3.6: Warmup sensitivity for PopFR-CV on SDT / CIFAR-10.

Hypothesis (dead-neuron trap):
    Applying PopFR-CV earlier in training should cause more neurons to be
    driven to fr = 0 before the task loss can pin them active, producing a
    larger "dead-neuron ratio" and a bigger Pop-CV collapse at convergence.
    A sufficiently long warmup should mitigate or eliminate the trap.

Sweep: pop_cv_start ∈ {0, 10, 20, 40} epochs, with λ_pop fixed at 5e-4.

For each run we record at the final epoch:
  acc            – top-1 accuracy
  mean_fr        – mean firing rate (efficiency)
  cv_fr          – population CV(firing rate)
  dead_ratio     – fraction of neurons with per-sample firing rate below
                   DEAD_THRESHOLD (default 1e-3), averaged across all LIF
                   hooks and the validation set
  dead_ratio_hard – same but threshold = 0 (strictly silent neurons)

Output: `warmup_sweep_results.json` + 4-panel summary figure.

Runtime: 4 starts × 100 ep ≈ 6.7 GPU-hours on a single 3090.  Pass
`--epochs 50` for a faster sanity-check.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from timm.data.auto_augment import rand_augment_transform
from timm.data.mixup import Mixup
from timm.models import create_model

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from model import sdt  # noqa
from spikingjelly.clock_driven import functional
import criterion_v2

# ── CLI ───────────────────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser()
_parser.add_argument("--epochs", type=int, default=100)
_parser.add_argument("--lambda-pop", type=float, default=5e-4)
_parser.add_argument("--dead-threshold", type=float, default=1e-3)
_parser.add_argument("--plot-only", action="store_true")
CLI = _parser.parse_args()

DEVICE = (
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)
N_EPOCHS     = CLI.epochs
LAMBDA_POP   = CLI.lambda_pop
DEAD_THR     = CLI.dead_threshold
BATCH_TRAIN  = 64
BATCH_VAL    = 128
LR           = 3e-4
MIN_LR       = 1e-5
LR_WARMUP    = min(20, N_EPOCHS // 5)
WEIGHT_DECAY = 0.06

WARMUP_GRID = [0, 10, 20, 40]
WARMUP_GRID = [w for w in WARMUP_GRID if w < N_EPOCHS]  # drop values ≥ epochs

RESULTS_FILE = os.path.join(REPO, "warmup_sweep_results.json")
FIGURE_FILE  = os.path.join(REPO, "warmup_sweep.png")


# ── Data & LR (same recipe as lambda_sweep) ──────────────────────────────────
MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2470, 0.2435, 0.2616)
aa_params = dict(translate_const=int(32 * 0.45),
                 img_mean=tuple(int(c * 255) for c in MEAN))
train_tf = T.Compose([
    T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
    rand_augment_transform("rand-m9-n1-mstd0.4-inc1", aa_params),
    T.ToTensor(), T.Normalize(MEAN, STD),
])
val_tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])

ds_train = torchvision.datasets.CIFAR10(os.path.join(REPO, "data"),
                                        train=True,  download=True, transform=train_tf)
ds_val   = torchvision.datasets.CIFAR10(os.path.join(REPO, "data"),
                                        train=False, download=True, transform=val_tf)

_gpu = torch.cuda.is_available()
_nw  = 4 if _gpu else 2
train_loader = torch.utils.data.DataLoader(ds_train, batch_size=BATCH_TRAIN,
                                           shuffle=True,  num_workers=_nw, pin_memory=_gpu)
val_loader   = torch.utils.data.DataLoader(ds_val,   batch_size=BATCH_VAL,
                                           shuffle=False, num_workers=_nw, pin_memory=_gpu)

mixup_fn = Mixup(mixup_alpha=0.5, cutmix_alpha=0.0, prob=1.0,
                 switch_prob=0.5, mode="batch",
                 label_smoothing=0.1, num_classes=10)


def make_model() -> nn.Module:
    return create_model(
        "sdt", T=4, num_classes=10, img_size_h=32, img_size_w=32,
        patch_size=4, embed_dims=256, num_heads=8, mlp_ratios=4, depths=2,
        sr_ratios=1, pooling_stat="0011", spike_mode="lif",
        dvs_mode=False, TET=True, in_channels=3,
        qkv_bias=False, drop_rate=0.0, drop_path_rate=0.2, drop_block_rate=None,
    ).to(DEVICE)


def get_lr(epoch):
    if epoch < LR_WARMUP:
        return LR * (epoch + 1) / LR_WARMUP
    t = (epoch - LR_WARMUP) / max(1, N_EPOCHS - LR_WARMUP)
    return MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * t))


def tet_loss(outputs, labels, ce_fn):
    return sum(ce_fn(outputs[t], labels) for t in range(outputs.size(0))) / outputs.size(0)


# ── Evaluation with dead-neuron counting ─────────────────────────────────────
@torch.no_grad()
def evaluate(model, dead_threshold: float) -> dict:
    model.eval()
    ce_fn = nn.CrossEntropyLoss().to(DEVICE)
    total_loss, total_correct, total_n = 0.0, 0, 0
    layer_fr, layer_pop = defaultdict(list), defaultdict(list)
    layer_dead_soft, layer_dead_hard = defaultdict(list), defaultdict(list)

    for imgs, labels in val_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        hook = {}
        outputs, hook = model(imgs, hook=hook)
        functional.reset_net(model)
        lm = outputs.mean(0)
        total_loss    += ce_fn(lm, labels).item() * imgs.size(0)
        total_correct += (lm.argmax(1) == labels).sum().item()
        total_n       += imgs.size(0)

        for k, s in hook.items():
            T_, B = s.shape[0], s.shape[1]
            flat = s.float().view(T_, B, -1)
            fr = flat.mean(dim=0)                                 # (B, N)
            layer_fr[k].append(fr.mean().item())

            mean_n = fr.mean(dim=1, keepdim=True) + 1e-8
            std_n  = fr.std(dim=1, keepdim=True, unbiased=False) + 1e-8
            layer_pop[k].append((std_n / mean_n).mean().item())

            # Dead-neuron definitions:
            #   soft: per-sample fr below dead_threshold (near-silent)
            #   hard: per-sample fr exactly 0 (fully silent in this batch)
            layer_dead_soft[k].append((fr < dead_threshold).float().mean().item())
            layer_dead_hard[k].append((fr == 0).float().mean().item())

    weight_cv = criterion_v2.compute_weight_cv(model)
    ce_loss = total_loss / total_n
    acc     = total_correct / total_n * 100.0
    mean_fr = float(np.mean([np.mean(v) for v in layer_fr.values()]))
    cv_fr   = float(np.mean([np.mean(v) for v in layer_pop.values()]))
    dead_s  = float(np.mean([np.mean(v) for v in layer_dead_soft.values()]))
    dead_h  = float(np.mean([np.mean(v) for v in layer_dead_hard.values()]))
    return {
        "acc": acc, "ce_loss": ce_loss, "mean_fr": mean_fr,
        "cv_fr": cv_fr, "weight_cv": weight_cv,
        "dead_ratio": dead_s, "dead_ratio_hard": dead_h,
        "ice": acc / (mean_fr + 1e-8),
    }


# ── Single warmup-start run ───────────────────────────────────────────────────
def train_run(pop_cv_start: int) -> dict:
    torch.manual_seed(0); np.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    model = make_model()
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    ce_fn = nn.CrossEntropyLoss(label_smoothing=0.1).to(DEVICE)
    t0 = time.time()
    dead_trace = []

    for epoch in range(N_EPOCHS):
        for pg in opt.param_groups:
            pg["lr"] = get_lr(epoch)
        model.train()
        apply_pop = epoch >= pop_cv_start
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            imgs, labels = mixup_fn(imgs, labels)
            opt.zero_grad()
            hook = {} if apply_pop else None
            outputs, hook = model(imgs, hook=hook)
            loss = tet_loss(outputs, labels, ce_fn)
            if apply_pop and hook:
                loss = loss + criterion_v2.firing_rate_cv_loss(list(hook.values()),
                                                               lambda_cv=LAMBDA_POP)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            functional.reset_net(model)

        if (epoch + 1) % 10 == 0 or epoch == N_EPOCHS - 1:
            m = evaluate(model, DEAD_THR)
            m["epoch"] = epoch + 1
            dead_trace.append(m)
            print(f"    ep {epoch+1}/{N_EPOCHS}  Acc={m['acc']:.2f}  "
                  f"MFR={m['mean_fr']:.4f}  CV_pop={m['cv_fr']:.3f}  "
                  f"dead={m['dead_ratio']:.3f}  dead0={m['dead_ratio_hard']:.3f}")

    final = evaluate(model, DEAD_THR)
    final["runtime_min"] = (time.time() - t0) / 60
    final["trace"] = dead_trace
    final["pop_cv_start"] = pop_cv_start
    return final


# ── Driver ────────────────────────────────────────────────────────────────────
def run_sweep() -> dict:
    results = {"epochs": N_EPOCHS, "lambda_pop": LAMBDA_POP,
               "dead_threshold": DEAD_THR, "runs": []}
    for w in WARMUP_GRID:
        print(f"\n=== pop_cv_start = {w} ===")
        r = train_run(w)
        results["runs"].append(r)
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)
    return results


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot(results: dict):
    runs = sorted(results.get("runs", []), key=lambda r: r["pop_cv_start"])
    if not runs:
        print("no runs to plot"); return

    starts = [r["pop_cv_start"] for r in runs]
    accs   = [r["acc"]          for r in runs]
    mfrs   = [r["mean_fr"]      for r in runs]
    cvs    = [r["cv_fr"]        for r in runs]
    deads  = [r["dead_ratio"]   for r in runs]
    ices   = [r["ice"]          for r in runs]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f"Warmup sensitivity — PopFR-CV on SDT / CIFAR-10  "
                 f"(λ_pop={LAMBDA_POP}, dead threshold={DEAD_THR:g})",
                 fontsize=13, fontweight="bold")

    def _bar(ax, xs, ys, title, ylabel, color, annotate_fmt="{:.3f}"):
        bars = ax.bar(range(len(xs)), ys, color=color, edgecolor="black", alpha=0.85)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels([f"start={x}" for x in xs])
        ax.set_ylabel(ylabel); ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.3, axis="y")
        for b, v in zip(bars, ys):
            ax.text(b.get_x() + b.get_width()/2, v, annotate_fmt.format(v),
                    ha="center", va="bottom", fontsize=8)

    _bar(axes[0, 0], starts, accs,  "Top-1 Accuracy ↑", "Acc %",      "#1565C0", "{:.2f}")
    _bar(axes[0, 1], starts, cvs,   "Population CV(FR)", "CV",         "#2E7D32", "{:.3f}")
    _bar(axes[0, 2], starts, deads, "Dead-neuron ratio ↓", "fraction",  "#C62828", "{:.3f}")
    _bar(axes[1, 0], starts, mfrs,  "Mean firing rate ↓", "MFR",        "#6A1B9A", "{:.4f}")
    _bar(axes[1, 1], starts, ices,  "ICE = Acc / MFR ↑", "ICE",         "#EF6C00", "{:.1f}")

    # Trace panel: CV over epochs for each warmup setting
    ax = axes[1, 2]
    for r, color in zip(runs, ["#1565C0", "#C62828", "#2E7D32", "#6A1B9A"]):
        trace = r.get("trace", [])
        if not trace: continue
        es = [t["epoch"] for t in trace]; vs = [t["cv_fr"] for t in trace]
        ax.plot(es, vs, color=color, marker="o", ms=3,
                label=f"start={r['pop_cv_start']}")
        ax.axvline(r["pop_cv_start"], color=color, ls="--", alpha=0.3)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Pop CV(FR)")
    ax.set_title("CV-collapse trajectory"); ax.grid(alpha=0.3); ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGURE_FILE, dpi=150, bbox_inches="tight")
    print(f"Figure → {FIGURE_FILE}")
    plt.close()


def main():
    if CLI.plot_only:
        if not os.path.exists(RESULTS_FILE):
            sys.exit(f"no {RESULTS_FILE} found")
        with open(RESULTS_FILE) as f:
            plot(json.load(f))
        return
    results = run_sweep()
    plot(results)


if __name__ == "__main__":
    main()
