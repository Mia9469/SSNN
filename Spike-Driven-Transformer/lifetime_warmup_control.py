"""
lifetime_warmup_control.py
==========================
P1.2 control: does LifetimeCV collapse when the onset is delayed?

In the main 5-way SDT ablation (scratch_train_compare.py), the LifetimeCV
condition collapses to ~10% (CIFAR-10) / ~1% (CIFAR-100) accuracy with the
default warmup of 20 epochs.  The open question for the paper is whether this
is (a) a fundamental incompatibility between lifetime-CV gradients and
surrogate-gradient SNNs (the "biophysical ceiling" story) or (b) an
onset-timing issue that later warmup can fix (the "dead-neuron trap" story —
same mechanism that bit PopFR-CV at warmup=0).

This script sweeps the LifetimeCV onset epoch over {20, 40, 60, 80} while
holding everything else fixed, trained for 120 epochs × 1 seed.  If collapse
disappears at a later onset, (b) wins and the paper's unified "dead-neuron
trap" story covers both PopCV-at-start=0 and LifetimeCV-at-start=20.  If
collapse persists at every onset, (a) wins and LifetimeCV is shown to be
fundamentally incompatible with binary-spike SNNs.

Outputs
  lifetime_warmup_control_results.json
  lifetime_warmup_control.png       (ICE / Acc / mean_fr vs onset epoch)
"""

from __future__ import annotations

import argparse
import copy
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

from model import sdt       # noqa: registers 'sdt' in the timm registry
from spikingjelly.clock_driven import functional
import criterion_v2

# ── CLI ───────────────────────────────────────────────────────────────────────
_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--epochs", type=int, default=120)
_ap.add_argument("--onsets", type=str, default="20,40,60,80",
                 help="comma-separated list of LifetimeCV onset epochs")
_ap.add_argument("--lambda-life", type=float, default=0.0005)
_ap.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
_ap.add_argument("--tag", type=str, default="")
_ap.add_argument("--plot-only", action="store_true")
CLI, _ = _ap.parse_known_args()

DATASET    = CLI.dataset
N_CLASSES  = 100 if DATASET == "cifar100" else 10
TAG        = f"_{CLI.tag}" if CLI.tag else ""
ONSETS     = [int(x) for x in CLI.onsets.split(",")]

DEVICE = (
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)
N_EPOCHS       = CLI.epochs
BATCH_TRAIN    = 64
BATCH_VAL      = 128
LR             = 3e-4
MIN_LR         = 1e-5
LR_WARMUP      = 20
WEIGHT_DECAY   = 0.06
LAMBDA_LIFE_CV = CLI.lambda_life

RESULTS_FILE = os.path.join(REPO, f"lifetime_warmup_control_results{TAG}.json")
FIGURE_FILE  = os.path.join(REPO, f"lifetime_warmup_control{TAG}.png")

print(f"Device  : {DEVICE}")
print(f"Dataset : {DATASET.upper()} ({N_CLASSES} classes)")
print(f"Epochs  : {N_EPOCHS}  |  LifetimeCV onsets: {ONSETS}  |  λ_life={LAMBDA_LIFE_CV}")

# ── Data ──────────────────────────────────────────────────────────────────────
MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2470, 0.2435, 0.2616)
aa_params = dict(translate_const=int(32 * 0.45),
                 img_mean=tuple(int(c * 255) for c in MEAN))
train_tf = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    rand_augment_transform("rand-m9-n1-mstd0.4-inc1", aa_params),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])
val_tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])

_DSCLS = torchvision.datasets.CIFAR100 if DATASET == "cifar100" else torchvision.datasets.CIFAR10
ds_train = _DSCLS(os.path.join(REPO, "data"), train=True,  download=True, transform=train_tf)
ds_val   = _DSCLS(os.path.join(REPO, "data"), train=False, download=True, transform=val_tf)

_gpu = torch.cuda.is_available()
_nw  = 4 if _gpu else 2
train_loader = torch.utils.data.DataLoader(ds_train, batch_size=BATCH_TRAIN,
                                           shuffle=True,  num_workers=_nw, pin_memory=_gpu)
val_loader   = torch.utils.data.DataLoader(ds_val,   batch_size=BATCH_VAL,
                                           shuffle=False, num_workers=_nw, pin_memory=_gpu)

mixup_fn = Mixup(mixup_alpha=0.5, cutmix_alpha=0.0, prob=1.0,
                 switch_prob=0.5, mode="batch",
                 label_smoothing=0.1, num_classes=N_CLASSES)


def make_model() -> nn.Module:
    return create_model(
        "sdt",
        T=4, num_classes=N_CLASSES, img_size_h=32, img_size_w=32,
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


@torch.no_grad()
def evaluate(model):
    model.eval()
    ce_fn = nn.CrossEntropyLoss().to(DEVICE)
    total_loss, total_correct, total_n = 0.0, 0, 0
    layer_fr, layer_pop, layer_life = defaultdict(list), defaultdict(list), defaultdict(list)
    dead_counts = defaultdict(list)

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
            fr   = flat.mean(dim=0)                             # (B, N)
            layer_fr[k].append(fr.mean().item())
            m = fr.mean(dim=1, keepdim=True) + 1e-8
            sd = fr.std(dim=1, keepdim=True, unbiased=False) + 1e-8
            layer_pop[k].append((sd / m).mean().item())
            per_neuron = fr.mean(dim=0)                         # (N,)
            mean_b = per_neuron + 1e-8
            std_b  = fr.std(dim=0, unbiased=False) + 1e-8
            layer_life[k].append((std_b / mean_b).mean().item())
            dead_counts[k].append(float((per_neuron < 1e-6).float().mean().item()))

    weight_cv = criterion_v2.compute_weight_cv(model)
    acc     = total_correct / total_n * 100.0
    mean_fr = float(np.mean([np.mean(v) for v in layer_fr.values()]))
    cv_fr   = float(np.mean([np.mean(v) for v in layer_pop.values()]))
    cv_life = float(np.mean([np.mean(v) for v in layer_life.values()]))
    dead    = float(np.mean([np.mean(v) for v in dead_counts.values()]))
    return {"acc": acc, "ce_loss": total_loss / total_n,
            "mean_fr": mean_fr, "cv_fr": cv_fr, "cv_life": cv_life,
            "weight_cv": weight_cv, "dead_ratio": dead,
            "ice": acc / (mean_fr + 1e-8)}


def train_one_run(onset: int) -> dict:
    torch.manual_seed(0); np.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    model = make_model()
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    ce_fn = nn.CrossEntropyLoss(label_smoothing=0.1).to(DEVICE)

    history = {"epoch": [], "acc": [], "mean_fr": [], "cv_life": [], "dead_ratio": [], "ice": []}
    t0 = time.time()
    for epoch in range(N_EPOCHS):
        lr_now = get_lr(epoch)
        for pg in opt.param_groups:
            pg["lr"] = lr_now
        model.train()
        apply_life = epoch >= onset
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            imgs, labels = mixup_fn(imgs, labels)
            opt.zero_grad()
            hook = {} if apply_life else None
            outputs, hook = model(imgs, hook=hook)
            loss = tet_loss(outputs, labels, ce_fn)
            if apply_life and hook:
                loss = loss + criterion_v2.lifetime_fr_cv_loss(list(hook.values()),
                                                               lambda_cv=LAMBDA_LIFE_CV)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            functional.reset_net(model)

        if (epoch + 1) % 10 == 0 or epoch == N_EPOCHS - 1:
            m = evaluate(model)
            history["epoch"].append(epoch + 1)
            for k in ("acc", "mean_fr", "cv_life", "dead_ratio", "ice"):
                history[k].append(m[k])
            print(f"  [onset={onset}] ep {epoch+1}/{N_EPOCHS}  "
                  f"Acc={m['acc']:.2f}  MFR={m['mean_fr']:.4f}  "
                  f"CV_L={m['cv_life']:.3f}  dead={m['dead_ratio']*100:.1f}%  "
                  f"ICE={m['ice']:.1f}")

    final = evaluate(model)
    final["runtime_min"] = (time.time() - t0) / 60
    final["onset"] = onset
    final["history"] = history
    return final


def run_sweep() -> dict:
    results = {"onsets": ONSETS, "epochs": N_EPOCHS,
               "lambda_life": LAMBDA_LIFE_CV, "dataset": DATASET,
               "runs": []}
    for onset in ONSETS:
        print(f"\n=== LifetimeCV onset = {onset} ===")
        m = train_one_run(onset)
        results["runs"].append(m)
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  final: Acc={m['acc']:.2f}  MFR={m['mean_fr']:.4f}  "
              f"ICE={m['ice']:.1f}  dead={m['dead_ratio']*100:.1f}%  "
              f"({m['runtime_min']:.1f} min)")
    return results


def plot(results: dict):
    runs = sorted(results["runs"], key=lambda r: r["onset"])
    onsets = [r["onset"]      for r in runs]
    accs   = [r["acc"]        for r in runs]
    mfrs   = [r["mean_fr"]    for r in runs]
    ices   = [r["ice"]        for r in runs]
    dead   = [r["dead_ratio"] for r in runs]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    fig.suptitle(f"LifetimeCV onset control — SDT / {results.get('dataset','cifar10').upper()} "
                 f"({results['epochs']} ep, λ={results['lambda_life']})",
                 fontsize=11, fontweight="bold")

    def _bar(ax, vals, title, ylabel, color):
        x = np.arange(len(onsets))
        bars = ax.bar(x, vals, color=color, edgecolor="black", lw=0.6)
        ax.set_xticks(x); ax.set_xticklabels([f"onset={o}" for o in onsets])
        ax.set_title(title, fontsize=10); ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        for xi, b, v in zip(x, bars, vals):
            ax.text(xi, b.get_height(), f"{v:.2f}" if v < 100 else f"{v:.1f}",
                    ha="center", va="bottom", fontsize=8)
        return bars

    _bar(axes[0], accs, "Top-1 Accuracy (%)",  "Acc", "#2E7D32")
    _bar(axes[1], ices, "ICE = Acc / MFR",     "ICE", "#1565C0")
    _bar(axes[2], [d * 100 for d in dead],
         "Dead-neuron ratio (%)",              "Dead %", "#C62828")

    # Mark collapse threshold (acc < 20%) on accuracy panel
    axes[0].axhline(20, color="#C62828", ls="--", lw=0.8, alpha=0.6)
    axes[0].text(len(onsets) - 0.5, 20, " collapse", color="#C62828",
                 fontsize=7, va="bottom", ha="right")

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
