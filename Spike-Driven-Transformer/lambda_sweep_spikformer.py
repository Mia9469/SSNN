"""
lambda_sweep_spikformer.py
==========================
P3.5b: λ sensitivity sweep for CV regularisation on *Spikformer* / CIFAR-10.

A direct analogue of ``lambda_sweep.py`` (which sweeps SDT) — lets us check
whether the ICE-vs-Acc Pareto frontier and the optimal λ band transfer across
spiking attention architectures.

Motivation
----------
The 1-seed P2.4 run showed that reusing SDT's λ_weight=1e-3 on Spikformer costs
~5pp accuracy while still boosting ICE 56%.  That suggests λ is too large for
Spikformer's full Q·K^T attention (more weights to regularise than K⊙V).  This
sweep finds Spikformer's own knee.

Families (same grids as the SDT sweep for direct comparability):
  A — WeightCV only : λ_weight ∈ {0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2}
  B — PopFR-CV only : λ_pop    ∈ {0, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3}

Usage
-----
    python lambda_sweep_spikformer.py                  # full 100-ep sweep + plot
    python lambda_sweep_spikformer.py --epochs 60      # faster
    python lambda_sweep_spikformer.py --family weight  # only family A
    python lambda_sweep_spikformer.py --family pop     # only family B
    python lambda_sweep_spikformer.py --plot-only      # re-render plot from JSON

Runtime estimate: 11 runs × ~50 min (100 ep) = ~9 GPU-hours on a single 3090.
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

import criterion_v2
import spikformer_model  # noqa: F401 — registers 'spikformer' in the timm registry
from spikingjelly.clock_driven import functional

# ── CLI ───────────────────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser()
_parser.add_argument("--epochs", type=int, default=100)
_parser.add_argument("--family", choices=["weight", "pop", "both"], default="both")
_parser.add_argument("--plot-only", action="store_true")
CLI = _parser.parse_args()

DEVICE = (
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)
N_EPOCHS        = CLI.epochs
BATCH_TRAIN     = 64
BATCH_VAL       = 128
LR              = 3e-4
MIN_LR          = 1e-5
WARMUP_EP       = min(20, N_EPOCHS // 5)
WEIGHT_DECAY    = 0.06
POP_CV_START    = WARMUP_EP

LAMBDA_WEIGHT_GRID = [0.0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
LAMBDA_POP_GRID    = [0.0, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3]

RESULTS_FILE = os.path.join(REPO, "lambda_sweep_spikformer_results.json")
FIGURE_FILE  = os.path.join(REPO, "lambda_sweep_spikformer.png")


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
    # Same dims as spikformer_compare.py for direct comparability.
    return create_model(
        "spikformer",
        T=4, num_classes=10, img_size_h=32, img_size_w=32,
        patch_size=4, embed_dims=256, num_heads=8, mlp_ratios=4, depths=2,
        pooling_stat="0011", spike_mode="lif",
        in_channels=3, TET=True,
    ).to(DEVICE)


def get_lr(epoch):
    if epoch < WARMUP_EP:
        return LR * (epoch + 1) / WARMUP_EP
    t = (epoch - WARMUP_EP) / max(1, N_EPOCHS - WARMUP_EP)
    return MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * t))


def tet_loss(outputs, labels, ce_fn):
    return sum(ce_fn(outputs[t], labels) for t in range(outputs.size(0))) / outputs.size(0)


# ── Evaluation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model):
    model.eval()
    ce_fn = nn.CrossEntropyLoss().to(DEVICE)
    total_loss, total_correct, total_n = 0.0, 0, 0
    layer_fr, layer_pop = defaultdict(list), defaultdict(list)

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
            fr = flat.mean(dim=0)
            layer_fr[k].append(fr.mean().item())
            layer_pop[k].append((fr.std(dim=1, unbiased=False) / (fr.mean(dim=1) + 1e-8)).mean().item())

    weight_cv = criterion_v2.compute_weight_cv(model)
    ce_loss = total_loss / total_n
    acc     = total_correct / total_n * 100.0
    mean_fr = float(np.mean([np.mean(v) for v in layer_fr.values()]))
    cv_fr   = float(np.mean([np.mean(v) for v in layer_pop.values()]))

    return {"acc": acc, "ce_loss": ce_loss, "mean_fr": mean_fr,
            "cv_fr": cv_fr, "weight_cv": weight_cv,
            "ice": acc / (mean_fr + 1e-8)}


# ── Training ──────────────────────────────────────────────────────────────────
def train_one_run(lambda_weight: float, lambda_pop: float) -> dict:
    torch.manual_seed(0); np.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    model = make_model()
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    ce_fn = nn.CrossEntropyLoss(label_smoothing=0.1).to(DEVICE)

    apply_pop = lambda_pop > 0.0
    t0 = time.time()
    for epoch in range(N_EPOCHS):
        lr_now = get_lr(epoch)
        for pg in opt.param_groups:
            pg["lr"] = lr_now
        model.train()
        apply_pop_now = apply_pop and epoch >= POP_CV_START
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            imgs, labels = mixup_fn(imgs, labels)
            opt.zero_grad()
            hook = {} if apply_pop_now else None
            outputs, hook = model(imgs, hook=hook)
            loss = tet_loss(outputs, labels, ce_fn)
            if lambda_weight > 0.0:
                loss = loss + criterion_v2.weight_cv_loss(model, lambda_weight_cv=lambda_weight)
            if apply_pop_now and hook:
                loss = loss + criterion_v2.firing_rate_cv_loss(list(hook.values()),
                                                               lambda_cv=lambda_pop)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            functional.reset_net(model)

        if (epoch + 1) % 20 == 0 or epoch == N_EPOCHS - 1:
            m = evaluate(model)
            print(f"    ep {epoch+1}/{N_EPOCHS}  Acc={m['acc']:.2f}  "
                  f"MFR={m['mean_fr']:.4f}  CV_pop={m['cv_fr']:.3f}  "
                  f"CV_w={m['weight_cv']:.3f}  ICE={m['ice']:.1f}")

    final = evaluate(model)
    final["runtime_min"] = (time.time() - t0) / 60
    return final


# ── Sweep driver ──────────────────────────────────────────────────────────────
def run_sweep() -> dict:
    results: dict = {"family_weight": [], "family_pop": [],
                     "epochs": N_EPOCHS, "model": "spikformer"}

    if CLI.family in ("weight", "both"):
        print("\n=== Family A: WeightCV sweep (Spikformer) ===")
        for lam in LAMBDA_WEIGHT_GRID:
            print(f"\n  λ_weight = {lam:.0e}")
            m = train_one_run(lambda_weight=lam, lambda_pop=0.0)
            m["lambda"] = lam
            results["family_weight"].append(m)
            with open(RESULTS_FILE, "w") as f:
                json.dump(results, f, indent=2)

    if CLI.family in ("pop", "both"):
        print("\n=== Family B: PopFR-CV sweep (Spikformer) ===")
        for lam in LAMBDA_POP_GRID:
            print(f"\n  λ_pop = {lam:.0e}")
            m = train_one_run(lambda_weight=0.0, lambda_pop=lam)
            m["lambda"] = lam
            results["family_pop"].append(m)
            with open(RESULTS_FILE, "w") as f:
                json.dump(results, f, indent=2)

    return results


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot(results: dict):
    """Pareto plot: one panel per CV family.

    Readability choices:
      * scatter only (no λ-ordered connecting line → no zigzag)
      * shared y-axis across panels so ICE magnitudes compare directly
      * horizontal dashed line at family baseline ICE (λ=0)
      * optimum marked with a star
      * ICE < baseline → tagged "collapse" (red tint annotation box)
      * marker size & colour encode |λ| (log scale) so reading order is clear
    """
    families = [
        ("family_weight", "WeightCV only: λ_w",  "#C62828"),
        ("family_pop",    "PopFR-CV only: λ_p",  "#2E7D32"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    fig.suptitle(f"λ sensitivity sweep on Spikformer / CIFAR-10 "
                 f"({results.get('epochs', N_EPOCHS)} epochs, 1 seed)",
                 fontsize=13, fontweight="bold")

    # Unified y-axis range from both families, padded
    all_ice = [e["ice"] for fam in ("family_weight", "family_pop")
               for e in results.get(fam, [])]
    if all_ice:
        ice_lo = min(all_ice) - 30
        ice_hi = max(all_ice) + 30
    else:
        ice_lo, ice_hi = 0, 1500

    for (key, title, color), ax in zip(families, axes):
        entries = results.get(key, [])
        if not entries:
            ax.text(0.5, 0.5, f"(no data for {key})", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(title); continue

        entries_sorted = sorted(entries, key=lambda d: d["lambda"])
        accs = np.array([e["acc"]    for e in entries_sorted])
        ices = np.array([e["ice"]    for e in entries_sorted])
        lams = np.array([e["lambda"] for e in entries_sorted])

        # Baseline ICE = the λ=0 run
        base_mask = lams == 0
        base_ice  = float(ices[base_mask][0]) if base_mask.any() else None
        best_idx  = int(np.argmax(ices))

        # Marker size by |λ| (smallest non-zero ~ 40, largest ~ 130)
        nz = lams[lams > 0]
        if len(nz):
            log_lo, log_hi = np.log10(nz.min()), np.log10(nz.max())
        else:
            log_lo, log_hi = -4, -2
        def msize(lam):
            if lam == 0:
                return 140
            t = (np.log10(lam) - log_lo) / max(1e-9, log_hi - log_lo)
            return 55 + 55 * t    # 55 → 110

        # Scatter (no connecting line)
        for a, i_, lam in zip(accs, ices, lams):
            is_collapse = base_ice is not None and i_ < base_ice - 5
            face = "#f0f0f0" if lam == 0 else color
            edge = "black"
            mk = "o"
            ax.scatter([a], [i_], s=msize(lam), color=face,
                       edgecolor=edge, linewidth=1.0, zorder=3, marker=mk)
            tag = "λ=0 (base)" if lam == 0 else f"λ={lam:.0e}"
            if is_collapse:
                tag = tag + "  ⚠"
                ax.annotate(tag, (a, i_), textcoords="offset points",
                            xytext=(8, -4), fontsize=8, color="#b71c1c",
                            bbox=dict(boxstyle="round,pad=0.15",
                                      fc="#ffebee", ec="#b71c1c", lw=0.6))
            else:
                ax.annotate(tag, (a, i_), textcoords="offset points",
                            xytext=(8, 4), fontsize=8)

        # Best point: star overlay
        ax.scatter([accs[best_idx]], [ices[best_idx]],
                   s=250, marker="*", color="#fdd835",
                   edgecolor="black", linewidth=0.8, zorder=4)

        # Baseline horizontal reference (label on the right edge, above the line)
        if base_ice is not None:
            ax.axhline(base_ice, color="gray", ls="--", lw=0.8, alpha=0.7)
            ax.text(0.985, base_ice + (ice_hi - ice_lo) * 0.008,
                    f"baseline ICE={base_ice:.0f}", fontsize=7.5,
                    color="gray", ha="right", va="bottom",
                    transform=ax.get_yaxis_transform())

            # Collapse band shading below baseline
            ax.axhspan(ice_lo, base_ice - 5, color="#b71c1c",
                       alpha=0.04, zorder=0)

        ax.set_xlabel("Top-1 Accuracy (%)")
        if ax is axes[0]:
            ax.set_ylabel("ICE = Acc / mean firing rate")
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.set_ylim(ice_lo, ice_hi)

    # Shared legend-ish note
    fig.text(0.5, -0.02,
             "★ = highest ICE in family    "
             "⚠ = ICE below baseline (collapse)    "
             "marker size ∝ log|λ|",
             ha="center", fontsize=8.5, color="#444")

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
