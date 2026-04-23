"""
fig1_plot.py
============
Figure 1 — Mechanism dichotomy of CV regularisation across scales.

Panel A (ANN / ResNet-20 / CIFAR-10, 3 seeds × 200 ep):
    Bar chart of ICE (Acc / mean|a|) across four conditions
    {Baseline, WeightCV, ActCV, Weight+ActCV}.  Annotated with Acc (%).
    Shows that activation-scale CV, not weight-scale CV, drives the gain,
    and that the two regularisers do not compose.

Panel B (SNN / SDT / CIFAR-10, warmup sweep, 100 ep):
    Bar chart of ICE across pop_cv_start ∈ {0, 10, 20, 40}.
    Annotated with dead-neuron ratio.  Shows the dead-neuron trap
    (start=0 collapses) and the U-shape around a moderate warmup.

Inputs:
    resnet20_compare_results.json
    warmup_sweep_results.json

Output:
    fig1.pdf / fig1.png  (saved next to this script)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        9,
    "axes.titlesize":   10,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":   0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

COLOR_BASELINE = "#9aa0a6"
COLOR_WEIGHT   = "#f4a261"
COLOR_ACT      = "#2a9d8f"
COLOR_COMBO    = "#264653"
COLOR_FAIL     = "#c44536"


# ── Panel A: ResNet-20 ─────────────────────────────────────────────────────────
def panel_a(ax) -> None:
    with open(HERE / "resnet20_compare_results.json") as fh:
        d = json.load(fh)
    h, cfgs, labels = d["hists"], d["configs"], d["labels"]

    ice      = [h[k]["ice"][-1]         for k in cfgs]
    ice_std  = [h[k]["ice_std"][-1]     for k in cfgs]
    acc      = [h[k]["acc"][-1]         for k in cfgs]
    acc_std  = [h[k]["acc_std"][-1]     for k in cfgs]
    names    = [labels[k] for k in cfgs]
    colors   = [COLOR_BASELINE, COLOR_WEIGHT, COLOR_ACT, COLOR_COMBO]

    x = np.arange(len(cfgs))
    bars = ax.bar(x, ice, yerr=ice_std, capsize=3, color=colors,
                  edgecolor="black", linewidth=0.6, error_kw=dict(lw=0.8))

    # annotate accuracy on top of each bar
    for xi, bar, a, s in zip(x, bars, acc, acc_std):
        h_bar = bar.get_height() + bar.get_yerr() if False else bar.get_height()
        ax.text(xi, h_bar + 18, f"{a:.2f}±{s:.2f}%",
                ha="center", va="bottom", fontsize=7.5)

    # highlight ActCV (winner) with a subtle star
    i_best = int(np.argmax(ice))
    ax.text(i_best, ice[i_best] - 50, "*", ha="center", va="center",
            fontsize=18, color="white", weight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("ICE = Acc / mean|a|")
    ax.set_title("A  ResNet-20 / CIFAR-10 — activation-scale CV wins")
    ax.set_ylim(0, max(ice) * 1.18)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)


# ── Panel B: warmup sweep ──────────────────────────────────────────────────────
def panel_b(ax) -> None:
    with open(HERE / "warmup_sweep_results.json") as fh:
        d = json.load(fh)
    runs = sorted(d["runs"], key=lambda r: r["pop_cv_start"])

    starts = [r["pop_cv_start"]       for r in runs]
    ice    = [r["ice"]                for r in runs]
    acc    = [r["acc"]                for r in runs]
    dead   = [r["dead_ratio"]         for r in runs]

    x = np.arange(len(starts))
    colors = [COLOR_FAIL if s == 0 else COLOR_ACT for s in starts]
    bars = ax.bar(x, ice, color=colors, edgecolor="black", linewidth=0.6)

    for xi, bar, a, dr in zip(x, bars, acc, dead):
        ax.text(xi, bar.get_height() + 25,
                f"Acc {a:.1f}%\ndead {dr*100:.0f}%",
                ha="center", va="bottom", fontsize=7.3, linespacing=1.05)

    # dashed guide at baseline (start=0 failure)
    ax.axhline(ice[0], color=COLOR_FAIL, lw=0.6, ls=":", alpha=0.7)
    ax.text(len(x) - 0.4, ice[0] + 8, "collapse", color=COLOR_FAIL,
            fontsize=7, va="bottom", ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels([f"start={s}" for s in starts])
    ax.set_xlabel("PopFR-CV activation epoch")
    ax.set_ylabel("ICE = Acc / mean FR")
    ax.set_title("B  SDT / CIFAR-10 — dead-neuron trap at early onset")
    ax.set_ylim(0, max(ice) * 1.25)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)


# ── Main figure ───────────────────────────────────────────────────────────────
def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6),
                             gridspec_kw=dict(wspace=0.28))
    panel_a(axes[0])
    panel_b(axes[1])

    fig.suptitle(
        "Figure 1.  CV regularisation acts on different scales in ANN vs SNN.",
        fontsize=10.5, y=1.02,
    )
    fig.tight_layout()

    out_pdf = HERE / "fig1.pdf"
    out_png = HERE / "fig1.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=220)
    print(f"wrote {out_pdf.relative_to(HERE)}")
    print(f"wrote {out_png.relative_to(HERE)}")


if __name__ == "__main__":
    main()
