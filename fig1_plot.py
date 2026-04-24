"""
fig1_plot.py
============
Figure 1 — Scale-dependence of CV regularisation across ANN and SNN.

Three panels:

  A. ANN / ResNet-20 / CIFAR-10 (3 seeds × 200 ep)
     Bar chart of ICE (Acc / mean|a|) across four conditions
     {Baseline, WeightCV, ActCV, Weight+ActCV}.  Accuracy on top of each bar.
     Shows activation-scale CV drives the gain, weight-scale CV does little,
     and the two do not compose.

  B. SNN / SDT / CIFAR-10 (1 seed × 300 ep; will be 3 seeds when finalised)
     Bar chart of ICE (Acc / mean FR) across the analogous four conditions
     {Baseline, WeightCV, PopFR-CV, Weight+PopCV}.  Shows population-FR-scale
     CV is the SNN analogue of ActCV, with much larger uplift (~+20% ICE).

  C. Dichotomy — 2×2 heatmap of ΔICE (%) relative to each model's baseline
     with rows = {ANN: ResNet, SNN: SDT} and cols = {WeightCV, Activation-CV}.
     Visualises the paper's central claim: weight-scale CV is the wrong lever
     for both; activation-scale CV is the right one for both; the magnitude is
     10× larger on the SNN.

Inputs:
    resnet20_compare_results.json   — Panel A, row "ANN" of Panel C
    scratch_compare_results.json    — Panel B, row "SNN" of Panel C

Output:
    fig1.pdf / fig1.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.8,
})

COLOR_BASELINE = "#9aa0a6"
COLOR_WEIGHT   = "#f4a261"
COLOR_ACT      = "#2a9d8f"
COLOR_COMBO    = "#264653"


def _load(name: str) -> dict:
    with open(HERE / name) as fh:
        return json.load(fh)


def _last(v):
    return v[-1] if isinstance(v, list) else v


# ── Panel A: ANN / ResNet-20 ──────────────────────────────────────────────────
def panel_a(ax) -> None:
    d = _load("resnet20_compare_results.json")
    h, cfgs, labels = d["hists"], d["configs"], d["labels"]

    ice     = [_last(h[k]["ice"])       for k in cfgs]
    ice_std = [_last(h[k].get("ice_std", [0])) for k in cfgs]
    acc     = [_last(h[k]["acc"])       for k in cfgs]
    acc_std = [_last(h[k].get("acc_std", [0])) for k in cfgs]
    names   = [labels[k] for k in cfgs]
    colors  = [COLOR_BASELINE, COLOR_WEIGHT, COLOR_ACT, COLOR_COMBO]

    x = np.arange(len(cfgs))
    bars = ax.bar(x, ice, yerr=ice_std, capsize=3, color=colors,
                  edgecolor="black", linewidth=0.6, error_kw=dict(lw=0.8))
    for xi, bar, a, s in zip(x, bars, acc, acc_std):
        ax.text(xi, bar.get_height() + max(ice) * 0.025,
                f"{a:.2f}±{s:.2f}%", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("ICE = Acc / mean|a|")
    ax.set_title("A  ANN · ResNet-20 / CIFAR-10")
    ax.set_ylim(0, max(ice) * 1.20)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)


# ── Panel B: SNN / SDT ────────────────────────────────────────────────────────
def panel_b(ax) -> None:
    d = _load("scratch_compare_results.json")
    h, cfgs, labels = d["hists"], d["configs"], d["labels"]

    # Only keep the 4 canonical conditions (drop LifetimeCV if present)
    keep = [c for c in cfgs if c in ("baseline", "weight_cv", "pop_cv", "weight_pop")]
    order = ["baseline", "weight_cv", "pop_cv", "weight_pop"]
    keep.sort(key=lambda c: order.index(c))

    acc     = [_last(h[k]["acc"])       for k in keep]
    acc_std = [_last(h[k].get("acc_std", [0])) for k in keep]
    mfr     = [_last(h[k]["mean_fr"])   for k in keep]
    ice     = [a / (m + 1e-8)           for a, m in zip(acc, mfr)]
    ice_std = [_last(h[k].get("ice_std", [0])) for k in keep]
    names   = [labels[k] for k in keep]
    colors  = [COLOR_BASELINE, COLOR_WEIGHT, COLOR_ACT, COLOR_COMBO]

    x = np.arange(len(keep))
    bars = ax.bar(x, ice, yerr=ice_std, capsize=3, color=colors,
                  edgecolor="black", linewidth=0.6, error_kw=dict(lw=0.8))
    for xi, bar, a, s in zip(x, bars, acc, acc_std):
        ax.text(xi, bar.get_height() + max(ice) * 0.025,
                f"{a:.2f}±{s:.2f}%", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("ICE = Acc / mean FR")
    ax.set_title("B  SNN · SDT / CIFAR-10")
    ax.set_ylim(0, max(ice) * 1.20)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)


# ── Panel C: dichotomy heatmap ────────────────────────────────────────────────
def _delta_ice(d: dict, base_key: str, cond_key: str, mfr_key: str) -> float:
    h = d["hists"]
    def get_ice(k):
        if "ice" in h[k]:
            return _last(h[k]["ice"])
        return _last(h[k]["acc"]) / (_last(h[k][mfr_key]) + 1e-8)
    b = get_ice(base_key); c = get_ice(cond_key)
    return (c - b) / b * 100.0


def panel_c(ax) -> None:
    ann = _load("resnet20_compare_results.json")
    snn = _load("scratch_compare_results.json")

    # rows = model, cols = intervention (weight-scale vs activation-scale)
    row_labels = ["ANN\n(ResNet-20)", "SNN\n(SDT)"]
    col_labels = ["WeightCV", "Activation-CV"]

    ann_w   = _delta_ice(ann, "baseline", "weight_cv", "mean_act")
    ann_act = _delta_ice(ann, "baseline", "act_cv",    "mean_act")
    snn_w   = _delta_ice(snn, "baseline", "weight_cv", "mean_fr")
    snn_act = _delta_ice(snn, "baseline", "pop_cv",    "mean_fr")

    M = np.array([[ann_w, ann_act],
                  [snn_w, snn_act]])

    # Diverging colour scale centred at 0, symmetric
    vmax = max(1.0, np.abs(M).max())
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    for i in range(2):
        for j in range(2):
            val = M[i, j]
            ax.text(j, i, f"{val:+.1f}%", ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="black" if abs(val) < vmax * 0.6 else "white")

    ax.set_xticks([0, 1]); ax.set_xticklabels(col_labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(row_labels)
    ax.set_title("C  ΔICE vs baseline (%)")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.04)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("ΔICE (%)", fontsize=8)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8),
                             gridspec_kw=dict(wspace=0.38,
                                              width_ratios=[1, 1, 0.95]))
    panel_a(axes[0])
    panel_b(axes[1])
    panel_c(axes[2])

    fig.suptitle(
        "Figure 1.  Activation-scale CV is the right lever in both ANN and SNN; "
        "the SNN lever is ~10× stronger.",
        fontsize=10.5, y=1.03,
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
