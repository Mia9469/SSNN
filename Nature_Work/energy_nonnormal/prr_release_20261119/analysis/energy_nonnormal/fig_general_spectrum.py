#!/usr/bin/env python3
"""Figure 2: the general-spectrum selection principle.

Three panels visualize the analytic results of the "General-spectrum selection
principle" section of the Supplemental Material:

  (a) C1  the stationary cost for stable real spectra collapses onto
          the weighted Frobenius budget sum_{i<k} w_ik M_ik^2, w_ik =
          1/[2 a_i a_k (a_i+a_k)], with only a cubic spread at larger amplitude;
  (b) C3  K^2 = 0 is the cost-truncation locus: the rank-one square-zero channel
          lies exactly on the quadratic cost law, while the feedforward triangle
          (K^2 != 0) deviates upward at finite amplitude;
  (c)     the minimal triangle has the larger onset slope (0.58 > 0.50) yet the
          rank-one channel dominates the peak squared gain over the displayed
          matched-cost range.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)


def stationary_cost(j: np.ndarray) -> float:
    n = j.shape[0]
    return float(np.trace(solve_continuous_lyapunov(j, -np.eye(n))))


def peak_gain(j: np.ndarray, t_max: float = 60.0, n_t: int = 3000) -> float:
    ts = np.linspace(0.0, t_max, n_t)
    return max(np.linalg.norm(expm(j * t), 2) ** 2 for t in ts)


def weighted_budget(a: np.ndarray, m: np.ndarray) -> float:
    n = a.size
    total = 0.0
    for i in range(n):
        for k in range(i + 1, n):
            total += m[i, k] ** 2 / (2.0 * a[i] * a[k] * (a[i] + a[k]))
    return total


def main() -> None:
    rng = np.random.default_rng(3)

    # ---- (a) general-spectrum weighted cost law ----
    predicted, numeric, amp_color = [], [], []
    for _ in range(500):
        n = int(rng.integers(4, 11))
        a = 0.5 + rng.random(n)                       # distinct positive rates
        m = np.triu(rng.standard_normal((n, n)), 1)
        m /= np.linalg.norm(m)                         # unit Frobenius direction
        amp = 10.0 ** rng.uniform(-2.0, -0.3)         # 0.01 .. 0.5
        d = np.diag(-a)
        c_normal = float(np.sum(1.0 / (2.0 * a)))
        numeric.append(stationary_cost(d + amp * m) - c_normal)
        predicted.append(weighted_budget(a, amp * m))  # sum_{i<k} w_ik (amp M_ik)^2
        amp_color.append(np.log10(amp))
    predicted = np.array(predicted)
    numeric = np.array(numeric)
    amp_color = np.array(amp_color)

    # ---- (b, c) cost truncation and matched-cost dominance (3x3, a=0.2) ----
    a0 = 0.2
    c_norm3 = 3.0 / (2.0 * a0)
    tri = np.array([[0.0, 1.0, 1.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0]]) / np.sqrt(3.0)   # ||K||_F = 1, K^2 != 0
    u = np.array([1.0, 0.0, 0.0])
    v = np.array([0.0, 0.0, 1.0])
    rank1 = np.outer(u, v)                              # ||K||_F = 1, square-zero

    amps = np.linspace(0.02, 1.4, 90)
    quad_law = 1.0 + amps**2 / (4.0 * a0**3) / c_norm3  # exact square-zero cost/Cn
    cost_r1 = np.array([stationary_cost(-a0 * np.eye(3) + s * rank1) / c_norm3 for s in amps])
    cost_tri = np.array([stationary_cost(-a0 * np.eye(3) + s * tri) / c_norm3 for s in amps])
    gain_r1 = np.array([peak_gain(-a0 * np.eye(3) + s * rank1) for s in amps])
    gain_tri = np.array([peak_gain(-a0 * np.eye(3) + s * tri) for s in amps])

    grid = np.linspace(1.02, 3.0, 80)                  # matched stationary cost
    g_r1_matched = np.interp(grid, cost_r1, gain_r1)
    g_tri_matched = np.interp(grid, cost_tri, gain_tri)

    slope_r1 = float(np.linalg.eigvalsh((rank1 + rank1.T) / 2.0).max())     # 0.5
    slope_tri = float(np.linalg.eigvalsh((tri + tri.T) / 2.0).max())        # 1/sqrt(3)

    # ---------------- plotting ----------------
    plt.rcParams.update({
        "font.size": 8.0,
        "axes.linewidth": 0.9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.labelsize": 7.4,
        "ytick.labelsize": 7.4,
        "axes.labelsize": 8.0,
    })
    fig = plt.figure(figsize=(7.1, 2.45), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, wspace=0.16)

    # (a)
    ax = fig.add_subplot(gs[0, 0])
    sc = ax.scatter(predicted, numeric, c=amp_color, cmap="viridis",
                    s=10, alpha=0.7, linewidths=0)
    lo = min(predicted.min(), numeric.min())
    hi = max(predicted.max(), numeric.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label=r"$y=x$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"weighted budget $\sum_{i<k} w_{ik} M_{ik}^2$")
    ax.set_ylabel(r"numerical excess $\mathrm{Tr}\,\Sigma - C_{\rm normal}$")
    ax.set_title(r"(a) General-spectrum cost", fontsize=8.3)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label(r"$\log_{10}$ amplitude", fontsize=7.4)
    cb.ax.tick_params(labelsize=7.2)
    ax.legend(frameon=False, fontsize=6.8, loc="upper left",
              bbox_to_anchor=(0.03, 0.98), borderaxespad=0.0)

    # (b)
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(amps, quad_law, "k--", lw=1.3, label="quadratic law")
    ax.plot(amps, cost_r1, color="#C62828", lw=2.0, label="rank-one ($K^2=0$)")
    ax.plot(amps, cost_tri, color="#1565C0", lw=2.0, label=r"triangle ($K^2\neq0$)")
    ax.set_yscale("log")
    ax.set_xlabel(r"non-normal amplitude")
    ax.set_ylabel(r"stationary cost $C/C_{\rm normal}$")
    ax.set_title("(b) Cost truncation", fontsize=8.3)
    ax.legend(frameon=False, fontsize=6.8, loc="upper left",
              bbox_to_anchor=(0.03, 0.98), borderaxespad=0.0)

    # (c)
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(grid, g_r1_matched, color="#C62828", lw=2.0, label="rank-one")
    ax.plot(grid, g_tri_matched, color="#1565C0", lw=2.0, label="triangle")
    ax.axhline(1.0, color="0.45", ls=":", lw=1)
    ax.set_xlabel(r"matched cost $C_0/C_{\rm normal}$")
    ax.set_ylabel(r"peak squared gain $G_{\max}^{(2)}$")
    ax.set_title("(c) Matched-cost gain", fontsize=8.3)
    ax.text(0.04, 0.96,
            f"onset slope\ntriangle {slope_tri:.2f} > rank-one {slope_r1:.2f}\n"
            "yet rank-one wins the gain",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.7,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.75", lw=0.8))
    ax.legend(frameon=False, fontsize=6.8, loc="upper left",
              bbox_to_anchor=(0.05, 0.54), borderaxespad=0.0,
              handlelength=1.6)

    fig.savefig(FIG_DIR / "fig_general_spectrum.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_general_spectrum.png", dpi=220, bbox_inches="tight")

    summary = {
        "cost_law_max_rel_dev_small_amp": float(
            np.max(np.abs(numeric[amp_color < -1.3] / predicted[amp_color < -1.3] - 1.0))
        ),
        "onset_slope_rank_one": slope_r1,
        "onset_slope_triangle": slope_tri,
        "rank_one_dominates_all_matched_cost": bool(
            np.all(g_r1_matched >= g_tri_matched - 1e-9)
        ),
    }
    (HERE / "general_spectrum_results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("saved figures/fig_general_spectrum.{pdf,png}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
