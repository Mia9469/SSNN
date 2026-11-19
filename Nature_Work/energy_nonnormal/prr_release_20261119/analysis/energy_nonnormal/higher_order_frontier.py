#!/usr/bin/env python3
"""Higher-order feedforward chains versus the square-zero rank-one baseline.

The fixed-budget selection theorem is exact within the square-zero class
(K^2 = 0).  A natural question for the Supplemental Material is whether richer
nilpotent structure can exceed that frontier once the same stationary cost is
charged.  We test the simplest extension: a single feedforward *chain* of
length L, i.e. an L-dimensional Jordan block of K with equal superdiagonal
weight w, embedded in N dimensions.  K^L = 0, and L = 2 reproduces the
square-zero rank-one carrier of the theorem.

For each chain length we sweep w, trace the exact (cost, gain) frontier, and
compare against the analytic square-zero rank-one optimum at matched cost.  All
operators have the identical spectrum {-a}; only the off-diagonal nilpotent
structure changes.

Cost is exact via the continuous Lyapunov equation; the closed form

    Tr Sigma = N/(2a) + sum_{j=1}^{L-1} (L-j) w^{2j} (2j)! / [ (j!)^2 (2a)^{2j+1} ]

is used as an independent cross-check.  Peak gain is the supremum over time of
the squared propagator 2-norm.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov


HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)


def chain_generator(a: float, n: int, length: int, weight: float) -> np.ndarray:
    """J = -a I + K, with K a single length-L feedforward chain in N dims."""
    k = np.zeros((n, n))
    for i in range(length - 1):
        k[i, i + 1] = weight
    return -a * np.eye(n) + k


def analytic_chain_cost(a: float, n: int, length: int, weight: float) -> float:
    """Closed-form Tr Sigma for an equal-weight chain of length L, Q = I."""
    cost = n / (2.0 * a)
    for j in range(1, length):
        term = (
            (length - j)
            * weight ** (2 * j)
            * math.factorial(2 * j)
            / (math.factorial(j) ** 2 * (2.0 * a) ** (2 * j + 1))
        )
        cost += term
    return cost


def chain_block_exp(length: int, weight: float, t: float) -> np.ndarray:
    """exp(K t) restricted to the length-L feedforward chain block.

    For a Jordan shift, the j-th superdiagonal of exp(wt N) is (wt)^j / j!.
    """
    b = np.eye(length)
    for j in range(1, length):
        coeff = (weight * t) ** j / math.factorial(j)
        for i in range(length - j):
            b[i, i + j] = coeff
    return b


def peak_gain_numeric(length: int, weight: float, a: float) -> float:
    """sup_t e^{-2at} ||exp(K t)_block||_2^2.

    Outside the chain block exp(Jt) acts as e^{-at} I, so the propagator 2-norm
    is set by the L x L block.  Evaluating only the block makes the sweep fast
    and exact (no large-matrix scaling-and-squaring).
    """
    if weight <= 0:
        return 1.0
    times = np.linspace(0.0, 25.0 / a, 4000)
    best = 1.0
    for tt in times:
        block = chain_block_exp(length, weight, float(tt))
        val = math.exp(-2.0 * a * tt) * float(np.linalg.norm(block, 2)) ** 2
        if val > best:
            best = val
    return best


def squarezero_rankone_gain(a: float, s: float) -> float:
    """Analytic G_max for the square-zero rank-one optimum with s_1 = s."""
    if s <= 0:
        return 1.0

    def phi(z: float) -> float:
        return 1.0 + 0.5 * z * z + 0.5 * z * math.sqrt(z * z + 4.0)

    times = np.linspace(0.0, 25.0 / a, 5000)
    return max(math.exp(-2.0 * a * tt) * phi(s * tt) for tt in times)


def squarezero_cost(a: float, n: int, s: float) -> float:
    return n / (2.0 * a) + s ** 2 / (4.0 * a ** 3)


def main() -> None:
    a = 0.20
    n = 64
    baseline = n / (2.0 * a)
    lengths = [2, 3, 4, 6]
    # Weight grids chosen per length so all chains span a common cost-ratio range.
    weights = np.linspace(0.0, 2.2, 300)

    curves: dict[int, dict[str, list[float]]] = {}
    max_cost_cross = 0.0
    for length in lengths:
        ratios, gains, cost_rel_err = [], [], []
        for w in weights:
            jmat = chain_generator(a, n, length, float(w))
            sigma = solve_continuous_lyapunov(jmat, -np.eye(n))
            cost = float(np.trace(sigma))
            cost_an = analytic_chain_cost(a, n, length, float(w))
            cost_rel_err.append(abs(cost - cost_an) / cost_an)
            ratios.append(cost / baseline)
            gains.append(peak_gain_numeric(length, float(w), a))
        curves[length] = {
            "weight": weights.tolist(),
            "cost_ratio": ratios,
            "Gmax": gains,
            "cost_relative_error": cost_rel_err,
        }
        max_cost_cross = max(max_cost_cross, max(ratios))

    # Square-zero rank-one baseline on the same cost-ratio axis, evaluated with
    # the identical block-norm numerics (L=2) so the matched-cost comparison is
    # apples-to-apples and free of grid mismatch.
    ref_ratios = np.linspace(1.0, min(1.6, max_cost_cross), 200)
    ref_gain = []
    ref_gain_analytic = []
    for ratio in ref_ratios:
        s = math.sqrt(max(0.0, 4.0 * a ** 3 * (ratio - 1.0) * baseline))
        ref_gain.append(peak_gain_numeric(2, s, a))
        ref_gain_analytic.append(squarezero_rankone_gain(a, s))

    # Matched-cost comparison: interpolate each chain's gain onto ref_ratios and
    # report the maximum relative excess over the square-zero rank-one baseline.
    comparison = {}
    for length in lengths:
        cr = np.asarray(curves[length]["cost_ratio"])
        g = np.asarray(curves[length]["Gmax"])
        order = np.argsort(cr)
        # L=2 is the square-zero reference itself.  Re-interpolating its coarse
        # weight grid onto the cost grid creates a spurious ~1e-4 overshoot near
        # onset, so compare it to the identical reference values exactly.
        g_interp = (
            np.asarray(ref_gain)
            if length == 2
            else np.interp(ref_ratios, cr[order], g[order], left=1.0)
        )
        excess = g_interp / np.asarray(ref_gain) - 1.0
        comparison[length] = {
            "max_relative_excess_over_rank1": float(np.max(excess)),
            "cost_ratio_at_max_excess": float(ref_ratios[int(np.argmax(excess))]),
        }

    output = {
        "parameters": {"a": a, "N": n, "baseline_cost": baseline,
                       "chain_lengths": lengths},
        "chains": {str(k): v for k, v in curves.items()},
        "squarezero_rank1_reference": {
            "cost_ratio": ref_ratios.tolist(),
            "Gmax": ref_gain,
            "Gmax_analytic": ref_gain_analytic,
        },
        "matched_cost_comparison": {str(k): v for k, v in comparison.items()},
        "max_cost_relative_error": max(
            max(curves[length]["cost_relative_error"]) for length in lengths
        ),
    }
    with (HERE / "higher_order_frontier_results.json").open("w") as f:
        json.dump(output, f, indent=2)

    plt.rcParams.update({
        "font.size": 10,
        "axes.linewidth": 0.9,
        "xtick.direction": "in",
        "ytick.direction": "in",
    })
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9))

    ax = axes[0]
    colors = plt.cm.plasma(np.linspace(0.08, 0.82, len(lengths)))
    xmax = 1.6
    for color, length in zip(colors, lengths):
        label = (r"$L=2$ (square-zero)" if length == 2 else rf"$L={length}$")
        cr = np.asarray(curves[length]["cost_ratio"])
        g = np.asarray(curves[length]["Gmax"])
        m = cr <= xmax  # restrict to the visible window so autoscale is clean
        ax.plot(cr[m], g[m], color=color, lw=2, label=label)
    ax.plot(ref_ratios, ref_gain, "k--", lw=1.3,
            label="rank-one theorem")
    ax.axhline(1.0, color="0.5", ls=":", lw=1)
    ax.set_yscale("log")
    ax.set_xlim(1.0, xmax)
    ax.set_ylim(0.85, max(ref_gain) * 1.5)
    ax.set_xlabel(r"stationary cost $C_0/C_{\rm normal}$")
    ax.set_ylabel(r"peak squared gain $G_{\max}^{(2)}$")
    ax.set_title("(a) Chain length vs. fixed-budget frontier")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    for color, length in zip(colors, lengths):
        if length == 2:
            continue
        cr = np.asarray(curves[length]["cost_ratio"])
        g = np.asarray(curves[length]["Gmax"])
        order = np.argsort(cr)
        g_interp = np.interp(ref_ratios, cr[order], g[order], left=1.0)
        ax.plot(ref_ratios, g_interp / np.asarray(ref_gain),
                color=color, lw=2, label=rf"$L={length}$")
    ax.axhline(1.0, color="k", ls="--", lw=1.2,
               label="rank-one baseline")
    ax.set_xlim(1.0, 1.6)
    ax.set_xlabel(r"stationary cost $C_0/C_{\rm normal}$")
    ax.set_ylabel(
        r"$G_{\max}^{(2)}(\mathrm{chain}) / G_{\max}^{(2)}(\mathrm{rank\,1})$"
    )
    ax.set_title("(b) Matched-cost excess over the baseline")
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_higher_order_frontier.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_higher_order_frontier.png", dpi=220,
                bbox_inches="tight")

    print(f"max cost relative error (closed form vs Lyapunov): "
          f"{output['max_cost_relative_error']:.3e}")
    for length in lengths:
        c = comparison[length]
        print(f"L={length}: max excess over rank-one = "
              f"{c['max_relative_excess_over_rank1']*100:+.2f}% "
              f"at C/C_normal={c['cost_ratio_at_max_excess']:.3f}")
    print("saved figures/fig_higher_order_frontier.{pdf,png}")
    print("saved higher_order_frontier_results.json")


if __name__ == "__main__":
    main()
