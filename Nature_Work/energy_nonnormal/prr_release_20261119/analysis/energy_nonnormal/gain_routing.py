#!/usr/bin/env python3
"""Exact gain-gated routing with fixed anatomical connectivity.

The construction sends the same input population to two alternative output
populations through one fixed anatomical matrix W.  Two context-dependent
diagonal gain profiles exchange the strengths of those output branches.
Because the branch strengths are merely permuted, the effective generators
have identical eigenvalue spectra, stationary covariance spectra, and
stationary fluctuation costs, while the context-dependent input-output maps
differ.

The script writes a generated comparison figure and a JSON record used by the
assertion-based verification suite.  The canonical main-text artwork
``figures/fig_gain_routing.pdf`` is manually curated and is deliberately never
overwritten by this script.
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from scipy.linalg import expm, solve_continuous_lyapunov


HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)


def group_vector(n: int, start: int, stop: int) -> np.ndarray:
    vector = np.zeros(n)
    vector[start:stop] = 1.0 / math.sqrt(stop - start)
    return vector


def peak_squared_gain(j: np.ndarray, times: np.ndarray) -> tuple[float, float]:
    gains = np.array([np.linalg.norm(expm(j * t), 2) ** 2 for t in times])
    index = int(np.argmax(gains))
    return float(gains[index]), float(times[index])


def build_model() -> dict[str, object]:
    # Some macOS Accelerate/SciPy builds emit spurious finite-matmul warnings;
    # every returned quantity is checked explicitly below.
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    n = 40
    group_size = 10
    a = 1.0
    gain_high = 3.0
    gain_low = 0.6
    route_time = 1.0 / a

    v = group_vector(n, 0, group_size)
    u1 = group_vector(n, 2 * group_size, 3 * group_size)
    u2 = group_vector(n, 3 * group_size, 4 * group_size)

    # One fixed anatomical operator routes the same cue to two alternative
    # downstream populations.  Context changes output gain, not anatomy.
    w_anatomy = np.outer(u1, v) + np.outer(u2, v)

    gains_a = np.ones(n)
    gains_b = np.ones(n)
    gains_a[2 * group_size:3 * group_size] = gain_high
    gains_a[3 * group_size:4 * group_size] = gain_low
    gains_b[2 * group_size:3 * group_size] = gain_low
    gains_b[3 * group_size:4 * group_size] = gain_high

    gain_profiles = [gains_a, gains_b]
    generators = []
    covariances = []
    propagators = []
    for gains in gain_profiles:
        k = np.diag(gains) @ w_anatomy
        if np.linalg.norm(k @ k) > 1e-12:
            raise AssertionError("gain-dressed routing operator is not square-zero")
        j = -a * np.eye(n) + k
        sigma = solve_continuous_lyapunov(j, -np.eye(n))
        generators.append(j)
        covariances.append(sigma)
        propagators.append(expm(j * route_time))

    covariance_spectra = [
        np.sort(np.linalg.eigvalsh(sigma))[::-1] for sigma in covariances
    ]
    generator_spectra = [
        np.sort_complex(np.linalg.eigvals(j)) for j in generators
    ]
    costs = [float(np.trace(sigma)) for sigma in covariances]

    task_inputs = [v, v]
    task_outputs = [u1, u2]
    task_transfer = np.zeros((2, 2))
    for task, (v, u) in enumerate(zip(task_inputs, task_outputs)):
        for state, propagator in enumerate(propagators):
            task_transfer[task, state] = abs(u @ propagator @ v) ** 2

    routed_outputs = [
        gain_high * u1 + gain_low * u2,
        gain_low * u1 + gain_high * u2,
    ]
    routed_outputs = [x / np.linalg.norm(x) for x in routed_outputs]
    leading_input_overlap = 1.0
    leading_output_overlap = abs(float(routed_outputs[0] @ routed_outputs[1]))
    routed_output_angle_degrees = float(
        np.degrees(np.arccos(np.clip(leading_output_overlap, -1.0, 1.0)))
    )

    times = np.linspace(0.0, 5.0 / a, 1000)
    peak_gains = [peak_squared_gain(j, times) for j in generators]

    covariance_spectrum_difference = float(
        np.max(np.abs(covariance_spectra[0] - covariance_spectra[1]))
    )
    generator_spectrum_difference = float(
        np.max(np.abs(generator_spectra[0] - generator_spectra[1]))
    )
    cost_difference = abs(costs[0] - costs[1])
    routing_selectivity = float(task_transfer[0, 0] / task_transfer[0, 1])

    if generator_spectrum_difference > 1e-12:
        raise AssertionError("gain states do not have identical generator spectra")
    if covariance_spectrum_difference > 1e-12:
        raise AssertionError("gain states do not have identical covariance spectra")
    if cost_difference > 1e-12:
        raise AssertionError("gain states do not have identical stationary costs")
    if abs(leading_input_overlap - 1.0) > 1e-12:
        raise AssertionError("the two contexts do not share the same input")
    if routed_output_angle_degrees < 60.0:
        raise AssertionError("context did not substantially redirect the common input")
    if routing_selectivity < 20.0:
        raise AssertionError("gain gating did not strongly switch task routing")

    return {
        "n": n,
        "group_size": group_size,
        "a": a,
        "gain_high": gain_high,
        "gain_low": gain_low,
        "route_time": route_time,
        "w_anatomy": w_anatomy,
        "gain_profiles": gain_profiles,
        "generators": generators,
        "covariances": covariances,
        "covariance_spectra": covariance_spectra,
        "generator_spectra": generator_spectra,
        "costs": costs,
        "task_transfer": task_transfer,
        "leading_input_overlap": leading_input_overlap,
        "leading_output_overlap": leading_output_overlap,
        "routed_output_angle_degrees": routed_output_angle_degrees,
        "peak_gains": peak_gains,
        "covariance_spectrum_difference": covariance_spectrum_difference,
        "generator_spectrum_difference": generator_spectrum_difference,
        "cost_difference": cost_difference,
        "routing_selectivity": routing_selectivity,
    }


def arrow(ax, start, end, color="0.35", lw=1.4, ms=11) -> None:
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=ms,
        linewidth=lw, color=color, shrinkA=5, shrinkB=5,
    ))


def plot_model(model: dict[str, object]) -> None:
    red = "#C53D4C"
    blue = "#2878B5"
    ink = "#263238"
    grey = "#96A1A8"
    pale = "#F3F5F6"
    gain_high = float(model["gain_high"])
    gain_low = float(model["gain_low"])
    covariance_spectra = model["covariance_spectra"]
    costs = model["costs"]
    task_transfer = np.asarray(model["task_transfer"])
    flywire = json.loads((HERE / "flywire_gain_routing_results.json").read_text())
    if not flywire["predeclared_gates"]["all_pass"]:
        raise AssertionError("FlyWire inset requested although its gates did not pass")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 8.2,
        "axes.linewidth": 0.8,
        "axes.edgecolor": ink,
        "axes.labelcolor": ink,
        "text.color": ink,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.color": ink,
        "ytick.color": ink,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "axes.labelsize": 8.1,
        "axes.titleweight": "semibold",
        "axes.titlesize": 8.4,
        "legend.fontsize": 6.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig = plt.figure(figsize=(7.15, 2.48))
    gs = fig.add_gridspec(
        1, 3, width_ratios=[1.58, 1.62, 1.02],
        left=0.025, right=0.985, bottom=0.23, top=0.82, wspace=0.42,
    )

    # (a) One recurrent circuit, two context-dependent effective routes.
    ax = fig.add_subplot(gs[0, 0])
    ax.set_axis_off()
    ax.set_title("(a) Context-gated routing", loc="left", pad=7)
    colors = [red, blue]
    ax.text(0.57, 0.90, r"same recurrent circuit $W$", ha="center",
            fontsize=6.7, color="#526169", weight="semibold")

    # The same network is redrawn in the two context rows.  All anatomical
    # paths remain grey; only the context-selected effective path is colored.
    for y0, context, active in [(0.53, "context A", 0), (0.15, "context B", 1)]:
        card = FancyBboxPatch(
            (0.02, y0), 0.96, 0.29,
            boxstyle="round,pad=0.014,rounding_size=0.025",
            fc="#FAFBFB", ec="#C9D0D4", lw=0.85,
        )
        ax.add_patch(card)
        ax.text(0.055, y0 + 0.245, context, ha="left", fontsize=6.8,
                weight="semibold")

        # Shared recurrent population.
        net = FancyBboxPatch(
            (0.34, y0 + 0.045), 0.34, 0.18,
            boxstyle="round,pad=0.012,rounding_size=0.04",
            fc="#F0F3F4", ec="#B9C3C8", lw=0.75,
        )
        ax.add_patch(net)
        node_xy = [(0.39, y0 + 0.15), (0.47, y0 + 0.09),
                   (0.55, y0 + 0.17), (0.63, y0 + 0.10)]
        for i, j in [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 1)]:
            ax.plot([node_xy[i][0], node_xy[j][0]],
                    [node_xy[i][1], node_xy[j][1]],
                    color="#C4CCD0", lw=0.65, zorder=1)
        for xy in node_xy:
            ax.add_patch(Circle(xy, 0.014, fc="white", ec="#8E9BA2",
                                lw=0.7, zorder=2))

        # The sensory input is identical in both contexts.  Gain selects which
        # downstream action population receives that common input.
        inp = (0.17, y0 + 0.13)
        ax.add_patch(Circle(inp, 0.027, fc="white", ec=ink, lw=1.15))
        arrow(ax, (inp[0] + 0.027, inp[1]), (0.34, inp[1]),
              color="#65757D", lw=1.5, ms=8)
        ax.text(inp[0], y0 + 0.060, "same cue", ha="center", fontsize=5.4,
                color=ink)

        for route, (offset, color) in enumerate(zip([0.17, 0.085], colors)):
            ypos = y0 + offset
            is_active = route == active
            alpha = 1.0 if is_active else 0.20
            lw = 2.7 if is_active else 1.0
            out = (0.84, ypos)
            ax.add_patch(Circle(out, 0.029, fc="white", ec=color,
                                alpha=alpha, lw=1.2))
            arrow(ax, (0.68, y0 + 0.13), (out[0] - 0.030, ypos),
                  color=color, lw=lw, ms=8)
            ax.patches[-1].set_alpha(alpha)
            ax.text(out[0], ypos - 0.052, f"action {route + 1}", ha="center",
                    fontsize=5.2, color=color, alpha=1.0 if is_active else 0.32)
            if is_active:
                ax.text(0.94, y0 + 0.245, rf"cue $\rightarrow$ action {route + 1}",
                        ha="right", fontsize=5.8, color=color,
                        weight="semibold")
        ax.text(0.51, y0 + 0.135, "population", ha="center", va="center",
                fontsize=5.5, color="#64737A", zorder=4)
    ax.annotate("gain changes", xy=(0.50, 0.50), xytext=(0.50, 0.47),
                ha="center", va="center", fontsize=5.8, color="#526169",
                arrowprops=dict(arrowstyle="-|>", color="#8D999F", lw=0.8))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # (b) Put the identifiability contrast in one visual unit: spontaneous
    # summaries overlap, whereas causal input-output transfer switches.
    sub = gs[0, 1].subgridspec(1, 2, width_ratios=[1.03, 0.92], wspace=0.48)
    ax = fig.add_subplot(sub[0, 0])
    ranks = np.arange(1, len(covariance_spectra[0]) + 1)
    ax.plot(ranks, covariance_spectra[0], color=red, lw=2.15, label="context A")
    ax.plot(ranks, covariance_spectra[1], color=blue, lw=1.65,
            ls=(0, (4, 2)), label="context B")
    ax.set_xlim(1, 14)
    ax.set_xlabel("PC rank")
    ax.set_ylabel("stationary variance")
    ax.set_title("(b) Spontaneous", loc="left", pad=7)
    ax.legend(frameon=False, loc="upper right", handlelength=1.8, fontsize=6.2)
    ax.text(
        0.05, 0.10,
        r"same $\sigma(J),\sigma(\Sigma)$" + "\n" +
        rf"$\mathrm{{Tr}}\,\Sigma={costs[0]:.2f}$",
        transform=ax.transAxes, fontsize=5.8, ha="left", va="bottom",
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#D1D6D8", lw=0.65),
    )
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(sub[0, 1])
    ax.imshow(task_transfer, cmap="Reds", vmin=0.0,
              vmax=float(task_transfer.max()), aspect="auto")
    ax.set_xticks([0, 1], ["A", "B"])
    ax.set_yticks([0, 1], ["cue→1", "cue→2"])
    ax.set_xlabel("context")
    ax.set_ylabel("tested output")
    ax.set_title("Perturbation", loc="left", pad=7)
    threshold = 0.55 * float(task_transfer.max())
    for i in range(2):
        for j in range(2):
            value = float(task_transfer[i, j])
            ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                    fontsize=6.2, color="white" if value > threshold else "0.2")
    for i in range(2):
        ax.add_patch(Rectangle((i - 0.48, i - 0.48), 0.96, 0.96,
                               fill=False, ec=ink, lw=1.0))
    ax.text(0.5, -0.24,
            f"{model['routing_selectivity']:.0f}× switch; "
            f"{model['routed_output_angle_degrees']:.0f}° output rotation",
            transform=ax.transAxes, ha="center", va="top", fontsize=6.2)

    # (c) The same route-switch statistic on the complete connectome.
    ax = fig.add_subplot(gs[0, 2])
    observed = float(flywire["switch"]["geometric_switch_factor"])
    null95 = float(flywire["null"]["q95"])
    nulls = np.asarray(flywire["null"]["switch_factors"], dtype=float)
    bins = np.linspace(1.0, max(observed * 1.06, nulls.max()), 18)
    ax.hist(nulls, bins=bins, density=True, color="#C7CED2", ec="white", lw=0.45)
    ax.axvline(null95, color="#66747C", ls=(0, (3, 2)), lw=1.1)
    ax.axvline(observed, color=red, lw=2.0)
    ax.set_xlim(1.0, observed * 1.10)
    ax.set_xlabel("route-switch score", labelpad=2)
    ax.set_yticks([])
    ax.set_title("(c) Full FlyWire test", loc="left", pad=7)
    ax.text(0.03, 0.97, "139,255 neurons\nmatched stability + cost",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.0,
            color="#526169")
    ax.text(null95, ax.get_ylim()[1] * 0.68, "null 95%", rotation=90,
            ha="right", va="center", fontsize=5.8, color="#66747C")
    ax.text(observed, ax.get_ylim()[1] * 0.68, f"observed\n{observed:.2f}",
            rotation=90, ha="right", va="center", fontsize=6.0, color=red,
            weight="semibold")
    ax.text(0.98, 0.97, f"$p={flywire['null']['p_one_sided']:.3g}$",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.3)
    ax.spines[["top", "right", "left"]].set_visible(False)

    # Keep generated plots separate from the manually curated main-text PDF.
    # This prevents a verification rerun from silently replacing final artwork.
    fig.savefig(FIG_DIR / "fig_gain_routing_generated.pdf", bbox_inches="tight")
    fig.savefig(
        FIG_DIR / "fig_gain_routing_generated.png",
        dpi=240,
        bbox_inches="tight",
    )


def serializable_summary(model: dict[str, object]) -> dict[str, object]:
    return {
        "parameters": {
            "n": int(model["n"]),
            "group_size": int(model["group_size"]),
            "leak_a": float(model["a"]),
            "gain_high": float(model["gain_high"]),
            "gain_low": float(model["gain_low"]),
            "route_time": float(model["route_time"]),
        },
        "matched_spontaneous_quantities": {
            "cost_state_a": float(model["costs"][0]),
            "cost_state_b": float(model["costs"][1]),
            "cost_abs_difference": float(model["cost_difference"]),
            "generator_spectrum_max_abs_difference": float(
                model["generator_spectrum_difference"]
            ),
            "covariance_spectrum_max_abs_difference": float(
                model["covariance_spectrum_difference"]
            ),
        },
        "routing": {
            "squared_task_transfer": np.asarray(model["task_transfer"]).tolist(),
            "routing_selectivity": float(model["routing_selectivity"]),
            "leading_input_abs_overlap": float(model["leading_input_overlap"]),
            "leading_output_abs_overlap": float(model["leading_output_overlap"]),
            "routed_output_angle_degrees": float(
                model["routed_output_angle_degrees"]
            ),
            "peak_squared_gain_state_a": float(model["peak_gains"][0][0]),
            "peak_squared_gain_state_b": float(model["peak_gains"][1][0]),
            "peak_time_state_a": float(model["peak_gains"][0][1]),
            "peak_time_state_b": float(model["peak_gains"][1][1]),
        },
    }


def main() -> None:
    model = build_model()
    plot_model(model)
    summary = serializable_summary(model)
    (HERE / "gain_routing_results.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print("saved figures/fig_gain_routing_generated.{pdf,png}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
