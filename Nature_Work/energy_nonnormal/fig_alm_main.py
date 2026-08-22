#!/usr/bin/env python3
"""Main-text figure for the prespecified multi-mouse ALM validation.

The figure deliberately uses mice, not sessions, as the inferential unit.
Session-level stability and gain are descriptive; the response-geometry panels
compare each mouse's mean full-operator statistic with exact same-spectrum
block-normal controls.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
RESULTS = {
    8: HERE / "alm_multisession_results_d8.json",
    12: HERE / "alm_multisession_results.json",
    16: HERE / "alm_multisession_results_d16.json",
}
OUTPUT = HERE / "figures" / "fig_alm_main"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def mouse_rows(report: dict) -> dict[str, dict]:
    return {row["mouse"]: row for row in report["summary"]["mouse_values"]}


def main() -> None:
    reports = {dimension: load(path) for dimension, path in RESULTS.items()}
    primary = reports[12]
    mice = sorted(mouse_rows(primary))
    colors = {
        mouse: plt.get_cmap("tab10")(index)
        for index, mouse in enumerate(mice)
    }

    plt.rcParams.update(
        {
            "font.size": 7.7,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.8,
            "xtick.labelsize": 7.1,
            "ytick.labelsize": 7.1,
            "axes.linewidth": 0.85,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(1, 4, figsize=(7.08, 2.05))

    # (a) Descriptive session-level stability and finite-time gain.
    ax = axes[0]
    for row in primary["sessions"]:
        ax.scatter(
            row["discrete_spectral_radius"],
            row["peak_squared_gain_0_to_1s"],
            s=24,
            color=colors[row["mouse"]],
            edgecolor="white",
            linewidth=0.35,
            alpha=0.88,
        )
    ax.axvline(1.0, color="0.35", ls="--", lw=0.8)
    ax.axhline(1.0, color="0.35", ls=":", lw=0.8)
    ax.set_xlabel(r"spectral radius $\rho(A)$")
    ax.set_ylabel(r"peak $\|A^k\|_2^2$")
    ax.set_title("(a) Fitted dynamics")

    def paired_mouse_panel(
        ax: plt.Axes,
        left_key: str,
        right_key: str,
        labels: tuple[str, str],
    ) -> None:
        rows = mouse_rows(primary)
        for mouse in mice:
            values = [rows[mouse][left_key], rows[mouse][right_key]]
            ax.plot([0, 1], values, color=colors[mouse], lw=1.0, alpha=0.9)
            ax.scatter([0, 1], values, color=colors[mouse], s=23, zorder=3)
        ax.axhline(0.0, color="0.5", lw=0.7)
        ax.set_xlim(-0.28, 1.28)
        ax.set_xticks([0, 1], labels)
        ax.set_ylabel("residual Spearman $r$")

    # (b) Prespecified primary endpoint.
    ax = axes[1]
    paired_mouse_panel(
        ax,
        "same_spectrum_normal_spearman_mean",
        "full_geometry_residual_spearman_mean",
        (r"same-$\lambda$" + "\nnormal", "full\ngeometry"),
    )
    ax.set_title("(b) Recovery prediction")

    # (c) Complete latent-dimension sensitivity, preserving the mouse unit.
    ax = axes[2]
    dimensions = np.array(sorted(reports))
    for mouse in mice:
        differences = []
        for dimension in dimensions:
            row = mouse_rows(reports[int(dimension)])[mouse]
            differences.append(
                row["full_geometry_residual_spearman_mean"]
                - row["same_spectrum_normal_spearman_mean"]
            )
        ax.plot(
            dimensions,
            differences,
            marker="o",
            ms=3.5,
            lw=0.9,
            color=colors[mouse],
        )
    ax.axhline(0.0, color="0.35", lw=0.8)
    ax.set_xticks(dimensions)
    ax.set_xlabel("latent dimension")
    ax.set_ylabel(r"full $-$ same-$\lambda$ $r$")
    ax.set_title("(c) Dimension robustness")

    # (d) Stricter condition-adjusted sensitivity at the primary dimension.
    ax = axes[3]
    paired_mouse_panel(
        ax,
        "conditioned_same_spectrum_normal_spearman_mean",
        "conditioned_full_geometry_residual_spearman_mean",
        (r"same-$\lambda$" + "\nnormal", "full\ngeometry"),
    )
    ax.set_title("(d) Condition adjusted")

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="-",
            color=colors[mouse],
            markersize=4.2,
            linewidth=1.0,
            label=mouse.replace("BAYLOR", ""),
        )
        for mouse in mice
    ]
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=len(mice),
        frameon=False,
        fontsize=6.7,
        handlelength=1.35,
        handletextpad=0.35,
        columnspacing=1.05,
        title="mouse identity",
        title_fontsize=6.7,
    )
    figure.tight_layout(rect=(0.0, 0.16, 1.0, 1.0), w_pad=1.25)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(OUTPUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
