#!/usr/bin/env python3
"""Plot the trained-SNN measurements used as an empirical scope check.

The figure is intentionally descriptive.  It does not claim that training
solves the analytic constrained problem; it tests three consequences that are
relevant to the theory: trained recurrent weights remain directed, the bare
weight and effective dynamical radii live at different levels, and covariance
exponents depend on the recorded variable.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / "trained_snn_metrics.json").read_text())
OUT = HERE / "figures" / "fig_empirical_anchor"


def midpoint_error(entry: dict) -> tuple[float, float]:
    mean = float(entry["mean"])
    low, high = map(float, entry["range"])
    return mean, max(mean - low, high - mean)


plt.rcParams.update({
    "font.size": 8.0,
    "axes.linewidth": 0.9,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.labelsize": 7.4,
    "ytick.labelsize": 7.4,
    "axes.labelsize": 8.0,
})

fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.45))

# (a) Recurrent-weight reciprocity.
ax = axes[0]
wr = DATA["weight_reciprocity"]
labels = ["Clust.\nbase", "Clust.\nrate", "MNIST\nbase", "MNIST\nrate"]
entries = [
    wr["clusters_N512"]["baseline"],
    wr["clusters_N512"]["sigma2_fm"],
    wr["mnist_N512"]["baseline"],
    wr["mnist_N512"]["sigma2_fm"],
]
means, errs = zip(*(midpoint_error(e) for e in entries))
x = np.arange(len(labels))
ax.bar(x, means, yerr=errs, capsize=3, color=["#3949AB", "#7986CB", "#00897B", "#80CBC4"])
ax.axhline(1.0, color="#B71C1C", ls="--", lw=1.2, label="symmetric")
ax.axhline(0.0, color="0.35", ls=":", lw=1)
ax.set_ylim(-0.10, 1.08)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=7.1)
ax.set_ylabel(r"weight reciprocity $\eta_W$")
ax.set_title("(a) Directed weights", fontsize=8.3)
ax.legend(frameon=False, fontsize=6.8, loc="upper right",
          bbox_to_anchor=(0.98, 0.86), borderaxespad=0.0)

# (b) Bare-weight and effective radii.
ax = axes[1]
sr = DATA["spectral_radii"]
groups = ["Clust.\nbase", "Clust.\nrate", "MNIST\nbase", "MNIST\nrate"]
weight_arrays = [
    sr["weight_rho_clusters"]["baseline"],
    sr["weight_rho_clusters"]["sigma2_fm"],
    sr["weight_rho_mnist"]["baseline"],
    sr["weight_rho_mnist"]["sigma2_fm"],
]
dmd_arrays = [
    sr["dmd_rho_clusters"]["baseline"],
    sr["dmd_rho_clusters"]["sigma2_fm"],
    sr["dmd_rho_mnist"]["baseline"],
    sr["dmd_rho_mnist"]["sigma2_fm"],
]
weight_mean = np.array([np.mean(v) for v in weight_arrays])
weight_std = np.array([np.std(v) for v in weight_arrays])
dmd_mean = np.array([np.mean(v) for v in dmd_arrays])
dmd_std = np.array([np.std(v) for v in dmd_arrays])
w = 0.36
ax.bar(x - w / 2, weight_mean, w, yerr=weight_std, capsize=3, color="#EF6C00", label=r"$\rho(W)$")
ax.bar(x + w / 2, dmd_mean, w, yerr=dmd_std, capsize=3, color="#1565C0", label=r"$\rho_{\rm eff}$")
ax.axhline(1.0, color="0.35", ls=":", lw=1)
ax.set_xticks(x)
ax.set_xticklabels(groups, fontsize=7.1)
ax.set_ylabel("spectral radius")
ax.set_title(r"(b) $\rho(W)$ vs $\rho_{\rm eff}$", fontsize=8.3)
ax.legend(frameon=False, fontsize=6.8)

# (c) Probe dependence of the covariance exponent.
ax = axes[2]
ce = DATA["covariance_exponents"]
mem_entries = [
    ce["nu_mem_clusters_baseline"],
    ce["nu_mem_mnist_baseline"],
]
spike_entries = [
    ce["nu_spike_clusters_baseline"],
    ce["nu_spike_mnist_baseline"],
]
mem_mean, mem_err = zip(*(midpoint_error(e) for e in mem_entries))
spike_mean, spike_err = zip(*(midpoint_error(e) for e in spike_entries))
x2 = np.arange(2)
ax.bar(x2 - w / 2, mem_mean, w, yerr=mem_err, capsize=3, color="#8E24AA", label="membrane")
ax.bar(x2 + w / 2, spike_mean, w, yerr=spike_err, capsize=3, color="#43A047", label="spikes")
ax.axhspan(0.68, 0.78, color="#EF5350", alpha=0.16, label="cortical band")
ax.set_xticks(x2)
ax.set_xticklabels(["Clusters", "MNIST"])
ax.set_ylabel(r"covariance exponent $\nu$")
ax.set_title("(c) Probe-dependent exponent", fontsize=8.3)
ax.legend(frameon=False, fontsize=6.8, loc="upper left")

fig.tight_layout()
fig.savefig(str(OUT) + ".pdf", bbox_inches="tight")
fig.savefig(str(OUT) + ".png", dpi=220, bbox_inches="tight")
print(f"saved {OUT}.{{pdf,png}}")
