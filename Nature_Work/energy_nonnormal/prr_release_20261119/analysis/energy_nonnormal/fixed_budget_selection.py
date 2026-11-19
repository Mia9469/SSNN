#!/usr/bin/env python3
"""Exact fixed-budget and pair-onset theory for non-normal response.

The analytically solvable family is

    J = -a I + K,        K^2 = 0,

with isotropic background drive unless stated otherwise.  Every member has the
same eigenvalue spectrum {-a}, while K controls feedforward, non-normal
coupling.  For singular values s_i of K,

    Tr Sigma = N/(2a) + sum_i s_i^2/(4a^3),

and the peak propagator norm is controlled by max_i s_i.  Under a stationary
cost cap, concentrating the available Frobenius norm into one singular direction
(rank-one feedforward coupling) therefore maximizes peak squared gain within this
class; above the amplification onset the optimum exhausts the cap.

For a directed Schur-mode pair with unequal decay rates,

    J_ik = [[-a_i, m], [0, -a_k]],

the stationary excess cost and transient-growth onset are also exact:

    Delta C_ik = m^2 / [2 a_i a_k (a_i + a_k)],
    Delta C_on,ik = 2 / (a_i + a_k).

The script produces the main fixed-budget figure and a compact JSON table used
by the PRL Letter.
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
from scipy.linalg import expm, solve_continuous_lyapunov
from scipy.optimize import minimize_scalar

from higher_order_frontier import analytic_chain_cost, peak_gain_numeric


HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)


def block_norm_sq(x: np.ndarray | float) -> np.ndarray | float:
    """Largest singular value squared of [[1, x], [0, 1]]."""
    x = np.asarray(x)
    return 1.0 + 0.5 * x**2 + 0.5 * x * np.sqrt(x**2 + 4.0)


def peak_gain(a: float, smax: float) -> tuple[float, float]:
    """Return max_t exp(-2at) ||I+tK||_2^2 and the maximizing time.

    For K^2=0 with orthogonal input/output singular subspaces, the full norm is
    the norm of the 2x2 block associated with the largest singular value.
    """
    if smax <= 0:
        return 1.0, 0.0

    def objective(t: float) -> float:
        return -math.exp(-2.0 * a * t) * float(block_norm_sq(smax * t))

    # The peak is O(1/a); 30/a is safely beyond it over the plotted range.
    result = minimize_scalar(
        objective, bounds=(0.0, 30.0 / a), method="bounded",
        options={"xatol": 1e-12}
    )
    gain = max(1.0, -float(result.fun))
    tpeak = 0.0 if gain == 1.0 else float(result.x)
    return gain, tpeak


def singular_value_from_budget(
    a: float, n: int, cost_ratio: float, rank: int
) -> float:
    """Equal singular value for a rank-r operator at a fixed total cost."""
    baseline = n / (2.0 * a)
    excess = max(0.0, cost_ratio * baseline - baseline)
    frob_sq = 4.0 * a**3 * excess
    return math.sqrt(frob_sq / rank) if rank > 0 else 0.0


def transfer_gain_at_peak(a: float, k: float) -> float:
    """Peak squared transfer |u^T exp(Jt) v|^2 for K=k u v^T, u⊥v."""
    return (k / (a * math.e)) ** 2


def pair_excess_cost(a_i: float, a_k: float, coupling: float) -> float:
    """Exact isotropic stationary excess cost of one directed Schur pair."""
    return coupling**2 / (2.0 * a_i * a_k * (a_i + a_k))


def pair_onset_cost(a_i: float, a_k: float) -> float:
    """Exact excess cost at Euclidean transient-growth onset for one pair."""
    return 2.0 / (a_i + a_k)


def mean_and_asymmetric_range(entry: dict) -> tuple[float, tuple[float, float]]:
    """Return mean and asymmetric error from a JSON {mean, range} entry."""
    mean = float(entry["mean"])
    low, high = map(float, entry["range"])
    return mean, (mean - low, high - mean)


def main() -> None:
    a = 0.20
    n = 64
    ranks = [1, 2, 4, 8, 16]
    ratios = np.linspace(1.0, 1.25, 121)
    baseline = n / (2.0 * a)

    curves: dict[int, dict[str, list[float]]] = {}
    for rank in ranks:
        gains, peaks, singular_values = [], [], []
        for ratio in ratios:
            s = singular_value_from_budget(a, n, float(ratio), rank)
            gain, tpeak = peak_gain(a, s)
            singular_values.append(s)
            gains.append(gain)
            peaks.append(tpeak)
        curves[rank] = {
            "cost_ratio": ratios.tolist(),
            "smax": singular_values,
            "Gmax": gains,
            "tpeak": peaks,
        }

    # Threshold for immediate transient amplification: omega=-a+smax/2 > 0.
    # For rank one this corresponds to Delta C=1/a.
    threshold_ratio_rank1 = 1.0 + (1.0 / a) / baseline

    matched_ratio = 1.10
    rank_grid = np.arange(1, n // 2 + 1)
    rank_gains = []
    rank_smax = []
    for rank in rank_grid:
        s = singular_value_from_budget(a, n, matched_ratio, int(rank))
        gain, _ = peak_gain(a, s)
        rank_smax.append(s)
        rank_gains.append(gain)

    # Exact onset law for a directed pair embedded in an arbitrary stable
    # real-eigenvalue spectrum.  The equal-decay point connects this law to the
    # square-zero panels above.
    pair_decay_sums = np.linspace(2.0 * a, 4.0, 400)
    pair_onset_costs = 2.0 / pair_decay_sums
    pair_validation = []
    for a_i, a_k in [(0.08, 0.31), (0.20, 0.20), (0.35, 0.90), (1.20, 2.40)]:
        coupling = 2.0 * math.sqrt(a_i * a_k)
        j_pair = np.array([[-a_i, coupling], [0.0, -a_k]])
        sigma_pair = solve_continuous_lyapunov(j_pair, -np.eye(2))
        baseline_pair = 1.0 / (2.0 * a_i) + 1.0 / (2.0 * a_k)
        numerical_excess = float(np.trace(sigma_pair) - baseline_pair)
        exact_excess = pair_excess_cost(a_i, a_k, coupling)
        onset_law = pair_onset_cost(a_i, a_k)
        omega_onset = float(
            np.linalg.eigvalsh((j_pair + j_pair.T) / 2.0).max()
        )
        pair_validation.append({
            "a_i": a_i,
            "a_k": a_k,
            "m_onset": coupling,
            "numerical_excess_cost": numerical_excess,
            "exact_onset_cost": exact_excess,
            "cost_relative_error": abs(numerical_excess - exact_excess)
            / exact_excess,
            "onset_law_relative_error": abs(exact_excess - onset_law)
            / onset_law,
            "omega_at_onset": omega_onset,
        })

    # Alignment rule.  At fixed excess background cost, transfer gain scales
    # as 1/q_v, where q_v is background variance along the amplified input.
    q_values = np.geomspace(0.05, 2.0, 120)
    excess_cost = 0.05 * baseline
    aligned_transfer = []
    k_values = []
    for q in q_values:
        k = math.sqrt(4.0 * a**3 * excess_cost / q)
        k_values.append(k)
        aligned_transfer.append(transfer_gain_at_peak(a, k))

    # Robustness beyond the square-zero class.  Compare the rank-one selection
    # frontier against single higher-order feedforward chains K^L=0 with matched
    # spectrum and matched stationary cost.  This panel makes explicit that the
    # square-zero result is not simply winning because no richer nilpotent
    # structures were tested.
    chain_lengths = [2, 3, 4, 6]
    chain_weights = np.linspace(0.0, 2.2, 220)
    chain_curves: dict[int, dict[str, list[float]]] = {}
    for length in chain_lengths:
        chain_ratios, chain_gains = [], []
        for weight in chain_weights:
            cost = analytic_chain_cost(a, n, length, float(weight))
            chain_ratios.append(cost / baseline)
            chain_gains.append(peak_gain_numeric(length, float(weight), a))
        chain_curves[length] = {
            "weight": chain_weights.tolist(),
            "cost_ratio": chain_ratios,
            "Gmax": chain_gains,
        }

    chain_ref_ratios = np.linspace(1.0, 1.30, 130)
    rank_one_ref_gain = []
    for ratio in chain_ref_ratios:
        s = singular_value_from_budget(a, n, float(ratio), 1)
        rank_one_ref_gain.append(peak_gain(a, s)[0])
    chain_ratio_to_rank1: dict[int, list[float]] = {}
    for length in [3, 4, 6]:
        cr = np.asarray(chain_curves[length]["cost_ratio"])
        g = np.asarray(chain_curves[length]["Gmax"])
        order = np.argsort(cr)
        interp_gain = np.interp(chain_ref_ratios, cr[order], g[order], left=1.0)
        chain_ratio_to_rank1[length] = (
            interp_gain / np.asarray(rank_one_ref_gain)
        ).tolist()

    # Empirical scope check for the main figure.  This is descriptive only: it
    # anchors the premises that trained recurrent SNN weights are directed, the
    # fitted effective propagator lives at a different operator level from bare
    # weights, and covariance exponents depend on the observation map.  It is not
    # used to prove the fixed-budget selection rule.
    trained = json.loads((HERE / "trained_snn_metrics.json").read_text())
    eta_entry = trained["weight_reciprocity"]["clusters_N512"]["baseline"]
    eta_mean, eta_err = mean_and_asymmetric_range(eta_entry)
    weight_rho = np.asarray(
        trained["spectral_radii"]["weight_rho_clusters"]["baseline"],
        dtype=float,
    )
    eff_rho = np.asarray(
        trained["spectral_radii"]["dmd_rho_clusters"]["baseline"],
        dtype=float,
    )
    nu_mem_entry = trained["covariance_exponents"]["nu_mem_clusters_baseline"]
    nu_spike_entry = trained["covariance_exponents"]["nu_spike_clusters_baseline"]
    nu_mem_mean, nu_mem_err = mean_and_asymmetric_range(nu_mem_entry)
    nu_spike_mean, nu_spike_err = mean_and_asymmetric_range(nu_spike_entry)
    empirical_anchor = {
        "condition": "clusters_N512_baseline",
        "seeds": 3,
        "eta_W_mean": eta_mean,
        "eta_W_range": eta_entry["range"],
        "rho_weight_mean": float(np.mean(weight_rho)),
        "rho_weight_std": float(np.std(weight_rho)),
        "rho_weight_values": weight_rho.tolist(),
        "rho_eff_mean": float(np.mean(eff_rho)),
        "rho_eff_std": float(np.std(eff_rho)),
        "rho_eff_values": eff_rho.tolist(),
        "nu_mem_mean": nu_mem_mean,
        "nu_mem_range": nu_mem_entry["range"],
        "nu_spike_mean": nu_spike_mean,
        "nu_spike_range": nu_spike_entry["range"],
    }

    output = {
        "parameters": {
            "a": a,
            "N": n,
            "baseline_cost": baseline,
            "matched_cost_ratio": matched_ratio,
            "rank_one_amplification_threshold_ratio": threshold_ratio_rank1,
            "alignment_excess_cost": excess_cost,
        },
        "fixed_budget_curves": {str(k): v for k, v in curves.items()},
        "rank_concentration": {
            "rank": rank_grid.tolist(),
            "smax": rank_smax,
            "Gmax": rank_gains,
        },
        "alignment": {
            "background_variance_qv": q_values.tolist(),
            "coupling_k": k_values,
            "peak_transfer_gain": aligned_transfer,
            "gain_per_excess_cost": (
                np.asarray(aligned_transfer) / excess_cost
            ).tolist(),
        },
        "general_spectrum_pair_onset": {
            "decay_rate_sum": pair_decay_sums.tolist(),
            "exact_onset_cost": pair_onset_costs.tolist(),
            "equal_decay_reference": {
                "a_i": a,
                "a_k": a,
                "decay_rate_sum": 2.0 * a,
                "exact_onset_cost": 1.0 / a,
            },
            "matrix_validation": pair_validation,
        },
        "higher_order_chains": {
            "cost_ratio_grid": chain_ref_ratios.tolist(),
            "rank_one_reference_Gmax": rank_one_ref_gain,
            "chains": {str(k): v for k, v in chain_curves.items()},
            "ratio_to_rank_one": {
                str(k): v for k, v in chain_ratio_to_rank1.items()
            },
            "max_ratio_to_rank_one": {
                str(k): float(np.max(v))
                for k, v in chain_ratio_to_rank1.items()
            },
        },
        "empirical_scope_check": empirical_anchor,
    }

    # Independent matrix-level validation of the closed forms.
    rng = np.random.default_rng(7)
    validation = []
    n_val = 32
    baseline_val = n_val / (2.0 * a)
    ratio_val = 1.12
    # Some Accelerate/SciPy builds emit spurious overflow warnings inside
    # scaling-and-squaring even though the returned finite matrices are valid.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        for rank in [1, 3, 8]:
            qmat, _ = np.linalg.qr(rng.standard_normal((n_val, 2 * rank)))
            umat = qmat[:, :rank]
            vmat = qmat[:, rank:2 * rank]
            sval = singular_value_from_budget(a, n_val, ratio_val, rank)
            kmat = umat @ (sval * np.eye(rank)) @ vmat.T
            jmat = -a * np.eye(n_val) + kmat
            k_squared_norm = float(np.linalg.norm(kmat @ kmat))
            sigma = solve_continuous_lyapunov(jmat, -np.eye(n_val))
            numerical_cost = float(np.trace(sigma))
            analytic_cost = baseline_val + rank * sval**2 / (4.0 * a**3)
            # The peak is near O(1/a); avoiding very late times also avoids
            # unnecessary scaling-and-squaring work in the independent expm check.
            times = np.linspace(0.0, 12.0 / a, 1200)
            numerical_gain = max(
                float(np.linalg.norm(expm(jmat * tt), 2) ** 2) for tt in times
            )
            analytic_gain, _ = peak_gain(a, sval)
            validation.append({
                "rank": rank,
                "K_squared_frobenius": k_squared_norm,
                "max_real_eigenvalue": float(np.max(np.linalg.eigvals(jmat).real)),
                "analytic_cost": analytic_cost,
                "numerical_cost": numerical_cost,
                "cost_relative_error": abs(numerical_cost - analytic_cost) / analytic_cost,
                "analytic_gain": analytic_gain,
                "numerical_gain_grid": numerical_gain,
                "gain_relative_error": abs(numerical_gain - analytic_gain) / analytic_gain,
            })
    output["matrix_validation"] = validation

    # Atomic replacement avoids forcing hydration of a stale iCloud placeholder
    # when regenerating this machine-readable output.
    results_path = HERE / "fixed_budget_selection_results.json"
    temporary_results_path = HERE / ".fixed_budget_selection_results.json.tmp"
    with temporary_results_path.open("w") as f:
        json.dump(output, f, indent=2)
    temporary_results_path.replace(results_path)

    plt.rcParams.update({
        "font.size": 8.0,
        "axes.linewidth": 0.9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.labelsize": 7.4,
        "ytick.labelsize": 7.4,
        "axes.labelsize": 8.0,
    })
    fig = plt.figure(figsize=(3.45, 3.55), constrained_layout=True)
    gs = fig.add_gridspec(
        2, 2, height_ratios=[1.0, 1.0],
        wspace=0.12, hspace=0.16
    )

    ax = fig.add_subplot(gs[0, 0])
    colors = plt.cm.viridis(np.linspace(0.08, 0.88, len(ranks)))
    rank_styles = {
        1: ("-", "o"),
        2: ("--", "s"),
        4: ("-.", "^"),
        8: (":", "D"),
        16: ((0, (5, 2, 1, 2)), "x"),
    }
    for index, (color, rank) in enumerate(zip(colors, ranks)):
        linestyle, marker = rank_styles[rank]
        ax.plot(
            ratios,
            curves[rank]["Gmax"],
            color=color,
            ls=linestyle,
            lw=1.75,
            marker=marker,
            ms=2.8,
            markevery=(3 * index, 22),
            markerfacecolor=("none" if marker != "x" else color),
            markeredgewidth=0.7,
            label=rf"$r={rank}$",
        )
    ax.axhline(1.0, color="0.45", ls=":", lw=1)
    ax.axvline(
        threshold_ratio_rank1, color="#B71C1C", ls="--", lw=1.2,
        label="onset"
    )
    ax.set_ylim(bottom=0.85)
    ax.set_xlabel(r"budget $C_0/C_{\rm normal}$")
    ax.set_ylabel(r"peak squared gain $G_{\max}^{(2)}$")
    ax.set_title("(a) Fixed cost", fontsize=8.3)
    ax.legend(frameon=False, fontsize=5.6, loc="upper left",
              handlelength=2.0, borderaxespad=0.2, labelspacing=0.2)
    # Visual anchor: the same Frobenius budget can be concentrated into a
    # single directed channel or diffused over many channels.
    k_rank1 = np.zeros((7, 7))
    k_rank1[1, 5] = 1.0
    k_diffuse = np.zeros((7, 7))
    for ii in range(1, 6):
        k_diffuse[ii, 6 - ii] = 1.0 / np.sqrt(5)
    for xpos, mat, lab in [(0.55, k_rank1, "one"), (0.78, k_diffuse, "diffuse")]:
        axins = ax.inset_axes([xpos, 0.20, 0.18, 0.22])
        axins.imshow(mat, cmap="Reds", vmin=0, vmax=1, interpolation="nearest")
        axins.set_xticks([])
        axins.set_yticks([])
        axins.set_title(lab, fontsize=4.9, pad=0.5, color="0.25")
        for spine in axins.spines.values():
            spine.set_linewidth(0.5)
            spine.set_edgecolor("0.55")

    ax = fig.add_subplot(gs[0, 1])
    colors_b = {"L=3": "#D81B60", "L=4": "#8E24AA", "L=6": "#3949AB"}
    chain_styles = {
        3: ("-", "o"),
        4: ("--", "s"),
        6: ("-.", "^"),
    }
    for index, length in enumerate([3, 4, 6]):
        linestyle, marker = chain_styles[length]
        ax.plot(
            chain_ref_ratios,
            chain_ratio_to_rank1[length],
            lw=1.8,
            ls=linestyle,
            marker=marker,
            ms=2.8,
            markevery=(4 * index, 24),
            markerfacecolor="none",
            markeredgewidth=0.7,
            color=colors_b[f"L={length}"],
            label=rf"$L={length}$",
        )
    ax.axhline(
        1.0,
        color="k",
        ls=(0, (1.5, 1.5)),
        lw=1.1,
        label="rank-one frontier",
    )
    ax.set_xlim(1.0, 1.30)
    ax.set_ylim(0.0, 1.08)
    ax.set_xlabel(r"cost $C_0/C_{\rm normal}$")
    ax.set_ylabel(r"chain/rank-one $G_{\max}^{(2)}$")
    ax.set_title("(b) Chain robustness", fontsize=8.3)
    ax.legend(frameon=False, fontsize=5.6, loc="lower left",
              handlelength=2.0, borderaxespad=0.2)

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(
        pair_decay_sums, pair_onset_costs, color="#1976B9", lw=2.2,
        label=r"$2/(a_i+a_k)$",
    )
    ax.scatter(
        [2.0 * a], [1.0 / a], s=22, color="#C62828", zorder=4,
        label=r"$a_i=a_k=a$",
    )
    ax.annotate(
        r"$\Delta C_{\rm on}=1/a$",
        xy=(2.0 * a, 1.0 / a), xytext=(1.00, 4.1),
        arrowprops=dict(arrowstyle="->", lw=0.7, color="0.25"),
        fontsize=5.5, color="0.20",
    )
    ax.set_xlim(0.25, 4.0)
    ax.set_ylim(0.30, 5.4)
    ax.set_xlabel(r"decay sum $a_i+a_k$")
    ax.set_ylabel(r"onset cost $\Delta C_{{\rm on},ik}$")
    ax.set_title("(c) Exact pair onset", fontsize=8.3)
    ax.legend(frameon=False, fontsize=5.3, loc="lower left",
              handlelength=1.2, borderaxespad=0.2, labelspacing=0.2)

    ax = fig.add_subplot(gs[1, 1])
    ax.loglog(
        q_values, aligned_transfer, color="#00897B", lw=2.2,
        label="exact optimum"
    )
    ref = aligned_transfer[0] * q_values[0] / q_values
    ax.loglog(q_values, ref, "k--", lw=1, label=r"$\propto q_v^{-1}$")
    ax.set_xlabel(r"variance $q_v$")
    ax.set_ylabel("task transfer")
    ax.set_title("(d) Task/noise", fontsize=8.3)
    ax.legend(frameon=False, fontsize=5.6, loc="upper right",
              handlelength=1.2, borderaxespad=0.2)

    # The trained-SNN scope check (empirical_anchor, computed above and written to
    # fixed_budget_selection_results.json) is rendered as a standalone figure by
    # fig_empirical_anchor.py; it is no longer a panel of this theory figure.
    fig.savefig(FIG_DIR / "fig_fixed_budget_selection.pdf", bbox_inches="tight")
    fig.savefig(
        FIG_DIR / "fig_fixed_budget_selection.png",
        dpi=220, bbox_inches="tight"
    )

    print(f"baseline cost = {baseline:.6g}")
    print(
        "rank-one amplification threshold C/C_normal = "
        f"{threshold_ratio_rank1:.6g}"
    )
    print(
        f"at C/C_normal={matched_ratio:.2f}: "
        f"Gmax(rank 1)={rank_gains[0]:.6g}, "
        f"Gmax(rank 4)={rank_gains[3]:.6g}, "
        f"Gmax(rank 16)={rank_gains[15]:.6g}"
    )
    print(
        "matrix validation max relative errors: "
        f"cost={max(v['cost_relative_error'] for v in validation):.3e}, "
        f"gain={max(v['gain_relative_error'] for v in validation):.3e}"
    )
    print(
        "pair-onset validation max errors: "
        f"cost={max(v['cost_relative_error'] for v in pair_validation):.3e}, "
        f"omega={max(abs(v['omega_at_onset']) for v in pair_validation):.3e}"
    )
    for length, ratios_to_rank1 in chain_ratio_to_rank1.items():
        print(
            f"L={length} chain max ratio to rank-one over displayed range = "
            f"{max(ratios_to_rank1):.6g}"
        )
    print(
        "empirical anchor (clusters baseline): "
        f"eta_W={eta_entry['range'][0]:.3f}-{eta_entry['range'][1]:.3f}, "
        f"rho(W)={empirical_anchor['rho_weight_mean']:.3f}, "
        f"rho_eff={empirical_anchor['rho_eff_mean']:.3f}, "
        f"nu_mem={nu_mem_entry['range'][0]:.3f}-{nu_mem_entry['range'][1]:.3f}, "
        f"nu_spike={nu_spike_entry['range'][0]:.3f}-{nu_spike_entry['range'][1]:.3f}"
    )
    print("saved figures/fig_fixed_budget_selection.{pdf,png}")
    print("saved fixed_budget_selection_results.json")


if __name__ == "__main__":
    main()
