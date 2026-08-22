#!/usr/bin/env python3
"""Assertion-based verification of the PRL Letter's core claims."""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np
from scipy.linalg import eigh, expm, null_space, solve_continuous_lyapunov

from fixed_budget_selection import (
    pair_excess_cost,
    pair_onset_cost,
    peak_gain,
    singular_value_from_budget,
)


HERE = Path(__file__).resolve().parent


def assert_close(x: float, y: float, tol: float, label: str) -> None:
    rel = abs(x - y) / max(1.0, abs(x), abs(y))
    if rel > tol:
        raise AssertionError(f"{label}: {x} vs {y}, relative error {rel}")


def main() -> None:
    # Some macOS Accelerate/SciPy combinations emit spurious overflow warnings
    # inside finite matrix products and expm scaling-and-squaring. All checks
    # below assert finite returned values explicitly.
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    rng = np.random.default_rng(11)

    # 1. Stable normal operators are contractive.
    n = 20
    raw = rng.standard_normal((n, n))
    sym = (raw + raw.T) / (2.0 * np.sqrt(n))
    j_normal = sym - (np.linalg.eigvalsh(sym).max() + 0.2) * np.eye(n)
    normal_gain = max(
        np.linalg.norm(expm(j_normal * t), 2) ** 2
        for t in np.linspace(0.0, 20.0, 250)
    )
    assert_close(normal_gain, 1.0, 1e-10, "normal contraction")

    # 2. Dense square-zero construction matches exact cost and gain.
    a = 0.2
    n = 24
    rank = 3
    q, _ = np.linalg.qr(rng.standard_normal((n, 2 * rank)))
    u, v = q[:, :rank], q[:, rank:2 * rank]
    s = 0.62
    k = u @ (s * np.eye(rank)) @ v.T
    if np.linalg.norm(k @ k) > 1e-12:
        raise AssertionError("constructed K is not square-zero")
    j = -a * np.eye(n) + k
    sigma = solve_continuous_lyapunov(j, -np.eye(n))
    cost_numeric = float(np.trace(sigma))
    cost_exact = n / (2.0 * a) + rank * s**2 / (4.0 * a**3)
    assert_close(cost_numeric, cost_exact, 1e-11, "square-zero cost")

    gain_exact, _ = peak_gain(a, s)
    gain_numeric = max(
        np.linalg.norm(expm(j * t), 2) ** 2
        for t in np.linspace(0.0, 30.0, 1000)
    )
    assert_close(gain_numeric, gain_exact, 2e-5, "square-zero gain")

    # 3. At one cost cap, rank concentration strictly orders the active gains.
    n = 64
    ratio = 1.10
    rank_gains = {}
    for r in [1, 2, 4, 8, 16]:
        sr = singular_value_from_budget(a, n, ratio, r)
        rank_gains[r], _ = peak_gain(a, sr)
    if not (
        rank_gains[1] > rank_gains[2] > rank_gains[4]
        and rank_gains[4] == rank_gains[8] == rank_gains[16] == 1.0
    ):
        raise AssertionError(f"rank ordering failed: {rank_gains}")

    # 4. Exact rank-one onset occurs at C*/Cnormal=1+2/N.
    threshold_ratio = 1.0 + 2.0 / n
    below = singular_value_from_budget(a, n, threshold_ratio - 1e-4, 1)
    above = singular_value_from_budget(a, n, threshold_ratio + 1e-4, 1)
    gain_below, _ = peak_gain(a, below)
    gain_above, _ = peak_gain(a, above)
    if gain_below != 1.0 or not gain_above > 1.0:
        raise AssertionError(
            f"threshold failed: below={gain_below}, above={gain_above}"
        )

    # 4b. A directed pair embedded in any stable real-eigenvalue spectrum has
    # an exact isotropic cost and onset Delta C=2/(a_i+a_k).
    for _ in range(32):
        a_i, a_k = np.exp(rng.uniform(np.log(0.05), np.log(3.0), size=2))
        m_on = 2.0 * math.sqrt(a_i * a_k)
        j_pair = np.array([[-a_i, m_on], [0.0, -a_k]])
        sigma_pair = solve_continuous_lyapunov(j_pair, -np.eye(2))
        baseline_pair = 1.0 / (2.0 * a_i) + 1.0 / (2.0 * a_k)
        numerical_excess = float(np.trace(sigma_pair) - baseline_pair)
        exact_excess = pair_excess_cost(a_i, a_k, m_on)
        onset_law = pair_onset_cost(a_i, a_k)
        assert_close(
            numerical_excess, exact_excess, 1e-11,
            "general-spectrum pair cost",
        )
        assert_close(exact_excess, onset_law, 1e-12, "pair onset law")
        omega_on = float(
            np.linalg.eigvalsh((j_pair + j_pair.T) / 2.0).max()
        )
        assert_close(omega_on, 0.0, 1e-11, "pair numerical-abscissa onset")
        for scale, should_amplify in [(0.999, False), (1.001, True)]:
            j_scaled = np.array([[-a_i, scale * m_on], [0.0, -a_k]])
            omega = float(
                np.linalg.eigvalsh((j_scaled + j_scaled.T) / 2.0).max()
            )
            if (omega > 0.0) != should_amplify:
                raise AssertionError(
                    f"pair onset bracket failed: scale={scale}, omega={omega}"
                )

    # 5. Fixed excess cost gives exact inverse-q task transfer.
    excess = 8.0
    q1, q2 = 0.1, 0.8
    k1 = math.sqrt(4.0 * a**3 * excess / q1)
    k2 = math.sqrt(4.0 * a**3 * excess / q2)
    transfer1 = (k1 / (a * math.e)) ** 2
    transfer2 = (k2 / (a * math.e)) ** 2
    assert_close(
        transfer1 / transfer2, q2 / q1, 1e-12,
        "inverse background-variance law"
    )

    # 5b. When the anisotropic-cost cross term is removed, the continuous
    # task/noise orientation problem is exactly a generalized Rayleigh quotient.
    n_align = 5
    raw_q = rng.standard_normal((n_align, n_align))
    q_metric = raw_q @ raw_q.T + np.eye(n_align)
    raw_r = rng.standard_normal((n_align, 2))
    r_metric = raw_r @ raw_r.T
    _, vectors = eigh(r_metric, q_metric)
    v_opt = vectors[:, -1]
    v_opt /= np.linalg.norm(v_opt)
    ratio_opt = float(v_opt @ r_metric @ v_opt / (v_opt @ q_metric @ v_opt))
    for _ in range(2000):
        trial = rng.standard_normal(n_align)
        trial /= np.linalg.norm(trial)
        ratio = float(trial @ r_metric @ trial / (trial @ q_metric @ trial))
        if ratio > ratio_opt + 1e-10:
            raise AssertionError("generalized task/noise mode is not optimal")
    u_basis = null_space(np.vstack([v_opt, q_metric @ v_opt]))
    if u_basis.shape[1] == 0:
        raise AssertionError("no output direction cancels the anisotropic cross term")
    u_opt = u_basis[:, 0]
    assert_close(float(u_opt @ v_opt), 0.0, 1e-10, "u-v orthogonality")
    assert_close(
        float(u_opt @ q_metric @ v_opt), 0.0, 1e-10,
        "anisotropic cross-term cancellation"
    )
    s_align = 0.4
    k_align = s_align * np.outer(u_opt, v_opt)
    j_align = -a * np.eye(n_align) + k_align
    sigma_align = solve_continuous_lyapunov(j_align, -q_metric)
    cost_align = float(np.trace(sigma_align))
    expected_align = (
        np.trace(q_metric) / (2.0 * a)
        + s_align**2 * float(v_opt @ q_metric @ v_opt) / (4.0 * a**3)
    )
    assert_close(cost_align, expected_align, 1e-10, "anisotropic reduced cost")

    results = json.loads(
        (HERE / "fixed_budget_selection_results.json").read_text()
    )
    reported = results["parameters"]["rank_one_amplification_threshold_ratio"]
    assert_close(reported, threshold_ratio, 1e-12, "reported threshold")
    for row in results["general_spectrum_pair_onset"]["matrix_validation"]:
        if not (
            row["cost_relative_error"] < 1e-11
            and row["onset_law_relative_error"] < 1e-12
            and abs(row["omega_at_onset"]) < 1e-11
        ):
            raise AssertionError(f"reported pair-onset validation failed: {row}")

    # 6. In the explicit robustness panel, higher-order single chains do not
    # exceed the square-zero rank-one frontier at matched stationary cost.
    for length, max_ratio in results["higher_order_chains"][
        "max_ratio_to_rank_one"
    ].items():
        if float(max_ratio) > 1.0005:
            raise AssertionError(
                f"L={length} chain exceeds rank-one frontier: {max_ratio}"
            )

    # 7. Even with known Q, a skew family of stable generators has the same
    # full stationary covariance.  This verifies the exact inverse-problem
    # bridge used in the Supplemental Material.
    n = 9
    raw_sigma = rng.standard_normal((n, n))
    sigma_target = raw_sigma @ raw_sigma.T + 0.5 * np.eye(n)
    q_diag = 0.5 + rng.random(n)
    q_drive = np.diag(q_diag)
    raw_omega = rng.standard_normal((n, n))
    omega = 0.35 * (raw_omega - raw_omega.T)
    j_cov = (-0.5 * q_drive + omega) @ np.linalg.inv(sigma_target)
    lyapunov_residual = (
        j_cov @ sigma_target + sigma_target @ j_cov.T + q_drive
    )
    if np.linalg.norm(lyapunov_residual) > 1e-10:
        raise AssertionError("covariance-equivalent generator fails Lyapunov identity")
    if np.max(np.real(np.linalg.eigvals(j_cov))) >= 0.0:
        raise AssertionError("covariance-equivalent generator is not stable")
    sigma_recovered = solve_continuous_lyapunov(j_cov, -q_drive)
    if not np.allclose(sigma_recovered, sigma_target, rtol=1e-9, atol=1e-9):
        raise AssertionError("known-drive covariance family did not recover Sigma")

    # 8. Heterogeneous diagonal gain can make a symmetric substrate
    # non-symmetric while preserving an exactly real spectrum through a
    # similarity to a symmetric matrix.
    n = 12
    raw_w = rng.standard_normal((n, n))
    w_sym = (raw_w + raw_w.T) / (2.0 * np.sqrt(n))
    gains = np.exp(np.linspace(-0.6, 0.6, n))
    g = np.diag(gains)
    sqrt_g = np.diag(np.sqrt(gains))
    inv_sqrt_g = np.diag(1.0 / np.sqrt(gains))
    symmetric_parent = sqrt_g @ w_sym @ sqrt_g
    gain_left = g @ w_sym
    gain_right = w_sym @ g
    if not np.allclose(gain_left, sqrt_g @ symmetric_parent @ inv_sqrt_g):
        raise AssertionError("left gain-dressed similarity identity failed")
    if not np.allclose(gain_right, inv_sqrt_g @ symmetric_parent @ sqrt_g):
        raise AssertionError("right gain-dressed similarity identity failed")
    eig_parent = np.sort(np.linalg.eigvalsh(symmetric_parent))
    for label, dressed in [("left", gain_left), ("right", gain_right)]:
        eig_dressed = np.linalg.eigvals(dressed)
        if np.max(np.abs(np.imag(eig_dressed))) > 1e-10:
            raise AssertionError(f"{label} gain-dressed spectrum is not real")
        if not np.allclose(np.sort(np.real(eig_dressed)), eig_parent, atol=1e-10):
            raise AssertionError(f"{label} gain-dressed spectrum changed")
        if np.linalg.norm(dressed - dressed.T) < 1e-3:
            raise AssertionError(f"{label} gain-dressed operator remained symmetric")

    # 9. With one fixed anatomical W, two diagonal gain states can redirect the
    # same input toward alternative outputs while matching generator spectrum,
    # covariance spectrum, and stationary fluctuation cost exactly.
    routing = json.loads((HERE / "gain_routing_results.json").read_text())
    matched = routing["matched_spontaneous_quantities"]
    switched = routing["routing"]
    if not (
        matched["cost_abs_difference"] < 1e-12
        and matched["generator_spectrum_max_abs_difference"] < 1e-12
        and matched["covariance_spectrum_max_abs_difference"] < 1e-12
    ):
        raise AssertionError(
            f"gain-routing spontaneous summaries do not match: {matched}"
        )
    if not (
        switched["routing_selectivity"] > 20.0
        and abs(switched["leading_input_abs_overlap"] - 1.0) < 1e-12
        and switched["routed_output_angle_degrees"] > 60.0
    ):
        raise AssertionError(f"gain-routing channel did not switch: {switched}")

    # 10. The same routing logic survives the full FlyWire v783 release under
    # the predeclared stability, cost, and stratified-null gates.
    # This is explicitly an in-silico intervention, not a measured gain state.
    flywire = json.loads(
        (HERE / "flywire_gain_routing_results.json").read_text()
    )
    if not (
        flywire["graph"]["neurons"] == 139255
        and flywire["graph"]["connected_model_nodes"] == 138639
        and flywire["graph"]["isolated_published_neurons_appended"] == 616
        and flywire["graph"]["edges"] == 15091983
        and flywire["graph"]["uses_all_edges"]
    ):
        raise AssertionError(f"FlyWire graph is incomplete: {flywire['graph']}")
    if "not an in-vivo" not in flywire["provenance"]["interpretation"]:
        raise AssertionError("FlyWire interpretation does not preserve the causal scope")
    if flywire["provenance"]["materialization_version"] != 783:
        raise AssertionError("FlyWire materialization is not v783")
    expected_mappings = {
        "720575940620900446": 720575940639259967,
        "720575940645521262": 720575940618238523,
        "720575940626307902": 720575940625054647,
    }
    actual_mappings = {
        old: row["v783_root"]
        for old, row in flywire["provenance"]["root_id_mappings"].items()
    }
    if actual_mappings != expected_mappings:
        raise AssertionError(f"unexpected v630-to-v783 root mapping: {actual_mappings}")
    fw_switch = flywire["switch"]
    fw_match = flywire["global_matching"]
    fw_null = flywire["null"]
    if not (
        flywire["predeclared_gates"]["all_pass"]
        and fw_switch["feeding_selectivity"] > 1.0
        and fw_switch["grooming_selectivity"] > 1.0
        and fw_switch["geometric_switch_factor"] > fw_null["q95"]
        and fw_null["p_one_sided"] <= 0.05
        and fw_match["spectral_radius_relative_difference"] < 0.02
        and fw_match["cost_relative_difference"] < 0.05
    ):
        raise AssertionError("full-connectome FlyWire routing gates failed")

    flywire_sensitivity = json.loads(
        (HERE / "flywire_gain_routing_sensitivity.json").read_text()
    )
    if not flywire_sensitivity["all_conditions_pass"]:
        raise AssertionError("FlyWire routing sensitivity summary failed")
    for row in flywire_sensitivity["conditions"]:
        if not (
            row["passes"]
            and row["feeding_selectivity"] > 1.0
            and row["grooming_selectivity"] > 1.0
            and row["geometric_switch_factor"] > row["null_q95"]
            and row["cost_relative_difference"] < 0.05
        ):
            raise AssertionError(f"FlyWire sensitivity condition failed: {row}")

    # 11. The empirical anchor panel is only a scope check: directed recurrent
    # weights, separation between bare and fitted radii, and readout-dependent
    # covariance exponents.  These assertions deliberately do not test
    # non-normal optimality.
    anchor = results["empirical_scope_check"]
    if not 0.0 <= anchor["eta_W_mean"] <= 0.06:
        raise AssertionError(f"unexpected eta_W: {anchor['eta_W_mean']}")
    if not anchor["rho_weight_mean"] > 1.2 * anchor["rho_eff_mean"]:
        raise AssertionError(
            "bare and fitted radii are not clearly separated: "
            f"{anchor['rho_weight_mean']} vs {anchor['rho_eff_mean']}"
        )
    if not anchor["rho_eff_mean"] < 1.0:
        raise AssertionError(f"fitted radius is not stable: {anchor['rho_eff_mean']}")
    nu_mem_low = float(anchor["nu_mem_range"][0])
    nu_spike_high = float(anchor["nu_spike_range"][1])
    if not nu_mem_low > nu_spike_high:
        raise AssertionError(
            "membrane/spike exponent ranges overlap: "
            f"{anchor['nu_mem_range']} vs {anchor['nu_spike_range']}"
        )

    # 12. Current multi-animal ALM scope check.  The manuscript uses these data
    # only for spectrum insufficiency and explicitly reports the negative
    # task/noise-orientation result.
    for dimension, filename in (
        (8, "alm_multisession_results_d8.json"),
        (12, "alm_multisession_results.json"),
        (16, "alm_multisession_results_d16.json"),
    ):
        alm = json.loads((HERE / filename).read_text())
        summary = alm["summary"]
        if summary["n_mice"] != 6 or summary["n_sessions"] != 17:
            raise AssertionError(f"unexpected ALM sample at d={dimension}: {summary}")
        if summary["full_gt_normal_mice"] != 6:
            raise AssertionError(
                f"full geometry does not exceed the surrogate median in all mice at d={dimension}"
            )
        if not math.isclose(
            summary["full_vs_normal_sign_p_greater"], 1.0 / 64.0
        ):
            raise AssertionError(f"unexpected exact ALM sign-test p at d={dimension}")
    primary_alm = json.loads((HERE / "alm_multisession_results.json").read_text())
    primary_summary = primary_alm["summary"]
    if not (
        primary_summary["noise_weighting_improves_mice"] == 3
        and math.isclose(
            primary_summary["noise_weighting_wilcoxon_p_greater"], 0.65625
        )
    ):
        raise AssertionError("the negative ALM task/noise-orientation result changed")

    print("PASS: stable normal contraction")
    print("PASS: exact square-zero cost and gain")
    print("PASS: fixed-budget rank concentration")
    print("PASS: exact response threshold")
    print("PASS: exact pair-supported onset for stable real-eigenvalue spectra")
    print("PASS: inverse-q task/noise alignment")
    print("PASS: generalized task/noise mode under explicit cross-term condition")
    print("PASS: higher-order chains below the rank-one frontier")
    print("PASS: known-drive covariance-equivalent stable family")
    print("PASS: gain-dressed real-spectrum effective dynamics")
    print("PASS: fixed-anatomy gain-gated routing")
    print("PASS: full-connectome FlyWire gain-gated routing and sensitivity")
    print("PASS: empirical scope-check quantities remain descriptive")
    print("PASS: multi-animal ALM spectrum-insufficiency scope check")
    print("PASS: negative ALM orientation result and latent-dimension sensitivity")


if __name__ == "__main__":
    main()
