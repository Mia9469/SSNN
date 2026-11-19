#!/usr/bin/env python3
"""Cross-session response-geometry check in bilateral ALM photoinhibition data.

The public Chen et al. data contain simultaneous bilateral ALM spiking during
left, right, or bilateral optogenetic inhibition.  For each session, this
analysis fits a latent one-step propagator only from non-photostimulation
trials in the late delay period.  It then asks whether the fitted geometry
predicts recovery after the photoinhibition offset better than real normal
operators with the same eigenvalues.

This is an independent, multi-animal premise check.  It does not identify the
latent drive covariance or prove the fluctuation-budget optimum.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.linalg import block_diag, eigh, qr, svdvals
from scipy.ndimage import gaussian_filter1d
from scipy.stats import binomtest, spearmanr, wilcoxon


DT = 0.050
REL_START = -3.50
REL_STOP = 0.05
SMOOTH_SIGMA = 0.050
N_COMPONENTS = 12
N_FOLDS = 5
N_SURROGATES = 200
RNG_SEED = 23

# Photoinhibition runs from delay onset (-1.7 s relative to the cue) to -0.9 s.
FIT_WINDOW = (-0.85, -0.10)
BACKGROUND_WINDOW = (-1.65, -0.95)
EARLY_WINDOW = (-0.90, -0.75)
POST_WINDOW = (-0.65, -0.45)


def _chars(dataset: h5py.Dataset) -> str:
    values = np.asarray(dataset).ravel(order="F")
    return "".join(chr(int(value)) for value in values if int(value))


def _cell_strings(handle: h5py.File, dataset: h5py.Dataset) -> list[str]:
    output: list[str] = []
    for reference in np.asarray(dataset).ravel(order="F"):
        item = handle[reference]
        if isinstance(item, h5py.Dataset) and item.dtype.kind != "O":
            output.append(_chars(item))
        else:
            output.append("")
    return output


def _hash_values(handle: h5py.File, group: h5py.Group) -> dict[str, np.ndarray]:
    keys = _cell_strings(handle, group["keyNames"])
    references = np.asarray(group["value"]).ravel(order="F")
    return {key: np.asarray(handle[reference]) for key, reference in zip(keys, references)}


def load_session(path: Path) -> dict:
    with h5py.File(path, "r") as handle:
        properties = _hash_values(handle, handle["obj/trialPropertiesHash"])
        starts = np.asarray(handle["obj/trialStartTimes"]).ravel(order="F")
        trial_matrix = np.asarray(handle["obj/trialTypeMat"])
        if trial_matrix.shape[0] != len(starts):
            trial_matrix = trial_matrix.T

        cue_raw = np.asarray(properties["CueTime"])
        cue = cue_raw[0] if cue_raw.ndim == 2 else cue_raw.ravel()
        good = np.asarray(properties["GoodTrials"]).ravel(order="F").astype(bool)
        photo = np.asarray(properties["PhotostimulationType"]).ravel(order="F")

        # Trial labels refer to the instructed pole location, not the animal's
        # choice: HitR and ErrL share one stimulus, HitL and ErrR the other.
        early_lick = trial_matrix[:, 6] > 0
        instructed_right = (trial_matrix[:, 0] > 0) | (trial_matrix[:, 3] > 0)
        instructed_left = (trial_matrix[:, 1] > 0) | (trial_matrix[:, 2] > 0)
        valid = (
            good
            & np.isfinite(cue)
            & ~early_lick
            & (instructed_right | instructed_left)
            & np.isin(photo, [0.0, 1.0, 2.0, 3.0])
        )
        selected = np.flatnonzero(valid)
        labels = np.where(instructed_right[selected], 1, -1)

        values = np.asarray(handle["obj/eventSeriesHash/value"]).ravel(order="F")
        units = []
        first_trial = int(selected.min() + 1)
        last_trial = int(selected.max() + 1)
        for reference in values:
            unit = handle[reference]
            quality = float(np.asarray(unit["manual_quality_score"]).ravel()[0])
            event_trials = np.asarray(unit["eventTrials"]).ravel(order="F")
            event_times = np.asarray(unit["eventTimes"]).ravel(order="F")
            in_session = event_trials[(event_trials > 0) & (event_trials <= len(starts))]
            if quality <= 0 or len(in_session) == 0:
                continue
            if np.min(in_session) > first_trial or np.max(in_session) < last_trial:
                continue
            units.append((event_times, event_trials.astype(int)))

    return {
        "path": path.name,
        "session": path.stem.replace("data_structure_", ""),
        "mouse": path.stem.replace("data_structure_", "").split("_")[0],
        "trial_indices": selected,
        "trial_starts": starts[selected],
        "cue_times": cue[selected],
        "labels": labels,
        "photo_types": photo[selected].astype(int),
        "units": units,
        "n_trials_total": int(len(starts)),
    }


def make_trial_rates(record: dict) -> tuple[np.ndarray, np.ndarray]:
    edges = np.arange(REL_START, REL_STOP + 0.5 * DT, DT)
    centers = (edges[:-1] + edges[1:]) / 2.0
    n_trials = len(record["trial_indices"])
    n_units = len(record["units"])
    rates = np.zeros((n_trials, len(centers), n_units), dtype=np.float32)

    for unit_index, (event_times, event_trials) in enumerate(record["units"]):
        for trial_position, trial_index in enumerate(record["trial_indices"]):
            trial_id = int(trial_index + 1)
            relative = (
                event_times[event_trials == trial_id]
                - record["trial_starts"][trial_position]
                - record["cue_times"][trial_position]
            )
            rates[trial_position, :, unit_index] = np.histogram(relative, bins=edges)[0] / DT

    rates = gaussian_filter1d(rates, SMOOTH_SIGMA / DT, axis=1, mode="nearest")
    return centers, rates.astype(float)


def residual_latent_scores(
    rates: np.ndarray,
    centers: np.ndarray,
    labels: np.ndarray,
    photo_types: np.ndarray,
    n_components_max: int = N_COMPONENTS,
) -> tuple[np.ndarray, dict]:
    control = photo_types == 0
    fit_time = (centers >= FIT_WINDOW[0]) & (centers < FIT_WINDOW[1])

    control_mean = {}
    for label in (-1, 1):
        subset = control & (labels == label)
        if np.sum(subset) < 8:
            raise ValueError(f"too few control trials for label {label}")
        control_mean[label] = rates[subset].mean(axis=0)

    residual = np.empty_like(rates)
    for index, label in enumerate(labels):
        residual[index] = rates[index] - control_mean[int(label)]

    scale_source = residual[control][:, fit_time, :].reshape(-1, residual.shape[-1])
    unit_scale = scale_source.std(axis=0, ddof=1)
    keep = unit_scale > 0.20
    if np.sum(keep) < 10:
        raise ValueError("fewer than ten variable units remain")
    residual = residual[:, :, keep] / unit_scale[keep]

    source = residual[control][:, fit_time, :].reshape(-1, np.sum(keep))
    _, singular_values, components = np.linalg.svd(source, full_matrices=False)
    n_components = min(n_components_max, components.shape[0])
    components = components[:n_components]
    scores = residual @ components.T
    explained = float(
        np.sum(singular_values[:n_components] ** 2) / np.sum(singular_values**2)
    )
    return scores, {
        "n_units_stable": int(rates.shape[-1]),
        "n_units_variable": int(np.sum(keep)),
        "n_components": int(n_components),
        "explained_variance": explained,
    }


def fit_operator(
    scores: np.ndarray,
    centers: np.ndarray,
    photo_types: np.ndarray,
) -> tuple[np.ndarray, dict]:
    control_indices = np.flatnonzero(photo_types == 0)
    pair_time = np.flatnonzero(
        (centers[:-1] >= FIT_WINDOW[0]) & (centers[1:] < FIT_WINDOW[1])
    )
    x0 = scores[control_indices][:, pair_time, :]
    x1 = scores[control_indices][:, pair_time + 1, :]
    n_trials, n_pairs, dimension = x0.shape
    folds = np.arange(n_trials) % min(N_FOLDS, n_trials)
    candidates = np.logspace(-6, 2, 28)
    scale = np.trace(x0.reshape(-1, dimension).T @ x0.reshape(-1, dimension)) / dimension

    errors = []
    for coefficient in candidates:
        error = 0.0
        count = 0
        for fold in np.unique(folds):
            train = folds != fold
            test = folds == fold
            train0 = x0[train].reshape(-1, dimension)
            train1 = x1[train].reshape(-1, dimension)
            operator = (train1.T @ train0) @ np.linalg.inv(
                train0.T @ train0 + coefficient * scale * np.eye(dimension)
            )
            residual = x1[test].reshape(-1, dimension) - x0[test].reshape(-1, dimension) @ operator.T
            error += float(np.sum(residual**2))
            count += residual.size
        errors.append(error / count)

    best = int(np.argmin(errors))
    coefficient = float(candidates[best])
    flat0 = x0.reshape(-1, dimension)
    flat1 = x1.reshape(-1, dimension)
    operator = (flat1.T @ flat0) @ np.linalg.inv(
        flat0.T @ flat0 + coefficient * scale * np.eye(dimension)
    )
    return operator, {
        "n_control_trials": int(n_trials),
        "fit_pairs": int(n_trials * n_pairs),
        "ridge_scale_multiplier": coefficient,
        "validation_mse": float(errors[best]),
        "validation_scheme": "trial-blocked cross-validation",
    }


def _window_mean(scores: np.ndarray, centers: np.ndarray, window: tuple[float, float]) -> np.ndarray:
    mask = (centers >= window[0]) & (centers < window[1])
    return scores[:, mask, :].mean(axis=1)


def _residualize(values: np.ndarray, covariate: np.ndarray) -> np.ndarray:
    covariate = np.asarray(covariate)
    if covariate.ndim == 1:
        covariate = covariate[:, None]
    design = np.c_[np.ones(len(values)), covariate]
    return values - design @ np.linalg.lstsq(design, values, rcond=None)[0]


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 8 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def _block_normal_same_spectrum(matrix: np.ndarray) -> np.ndarray:
    values = np.linalg.eigvals(matrix)
    blocks = []
    used = np.zeros(len(values), dtype=bool)
    for index, value in enumerate(values):
        if used[index]:
            continue
        if abs(value.imag) < 1e-9:
            blocks.append(np.array([[value.real]]))
            used[index] = True
            continue
        if value.imag < 0:
            partner = int(np.argmin(np.abs(values - np.conjugate(value))))
            if not used[partner]:
                continue
            used[index] = True
            continue
        partner = int(np.argmin(np.abs(values - np.conjugate(value))))
        used[index] = True
        used[partner] = True
        blocks.append(np.array([[value.real, -abs(value.imag)], [abs(value.imag), value.real]]))
    output = block_diag(*blocks)
    if output.shape != matrix.shape:
        raise RuntimeError("failed to construct a real same-spectrum normal control")
    return output


def _abs_cosine(first: np.ndarray, second: np.ndarray) -> float:
    return float(abs(np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second))))


def analyze_session(
    path: Path,
    n_surrogates: int = N_SURROGATES,
    n_components: int = N_COMPONENTS,
) -> dict:
    record = load_session(path)
    if len(record["units"]) < 20:
        raise ValueError("fewer than twenty stable high-quality units")
    centers, rates = make_trial_rates(record)
    scores, pca = residual_latent_scores(
        rates,
        centers,
        record["labels"],
        record["photo_types"],
        n_components_max=n_components,
    )
    operator, fit = fit_operator(scores, centers, record["photo_types"])

    rho = float(np.max(np.abs(np.linalg.eigvals(operator))))
    early = _window_mean(scores, centers, EARLY_WINDOW)
    post = _window_mean(scores, centers, POST_WINDOW)
    stimulated = np.isin(record["photo_types"], [1, 2, 3])
    x0 = early[stimulated]
    observed_state = post[stimulated]
    photo_labels = record["photo_types"][stimulated]
    task_labels = record["labels"][stimulated]

    early_midpoint = 0.5 * sum(EARLY_WINDOW)
    post_midpoint = 0.5 * sum(POST_WINDOW)
    prediction_step = int(round((post_midpoint - early_midpoint) / DT))
    propagator = np.linalg.matrix_power(operator, prediction_step)
    predicted_state = x0 @ propagator.T
    early_energy = np.sum(x0**2, axis=1)
    observed_energy = np.sum(observed_state**2, axis=1)
    predicted_energy = np.sum(predicted_state**2, axis=1)

    log_early = np.log(early_energy + 1e-9)
    observed_residual = _residualize(np.log(observed_energy + 1e-9), log_early)
    predicted_residual = _residualize(np.log(predicted_energy + 1e-9), log_early)
    full_r = _safe_spearman(predicted_residual, observed_residual)

    # A stricter sensitivity check removes mean differences among the three
    # photoinhibition types and the two instructed trial types in addition to
    # early response magnitude.  The primary statistic above intentionally
    # retains between-perturbation directional information.
    condition_covariates = np.c_[
        log_early,
        photo_labels == 2,
        photo_labels == 3,
        task_labels > 0,
    ]
    observed_conditioned = _residualize(
        np.log(observed_energy + 1e-9), condition_covariates
    )
    predicted_conditioned = _residualize(
        np.log(predicted_energy + 1e-9), condition_covariates
    )
    full_conditioned_r = _safe_spearman(predicted_conditioned, observed_conditioned)

    rng = np.random.default_rng(RNG_SEED + sum(path.name.encode()))
    normal = _block_normal_same_spectrum(operator)
    surrogate_r = np.empty(n_surrogates)
    surrogate_conditioned_r = np.empty(n_surrogates)
    for index in range(n_surrogates):
        rotation, _ = qr(rng.normal(size=operator.shape), mode="economic")
        surrogate = rotation @ normal @ rotation.T
        surrogate_prop = np.linalg.matrix_power(surrogate, prediction_step)
        surrogate_energy = np.sum((x0 @ surrogate_prop.T) ** 2, axis=1)
        surrogate_residual = _residualize(np.log(surrogate_energy + 1e-9), log_early)
        surrogate_r[index] = _safe_spearman(surrogate_residual, observed_residual)
        surrogate_conditioned = _residualize(
            np.log(surrogate_energy + 1e-9), condition_covariates
        )
        surrogate_conditioned_r[index] = _safe_spearman(
            surrogate_conditioned, observed_conditioned
        )
    surrogate_r = surrogate_r[np.isfinite(surrogate_r)]
    surrogate_conditioned_r = surrogate_conditioned_r[
        np.isfinite(surrogate_conditioned_r)
    ]

    singular_values = svdvals(propagator)
    left, _, right_h = np.linalg.svd(propagator)
    top_right = right_h[0]
    top_left = left[:, 0]
    input_fraction = (x0 @ top_right) ** 2 / np.maximum(early_energy, 1e-12)
    output_fraction = (observed_state @ top_left) ** 2 / np.maximum(observed_energy, 1e-12)

    task_matrix = np.zeros_like(operator)
    for photo_type in np.unique(photo_labels):
        subset = x0[photo_labels == photo_type]
        mean = subset.mean(axis=0)
        task_matrix += (len(subset) / len(x0)) * np.outer(mean, mean)
    control = record["photo_types"] == 0
    background_mask = (centers >= BACKGROUND_WINDOW[0]) & (centers < BACKGROUND_WINDOW[1])
    background_samples = scores[control][:, background_mask, :].reshape(-1, scores.shape[-1])
    background = np.cov(background_samples.T)
    regularizer = max(1e-8, 1e-5 * np.trace(background) / background.shape[0])
    background_regularized = background + regularizer * np.eye(background.shape[0])
    dimension = operator.shape[0]
    task_only = eigh(task_matrix, subset_by_index=[dimension - 1, dimension - 1])[1][:, 0]
    task_noise = eigh(
        task_matrix,
        background_regularized,
        subset_by_index=[dimension - 1, dimension - 1],
    )[1][:, 0]
    task_only /= np.linalg.norm(task_only)
    task_noise /= np.linalg.norm(task_noise)

    gain = []
    power = np.eye(dimension)
    for _ in range(21):
        gain.append(float(svdvals(power)[0] ** 2))
        power = power @ operator

    return {
        "session": record["session"],
        "mouse": record["mouse"],
        "file": path.name,
        "n_trials_total": record["n_trials_total"],
        "n_trials_selected": int(len(record["trial_indices"])),
        "n_trials_control": int(np.sum(record["photo_types"] == 0)),
        "n_trials_photoinhibition": int(np.sum(stimulated)),
        **pca,
        **fit,
        "discrete_spectral_radius": rho,
        "peak_squared_gain_0_to_1s": float(np.max(gain)),
        "prediction_step": prediction_step,
        "prediction_horizon_s": prediction_step * DT,
        "full_geometry_residual_spearman": full_r,
        "same_spectrum_normal_median_spearman": float(np.median(surrogate_r)),
        "same_spectrum_normal_ci95": np.percentile(surrogate_r, [2.5, 97.5]).tolist(),
        "same_spectrum_normal_fraction_ge_full": float(
            (1 + np.sum(surrogate_r >= full_r)) / (len(surrogate_r) + 1)
        ),
        "conditioned_full_geometry_residual_spearman": full_conditioned_r,
        "conditioned_same_spectrum_normal_median_spearman": float(
            np.median(surrogate_conditioned_r)
        ),
        "conditioned_same_spectrum_normal_fraction_ge_full": float(
            (1 + np.sum(surrogate_conditioned_r >= full_conditioned_r))
            / (len(surrogate_conditioned_r) + 1)
        ),
        "leading_prediction_singular_value": float(singular_values[0]),
        "leading_input_fraction_mean": float(np.mean(input_fraction)),
        "leading_output_fraction_mean": float(np.mean(output_fraction)),
        "isotropic_fraction": float(1.0 / dimension),
        "singular_vs_perturbation_mode_alignment": _abs_cosine(top_right, task_only),
        "singular_vs_noise_weighted_mode_alignment": _abs_cosine(top_right, task_noise),
    }


def aggregate(reports: list[dict]) -> dict:
    mice = sorted({report["mouse"] for report in reports})

    def session_values(key: str) -> np.ndarray:
        return np.array([report[key] for report in reports], dtype=float)

    def mouse_means(key: str) -> np.ndarray:
        return np.array(
            [
                np.mean([report[key] for report in reports if report["mouse"] == mouse])
                for mouse in mice
            ],
            dtype=float,
        )

    session_full = session_values("full_geometry_residual_spearman")
    session_normal = session_values("same_spectrum_normal_median_spearman")
    session_null_tail = session_values("same_spectrum_normal_fraction_ge_full")
    session_conditioned_null_tail = session_values(
        "conditioned_same_spectrum_normal_fraction_ge_full"
    )
    conditioned_full = mouse_means("conditioned_full_geometry_residual_spearman")
    conditioned_normal = mouse_means(
        "conditioned_same_spectrum_normal_median_spearman"
    )
    full = mouse_means("full_geometry_residual_spearman")
    normal = mouse_means("same_spectrum_normal_median_spearman")
    task = mouse_means("singular_vs_perturbation_mode_alignment")
    weighted = mouse_means("singular_vs_noise_weighted_mode_alignment")
    input_fraction = mouse_means("leading_input_fraction_mean")
    isotropic = mouse_means("isotropic_fraction")
    stable = np.array([report["discrete_spectral_radius"] < 1 for report in reports])
    amplifying = np.array([report["peak_squared_gain_0_to_1s"] > 1.001 for report in reports])
    stable_reports = [
        report for report in reports if report["discrete_spectral_radius"] < 1
    ]
    stable_mice = sorted({report["mouse"] for report in stable_reports})
    stable_full = np.array(
        [
            np.mean(
                [
                    report["full_geometry_residual_spearman"]
                    for report in stable_reports
                    if report["mouse"] == mouse
                ]
            )
            for mouse in stable_mice
        ]
    )
    stable_normal = np.array(
        [
            np.mean(
                [
                    report["same_spectrum_normal_median_spearman"]
                    for report in stable_reports
                    if report["mouse"] == mouse
                ]
            )
            for mouse in stable_mice
        ]
    )

    mouse_reports = []
    for mouse in mice:
        subset = [report for report in reports if report["mouse"] == mouse]
        mouse_reports.append(
            {
                "mouse": mouse,
                "n_sessions": len(subset),
                "full_geometry_residual_spearman_mean": float(
                    np.mean([item["full_geometry_residual_spearman"] for item in subset])
                ),
                "same_spectrum_normal_spearman_mean": float(
                    np.mean(
                        [item["same_spectrum_normal_median_spearman"] for item in subset]
                    )
                ),
                "conditioned_full_geometry_residual_spearman_mean": float(
                    np.mean(
                        [
                            item["conditioned_full_geometry_residual_spearman"]
                            for item in subset
                        ]
                    )
                ),
                "conditioned_same_spectrum_normal_spearman_mean": float(
                    np.mean(
                        [
                            item["conditioned_same_spectrum_normal_median_spearman"]
                            for item in subset
                        ]
                    )
                ),
                "perturbation_mode_alignment_mean": float(
                    np.mean(
                        [item["singular_vs_perturbation_mode_alignment"] for item in subset]
                    )
                ),
                "noise_weighted_mode_alignment_mean": float(
                    np.mean(
                        [
                            item["singular_vs_noise_weighted_mode_alignment"]
                            for item in subset
                        ]
                    )
                ),
            }
        )

    def paired_wilcoxon(first: np.ndarray, second: np.ndarray, alternative: str) -> float:
        if len(first) < 3 or np.allclose(first, second):
            return float("nan")
        return float(wilcoxon(first, second, alternative=alternative).pvalue)

    def paired_sign_test(first: np.ndarray, second: np.ndarray) -> float:
        differences = first - second
        differences = differences[np.abs(differences) > 1e-12]
        if len(differences) == 0:
            return float("nan")
        result = binomtest(
            np.sum(differences > 0), len(differences), 0.5, alternative="greater"
        )
        return float(result.pvalue)

    return {
        "n_sessions": len(reports),
        "n_mice": len(mice),
        "inference_unit": "mouse; session metrics averaged within mouse",
        "mouse_values": mouse_reports,
        "stable_sessions": int(np.sum(stable)),
        "stable_and_amplifying_sessions": int(np.sum(stable & amplifying)),
        "stable_subset_n_mice": len(stable_mice),
        "stable_subset_full_minus_normal_median": (
            float(np.median(stable_full - stable_normal))
            if len(stable_mice)
            else float("nan")
        ),
        "stable_subset_full_gt_normal_mice": int(
            np.sum(stable_full > stable_normal)
        ),
        "stable_subset_full_vs_normal_wilcoxon_p_greater": paired_wilcoxon(
            stable_full, stable_normal, "greater"
        ),
        "session_full_geometry_residual_spearman_median": float(np.median(session_full)),
        "session_same_spectrum_normal_spearman_median": float(np.median(session_normal)),
        "session_full_minus_normal_median": float(
            np.median(session_full - session_normal)
        ),
        "session_full_gt_normal": int(np.sum(session_full > session_normal)),
        "sessions_full_above_null_95th_percentile": int(
            np.sum(session_null_tail < 0.05)
        ),
        "session_null_tail_fraction_median": float(np.median(session_null_tail)),
        "sessions_conditioned_full_above_null_95th_percentile": int(
            np.sum(session_conditioned_null_tail < 0.05)
        ),
        "session_conditioned_null_tail_fraction_median": float(
            np.median(session_conditioned_null_tail)
        ),
        "full_geometry_residual_spearman_median": float(np.median(full)),
        "same_spectrum_normal_spearman_median": float(np.median(normal)),
        "full_minus_normal_median": float(np.median(full - normal)),
        "full_gt_normal_mice": int(np.sum(full > normal)),
        "full_vs_normal_wilcoxon_p_greater": paired_wilcoxon(full, normal, "greater"),
        "full_vs_normal_sign_p_greater": paired_sign_test(full, normal),
        "conditioned_full_geometry_residual_spearman_median": float(
            np.median(conditioned_full)
        ),
        "conditioned_same_spectrum_normal_spearman_median": float(
            np.median(conditioned_normal)
        ),
        "conditioned_full_minus_normal_median": float(
            np.median(conditioned_full - conditioned_normal)
        ),
        "conditioned_full_gt_normal_mice": int(
            np.sum(conditioned_full > conditioned_normal)
        ),
        "conditioned_full_vs_normal_wilcoxon_p_greater": paired_wilcoxon(
            conditioned_full, conditioned_normal, "greater"
        ),
        "conditioned_full_vs_normal_sign_p_greater": paired_sign_test(
            conditioned_full, conditioned_normal
        ),
        "perturbation_mode_alignment_median": float(np.median(task)),
        "noise_weighted_mode_alignment_median": float(np.median(weighted)),
        "noise_weighting_improvement_median": float(np.median(weighted - task)),
        "noise_weighting_improves_mice": int(np.sum(weighted > task)),
        "noise_weighting_wilcoxon_p_greater": paired_wilcoxon(weighted, task, "greater"),
        "leading_input_fraction_median": float(np.median(input_fraction)),
        "isotropic_fraction_median": float(np.median(isotropic)),
        "input_fraction_vs_isotropic_wilcoxon_p_greater": paired_wilcoxon(
            input_fraction, isotropic, "greater"
        ),
    }


def make_figure(reports: list[dict], summary: dict, output: Path) -> None:
    rho = np.array([report["discrete_spectral_radius"] for report in reports])
    gain = np.array([report["peak_squared_gain_0_to_1s"] for report in reports])

    mice = sorted({report["mouse"] for report in reports})
    mouse_colors = {
        mouse: plt.get_cmap("tab10")(index % 10) for index, mouse in enumerate(mice)
    }
    marker_cycle = ("o", "s", "^", "D", "P", "X")
    mouse_markers = {
        mouse: marker_cycle[index % len(marker_cycle)]
        for index, mouse in enumerate(mice)
    }

    def paired_panel(ax, key_left: str, key_right: str, labels: list[str]) -> None:
        for report in reports:
            ax.plot(
                [0, 1],
                [report[key_left], report[key_right]],
                color=mouse_colors[report["mouse"]],
                alpha=0.22,
                lw=0.7,
            )
        for mouse in mice:
            subset = [report for report in reports if report["mouse"] == mouse]
            values = [
                np.mean([report[key_left] for report in subset]),
                np.mean([report[key_right] for report in subset]),
            ]
            ax.plot([0, 1], values, color=mouse_colors[mouse], lw=1.1, zorder=3)
            ax.scatter(
                [0, 1],
                values,
                color=mouse_colors[mouse],
                marker=mouse_markers[mouse],
                s=22,
                zorder=4,
            )
        ax.set_xticks([0, 1], labels)

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.linewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(1, 4, figsize=(7.1, 2.18))

    ax = axes[0]
    for mouse in mice:
        indices = [
            index for index, report in enumerate(reports)
            if report["mouse"] == mouse
        ]
        ax.scatter(
            rho[indices],
            gain[indices],
            s=24,
            color=mouse_colors[mouse],
            marker=mouse_markers[mouse],
            edgecolor="white",
            linewidth=0.4,
        )
    ax.axvline(1, color="0.4", ls="--", lw=0.9)
    ax.axhline(1, color="0.4", ls=":", lw=0.9)
    ax.set_xlabel(r"spectral radius $\rho(A)$")
    ax.set_ylabel(r"peak $\|A^k\|_2^2$")
    ax.set_title("(a) Stability and gain", fontsize=8.1)

    ax = axes[1]
    paired_panel(
        ax,
        "same_spectrum_normal_median_spearman",
        "full_geometry_residual_spearman",
        ["same $\\lambda$\nnormal", "full\ngeometry"],
    )
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_ylabel("residual Spearman $r$")
    ax.set_title("(b) Recovery prediction", fontsize=8.1)

    ax = axes[2]
    paired_panel(
        ax,
        "singular_vs_perturbation_mode_alignment",
        "singular_vs_noise_weighted_mode_alignment",
        ["perturb.\nonly", "perturb. /\nnoise"],
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel(r"alignment with $v_1$")
    ax.set_title("(c) Noise weighting", fontsize=8.1)

    ax = axes[3]
    paired_panel(
        ax,
        "isotropic_fraction",
        "leading_input_fraction_mean",
        ["isotropic", "observed\ninput"],
    )
    ax.set_ylim(bottom=0)
    ax.set_ylabel("leading-channel fraction")
    ax.set_title("(d) Channel concentration", fontsize=8.1)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=mouse_markers[mouse],
            linestyle="-",
            color=mouse_colors[mouse],
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
    figure.tight_layout(rect=(0.0, 0.15, 1.0, 1.0), w_pad=1.45)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=240, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--output-json", type=Path, default=Path("alm_multisession_results.json"))
    parser.add_argument("--figure", type=Path, default=Path("figures/fig_alm_multisession"))
    parser.add_argument("--surrogates", type=int, default=N_SURROGATES)
    parser.add_argument("--components", type=int, default=N_COMPONENTS)
    parser.add_argument(
        "--expected-files",
        type=int,
        default=0,
        help="fail unless this many session files are present (0 disables the check)",
    )
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    reports = []
    failures = []
    paths = sorted(args.data_dir.glob("data_structure_*.mat"))
    if args.expected_files and len(paths) != args.expected_files:
        raise RuntimeError(
            f"found {len(paths)} session files, expected {args.expected_files}; "
            "refusing a partial-record analysis"
        )
    for path in paths:
        try:
            report = analyze_session(
                path,
                n_surrogates=args.surrogates,
                n_components=args.components,
            )
            reports.append(report)
            print(
                report["session"],
                f"units={report['n_units_variable']}",
                f"stim={report['n_trials_photoinhibition']}",
                f"rho={report['discrete_spectral_radius']:.3f}",
                f"gain={report['peak_squared_gain_0_to_1s']:.2f}",
                f"r_full={report['full_geometry_residual_spearman']:.2f}",
                f"r_null={report['same_spectrum_normal_median_spearman']:.2f}",
            )
        except Exception as error:  # keep a transparent exclusion log
            failures.append({"file": path.name, "reason": repr(error)})
            print(f"SKIP {path.name}: {error}")

    if not reports:
        raise RuntimeError("no session passed the prespecified inclusion checks")
    summary = aggregate(reports)
    payload = {
        "dataset": {
            "name": "Chen et al. bilateral ALM photoinhibition",
            "doi": "10.5281/zenodo.6713616",
            "record_scope": "135 sessions from 39 mice in the complete archived release",
            "file_selection_rule": (
                "all data_structure_*.mat session files exposed individually by the "
                "Zenodo record; no selection based on fitted outcomes"
            ),
            "n_session_files_discovered": len(paths),
            "n_session_files_included": len(reports),
            "n_session_files_excluded": len(failures),
            "analysis_scope": (
                "independent multi-session premise check; control-trial operator "
                "predicts recovery after photoinhibition offset"
            ),
            "quality_exclusions": (
                "prespecified unit/trial support checks only; excluded files and reasons "
                "are retained below"
            ),
        },
        "parameters": {
            "dt_s": DT,
            "smooth_sigma_s": SMOOTH_SIGMA,
            "fit_window_relative_to_cue_s": FIT_WINDOW,
            "background_window_relative_to_cue_s": BACKGROUND_WINDOW,
            "early_window_relative_to_cue_s": EARLY_WINDOW,
            "post_window_relative_to_cue_s": POST_WINDOW,
            "n_components_max": args.components,
            "n_same_spectrum_surrogates_per_session": args.surrogates,
        },
        "summary": summary,
        "sessions": reports,
        "excluded": failures,
    }
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    make_figure(reports, summary, args.figure)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
