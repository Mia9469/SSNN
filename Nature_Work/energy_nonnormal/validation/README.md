# Empirical validation and scope checks

The directory now separates three empirical questions that should not be
collapsed into one claim:

1. Does fitted response geometry contain information absent from an exact
   same-spectrum normal control?  See `ALM_ANALYSIS_PROTOCOL.md`.
2. Does task engagement reorient low-dimensional stimulus representations
   relative to ongoing noise?  See `IBL_TASK_NOISE_ORIENTATION_NOTE.md`; an
   exploratory long-window contrast was positive but failed the
   short-latency/pre-movement control and is not manuscript evidence.
3. Do available causal S1--M1 perturbations directly support the proposed
   task/noise orientation rule?  See `S1_M1_DATASET_NOTE.md`; the current answer
   is no.

None of these datasets measures a full response--cost frontier, so none proves
global biological optimization of a fluctuation budget.
The consolidated verdict and dataset-by-dataset matrix are in
`TASK_NOISE_EVIDENCE_AUDIT.md`.

## Effective-operator proof of concept

Tests the Letter's **premise** on real cortical population data: is the fitted
effective operator *non-normal*, i.e. does it transiently amplify at a fixed
stable spectrum (numerical abscissa ω(J) > spectral abscissa α(J),
peak gain `G_max > 1`)? A stable **normal** operator has `G_max = 1` exactly.

It does **not** test rank-one optimality.  The generic fitter is retained as a
synthetic-verified diagnostic; manuscript evidence uses dataset-specific,
held-out protocols rather than treating an arbitrary population array as a
continuous stationary recording.

## Pipeline is verified

```
python3 effective_operator_analysis.py          # runs the synthetic self-test
```

The self-test builds a known square-zero non-normal operator and a matched
normal one, fits them from simulated data, and checks the pipeline recovers
`G_max > 1`, `ω > α`, low reciprocity for the former and `G_max ≈ 1`, `ω ≈ α`
for the latter. All checks PASS.

## Local S1--M1 archive status

The Piña Novo, Shepherd & Barrett (*eLife*, 2025) S1/M1 archive from Zenodo
record `14983570` is downloaded and checksum-verified on MimiHDD.  Its raw
trial structure is not a single stationary `neurons × time` array, so the
generic command below is only for genuinely continuous matrices:

```
python3 effective_operator_analysis.py path/to/file.mat --field <var> --dt <sec>
```

Output (`effective_operator_report.json`) reports, per fitted operator:

| field | meaning | theory expectation |
|---|---|---|
| `peak_transient_gain_Gmax` | `sup_t ‖e^{Jt}‖²` | `> 1` ⇒ non-normal amplification |
| `numerical_abscissa_omega` − `spectral_abscissa_alpha` | one-sided non-normality witness | `> 0` ⇒ non-normal |
| `eigenvector_condition_kappaV`, `henrici_departure` | departure from normality | large ⇒ strongly non-normal |
| `operator_reciprocity_eta` | 1 = symmetric, 0 = directed | low ⇒ directed |

For a perturbation→response (influence) matrix `R` (responder × stimulated), use
`perturbation_response_report(R)`: low `reciprocity_eta` + low
`singular_participation_ratio` ⇒ few dominant **directed** channels (the theory's
prediction).

## Honest scope

- The eLife set is peripheral/PV photostimulation ephys, so the natural object
  is a trial-resolved cross-area transfer analysis, **not** a neuron-to-neuron
  influence matrix.  Its completed orientation tests are negative/null; see
  `S1_M1_DATASET_NOTE.md`.
- Directly on-point single-neuron influence data (Chettih & Harvey 2019;
  Wagenmaker 2024 holographic; O'Shea/Shenoy 2022) is **request-only** — none is
  publicly deposited. Use `perturbation_response_report` if one of those is
  obtained.
