# Flexible neural routing PRL draft

Canonical files:

- `main_prl.tex` — active PRL Letter draft. Its central question is how one
  anatomical circuit can switch task-relevant input--output routes across states
  without rewiring or globally approaching spectral criticality.  The narrative
  hierarchy is: flexible routing (phenomenon), response geometry (hidden state),
  gain (control), and fluctuation budget (selection).
- `supplemental.tex` — PRL-synchronized Supplemental Material: definitions,
  proofs, numerical checks, scope, and extensions.
- `gain_routing.py` — exact fixed-anatomy construction underlying Fig. 1.
  It writes `fig_gain_routing_generated.{pdf,png}` for numerical checking and
  never overwrites the manually curated canonical artwork
  `figures/fig_gain_routing.pdf`.
- `flywire_gain_routing.py` — full-release generalization on FlyWire v783
  (139,255 published neurons; 138,639 connected model nodes plus 616 zero-edge
  nodes; 15,091,983 directed model edges),
  with published feeding/grooming endpoints and 999 route-label nulls jointly
  stratified by transmitter sign and separate in/out-strength/degree bins in
  the primary test.
- `validation/prepare_flywire_v783_inputs.py` — downloads the pinned v783
  connectivity table and the complete 139,255-neuron Codex table, verifies
  SHA-256 hashes, and materializes the exact two FlyWire input files expected
  by the analysis.
- `validation/FLYWIRE_DATA_PROVENANCE.md` — immutable source URLs, commit,
  transformations, expected hashes, and rerun instructions for the FlyWire
  inputs.
- `flywire_gain_routing_sensitivity.py` — robustness to module size, gain,
  propagator balance, communicability horizon/damping, ranking rule, and the
  numerical zero guard for the full-connectome routing result, using 99 new
  stratified nulls per condition.
- `fixed_budget_selection.py` — Fig. 2: exact fixed-spectrum/fixed-budget
  selection, matched-cost higher-order-chain checks, the exact pair-supported
  onset law for unequal decay rates, and the task/noise allocation law.
- `higher_order_frontier.py` — extended sweep of higher-order feedforward chains.
- `verify_fixed_budget_claims.py` — assertion-based checks of the central claims,
  including the exact general-spectrum pair onset, covariance-equivalent stable
  generators, gain-dressed real-spectrum effective dynamics, higher-order
  matched-cost comparisons, current FlyWire outputs, and the multi-animal ALM
  scope check, including its negative orientation result.
- `multisession_alm_validation.py` — prespecified multi-session validation in
  bilateral mouse ALM photoinhibition data.  The fitted operator uses only
  non-photostimulation trials; recovery after inhibition is compared with
  real block-normal controls having exactly the same discrete eigenvalues.
- `validation/ALM_ANALYSIS_PROTOCOL.md` — analysis and exclusion rules frozen
  before inspecting the complete 18-file ALM result.
- `validation/S1_M1_DATASET_NOTE.md` — checksum, extraction, and scope record
  for the 19.8-GB S1--M1 photostimulation archive.  Its completed hand- and
  PV-stimulation orientation tests are negative/null and are not used as
  positive evidence in the Letter.
- `validation/TASK_NOISE_EVIDENCE_AUDIT.md` — consolidated audit showing why no
  currently available dataset establishes biological task/noise orientation or
  global fluctuation-budget optimization.
- `validation/download_zenodo_record.py` and
  `validation/download_zenodo_parallel.py` — resumable, checksum-verified
  Zenodo download helpers.
- `fig_empirical_anchor.py` — standalone trained SNN scope-check figure retained
  for the Supplemental Material.
- `fig_alm_main.py` and `fig_general_spectrum.py` — plotting scripts for the
  multi-animal and general-spectrum panels.

Generated outputs:

- `main_prl.pdf`
- `supplemental.pdf`
- `figures/fig_gain_routing.pdf`
- `figures/fig_gain_routing_submission.pdf` — submission copy of the manually
  curated Fig. 1 with the corrected joint-stratification permutation result;
  the original curated file remains untouched.
- `figures/fig_flywire_gain_routing.pdf`
- `figures/fig_flywire_gain_routing_sensitivity.pdf`
- `figures/fig_fixed_budget_selection.pdf`
- `figures/fig_alm_multisession.pdf`
- `figures/fig_higher_order_frontier.pdf`
- `figures/fig_empirical_anchor.pdf`
- `fixed_budget_selection_results.json`
- `gain_routing_results.json`
- `flywire_gain_routing_results.json`
- `flywire_gain_routing_sensitivity.json`
- `higher_order_frontier_results.json`
- `alm_multisession_results.json`
- `alm_multisession_results_d8.json`
- `alm_multisession_results_d16.json`

The Letter selectively incorporates the general scientific premise developed
around the earlier Nature Matters Arising, while remaining an independent
positive-mechanism paper:

- no Matters Arising text, figures, correspondence, or Nature-specific numerical
  analyses are reused;
- spontaneous summaries are treated as defining a feasible class rather than a
  unique mechanism;
- state-dependent neuronal gain is recast as a control of population response
  geometry, not merely scalar sensitivity;
- the full v783 connectome test shows that equal-size, route-defined gain
  fields can switch published feeding and grooming transformations while global
  stability and finite-horizon fluctuation cost remain matched; this is an
  in silico anatomical-realizability test, not evidence that these gain states
  were measured in vivo;
- the fluctuation-budget rule explains why limited response capacity is
  concentrated into task-relevant, low-background-variance routes;
- the independent mouse ALM cohort tests whether geometry predicts
  post-inhibition recovery beyond exact same-spectrum normal controls.  Its
  exploratory task/noise-orientation result is negative and neither quantity is
  presented as proof of fixed-budget optimality.

Build with:

```bash
tectonic main_prl.tex
tectonic supplemental.tex
```

The archived numerical environment used Python 3.9.6 with the package versions
in `requirements-reproducibility.txt`.

Verify with:

```bash
python3 fixed_budget_selection.py
python3 gain_routing.py
python3 flywire_gain_routing.py
python3 flywire_gain_routing_sensitivity.py
python3 higher_order_frontier.py
python3 multisession_alm_validation.py /path/to/zenodo_6713616/sessions \
  --expected-files 18 --surrogates 200 --components 12
python3 verify_fixed_budget_claims.py
python3 fig_empirical_anchor.py
```

The ALM source files are the 18 individually exposed
`data_structure_*.mat` sessions in Zenodo record `6713616` (six mice).  They
can be fetched without outcome-based selection using:

```bash
python3 validation/download_zenodo_record.py 6713616 /path/to/sessions \
  --pattern 'data_structure_*.mat' --concurrent-files 3 --workers-per-file 1
```

The full-connectome scripts expect `../flywire_spontaneous/Connectivity_783.parquet`
and `../flywire_spontaneous/Completeness_783_all_139255.csv`.  Recreate and
verify both files directly from the pinned public sources with:

```bash
python3 validation/prepare_flywire_v783_inputs.py ../flywire_spontaneous
```

The node table preserves the official Codex v783 order.  The 138,639 neurons
represented in the Shiu connectivity table carry every model edge; the other
616 published neurons enter as zero-edge nodes.  Exact URLs, transformations,
and hashes are recorded in `validation/FLYWIRE_DATA_PROVENANCE.md`, while the
primary result JSON additionally records v630-to-v783 root mappings and every
task/gain-module neuron ID.

The tagged public release contains the active manuscript source, code,
machine-readable outputs, figures, and provenance documents.  It excludes local
recordings, scratch products, superseded drafts, and historical v630 analyses.
