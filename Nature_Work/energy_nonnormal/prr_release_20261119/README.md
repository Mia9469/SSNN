# The Stronger Route Can Be the Worse Readout

**Flexible Routing under a Fluctuation Budget** — Mingyi Huang and Yuguo Yu.

This is the frozen source and numerical-record archive for the manuscript
prepared for *Physical Review Research*, not a record of journal acceptance.
The release includes the decision-feedback tests (Supplement S9) and the
activity-dependent gain loop (S10). No earlier ZIP is required.

- [Main paper](paper/main_submitted.pdf)
- [Supplemental Material](paper/supplement_submitted.pdf)
- [Article TeX](manuscript/main.tex)
- [Supplement TeX](manuscript/supplemental.tex)
- [File checksums](MANIFEST.json)

Fixed release: [`prr-flexible-routing-2026-11-19`](https://github.com/Mia9469/SSNN/releases/tag/prr-flexible-routing-2026-11-19).

## Contents and scope

`manuscript/` contains the current TeX, all figures, saved synthetic results,
and the newest analysis programs. `analysis/energy_nonnormal/` preserves the
relative paths of the original model and supplemental empirical pipelines.
`paper/` contains the two author-supplied submission PDFs byte-for-byte.
The public TeX title is synchronized with those PDFs. Figure PDFs include the
author's final artwork: programmatically regenerated layouts may differ.

This repository is not evidence that biological circuits optimize a variance
budget. The exact theory concerns the stated one-way module class; the gain-loop
tests retain finite-sample and transient-cost errors. Raw ALM/Allen recordings,
FlyWire connectivity tables, trial caches, private correspondence and signatures
are not redistributed. The empirical analyses require those separate inputs.

## Build the manuscript

Use a REVTeX 4.2-capable LaTeX installation. From `manuscript/`:

```sh
tectonic main.tex
tectonic supplemental.tex
```

Alternatively use LaTeX/BibTeX or upload that complete directory to Overleaf.
The supplement includes `allocation_supp.tex`, `strengthening_supp.tex`, and
`local_gain_loop_supp.tex`; retain all three inputs and the figures directory.

## Inspect and reproduce the current synthetic results

Install `requirements.txt` with Python 3.10+ (the model tests can also run
on the documented test runtime). Verify file hashes before running analyses:

```sh
python3 verify_release.py
cd manuscript
python3 strengthening_analysis.py
python3 feedback_allocation.py
python3 recurrent_feedback.py
python3 local_gain_loop.py
```

These programs overwrite their own derived JSON, not the input data. Work in a
copy to retain the frozen results. `strengthening_analysis.py` covers eight
synthetic recurrent networks and 48 window–budget settings. The following two
programs reproduce S9. `local_gain_loop.py` reproduces S10, including the
eight-route short-window residual error and the slower-actuator control.
It requires Numba and may take several minutes. Cross-platform floating-point
differences are expected; bitwise equality is not a mathematical requirement.

To regenerate the added figures after the analyses:

```sh
python3 strengthening_figures.py
python3 recurrent_feedback_figure.py
python3 local_gain_loop_figure.py
```

Figures use saved author-art components in `author_figures/` and model records
in `source_records/`; both are included. The names of supplementary figures in
the PDF are determined by LaTeX order, not by their file stems.

## Earlier theory and readout checks

From `analysis/energy_nonnormal/` run `fixed_budget_selection.py`,
`higher_order_frontier.py`, and `general_spectrum_selection.py` for their
respective checks. From its `prr_revision_routing_polish/` subdirectory run:

```sh
python3 verify_readout_budget.py
python3 finite_time_allocation.py
python3 window_discrimination.py
python3 window_crossover.py
python3 plot_cost_frontier.py
```

These reproduce pair costs, window-wise discrimination, temporal-decoder
controls and the modular allocation identities. Plotting scripts redraw
computed results, not the author's manual figure edits.

## Supplemental empirical analyses

- **FlyWire:** follow `analysis/energy_nonnormal/validation/FLYWIRE_DATA_PROVENANCE.md`
  and `prepare_flywire_v783_inputs.py`. Raw graph/node tables are obtained from
  the cited release. The routing script accepts explicit `--graph` and `--nodes`
  paths. Saved results include input hashes and selected node identifiers.
- **ALM:** obtain Zenodo dataset 6713616, following
  `analysis/energy_nonnormal/validation/ALM_ANALYSIS_PROTOCOL.md`. Supply its
  session directory to `alm_normal_control.py /path/to/sessions --dimension 12`
  in `analysis/energy_nonnormal/prr_revision_routing_polish/`; repeat for 8 and 16.
  This refits raw recordings. The supplied JSON alone is not trial-level data.
- **Allen:** acquisition, matching and analysis instructions are in
  `analysis/energy_nonnormal/validation/allen_routing/`. Raw NWBs and derived
  trial arrays are needed to refit. Change the documented external-drive defaults
  or use command-line path overrides. No download is launched by a LaTeX build.

Frozen summaries support inspection of reported numbers; they do not substitute
for raw-data refitting. `BUILD_VERIFICATION.json` specifies what was actually
checked for this release. The archive grants no new license to third-party data.
