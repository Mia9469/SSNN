# Locomotion versus stationary: data acquisition and pilot

Requested 2026-09-29. This is an independent extension of the frozen PRL draft.
Read `PROTOCOL.md` for the outcome-blind plan and interpretation gates.

Data destination:
`/Volumes/MimiHDD/energy_nonnormal_data/allen_visual_coding/`

Official Visual Coding release: 58 session NWBs, 144,965,967,784 bytes
(approximately 135.01 GiB; exact inventory is `download_manifest.json`).
No raw AP, probe LFP or image/movie templates are downloaded.
This is the same public resource used by Zhu & Jia, but their exact analysis
cohort has not yet been independently recovered.

## Execution

Run using the system Python with numpy/scipy/pandas/h5py/requests:

```
python3 download_allen.py --workers 2 --analyze
```

The current detached worker writes to `download.log` and its PID is in
`worker.json` on the external drive. Do not launch duplicate workers.
The Mac must remain awake, network-connected, with the drive mounted.
Each 512-KiB segment has its own cache key; Content-Range, byte length and
ETag are validated. Completed sessions receive SHA-256 and HDF5/units-table
checks. SHA-256 is a local integrity record, not a publisher-provided hash.
Downloads resume from validated segments after interruption. Partial files
are not fed into the analysis. Full-session files are followed automatically
by feasibility gates and a pilot, with per-session analysis logs.

Output files: `analysis/session_*_report.json`, matched trial tables and
preprocessed trial arrays. Rejections and failures are retained.
For each eligible session, the pilot uses identical neurons, matched grating
conditions and trial counts, a shared training-only PCA basis, and held-out
trials. It compares pooled, scalar-gain, diagonal-gain and separate operators.
The diagonal-gain comparator acts in latent coordinates and is not evidence
for cellular gain modulation on an identified anatomical W.

## Validation and limitations

`python3 analyze_session.py --selftest` tests recovery of a known synthetic
state-dependent channel and the Lyapunov residual. It is NOT a real-data result.
NWB HDF5 group names were verified remotely against session 715093703.
Acquisition and the bounded biological extension are complete (2026-10-02).
Read `RESULTS_20261002.md` for all effect sizes and limitations; final external
outputs are in `state_inference_final/` and `pupil_robustness/`, not the earlier
preliminary `state_inference/` directory.
On this Mac, NumPy/Accelerate may emit floating-point matmul warnings; non-finite
results are rejected by strict JSON serialization and never reported as success.

This first analysis is exploratory. A positive prediction gap does not establish
causal rerouting, actual task input-output maps, matched-spectrum/matched-cost
biology, low-rank optimality, or brain optimization of a fluctuation budget.
Direction reliability, time-blocked null refits, nuisance controls and per-mouse
aggregation are required before manuscript integration. No original figures or
manuscript text are overwritten.

Final inference, using existing cached trial arrays:

```
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 state_inference.py --workers 3 --permutations 999 --out /Volumes/MimiHDD/energy_nonnormal_data/allen_visual_coding/state_inference_final
VECLIB_MAXIMUM_THREADS=1 python3 pupil_check.py
python3 plot_state_evidence.py
```

Final cohort: 22 mice, with 14 supporting strict local time/stimulus matching.
Held-out state prediction is positive beyond common/scalar/latent-diagonal gain.
Independent-half channel direction has limited support; pupil-controlled
causality and biological budget optimization remain unestablished. The report
includes both positive and inconclusive checks and a bounded SI-ready paragraph.
