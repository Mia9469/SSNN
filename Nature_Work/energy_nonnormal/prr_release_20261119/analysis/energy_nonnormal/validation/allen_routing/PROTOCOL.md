# Allen Visual Coding: state-dependent response geometry

Frozen before accessing routing outcomes, 2026-09-29.

## Question and scope

Does locomotion change finite-time directed response geometry beyond a common
operator or scalar gain, in the same neurons under matched visual stimulation?
User requested this extension following Zhu & Jia (2026),
https://doi.org/10.64898/2026.07.19.737501. It is high-leverage empirical support
for state dependence, not required to establish the existing mathematical laws.
The PRL manuscript and author-edited figures remain frozen pending results.

Data: Allen Brain Observatory Visual Coding Neuropixels, not Visual Behavior.
Official source https://allensdk.readthedocs.io/en/latest/visual_coding_neuropixels.html.
Download session NWBs (spikes, stimuli, running, unit QC) only; omit raw AP/LFP
and stimulus movie pixels. Keep full session inventory and file provenance on
/Volumes/MimiHDD/energy_nonnormal_data/allen_visual_coding.
The authors' exact session inclusion list has not yet been recovered: this is
a new analysis of the same public resource, not an exact reproduction claim.

## Predefined feasibility and analysis

1. Use recorded VISp, VISl, VISrl, VISal, VISpm, VISam units, QC presence ratio
   >= .95, amplitude cutoff <= .1, ISI violations <= .5, good quality if available.
   Retain identical units across both states. No outcome-based session selection.
2. Initial endpoint: drifting-grating repeats (orientation and temporal frequency
   matched), running >1 cm/s versus <=1 cm/s using mean stimulus-window speed.
   Require >=20 trials/state overall and >=5 trials/state in each retained
   stimulus condition, >=40 units and >=2 visual areas. Record all exclusions.
   If primary support is insufficient, report failure; natural movies would be
   a separately frozen follow-up, not an unreported replacement.
3. Bin spikes at 20 ms. Primary fit excludes first 100 ms of stimulus onset.
   No acausal temporal smoothing: smoothing can manufacture transient gain.
   Train/test split by trial, stratified by stimulus and state; equal train
   trials within each condition/state. Use a common training-only feature scale
   and a single shared PCA basis (12 dimensions; 8 and 16 sensitivity later).
   Subtract condition-and-state mean time courses estimated only from training
   trials. This isolates residual dynamics; do not call it stimulus transmission.
4. Fit discrete operators by ridge regression, hyperparameter selected using
   training-only trial-blocked validation. Compare separate state operators to
   (i) pooled operator and (ii) scalar multiples of pooled operator fitted on
   training data. Test one-step and 100-ms prediction without refitting on test.
5. Primary endpoint: held-out 100-ms residual-state prediction improvement of
   separate state models over pooled model and scalar-gain control, paired per
   session/animal. Report negative results and unstable fitted operators.
6. Describe A^k at 100 ms: spectra, squared singular gain, right/left singular
   direction overlap, participation ratio. Nearly degenerate leading singular
   values require subspace analysis, not interpreting arbitrary vector rotations.
   A label-shuffle null must preserve stimulus and trial/time-block structure and
   refit the whole pipeline. Uncorrected angles alone are not evidence.
7. Noise covariance of innovations is Q_epsilon; residual activity covariance
   is Sigma. Never substitute Sigma for latent drive Q without assumptions.
   For rho(A)<1 only, compute discrete Lyapunov cost using estimated Q_epsilon
   and separately a fixed common Q. Report these as model-dependent costs.
   Do not enforce stability by clipping eigenvalues and then call it empirical.

## Interpretation gates and stopping rule

State-specific prediction plus reproducible direction differences supports
state-dependent effective response geometry. It does not identify anatomical
W, cellular gains, or causal inter-area routing (shared drive remains possible).
A^k on PCA residuals is not by itself the measured input-output map C A^k B.
Actual sensory-routing claims additionally require train-defined fixed sensory
input/output/readout directions and independent held-out stimulus responses,
with controls for common visual drive, firing rate, pupil/arousal and time drift.

First deliver acquisition and cohort feasibility, then a clearly labelled pilot.
Lock any expanded confirmatory analysis after inspecting feasibility only.
Across-state spectra and fluctuation costs are measured, not assumed matched.
Same-spectrum controls test spectral insufficiency; they do not establish
biological budget optimization. Edge sparsity, effective rank and singular
channel concentration are distinct. Pure positive diagonal gain cannot create
literal new zero anatomical edges, although thresholds can change detected edges.
No claim that locomotion must lower effective rank is prespecified.

The existing square-zero finite-budget optimum and pair-supported onset law
cannot be tested as universal cortical laws with these observational recordings.
Budget optimization requires a specified feasible alternative set and controlled
cost/drive; a Pareto plot alone is not sufficient. Incompatibility is a valid
outcome, not a reason to scan for a positive metric.
