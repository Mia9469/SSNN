# State-routing inference plan, 2026-10-02

This extends the 2026-09-29 outcome-blind protocol after acquisition. The
20 available pilot outcomes have been inspected: most sessions improve residual
prediction slightly. Therefore the new analyses are exploratory robustness
analyses, not a new preregistered confirmatory study. No session is selected
on predictive improvement, gain, angle, or null significance.

## Question and minimum evidence

Does the same recorded visual population have reproducible, state-dependent
100-ms response geometry beyond a pooled operator and scalar gain?

Use all 58 official NWBs and retain all feasibility exclusions. The original
drifting-grating gates, unit QC, 20-ms bins, first-100-ms exclusion, no smoothing,
and same neurons/coordinates apply. Each session is a distinct mouse, verified
from specimen metadata. Models describe condition-centered residual dynamics.

Primary controls: pooled dynamics and scalar gain. A nonnegative diagonal-gain
control in latent coordinates is a secondary comparator, not a cellular model.
State-specific operators need held-out predictive improvement; arbitrary
differences between fitted matrices or singular vectors are insufficient.

## Fixed robustness analyses

1. Repeated matched trial splits, dimension 12, five fixed seeds. Training-only
   condition/state means, common feature scale and PCA; inner trial validation
   selects ridge independently for each model. Report 20-ms and 100-ms errors.
   The first protocol split is reproducible; averaging five splits is a new
   robustness summary. Never treat folds/time bins as independent animals.
2. Local matching: pair stationary and locomotion trials within the same
   stimulus condition and a 300-s acquisition block, using nearest timestamps
   without neural outcomes. Require >=20 pairs and >=2 stimulus conditions.
   Fold allocation keeps each pair together; no train/test pairs overlap.
   A 600-s block is a declared sensitivity check; report reduced support.
3. Condition-by-time-block, paired label swaps, 999 permutations per eligible
   mouse. Refit condition/state means, common scale/PCA, ridge, and all models.
   Use the same held-out pair allocation in observed and shuffled fits. Report
   group nulls and plus-one p values; animal-level results also receive BH FDR.
   These swaps approximate local exchangeability; behavioral episode structure
   and unmeasured arousal can remain confounded. They do not establish causality.
4. Geometry: normalize each A^5 by its Frobenius norm; compare state maps and
   differences from scalar rescaling. Independent balanced training halves
   assess the cosine reproducibility of the *difference* of normalized maps.
   Leading vectors are descriptive; use rank-3 principal-subspace overlap when
   near-degenerate. Bootstrap/nulls refit, not just rotate finished operators.
   Independent-half centering requires >=2 trials/state/condition in each half
   and >=5 pairs/half overall. Mark unsupported halves as unavailable, never zero.
5. Check 8/16 dimensions, temporal-block held-out folds, and per-trial population
   mean removal. This last check reduces common rate fluctuations without
   claiming full control of cell-specific gain or arousal. Pupil tracking was
   absent in the first files inspected; later audit found usable eye-tracking
   interfaces in 17/22 eligible NWBs. Record missing measurements explicitly.

## Scope verdict and expansion record

Verdict B: focused, with a claim-critical robustness extension. State comparison
constitutes the claim; scalar gain, temporal matching and reliable geometry are
essential. Directed V1-to-higher-area maps are model-dependent summaries and do
not identify causal influence under shared drive. No additional dataset,
metabolic explanation, sparsity claim, or biological budget optimum is added.

Original judgment: state-dependent routing is worth testing on the downloaded
public recordings. External request: the user wants evidence from the same
population across behavior. Evidence-based decision: complete prediction and
geometry controls before proposing an insertion into the manuscript.

Freeze: one bounded empirical extension, one standalone figure and one report.
The existing manuscript and author-made figures stay intact. If held-out
advantage or reproducibility fails, report the failure and do not add routing
validation language. A positive finding supports behavior-associated effective
response geometry; gain-mediated causal routing and fixed-budget selection
require independently designed interventions.

Numerical audit: the first 199-permutation run reached its p-value resolution
floor for the group predictive endpoint. The final run uses 999 permutations
for every locally eligible session, with unchanged prediction endpoints and
pair selection. The geometry half-estimate availability gate above was added
after detecting that single-trial centering produced identically zero residuals
in under-supported halves. The preliminary 199-null outputs are retained in a
separate directory and are not the final geometry inference.

Pupil check, added after metadata discovery: filtered pupil-area coverage >=80%
per trial, mean-area caliper 0.5 session SD, same stimulus/300-s block, unchanged
training-support and >=20-pair gates. If this lacks support, record that failure.
Secondary additive adjustment fits linear/quadratic pupil effects to training
neural residuals only and compares adjusted and unadjusted data using exactly
the same valid-trial subset and held-out split. No caliper scan or new cohort is
used to search for a positive effect. These checks were not preregistered.
