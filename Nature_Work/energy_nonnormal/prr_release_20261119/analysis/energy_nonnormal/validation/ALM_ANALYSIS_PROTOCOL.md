# Bilateral ALM photoinhibition: analysis protocol

This protocol was fixed before the complete 18-session result was inspected.
The analysis is an independent scope check of the response-geometry premise,
not a test of fixed-budget optimality.

## Dataset and file rule

- Source: Chen et al., *Cell* (2021), Zenodo record `6713616`.
- Use every `data_structure_*.mat` file exposed individually by the Zenodo
  record (18 sessions from six mice). No session is selected by a fitted
  outcome.
- Require a manual quality score above zero and coverage from the first to the
  last selected task trial. Exclude a session only if fewer than 20 such units
  remain, fewer than eight control trials remain for either instructed trial
  type, or fewer than ten variable units survive scaling. Keep an explicit
  exclusion log.

## Operator and perturbation test

- Bin spikes at 50 ms and smooth with a 50-ms Gaussian kernel.
- Remove the instructed-trial mean separately for the two pole locations.
- Fit a 12-dimensional PCA basis and a ridge-regularized one-step propagator
  using only non-photostimulation trials in the late delay period
  (`-0.85` to `-0.10` s relative to the cue). Select ridge strength by
  trial-blocked cross-validation.
- Use 12 dimensions for the primary result; repeat the complete analysis at 8
  and 16 dimensions as a prespecified sensitivity check.
- Predict recovery from the immediate post-inhibition state (`-0.90` to
  `-0.75` s) to a later post-inhibition state (`-0.65` to `-0.45` s).
- Primary endpoint: Spearman correlation between observed and predicted log
  response energy after both are residualized against early response
  magnitude.
- Compare with real block-normal operators having exactly the fitted discrete
  eigenvalue spectrum and randomized orthogonal orientation.
- Strict sensitivity endpoint: additionally remove mean differences among
  left, right, and bilateral inhibition and the two instructed trial types.

## Inference and scope

- The inferential unit is the mouse: average session metrics within mouse,
  then use paired one-sided Wilcoxon and exact sign tests.
- Report session-level values and the number of fitted operators that are
  stable and transiently amplifying, but do not exclude sessions based on
  spectral radius, gain, or prediction outcome.
- Perturbation/noise alignment and channel concentration are exploratory
  secondary summaries because this experiment does not independently specify
  the theorem's task matrix or latent drive covariance.
- A positive primary result supports the claim that response geometry contains
  perturbational information absent from the eigenvalue spectrum. It does not
  establish the fluctuation-budget selection rule or anatomical non-normality.
