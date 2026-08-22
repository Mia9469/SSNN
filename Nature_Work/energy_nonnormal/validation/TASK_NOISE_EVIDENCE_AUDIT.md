# Task/noise orientation and fluctuation-budget evidence audit

## Bottom line

No dataset currently available on MimiHDD provides defensible positive
evidence that a biological circuit implements the proposed task/noise
orientation rule, and none can establish global optimization of a fluctuation
budget.  This is a data-identifiability limitation, not a failure of the
mathematical result.

## What each dataset can and cannot test

| Dataset | Animals/sessions | Perturbation | Task metric independent of fit | Stationary-noise proxy | Matched-spectrum geometry null | Result relevant here |
|---|---:|---|---|---|---|---|
| Bilateral ALM | 6 mice, 17 usable sessions | causal photoinhibition | incomplete | yes | yes | geometry beats same-spectrum normal controls; orientation negative |
| Macaque Area 2 | 2 monkeys, 4 sessions | mechanical bump | proxy only | yes | yes | no cross-session geometry/null or orientation support |
| S1--M1 hand stimulation | 9 mice, 13 recordings | peripheral hand stimulus | hand direction | yes | no | orientation negative |
| S1--M1 PV dose | 3 mice, 8 directional recordings | cortical PV activation | incomplete | yes | no | orientation null |
| IBL active/passive | 29 mice, 48 usable sessions | none | visual class labels | yes | analytic/label axes | late-window effect fails early and fixed-axis controls |
| FlyWire connectome | complete adult-fly graph | simulated gain/routing | synthetic | no measured task-state noise | degree/reciprocity nulls | anatomical controllability only; cannot test budget optimization |

## Defensible empirical claim retained in the Letter

The multi-animal ALM analysis supports **spectrum insufficiency**: recovery from
a causal displacement contains information in fitted response geometry that
exact same-spectrum normal operators omit.  It does not support task/noise
orientation or adaptive budget optimization.

## Why none of the current datasets proves optimization

Evidence for optimization requires more than a positive task/noise ratio.  At
minimum, the same circuit must be observed across multiple accessible response
geometries or gain states while the following are measured independently:

1. the stable eigenvalue spectrum;
2. stationary fluctuation cost (or a justified biophysical proxy);
3. finite-time responses to multiple controlled input directions;
4. task relevance and background-noise structure; and
5. same-spectrum/random-orientation nulls and a response--cost frontier.

The observed state would then need to lie near the held-out frontier, and
state-dependent rotations would need to track changes in task/noise structure
without being explained by spectrum, movement, or global excitability.  The
local public datasets each omit at least one indispensable component.

## Recommended experimental test

Use repeated two-photon holographic perturbations of orthogonal population
patterns while recording the same neurons across controlled gain or attention
states.  Fit the effective propagator from non-perturbed trials, estimate
background covariance independently, and measure the perturbational map on
held-out trials.  The preregistered primary endpoints should be:

- alignment of the measured leading input singular direction with the
  generalized task/noise mode, relative to exact same-spectrum orientation
  controls;
- transient gain at matched stationary fluctuation cost;
- proximity to a response--cost Pareto frontier; and
- rotation of the channel when task/noise structure changes, without a
  corresponding eigenvalue shift toward instability.

That design could strongly test biological optimization.  Reanalysing the
current observational datasets cannot supply the missing intervention and
frontier information.

