# Phase 8, Step 3 — Combined Pure Premium: Four-Way Comparison

Compares baseline, the GLM Frequency × Severity pipeline (Phase 7 champion), the direct
Tweedie GLM (Phase 7), and a new boosted Frequency × Severity pipeline (paid-frequency
boosting from step 1, severity boosting from step 2), on validation.

## Results

| Candidate | Deviance (power=1.5) | O/E | Gini |
|---|---|---|---|
| Baseline | 66.8503 | 0.8408 | -0.0230 |
| GLM Frequency × Severity (Phase 7 champion) | 65.4363 | 0.8430 | 0.3016 |
| Tweedie GLM | 67.8935 | 0.6813 | 0.2584 |
| **Boosted Frequency × Severity** | **64.5682 (best)** | **1.0108 (best)** | **0.3283 (best)** |

**The boosted pipeline wins on every aggregate metric.** Deviance improves 1.3% over the GLM
champion; Gini improves 8.9%; and calibration improves dramatically — O/E of 1.0108 is
essentially exact, versus the GLM's 0.8430 (16% under-calibrated) and Tweedie's 0.6813 (32%
under-calibrated). The combined effect is larger than either component challenger's individual
edge (frequency: +3.0% deviance, Step 1; severity: +0.38% deviance, Step 2) — consistent with
two modest, independent improvements compounding through multiplication.

## But the aggregate numbers don't tell the whole story — checked, not assumed

The same segment-level diligence applied to the GLM champion in Phase 7 step 4 was applied
here too, specifically because a good aggregate result can hide real problems (the plan's own
repeated warning throughout this project).

### Decile calibration is noisier than the GLM's, not better

| Decile | Observed rate | Predicted rate |
|---|---|---|
| 0 (lowest) | €36 | €60 |
| 1 | €83 | €77 |
| 2 | €122 | €87 |
| 3 | €86 | €98 |
| 4 | €173 | €110 |
| 5 | €98 | €125 |
| 6 | €140 | €146 |
| 7 | €243 | €175 |
| 8 | €176 | €229 |
| 9 (highest) | €421 | €455 |

Non-monotonic in the observed column (decile 2's €122 exceeds decile 3's €86, despite being
ranked lower-risk), and decile 0 under-shoots notably (observed €36 vs. predicted €60). The
GLM's own decile table (Phase 7 step 4) was also noisy in the middle, but this is noisier
still, not an improvement at this level of granularity.

### Regional calibration is not better, and "Other" is worse

| Region | GLM O/E (Phase 7) | Boosted O/E |
|---|---|---|
| R41 | 0.46 | 0.60 |
| R24 (largest region) | 0.76 | 0.80 |
| R11 | 0.73 | 0.94 |
| R93 | 1.20 | 1.50 |
| **Other** | **1.83** | **2.54 (worse)** |

Some regions improve (R11, R41 move closer to 1.0), but others get worse, and the most
extreme segment (`Other`, the pooled low-credibility regions from Phase 4) is *more*
miscalibrated for the boosted model (2.54× under-prediction) than for the GLM (1.83×) - a
real, checked finding, not assumed to automatically improve just because the aggregate numbers
did.

## What this means

**The boosted pipeline is the better choice by every portfolio-level metric that matters for
overall pricing accuracy and risk-ranking**, and severity boosting brought a genuine structural
advantage (automatic protection against the Region/VehBrand overfitting that needed manual
fixing for the GLM). But it does **not** resolve, and in the "Other" segment specifically makes
worse, the same segment-level calibration problem already disclosed for the GLM champion. A
model that is better in aggregate is not automatically better everywhere a real pricing
decision would look - this is recorded honestly for Phase 8's eventual champion-challenger
recommendation (step 5), not smoothed into a simpler "ML wins" conclusion.
