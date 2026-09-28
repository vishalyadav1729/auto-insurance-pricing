# Phase 8, Step 1 — Frequency Gradient Boosting Challenger

Compares a Poisson-loss `HistGradientBoostingRegressor` against Phase 5's Poisson GLM champion,
on the **validation split** (the GLM was never re-fit here; this reuses the exact same
champion from Phase 5).

## A real mechanical difference from the GLMs, verified before trusting it

`HistGradientBoostingRegressor` has no offset parameter. The standard fix (used in
scikit-learn's own official insurance tutorial, this project's reference point): predict the
annualized rate directly, and pass `sample_weight=Exposure` to `.fit()`, which weights each
policy's contribution to the loss by its time-at-risk — the same role an offset plays in a
GLM. Verified on synthetic data before running on the real portfolio: a quick smoke test
reconciled to within 1%.

The model also uses the engineered categorical columns (`DrivAgeBand`, `VehAgeBand`,
`BonusMalusBand`, `RegionGrouped`, `VehBrandGrouped`) **natively**, via
`categorical_features="from_dtype"` — these are already proper pandas Categoricals (Phase 5's
fix for single-row prediction), so no separate one-hot encoding step is needed here, unlike
the GLMs' patsy-based formulas.

## Hyperparameter tuning — a small, deliberate grid, not library defaults

The plan explicitly warns against comparing a *tuned* GLM to an *untuned* ML model. A grid of
`max_leaf_nodes` (15/31/63) × `learning_rate` (0.05/0.1) — 6 configurations, each using
scikit-learn's own internal early-stopping to pick the number of boosting iterations — was
fit on training data and scored on validation using the same exposure-weighted Poisson
deviance metric used throughout Phase 5:

| `max_leaf_nodes` | `learning_rate` | `n_iter_` | Validation deviance |
|---|---|---|---|
| 15 | 0.05 | 193 | 0.578227 |
| 15 | 0.10 | 115 | 0.578161 |
| 31 | 0.05 | 131 | 0.577315 |
| 31 | 0.10 | 76 | 0.577674 |
| **63** | **0.05** | **100** | **0.576589 (best)** |
| 63 | 0.10 | 53 | 0.577638 |

All six configurations land within ~0.3% of each other — the model is not especially
sensitive to these specific hyperparameters in this range, similar to Phase 7's finding for
Tweedie's power parameter. `max_leaf_nodes=63, learning_rate=0.05` was adopted.

## Comparison to the GLM champion

| Model | Validation deviance |
|---|---|
| Poisson GLM (Phase 5 champion) | 0.594235 |
| **Gradient boosting (tuned)** | **0.576589** |

**A real, modest improvement: ~3.0% lower deviance.** Not dramatic, but genuine — this is
exactly the kind of result the plan anticipates: gradient boosting can find non-linear
patterns (interactions, threshold effects) a GLM's hand-specified formula might miss, even
one as carefully constructed as Phase 5's (banded ages, grouped rare categories).

## Calibration — checked, not assumed to follow from a good deviance score

| Check | Result |
|---|---|
| Training reconciliation | 25,375.73 predicted vs. 25,227 observed (ratio 1.0059) |
| Validation reconciliation | 5,441.51 predicted vs. 5,404 observed (ratio 1.0069) |
| Observed/Expected (validation) | 0.9931 |

Unlike a canonical-link Poisson GLM, gradient boosting has no guaranteed exact-match property
— these ratios are close (within ~0.7%) but not exact, which is expected, not a flaw.

**Calibration by decile (validation)** tracks closely across the entire risk range:

| Decile | Observed rate | Predicted rate |
|---|---|---|
| 0 (lowest) | 0.0443 | 0.0447 |
| 1 | 0.0547 | 0.0574 |
| 2 | 0.0644 | 0.0669 |
| 3 | 0.0714 | 0.0744 |
| 4 | 0.0808 | 0.0809 |
| 5 | 0.0949 | 0.0881 |
| 6 | 0.0960 | 0.0984 |
| 7 | 0.1104 | 0.1159 |
| 8 | 0.1453 | 0.1490 |
| 9 (highest) | 0.3544 | 0.3469 |

No decile shows the kind of severe miscalibration found for Gamma severity in Phase 6 or
Tweedie pure premium in Phase 7 — this model is both more accurate (lower deviance) and
well-calibrated, not one at the expense of the other.

## What this means, and what's still open

A real predictive edge exists for frequency, but it is modest (3%), not transformative. This
is one data point toward Phase 8's eventual champion-challenger recommendation — not a
decision on its own. The plan's own framing matters here: the question is not "did ML win,"
it's "does a 3% deviance improvement justify losing the GLM's explicit relativities, stable
extrapolation, and straightforward governance story" — a question properly answered once
severity (step 2) and the combined pure-premium comparison (step 3) are also in hand.
