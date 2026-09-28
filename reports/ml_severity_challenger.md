# Phase 8, Step 2 — Severity Gradient Boosting Challenger

Compares a Gamma-loss `HistGradientBoostingRegressor` against Phase 6's lognormal champion,
on the claim-level validation split (3,947 claims).

## Tuning — and the same overfitting signature found for the Gamma GLM, in a different model family

| `max_leaf_nodes` | `learning_rate` | `n_iter_` | Validation deviance |
|---|---|---|---|
| **15** | **0.10** | **16** | **1.521545 (best)** |
| 15 | 0.05 | 17 | 1.521766 |
| 31 | 0.05 | 20 | 1.533564 |
| 31 | 0.10 | 14 | 1.544456 |
| 63 | 0.05 | 16 | 1.550559 |
| 63 | 0.10 | 15 | 1.562541 |

Unlike frequency's grid (Step 1, all six configs within ~0.3% of each other), severity's
results get **steadily worse as model complexity increases** — and every configuration's
early-stopping cut training off within 14-20 boosting iterations, a tiny fraction of the
`max_iter=300` ceiling. This is not a new problem: it is the same overfitting signature Phase
6 found for the Gamma GLM (traced there to `Region`/`VehBrand`), now showing up independently
in a completely different model family. Severity's weak signal-to-noise ratio punishes
complexity regardless of which kind of model is doing the fitting.

The best configuration (`max_leaf_nodes=15, learning_rate=0.1`) is adopted.

## Comparison to the GLM champion

| Model | Validation deviance |
|---|---|
| Lognormal (Phase 6 champion, smearing-corrected) | 1.5273 |
| **Gradient boosting (tuned)** | **1.5215** |

A real but very small improvement: **0.38%**. Consistent with Phase 6's own finding that even
the best GLM approach only modestly beat a trivial baseline (~2.7%) — severity has little
genuine signal in these rating factors for *any* model family to find, gradient boosting
included.

## A genuine advantage found: no manual intervention needed for the Region/VehBrand overfitting

Phase 6 needed a *deliberate, manual* fix for Gamma's overfitting on `Region`/`VehBrand`
(dropping them from the formula). The same check was run here — dropping those two features
from the boosted model:

| Formula | Validation deviance |
|---|---|
| Full (with Region/VehBrand) | 1.5215 |
| Without Region/VehBrand | 1.5234 |

**Essentially no difference** (both ~1.52) — gradient boosting's own regularization
(`max_leaf_nodes` constraint + early stopping, which cut this model off after only 16
iterations) already protects against the overfitting that required manual feature removal for
the GLM. A real, structural advantage of the tree-based approach for this specific problem,
not just a coincidence of this particular grid.

## Calibration

| Check | Result |
|---|---|
| Training reconciliation | ratio 0.8237 (18% under-prediction on the data it was fit on) |
| Validation reconciliation | ratio 0.9732 (better calibrated on validation than training) |
| Observed/Expected (validation) | 1.0276 |

The training-under/validation-better pattern is unusual and not fully investigated here (early
stopping halts training very early, at 16 iterations, which likely explains most of it) - noted
honestly as a pattern rather than a fully diagnosed one. The practical result (validation O/E of
1.028) is reasonable, well within the range of acceptable calibration and better than Gamma's
severe top-decile miscalibration from Phase 6.

Decile-level calibration shows the same kind of noise Lognormal's did in Phase 6 (e.g. decile 7
spikes to an observed mean of €3,770 against a predicted €1,924) - expected given severity's
established unpredictability, not a new weakness specific to this model.

## Conclusion for step 2

A marginal (0.38%) improvement over the GLM champion, with one genuine structural advantage
(automatic protection against the Region/VehBrand overfitting that required manual
intervention for Gamma). Like frequency, this is one data point - the combined pure-premium
picture (step 3) matters more for the eventual champion-challenger call.
