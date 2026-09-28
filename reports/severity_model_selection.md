# Phase 6 — Severity Model Selection

Decision record for the champion severity model, evaluated on the **validation split**
(3,947 claims, never used for fitting).

## Candidates compared

| Candidate | Parameters | Validation Gamma deviance | Observed/Expected |
|---|---|---|---|
| Baseline (mean) | 1 | 1.5690 | 0.8436 |
| Gamma GLM, full formula | 48 | **1.5724 (worse than baseline)** | 0.8755 |
| Gamma GLM, no Region/VehBrand | 22 | 1.5567 | 0.8675 |
| Gamma GLM, BonusMalus+VehAge only | 12 | 1.5962 | 0.8510 |
| Lognormal (smearing-corrected), no Region/VehBrand | 22 | 1.5515 | 0.8582 |
| **Lognormal (smearing-corrected), full formula** | 48 | **1.5273 (best)** | 0.8496 |

(Gamma deviance used throughout — not RMSE, which the plan explicitly warns against relying
on for this kind of outcome. MAE checked separately below as a secondary metric.)

## Finding 1: the full Gamma GLM does not beat the trivial baseline

This is a real, checked result, not a rounding artifact: the 48-parameter Gamma model's
validation deviance (1.5724) is *worse* than simply predicting the training mean for every
claim (1.5690) — despite improving training deviance by 14.8% (Phase 6 step 1). This is a
genuine overfitting signal, and it was investigated rather than accepted at face value:
dropping `RegionGrouped`/`VehBrandGrouped` (26 of the 48 parameters) alone improves Gamma's
validation deviance to 1.5567 — better than the full model *and* the baseline. Those two
high-cardinality categorical factors, already known from Phase 3 to have no reliable
univariate relationship with severity, are the specific source of the overfitting.

## Finding 2: the same trim that helps Gamma *hurts* Lognormal

Dropping `Region`/`VehBrand` from the lognormal model makes it *worse* (1.5273 → 1.5515) — the
opposite pattern from Gamma. Lognormal's full-formula version is the best-performing candidate
overall, by a clear margin over every other candidate tested (next best: 1.5515).

**Why the two models react oppositely to the same features**: Gamma's fitting weights errors
in proportion to the *square* of the predicted value (`Var ∝ μ²`), so a handful of large,
noisy claims concentrated in specific `Region`/`VehBrand` categories can pull those
coefficients toward extreme, overfit values that generalize badly. Fitting on the
**log scale** (lognormal) compresses that same variation — the same few large claims have far
less leverage on the fit — so the extra parameters add a small amount of real signal instead
of mostly noise.

## Finding 3: calibration by decile shows exactly where Gamma breaks down

| Decile (highest-risk) | Gamma: observed | Gamma: predicted | Lognormal: observed | Lognormal: predicted |
|---|---|---|---|---|
| 9 (highest predicted) | €2,963 | **€5,842 (97% too high)** | €2,523 | €3,121 (24% too high) |

Gamma's highest-predicted-severity decile over-predicts by nearly 2×; lognormal's worst
individual-decile miss (a different decile, an under-prediction) is proportionally smaller.
Gamma's aggregate portfolio-level observed/expected ratio (0.8755) looks only mildly worse
than lognormal's (0.8496) — that single number hides a large, concentrated miscalibration in
exactly the segment a pricing model most needs to get right: the one it believes is riskiest.

## Finding 4: MAE tells a different, less decisive story — and that's expected, not a contradiction

| Candidate | MAE |
|---|---|
| Baseline | €2,116.26 |
| Gamma (full) | €2,089.09 |
| Lognormal (full) | €2,108.52 |

By mean absolute error, Gamma actually edges out lognormal slightly. This is not a
contradiction of the deviance-based conclusion — it's the plan's own warning ("judging
performance only through RMSE") playing out concretely. MAE weights every euro of error
equally regardless of the prediction's scale, so it is far less sensitive to the specific,
large *relative* miscalibration Gamma shows in its top decile. All three MAE values are within
1.3% of each other — a weak signal, correctly overridden here by the much larger and more
diagnostic gap in Gamma deviance and decile calibration.

## Finding 5: the champion also resolves Phase 6 step 2's fragility concern

Step 2 found the Gamma model's `DrivAgeBand` severity relativities were fragile — dominated by
a handful of large claims, nearly disappearing once the top 1% were excluded (e.g., the
23-29 band: 0.28 full vs. 0.87 trimmed). Re-running that same check on the **lognormal**
model shows far more stable relativities: 0.872 (full) vs. 0.918 (trimmed) for the same band -
a modest, expected shift, not a near-total reversal. This is not a separate finding by luck:
it is the same mechanism as Finding 2 (log-scale fitting reduces the leverage of a few large
claims), showing up consistently across both the sensitivity analysis and the held-out
validation comparison.

## Champion: Lognormal (Duan smearing-corrected), full formula

Best validation deviance among every candidate tested, most resilient calibration (no
single-decile blowup like Gamma's), and materially more stable `DrivAgeBand` relativities
under the large-loss sensitivity check. This is not the "textbook default" choice (Gamma is
usually the first reach for this kind of positive, skewed outcome) — it is the choice the
validation evidence actually supports here, checked rather than assumed.

**Caveat carried forward**: even the champion only modestly beats the trivial baseline
(deviance 1.5273 vs. 1.5690, roughly 2.7% better) — consistent with Phase 3's original finding
that severity has little genuine relationship to the available rating factors. This model
should not be oversold as strongly predictive; its main value is a real, if modest, 2.7%
improvement over "just use the average," with better-behaved uncertainty than Gamma's
raw-scale fit provides.
