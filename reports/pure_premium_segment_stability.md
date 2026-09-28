# Phase 7, Step 4 — Segment Stability and the Final Test-Set Check

This is the first point in the entire project where the **test split** is used. Every prior
decision — cleaning thresholds, feature engineering, model family, regularization, champion
selection — was made using only training and validation data. Test is touched here exactly
once, to get an honest final read on the *already-chosen* champion (Frequency × Severity,
`reports/pure_premium_model_comparison.md`), not to compare candidates or pick a different one.
Doing otherwise would quietly undo the discipline this project has maintained since Phase 4.

## Pure premium calibration by decile (validation)

| Decile | Predicted rate | Observed rate |
|---|---|---|
| 0 (lowest) | €67 | €101 |
| 1 | €91 | €85 |
| 2 | €107 | €75 |
| 3 | €120 | €69 |
| 4 | €133 | €104 |
| 5 | €150 | €176 |
| 6 | €175 | €135 |
| 7 | €215 | €144 |
| 8 | €287 | €248 |
| 9 (highest) | €560 | €454 |

Real separation exists between the extremes (decile 0 vs. decile 9 differ by ~4.5× in both
predicted and observed terms), but the middle deciles are visibly noisy — not smoothly
monotonic the way Phase 5's frequency calibration was. This is expected, not a new problem:
Phase 6 already established that severity carries very little genuine predictive signal from
these rating factors, and pure premium inherits that noise on top of frequency's own signal.

## Segment-level observed/expected — a real, disclosed limitation

### By BonusMalus band

| Band | O/E |
|---|---|
| 50 (best) | 0.947 |
| 51-59 | 1.027 |
| 60-79 | 0.669 |
| 80-99 | 0.665 |
| 100-129 | 0.862 |
| 130+ | 0.764 |

The best and near-best bands are well-calibrated; the middle bands (60-99) show the model
over-predicting by roughly a third.

### By Region — checked against segment size, not assumed to be sampling noise

| Region | Exposure (policy-years) | O/E |
|---|---|---|
| R41 | 1,223 | 0.46 |
| R73 | 1,070 | 0.51 |
| R52 | 3,326 | 0.53 |
| R72 | 2,084 | 0.55 |
| R24 (largest region) | 15,439 | 0.76 |
| R11 | 4,487 | 0.73 |
| R93 | 5,318 | 1.20 |
| R82 | 6,767 | 0.95 |
| Other | 1,418 | 1.83 |

**This was checked against exposure size specifically because the obvious first guess — "the
extreme ratios are just small-segment noise" — turns out to be incomplete.** `R24`, the
*largest* region in the entire portfolio (15,439 policy-years, far more than most individual
regions), still shows a 24% over-prediction. `R11` (4,487 policy-years, a mid-sized,
reasonably well-populated region) shows 27% over-prediction. If every region were simply
scaled by the portfolio's overall ~16% over-prediction, O/E would sit near 0.84 everywhere —
instead it ranges from 0.46 to 1.83, real heterogeneity beyond the aggregate shift, not
explained away by segment size alone.

**This is disclosed as a genuine, open limitation, not resolved here.** Consistent with how
Tweedie's unexplained calibration reversal was handled in step 3: one plausible explanation
was checked (segment size) and found insufficient to fully explain the pattern, and rather
than force a tidier story, this is recorded as real evidence that **regional-level pricing
decisions should not be taken directly from this model without further investigation** — even
though the model's portfolio-level and ranking performance are genuinely good. This is exactly
the scenario the plan warns about: a reasonable aggregate observed-to-expected ratio can hide
real miscalibration in specific segments.

## The test-set check

| Split | Deviance (power=1.5) | O/E | Gini |
|---|---|---|---|
| Validation (already seen, used for selection) | 65.4363 | 0.8430 | 0.3016 |
| **Test (touched here, once, for the first time)** | **59.5602** | **0.8497** | **0.2701** |

All three metrics land close to their validation values — no dramatic reversal like Tweedie's
train-to-validation swing (Phase 7, step 3). Deviance is actually slightly better on test;
O/E is nearly identical (0.843 vs. 0.850); Gini is modestly lower but the same order of
magnitude. This is the reassuring result: the champion's validation-based selection was not
an artifact of validation-specific noise, confirmed on data that had never influenced any
decision in this project until this moment.

## What carries forward

- The champion (Frequency × Severity) is confirmed stable across train, validation, and test.
- Portfolio-level and ranking performance are good; **segment-level calibration, especially by
  Region, is not** — a real limitation to disclose in any deployment or business
  communication of this model, not something the aggregate metrics alone would reveal.
- Consistent with the whole project's practice: this limitation is recorded, not hidden, and
  not force-resolved with a guess that hasn't been checked.
