# Phase 7 — Pure Premium Model Comparison

Decision record comparing three pure-premium candidates on the **validation split**
(101,702 policies, never used for fitting).

## Candidates

| Candidate | What it is |
|---|---|
| Baseline | One flat annual rate (€175.40, train mean) for every policy |
| Frequency × Severity | Paid-frequency (Phase 7 step 1) × Lognormal severity (Phase 6 champion) |
| Direct Tweedie GLM | Single model predicting `ClaimAmountSum` directly, `power=1.5`, `log(Exposure)` offset |

## Results

| Candidate | Tweedie deviance (power=1.5) | Observed/Expected | Gini (ranked by annualized rate) |
|---|---|---|---|
| Baseline | 66.8503 | 0.8408 | -0.0230 |
| **Frequency × Severity** | **65.4363 (best)** | 0.8430 | **0.3016 (best)** |
| Direct Tweedie | 67.8935 (worst) | **0.6813 (worst)** | 0.2584 |

**Frequency × Severity wins on every metric tested.** It has the lowest deviance, calibration
comparable to the baseline (both moderately under-predict the validation portfolio), and by
far the best risk-ranking power. The direct Tweedie GLM is the worst performer on both
deviance and calibration, despite being fit with the same features and a defensibly-chosen
power parameter.

## Choosing the Tweedie fitting power, honestly

A fixed-evaluation-power grid search (fitting powers 1.1/1.3/1.5/1.7, all scored at a shared
evaluation power of 1.5 for a fair comparison — the deviance formula itself is a different
function at different powers, so scoring each candidate "at its own power" would not be a fair
comparison) found all four fitting powers within ~1.6% of each other (67.65 to 68.78). `1.5`
was adopted: the standard actuarial default, and empirically only 0.36% behind the
best-tested value. `power=1.9` failed to converge during exploration and was not pursued
further.

## A real surprise, investigated rather than assumed away: Tweedie's calibration reverses between train and validation

On **training** data, the Tweedie GLM *over*-predicts total claim cost by 23% (ratio 1.2315).
On **validation**, it *under*-predicts by 32% (ratio 0.6813) — a dramatic reversal in
direction, not just magnitude.

Phase 6 found an analogous problem for the Gamma severity model: overfitting traced to
`Region`/`VehBrand` (26 of 48 parameters), fixed by dropping them. The same fix was tried here
and **did not resolve it**: dropping `Region`/`VehBrand` from the Tweedie formula barely moved
validation O/E (0.6813 → 0.6786) — a materially different result from Gamma's case, checked
directly rather than assumed to generalize. The precise mechanism behind Tweedie's calibration
gap was not fully pinned down within the scope of this comparison. This is disclosed
honestly rather than either overclaiming a diagnosis or hiding the gap: **the practical
conclusion (Frequency × Severity is the better-calibrated, better-discriminating, lower-deviance
champion) holds regardless of the precise cause**, since it is supported by three independent,
converging metrics, not just one.

## A methodology correction made before trusting the Gini numbers

The first computation of Gini ranked policies by raw **expected loss** (`rate × Exposure`).
This produced a baseline Gini of **-0.31** — a strongly negative value for a model that
predicts the same flat rate for everyone and therefore has zero genuine discriminating power
(a non-discriminating model should show a Gini near 0). Investigated rather than reported at
face value: for the baseline, `expected loss` is mechanically identical to `Exposure` (since
its rate is constant), and `Exposure` has essentially zero real correlation with risk in this
data (0.006) — but a handful of very-short-exposure policies happen to have one large claim
(the "short policy + one big claim" pattern documented repeatedly since Phase 3), which then
dominates the low end of an exposure-ranked curve.

**Fix**: rank by **annualized rate** (`auto_pricing.pure_premium.predict_annual_pure_premium`'s
output), not raw expected loss — the same distinction Phase 7 step 1 already built into two
separate, clearly-named functions for exactly this reason. Re-ranked this way, the baseline's
Gini becomes -0.023, correctly close to zero. `src/auto_pricing/evaluation.py`'s
`lorenz_curve`/`gini_index` docstrings and parameter names were corrected to require an
annualized rate, with this exact finding recorded in the docstring so the mistake cannot be
quietly reintroduced later.

## Gini measures ranking, not calibration — demonstrated directly, not just asserted

The plan explicitly warns that Gini can preserve perfect risk-ranking while hiding badly wrong
price *levels*. Demonstrated concretely: scaling the Frequency × Severity champion's
predictions by a constant factor (e.g. ×3, a stand-in for "what if this model were badly
miscalibrated but still ranked correctly") leaves Gini **exactly unchanged** (scaling every
prediction by the same factor cannot change their relative order), while the
observed-to-expected ratio moves substantially. See
`notebooks/05_pure_premium_evaluation.ipynb` for the live demonstration on real validation
predictions, and `tests/test_evaluation.py::test_gini_index_is_unchanged_by_scaling_predictions_but_calibration_is_not`
for the synthetic proof.

## Champion: Frequency × Severity

Best deviance, best risk-ranking, and calibration matching (not worse than) the trivial
baseline. The direct Tweedie GLM — despite being simpler to maintain, as the plan notes it can
be — does not out-perform the two-model pipeline here, on any metric checked.
