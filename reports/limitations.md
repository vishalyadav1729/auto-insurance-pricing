# Limitations — Consolidated Reference

Phase 9, step 2. Every limitation disclosed across this project's eight prior phases,
gathered into one reference document rather than left scattered across sixteen separate
reports. Nothing here is new — each item links to where it was originally found, checked,
and disclosed. Organized by category: data, modelling, evaluation, fairness, and
deployment.

## Data limitations

**1. The dataset is a single historical snapshot with no usable temporal structure.**
The train/validation/test split (Phase 4) is a random 70/15/15 split stratified by claim
occurrence, not a chronological holdout, because no reliable claim-date field was used in
modelling. This means every model in this project has been validated against
*contemporaneous* data only — none of it demonstrates how these models would perform
against a genuinely *future* period, which is what a real deployment actually needs.
(`split.py`; `reports/model_card.md`)

**2. Six known data anomalies, decided on in Phase 3, each with a residual limitation:**

| Anomaly | Decision | What remains a limitation |
|---|---|---|
| `Exposure` > 1.0 (1,224 policies) | Clipped to 1.0 | None — evidence showed these aren't a distinct risk group |
| Implausible `ClaimNb` (9 policies, up to 16 claims) | Capped at 4 | None — matches a published precedent for this exact dataset |
| `ClaimNb` vs. paid-claim mismatch (9,123 policies) | Two targets used for two purposes | **The structural bias this creates is never fully eliminated — see item 6 below** |
| 6 orphaned policies, 195 claim rows, ~€789k | Excluded from modelling | **~1.3% of total claim value in this dataset is permanently unusable by any policy-level model built here** |
| `ClaimAmount` long right tail (max ≈3,500× median) | Not capped; sensitivity-tested instead | **Large-loss concentration remains a first-order driver of model behavior — see item 8** |
| `VehAge` top-coded at 99/100 | Binned as a feature, not deleted | Top-coding means the true age of the oldest vehicles is unknown, only that it's "99 or more" |

(`reports/data_dictionary.md`, `reports/cleaning_policy.md`)

**3. No demographic data exists in this dataset** — no race, gender, income, marital
status, or occupation field. This is a hard ceiling on what any fairness analysis of this
project can establish; see the Fairness section below. (`reports/fairness_analysis.md`)

**4. Single insurer, single country, single era.** All 678,013 policies come from one
anonymized French insurer, using French-specific conventions (the `BonusMalus` scale,
`Region` codes) from a dataset now well over a decade old. Nothing in this project
demonstrates generalizability to other insurers, countries, or the present day's driving
patterns, vehicle mix, or claims-cost inflation. (`reports/data_dictionary.md`,
`reports/model_card.md`)

## Modelling limitations

**5. Severity has limited genuine predictive signal from the available rating factors,
full stop.** Established in Phase 3's exploratory analysis, reconfirmed by every severity
model built since: the Gamma GLM overfit on `Region`/`VehBrand` (26 of 48 parameters,
Phase 6), the boosted severity challenger needed aggressive regularization (early
stopping at 16 of a possible 300 iterations, Phase 8) and, as a direct consequence,
failed to learn a real, independently-validated relationship (`BonusMalus`, confirmed
twice — Phase 8 steps 4 and 5). No model family in this project overcame this — it is a
property of the data's relationship between these rating factors and claim cost, not a
fixable modelling choice. (`reports/severity_model_selection.md`,
`reports/ml_interpretation.md`)

**6. The frequency/severity decomposition carries a structural bias that is reduced, but
not eliminated.** `ClaimNb` (reported claims) includes claims that were reported but
never paid; multiplying it by average paid severity overstated true pure premium by
36.3% (Phase 3, resolved Phase 7). The Phase 7 fix — a second frequency model targeting
`ClaimNbFromSev` (paid claims) — reconciles the frequency component *exactly* (train
ratio 1.0000), but a small ~0.7% gap remains in the combined pure-premium figure (train
ratio 0.9931), attributable to the severity model's own known imperfect reconciliation.
This residual gap was never re-checked at the segment level — it could plausibly vary by
segment the way the underlying payment rate does (0.65 to 1.00 by `BonusMalus`, 0.61 to
0.89 by `DrivAge`, `reports/pure_premium_reconciliation.md`).

**7. Large-loss sensitivity was checked, but the underlying concentration itself is not
addressed by any model here.** The top 1% of claims by value account for 38% of total
claim value portfolio-wide (Phase 3) and 41% within the training split specifically
(Phase 6). This is treated correctly as a genuine feature of third-party liability claims
(not capped or removed, per the plan's own guidance), but it means overall model accuracy
is disproportionately driven by how well a handful of very large, inherently hard-to-
predict claims are handled — and severity models in this project have limited genuine
predictive power to begin with (item 5). (`reports/severity_large_loss_sensitivity.md`)

**8. No model in this project accounts for policy-level dependence beyond the rating
factors used.** Standard independence assumptions (each policy's claims are independent
given its rating factors) underlie every GLM and boosting model here; no spatial or
temporal correlation structure was tested for or modelled. (`reports/model_card.md`)

**9. The ML challenger comparison (Phase 8) surfaced a governance-relevant modelling
risk, not just a performance one.** Gradient boosting for severity won narrowly on
aggregate deviance (−0.38%) while silently failing to learn a real, validated
relationship — a finding invisible from the deviance comparison alone and only surfaced
by dedicated interpretation work (permutation importance, partial dependence). This isn't
specific to this dataset: it's a general illustration of why aggregate accuracy metrics
alone are an insufficient basis for model selection, one this project observed directly
rather than merely warns about abstractly. (`reports/ml_interpretation.md`,
`reports/ml_champion_challenger_recommendation.md`)

## Evaluation limitations

**10. Segment-level pure-premium miscalibration is real, checked against sample size
specifically (not assumed to be noise), and remains unresolved for both model families:**

| Segment | O/E range | Explained by sample size? |
|---|---|---|
| Region (GLM champion) | 0.46 – 1.83 | No — the largest region (15,439 policy-years) still over-predicts by 24% |
| Region (boosted challenger) | up to 2.54× in the extreme case | No — worse than the GLM, not better, despite better aggregate metrics |
| BonusMalus | 0.665 – 1.027 | Middle bands (60-99) over-predict by roughly a third |
| DrivAge | 0.57 – 1.33 | No — the 60-69 band has 6× the exposure of the best-calibrated 18-22 band, yet is far worse calibrated |

Region's miscalibration was checked against DrivAge composition specifically (Phase 9
step 1) and found *not* to reduce to an age-composition effect — it remains its own,
separately unexplained problem. (`reports/pure_premium_segment_stability.md`,
`reports/ml_pure_premium_comparison.md`, `reports/fairness_analysis.md`)

**11. Gini/Lorenz-curve ranking has its own, narrower failure mode that was caught once
and must be guarded against.** Ranking by raw expected loss (rather than annualized rate)
mechanically ties the ranking to `Exposure`, which has ~0 real correlation with risk in
this data — this produced a misleadingly negative Gini (−0.31) for a non-discriminating
baseline before the bug was found and fixed. The fix is now locked in by a regression
test, but it's a reminder that ranking metrics can fail silently in ways portfolio-level
calibration metrics won't catch. (`reports/pure_premium_model_comparison.md`)

**12. The direct Tweedie GLM showed an unexplained calibration reversal between train
and validation** (+23% train, −32% validation) that was investigated (ruled out Phase
6's Gamma-overfitting mechanism specifically) but never fully explained. It was not
selected as champion for other reasons (worse deviance, worse Gini), so this was disclosed
as an open finding rather than a blocking one — but it means Tweedie GLMs, as a class,
were not as thoroughly understood in this project as the Frequency × Severity champion
was. (`reports/pure_premium_model_comparison.md`)

**13. Test-set confirmation happened exactly once per pipeline, by design — which means
there's no second independent check.** The project's strict train/validation/test
discipline (validation for all selection and tuning, test touched once to confirm) is a
methodological strength, but it also means each champion's test-set performance
(GLM: O/E 0.843→0.850, Gini 0.3016→0.2701; boosted: O/E 1.011→1.023, Gini 0.328→0.252)
is a single data point, not a distribution — ordinary sampling variability in that one
check is not distinguishable from a true difference. (`reports/pure_premium_segment_stability.md`,
`reports/ml_interpretation.md`)

## Fairness limitations

Full detail in `reports/fairness_analysis.md` (Phase 9 step 1); summarized here for
completeness:

**14. `DrivAge` — an explicit age variable — is genuinely miscalibrated** (O/E 0.57 to
1.33, over-predicting ages 40-69 and under-predicting 70+), not explained by sample size.
This is the single most fairness-relevant finding in this project, since `DrivAge` is the
rating factor most directly analogous to a legally protected characteristic used here.

**15. No proxy-variable analysis is possible on this dataset.** Whether `Region`,
`Density`, or `Area` function as proxies for a protected characteristic cannot be
verified either way, because the dataset contains no demographic fields to check against.
This is a permanent limitation of this dataset, not a gap that more analysis within this
project could close.

**16. Low-exposure segments carry an unaddressed individual-fairness risk.** Rare
`Region`/`VehBrand` categories and the `BonusMalus` `130+` band (364 training
policy-years) were pooled or flagged for low statistical credibility throughout this
project (Phase 4–7) — a sound approach for aggregate model stability, but it does not by
itself guarantee fair, stable pricing for the individual policyholders inside those
pooled or thin groups.

**17. No jurisdiction-specific legal review was performed for any rating factor used.**
Age-based and territory-based rating in particular face materially different
restrictions across jurisdictions; this project did not check any specific jurisdiction's
actual rules before using `DrivAge`, `Region`, `Area`, or `Density` as rating factors.

## Deployment limitations

**18. This model has never been tested against genuinely new (future, unseen-at-
training-time) data of any kind** — not a future time period (item 1), not a different
insurer, not a different country. Every "test-set" check in this project used data drawn
from the exact same historical snapshot as the training data.

**19. No production monitoring exists or was simulated.** The retraining and monitoring
recommendations in `reports/model_card.md` (periodic retraining, O/E monitoring by
region and by DrivAge band, periodic large-loss sensitivity re-checks) are
recommendations for a hypothetical deployment, not something this project implemented or
validated by simulating drift.

**20. The output is a technical pure premium, not a final premium.** Expense loading,
profit margin, reinsurance cost, and regulatory rate-filing constraints are entirely
absent from every number in this project. Any figure quoted anywhere in these reports
should not be read as "what a customer would actually pay."

## What this document deliberately does not do

It does not re-litigate or re-decide any of the above — every item here was already
investigated, checked against plausible alternative explanations where relevant (segment
size for Region and DrivAge O/E; Gamma-overfitting for Tweedie's reversal; age
composition for Region's miscalibration), and disclosed at the point it was found. This
document's only job is to make all of it findable in one place, for anyone evaluating
this project or its model card without having read all sixteen prior reports.
