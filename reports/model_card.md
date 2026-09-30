# Model Card — RiskRate Pure Premium Pipeline

**A note on sequencing.** This card was written first (Phase 9 step 3, before steps
1–2, at explicit request), initially grounded only in what Phases 1–8 already
established. It has since been updated once, after Phase 9 step 1 (fairness analysis)
produced real findings — the Fairness considerations section below reflects that
update. The Limitations section is updated again after Phase 9 step 2's consolidated
limitations document.

## Intended use

An educational / portfolio demonstration of actuarial pricing methodology: given French
motor third-party-liability policy characteristics, estimate **pure premium** — the
expected claim cost per policy-year — by separately modelling claim frequency and claim
severity and combining them. It demonstrates the full pipeline (data cleaning, feature
engineering, GLM fitting, ML challenger comparison, calibration checking) end to end on
a public dataset.

**This model's output is a technical pure-premium estimate, not a final consumer
premium.** A real premium would additionally need expense loading, profit margin,
reinsurance cost, and regulatory rate-filing constraints — none of which are modelled
here.

## Prohibited use

- **Do not use to set real insurance premiums or make real underwriting decisions.**
  It was built on a public research dataset (2011–era, single anonymized French
  insurer) with none of the governance, ongoing monitoring, or regulatory review a
  production pricing model requires.
- **Do not use to determine real coverage eligibility or claims decisions for
  individuals.**
- **Do not deploy in any jurisdiction without independent legal review of which rating
  factors are permitted.** Age-based and location-based rating in particular face very
  different restrictions across jurisdictions (see Fairness section below); this
  project did not check any specific jurisdiction's rules.
- **Do not cite any relativity in this project as a causal claim about what makes
  driving risky.** Every finding below is an association learned from one historical
  dataset (Phase 8 step 4 states this caution explicitly and it applies throughout).

## Dataset population

- **Source**: `freMTPL2freq` (OpenML data_id 41214, 678,013 policies) and
  `freMTPL2sev` (OpenML data_id 41215, 26,639 claims), joined on `IDpol`
  (`reports/data_dictionary.md`).
- **Population**: private motor third-party-liability policies from a single,
  anonymized French insurer. Region codes, vehicle brands, and bonus-malus scale are
  all French-market-specific conventions.
- **Not necessarily generalizable** to other countries (different driving laws, vehicle
  safety standards, claims-cost inflation, legal liability regimes), other insurers
  (different underwriting mix), or the present day (the dataset predates this project by
  well over a decade; nothing about claims inflation or driving-pattern shifts since
  then is reflected).
- **No temporal structure was available or used.** The dataset is a single
  cross-sectional snapshot, so the train/validation/test split (Phase 4) is a random
  70/15/15 stratified split by claim occurrence, not a chronological holdout. A real
  deployment would need to validate against a genuinely *future* period, which this
  project could not do.

## Target definition

Two different frequency targets are used deliberately, for two different purposes —
worth stating plainly since it's the single most non-obvious design decision in the
project:

- **`ClaimNb`** (capped at 4, Phase 3 rule 2a): count of *reported* claims. Used for the
  Phase 5 frequency champion, appropriate for claims-volume/workload questions.
- **`ClaimNbFromSev`**: count of claims that actually appear in the severity table (i.e.
  were *paid*). A second, purpose-built frequency model was fit to this target in Phase
  7 specifically because `ClaimNb` includes claims reported but never paid (9,117 of
  9,123 mismatched policies, cleaning_policy.md rule 2b) — multiplying `ClaimNb` ×
  average paid severity overstates pure premium by 36.3% (Phase 3, resolved Phase 7).
- **Severity target**: `ClaimAmount`, conditional on a claim being paid (lognormal
  champion, Duan-smearing corrected for retransformation bias — naive exponentiation of
  log-scale predictions understated cost by 60% before this correction, Phase 6).
- **Pure premium**: `predict_frequency(paid-claim model) × predict_severity`, reported
  both as expected loss over the observed exposure period and as an annualized rate
  (used for ranking/Gini, since raw expected loss is mechanically tied to `Exposure`
  and doesn't reflect per-year risk — a bug caught and fixed in Phase 7).

## Model assumptions

- **Frequency**: Poisson GLM, canonical log link, `Exposure` as a log-offset (not an
  ordinary predictor). Overdispersion is real and measured (Pearson ratio 2.31,
  Negative Binomial α=0.8115, p≈1.1×10⁻⁶⁰) but the Poisson point predictions are
  statistically indistinguishable from the Negative Binomial's (validation deviance
  0.594235 vs. 0.594267) — NB's benefit here is calibrated standard errors, not
  different point estimates, so Poisson remains the champion (`frequency_model_selection.md`).
- **Severity**: Lognormal linear model on log(`ClaimAmount`), Duan-smearing corrected,
  chosen over a Gamma GLM because the Gamma GLM overfit on `Region`/`VehBrand`'s 26 of
  48 parameters on this data (Phase 6).
- **Multiplicative relativities** throughout (log-link structure): each rating factor's
  effect is assumed multiplicative, not additive, and relative to a fixed reference
  category.
- **Rare categories pooled** at a fixed 2,000-training-policy-year threshold, fit on
  training data only (`Region` R21/R42/R43/R74/R83/R94 and `VehBrand` B14 grouped into
  `"Other"` — Phase 4).
- **Independence across policies** is assumed (no explicit spatial or temporal
  correlation structure), standard for this class of GLM but not separately tested.
- **Random, non-chronological data split** — see Dataset population above; this is a
  structural assumption of the whole project, not fixable without different data.

## Training period

Not applicable in the usual sense of a rolling training window — the dataset is one
historical snapshot with no reliable claim date field used in modelling. What was
controlled instead:

- 678,013 policies split 70% train / 15% validation / 15% test, stratified by claim
  occurrence (`split.py`; claim rate matches to 4 decimal places — 5.0235% — across all
  three splits).
- Every cleaning threshold, feature-engineering rule, and regularization/model-family
  choice was fit or selected using **training and validation data only**. Test was
  touched exactly once per pipeline (Phase 7 step 4 for the GLM, Phase 8 step 4 for the
  boosted challenger), to confirm — never to reselect — an already-chosen champion.

## Evaluation metrics

Exposure-weighted deviance (Poisson for frequency, Gamma for severity, Tweedie
power=1.5 for combined pure premium), observed-to-expected ratio, calibration by risk
decile, and Gini index computed on annualized rate (not raw expected loss — see Target
definition). Headline validation → test results for the GLM champion pipeline
(`pure_premium_segment_stability.md`):

| Metric | Validation | Test |
|---|---|---|
| Tweedie deviance (power=1.5) | 65.4363 | 59.5602 |
| O/E ratio | 0.8430 | 0.8497 |
| Gini (ranked by annualized rate) | 0.3016 | 0.2701 |

All three are stable test-vs-validation (no reversal), confirming the champion
selection wasn't an artifact of validation-specific noise.

## Subgroup results

- **BonusMalus**: the single most robust rating factor across both frequency (6.43×
  relativity at the worst score) and severity (1.66× at the worst score, confirmed
  stable under a large-loss sensitivity check that excludes the top 1% of claims by
  value) — independently reconfirmed for frequency by gradient boosting (Phase 8), but
  boosting **failed to learn this severity relationship at all** (permutation
  importance indistinguishable from zero, flat partial dependence, confirmed twice —
  Phase 8 steps 4–5), a concrete illustration of an ML governance risk this project
  actually observed rather than a hypothetical one.
- **DrivAge**: strong, non-linear young-driver frequency effect; a severity effect that
  Phase 6's large-loss sensitivity check found only *partially* robust (shifts, not
  dramatically, when the top 1% of claims are excluded) — a weaker signal than
  BonusMalus, and one the boosted severity model over-relies on relative to its actual
  reliability (Phase 8 step 4).
- **VehAge**: a strong, non-monotonic frequency effect (new vehicles are the riskiest,
  not the safest — the single most surprising finding of the project, independently
  reproduced by an unrelated gradient-boosting model in Phase 8); **no severity effect
  at all**.
- **Area/Density/VehPower/VehGas**: contribute to frequency (Area/Density/Region jointly
  significant, p≈6.2×10⁻³³, despite looking weak individually due to multicollinearity)
  but show **no measurable relationship to severity** — established in Phase 6 and never
  contradicted by any later analysis.
- **Region (segment calibration — a genuine, unresolved gap)**: portfolio-level and
  ranking performance are good, but segment-level pure-premium O/E ranges from 0.46 to
  1.83 across regions, and this is **not** explained by segment size — the *largest*
  region (R24, 15,439 policy-years) still shows 24% over-prediction. The boosted
  challenger's segment-level calibration was checked too and found **worse**, not
  better, in the extreme case (`Other` region: 2.54× vs. the GLM's 1.83× under-prediction,
  Phase 8 step 3).

## Limitations

Full, consolidated treatment of every limitation disclosed across this project — data,
modelling, evaluation, fairness, and deployment — is in `reports/limitations.md` (Phase 9
step 2). Headlines:

- Segment-level pure-premium miscalibration is real, checked against sample size, and
  unresolved for both `Region` (O/E 0.46–1.83) and `DrivAge` (O/E 0.57–1.33), and for
  both the GLM champion and the boosted challenger.
- Severity has limited genuine predictive signal from the available rating factors
  overall — established in Phase 3, reconfirmed by every severity model built since;
  this is a hard ceiling on pure-premium accuracy that no model family in this project
  overcame.
- No temporal/chronological validation was possible (single-snapshot dataset) — real
  deployment drift is entirely unmodelled.
- Large-loss concentration: the top 1% of claims account for roughly 38–41% of total
  claim value depending on which model's sensitivity check is used; overall accuracy is
  disproportionately driven by a handful of the largest claims.
- Known structural bias from the frequency/severity decomposition itself: `ClaimNb`
  includes claims reported but never paid, so the naive `ClaimNb × severity` product
  overstated pure premium by 36.3% before Phase 7's paid-frequency correction. That
  correction reconciles the frequency component *exactly* (train ratio 1.0000), but a
  small ~0.7% gap remains in the *combined* figure (train ratio 0.9931), attributable
  to the severity model's own known imperfect reconciliation (Phase 6) — not a new,
  unexplained discrepancy, but not exactly zero either, and not re-checked at the
  segment level.
- Geographic/temporal generalizability is unverified (see Dataset population).
- No demographic data exists in this dataset, so no proxy-variable analysis is possible
  for `Region`/`Density`/`Area` — a permanent data limitation, not a completed check.

## Fairness considerations

Full analysis and evidence in `reports/fairness_analysis.md` (Phase 9 step 1). Summary:

- **`DrivAge` segment calibration was checked for the first time in Step 1, and is
  genuinely miscalibrated**: observed-to-expected ratio ranges from 0.57 (ages 60-69,
  over-predicted) to 1.33 (ages 70+, under-predicted) on validation — a ~2.3× spread not
  explained by sample size (the 60-69 band has six times the exposure of the
  best-calibrated 18-22 band, yet is far worse calibrated). `DrivAge` is the rating
  factor most directly analogous to a legally protected characteristic used in this
  project, so this is a concrete, disclosed fairness-relevant finding, not a
  hypothetical concern. Several jurisdictions additionally restrict or prohibit
  age-based auto rating for adult drivers outright; this project did not check any
  specific jurisdiction's rules, consistent with the Prohibited Use section above.
- **`Region`'s known miscalibration (0.46 to 1.83, Phase 7 step 4) does not reduce to
  age composition.** Step 1 checked directly whether regions with older or younger mean
  `DrivAge` inherited the DrivAge miscalibration pattern above, and found no clean
  relationship (e.g. `R41` and `R93` have nearly identical mean ages but opposite O/E
  directions) — Region's miscalibration remains its own separate, unexplained problem.
- **No proxy-variable analysis is possible on this dataset.** It contains no race,
  gender, income, or occupation field, so whether `Region`/`Density`/`Area` function as
  proxies for a protected characteristic **cannot be verified either way** — this is a
  hard data limitation, stated plainly in Step 1 rather than assumed away.
- **Low-exposure segments** (rare `Region`/`VehBrand` categories, the `130+`
  `BonusMalus` band at only 364 policy-years) were pooled or flagged for low
  credibility throughout the project (Phase 4–7), which limits — but does not
  eliminate — the risk of unstable, individually unfair pricing for policyholders in
  those groups.
- **BonusMalus itself is not a neutral variable**: it is French regulation's own
  encoded summary of a driver's claims history, so relying on it heavily (as this
  project's champion does) largely re-prices past claims rather than introducing new
  demographic risk factors — the most defensible of this project's strong rating
  factors on fairness grounds, though this framing was not independently audited.

## Retraining and monitoring recommendations

- **Recommend periodic retraining** (e.g., annually) in any real deployment, given the
  complete absence of temporal validation here — this project cannot say how quickly
  this model's accuracy would degrade over time, because it was never tested against
  data from a later period.
- **Recommend production monitoring of observed-to-expected ratio by region and by
  DrivAge band**, specifically, given the documented and unresolved segment-level
  miscalibration in both (Phase 7 step 4; Phase 9 step 1) — aggregate portfolio-level
  O/E being healthy does not imply either dimension's health (this project found the
  opposite directly for both). **Implemented**, not just recommended: a scheduled
  AWS Lambda publishes O/E for the 6 worst-known segments to CloudWatch, with alarms
  tuned to the already-known-bad values — all 6 fired as designed once the daily
  evaluation period completed. See `reports/monitoring_architecture.md`.
- **Recommend re-running the large-loss sensitivity check periodically** on any
  retrained model, given how repeatedly this project found top-1%-of-claims sensitivity
  to be diagnostic (it's what first revealed the Gamma GLM's Region/VehBrand overfitting
  in Phase 6, and what confirmed the boosted severity model's BonusMalus failure in
  Phase 8).
- **A fairness/proxy audit (Phase 9 step 1) is a prerequisite**, not a nice-to-have,
  before this pipeline is used in any context beyond the educational one it was built
  for.

## Provenance

Built across Phases 1–8 of this project; full technical detail in
`reports/frequency_model_selection.md`, `reports/frequency_relativities.md`,
`reports/severity_model_selection.md`, `reports/severity_relativities.md`,
`reports/severity_large_loss_sensitivity.md`, `reports/pure_premium_reconciliation.md`,
`reports/pure_premium_model_comparison.md`, `reports/pure_premium_segment_stability.md`,
`reports/ml_frequency_challenger.md`, `reports/ml_severity_challenger.md`,
`reports/ml_pure_premium_comparison.md`, `reports/ml_interpretation.md`, and
`reports/ml_champion_challenger_recommendation.md`.
