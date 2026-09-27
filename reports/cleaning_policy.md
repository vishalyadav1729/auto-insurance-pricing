# Phase 3 — Cleaning Policy

This is the decision record for the 6 anomalies documented in
`reports/data_dictionary.md`. Each entry states the rule, the evidence behind it,
how many rows it touches, and — where relevant — what is deliberately *not* done.
Implementation of these rules happens in Step 2 (`src/auto_pricing/features.py` /
`scripts/prepare_data.py`), so this document is the source of truth for *why*, not
the code for *how*.

## 1. `Exposure` > 1.0 (1,224 policies, max 2.01)

**Rule: clip `Exposure` to a maximum of 1.0.**

Evidence considered before deciding:
- The over-exposed group's mean `ClaimNb` (0.044) is *lower* than the portfolio
  mean (0.053) — these are not a distinct high-risk subgroup, just ordinary
  policies with a slightly-too-large recorded exposure.
- 75% of the over-exposed group sits at ≤ 1.15; the excess is small for most of
  them (median 1.04). Only 13 policies exceed 1.81.
- Exposure is defined as a fraction of a policy-year, so any value above 1.0 is
  logically impossible, not just unusual — most likely an accounting/reporting
  artifact around policy-year boundaries (renewals, mid-term adjustments).

Clipping (rather than deleting) preserves 1,224 otherwise-ordinary policies and
their rating-factor information. Deleting them would throw away real driver/
vehicle data to fix a single mis-recorded field.

**Impact: 0.18% of policies (1,224 / 678,013) have their `Exposure` value changed; nothing removed.**

## 2. `ClaimNb` vs. actual severity-row count (9,123 policies)

Two separate decisions here, because this anomaly is really two different problems.

### 2a. Implausible extreme `ClaimNb` values

**Rule: cap `ClaimNb` at 4 for modelling.**

Evidence: the distribution is `0: 643,953 / 1: 32,178 / 2: 1,784 / 3: 82 / 4: 7`,
then a long thin tail of `5, 6, 8, 9, 11×3, 16` — 16 policies total, each with an
exposure too short to plausibly explain the claim count (e.g. `IDpol=2241683`:
16 claims in 0.33 years). This matches the same cap used in scikit-learn's own
tutorial on this dataset, for the same reason: a handful of extreme points would
otherwise dominate a Poisson GLM's log-likelihood out of proportion to what they
can teach the model about the other 99.998% of policies.

**Impact: 16 policies (0.0024% of the portfolio) have `ClaimNb` reduced from 5–16 down to 4. No policies removed.**

### 2b. The broader mismatch between `ClaimNb` and severity row count

Before deciding, the 9,123 mismatched policies were split by *direction* of
disagreement, because a mismatch that goes one way consistently means
something different from one that goes both ways randomly:

| Direction | Count | Shape |
|---|---|---|
| `ClaimNb` ≥ 1, but the severity table has **zero** rows for that policy | 9,117 | 9,116 of these have *no* severity rows at all (not a partial undercount) |
| Severity table has rows, but `ClaimNb` = 0 | 6 | Exactly the 6 orphan policies from Anomaly 3 |

The 9,117-policy direction is a near-total, one-directional pattern, not
noise: a claim was logged, but no money was ever recorded against it. That
shape is consistent with a claim being **reported and counted**, then later
**closed with zero payment** (denied, withdrawn, no covered damage) — a
severity/payment table naturally has no row for a claim that paid nothing,
while a claim-count field that counts *reported* claims naturally still
includes it. The other 6 policies are a different problem in kind (an entire
policy record missing from one table, not a payment-vs-report distinction)
and are already handled under Anomaly 3.

**Rule: `ClaimNb` (capped, per 2a) remains the frequency-model training
target — it is the broader "a claim was reported" definition, and it's what
the frequency model's offset and rating factors were captured alongside. The
severity table's `ClaimAmountSum` per policy — not a recount of `ClaimNb` —
is used as the ground truth when evaluating pure premium in Phase 7.**

**Documented consequence, not swept under the rug:** because `ClaimNb`
includes some zero-payout claims that `ClaimAmount` by definition cannot,
the multiplicative identity `Pure Premium = frequency × severity` will run
**slightly high** — it multiplies "rate of any reported claim" by "average
cost of a *paid* claim." This is a known nuance of frequency-severity
decomposition, not a bug, and will be stated explicitly as a limitation in
Phase 7 rather than corrected by force-matching the two tables.

This is the plan's "don't assume one source is simply right" principle applied
literally: neither table is treated as authoritative for everything; each is
used for what it was actually built to measure.

**Impact: no rows changed by this decision. It governs which column is trusted for which downstream use.**

## 3. Orphaned claims — 6 policies, 195 claim rows, ~€789k (freMTPL2sev only)

**Rule: exclude these 195 claim rows from any policy-characteristic model (frequency, severity, or pure premium), and disclose the excluded total (~1.3% of total claim value) explicitly in the modelling report.**

This is close to a technical necessity rather than a judgment call: there is no
`VehPower`/`DrivAge`/`BonusMalus`/etc. for these 6 policies to join against, so a
model that uses rating factors literally cannot use these claims. The judgment
call is what to do with the fact that we're excluding real money — the answer is
to say so plainly in the final report rather than let it disappear silently.

**Impact: 195 / 26,639 claim rows excluded (0.73% of claims), representing €788,713.18 / €60,697,930.68 of total claim value (1.30%).**

## 4. Long right tail in `ClaimAmount` (max ≈ 3,500× the median)

**Rule: no capping or Winsorizing of `ClaimAmount` for the primary severity
model. A separate, explicit large-loss sensitivity analysis (fit with vs.
without the top claims, compare) is done in Phase 6.**

The plan is explicit that removing large claims "because they make plots
unattractive" is a mistake to avoid — large losses are a real and expected
feature of third-party liability claims, not noise. Capping them by default
would understate the very tail risk a pricing model is supposed to capture.

**Impact: no rows changed or removed now; revisited explicitly, with results reported both ways, in Phase 6.**

## 5. `VehGas` values contain literal embedded quote characters

**Rule: strip the stray `'` characters from `VehGas` values (`"'Diesel'"` → `"Diesel"`, `"'Regular'"` → `"Regular"`).**

This is a parsing artifact, not a modelling decision — the source ARFF file's
quoting leaked into the string content. Left unfixed, one-hot encoding would
treat it as intended (harmless numerically, since there are still only 2
distinct values either way) but it would be a wrong, unreadable value to show
in any relativity table or the Streamlit app later.

**Impact: 678,013 values reformatted (all rows); the number of distinct categories (2) does not change.**

## 6. `VehAge` maximum of 100 (25 policies at 100, 23 at 99, near-empty 69–98)

**Rule: do not delete or "correct" these policies. Treat `VehAge` as likely
top-coded near 99/100 (censored at some maximum rather than truly measured),
and address it via binning/capping as a *feature* in Phase 4, not by removing
rows in Phase 3.**

Evidence: the sharp clump at 99–100 versus a near-empty 69–98 range is a
top-coding signature, not organic data (a genuinely old-but-still-registered
vehicle population would taper off gradually, not spike at a round number).
`DrivAge` shows the same signature at 99 and is left alone for the same reason
— age up to 100 is not impossible for a human, so there is even less reason to
treat it as an error requiring row removal.

**Impact: no rows removed; 48 `VehAge` values (99/100) and the broader age
range will be bucketed/binned as a documented feature-engineering step in
Phase 4, not deleted here.**

## Summary table

| # | Anomaly | Rule | Rows affected | Rows removed |
|---|---|---|---|---|
| 1 | Exposure > 1 | Clip to 1.0 | 1,224 (0.18%) | 0 |
| 2a | Implausible ClaimNb | Cap at 4 | 16 (0.002%) | 0 |
| 2b | ClaimNb/severity mismatch | Use ClaimNb for training, ClaimAmountSum for evaluation | 0 | 0 |
| 3 | Orphaned claims | Exclude from modelling, disclose value | 195 claim rows (0.73%) | 195 |
| 4 | ClaimAmount tail | No capping now; sensitivity analysis in Phase 6 | 0 | 0 |
| 5 | VehGas quoting | Strip quote characters | 678,013 (100%, cosmetic) | 0 |
| 6 | VehAge top-coding | No deletion; bin in Phase 4 | 0 now | 0 |

**Net effect on the frequency table: 0 policies removed.** The only row-level
exclusion anywhere is the 195 orphaned severity records, which were never
usable for a rating-factor model regardless of this policy.
