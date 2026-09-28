# Phase 6 — Large-Loss Sensitivity Analysis

This is the analysis `reports/cleaning_policy.md` (Phase 3, anomaly 4) explicitly deferred to
Phase 6: rather than cap or remove large claims by default, fit the model with and without
them and report what changes. Threshold: the top 1% of claims **by training-split value
only** (fit-on-train discipline, same as every other threshold in this project).

## The threshold and what it removes

| | Value |
|---|---|
| Threshold (99th percentile, training claims) | €16,795.72 |
| Claims excluded | 186 / 18,523 (1.00%) |
| **Value excluded** | **€17,988,204.64 / €44,004,810.78 (40.9%)** |

The top 1% of claims by count represent **41% of total claim value** — consistent with Phase
3's portfolio-wide finding (top 1% = 38% of total value across all splits combined). A model's
accuracy on total portfolio cost is dominated by how it handles these 186 claims, not by how
well it fits the other 18,337.

## Finding: the DrivAge severity effect is almost entirely an artifact of a few large claims

This is the headline result, and it changes how the full model's `DrivAgeBand` relativities
(reports/frequency_relativities.md's sibling severity table, forthcoming in step 4) should be
read.

| Band | Relativity (full model) | Relativity (top 1% excluded) |
|---|---|---|
| 18-22 (reference) | 1.00 | 1.00 |
| 23-29 | 0.28 | 0.87 |
| 30-39 | 0.27 | 0.91 |
| 40-49 | 0.26 | 0.93 |
| 50-59 | 0.26 | 0.94 |
| 60-69 | 0.28 | 0.95 |
| 70+ | 0.33 | 0.99 |

In the full model, every age band 23+ appears to cost 65-75% *less* than the youngest
reference group — a dramatic, striking pattern. Excluding just the top 1% of claims, that
pattern **almost completely disappears**: every band lands within about 5-13% of the
reference, a far more modest and plausible effect.

**What this means**: the apparent "young drivers have wildly more severe claims" relationship
is not a real, broad pattern across the youngest driver population. It is the arithmetic
consequence of a handful of very large claims happening to be concentrated in the 18-22 group
specifically. Remove those few claims, and the age-severity relationship nearly vanishes -
consistent with, and now a direct proof of, Phase 3's original warning that apparent
severity-by-segment patterns were traceable to a handful of large claims distorting small
samples, not genuine relationships.

`VehBrandGrouped[B11]` shows the same signature (relativity 1.99 -> 0.97): a large,
attention-grabbing effect in the full model that is not robust to excluding the largest few
claims.

## Finding: the baseline severity level itself is highly sensitive to the largest claims

The reference-category baseline severity (what the model predicts for an 18-22 driver with
every other factor at its reference level) drops from **€2,866.54 to €1,336.44** - a 53%
reduction - once the top 1% of claims are excluded. This is expected given how much value
those 186 claims represent (41% of the training total), but it is a concrete illustration of
exactly why capping or removing large claims by default (which Phase 3 declined to do) would
have quietly halved the model's sense of "typical" severity - a serious understatement of real
tail risk, not a cleaner model.

## What did NOT change much

Relativities for `BonusMalusBand`, `VehAgeBand`, `AreaOrdinal`, and most `Region`/`VehBrand`
levels shifted only modestly between the two fits - these effects, where they exist at all,
are not artifacts of the handful of largest claims the way the DrivAge and B11 effects are.

## Decision: no change to the primary model

Consistent with `cleaning_policy.md` rule 4, the primary severity model (Phase 6, step 1)
continues to use the full, uncapped claim data. This analysis is not a reason to cap claims -
large losses are a real and expected feature of third-party liability claims, and removing
them by default would understate genuine tail risk. It **is** a reason to flag the model's
`DrivAgeBand` severity relativities specifically as fragile and not to be trusted at face
value for pricing decisions, and to disclose this sensitivity explicitly in Phase 6's final
interpretation (step 4) rather than let the full model's dramatic-looking age effect stand
unqualified.
