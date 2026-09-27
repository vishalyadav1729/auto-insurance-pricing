# Data Dictionary — freMTPL2

Source: OpenML, via `scikit-learn`'s `fetch_openml` (`scripts/download_data.py`).
Downloaded 2026-09-27. Two tables, joined by `IDpol`.

| Table | OpenML data_id | Rows | Grain |
|---|---|---|---|
| `freMTPL2freq.csv` | 41214 | 678,013 | one row per **policy** |
| `freMTPL2sev.csv` | 41215 | 26,639 | one row per **claim** |

## `freMTPL2freq` — policy-level table

| Column | Type | Observed range | Description |
|---|---|---|---|
| `IDpol` | int (stored as float in source) | unique, no duplicates | Policy identifier. Join key to `freMTPL2sev`. |
| `ClaimNb` | int | 0–16 (mean 0.053) | Number of claims reported for the policy. 643,953 of 678,013 policies (95.0%) have 0. See **Anomaly 2**: this does not always agree with the actual claim rows in the severity table. |
| `Exposure` | float | 0.0027–2.01 (mean 0.529) | Fraction of a year the policy was in force and observed. Central to the whole project: this is why claim frequency is modelled with exposure as a log-offset rather than as an ordinary predictor (Phase 5) — a policy on the books for one month cannot be compared to one on the books for a year without adjusting for time-at-risk. See **Anomaly 1**: values should not exceed 1.0. |
| `Area` | ordered categorical | `A`–`F` (6 levels) | Area type code, roughly a rural→urban density banding. |
| `VehPower` | int, ordinal | 4–15 (median 6) | Vehicle engine power class. |
| `VehAge` | int | 0–100 (median 6) | Vehicle age in years. A max of 100 is implausible for a car and worth revisiting in Phase 3. |
| `DrivAge` | int | 18–100 (median 44) | Driver age in years. Range itself is plausible. |
| `BonusMalus` | int | 50–230 (median 50) | French no-claims bonus/malus score. 50 = best (max discount), 100 = neutral, >100 = penalized for claims history. A strong proxy for prior risk. |
| `VehBrand` | categorical | `B1`–`B14`, 11 levels present (`B7`–`B9` absent) | Anonymized vehicle brand code. |
| `VehGas` | categorical | 2 levels | Fuel type. See **Anomaly 3**: values arrive as the literal strings `'Diesel'` and `'Regular'` *including the single-quote characters* — an artifact of the source ARFF encoding, not a real third category. |
| `Density` | int | 1–27,000 | Population density (inhabitants/km²) of the policyholder's commune. Heavily right-skewed — the plan calls for a log transform before modelling (Phase 4). |
| `Region` | categorical | 22 levels (`R11`...`R94`) | French administrative region code. |

No missing values and no duplicate `IDpol` were found in this table.

## `freMTPL2sev` — claim-level table

| Column | Type | Observed range | Description |
|---|---|---|---|
| `IDpol` | int | — | Policy the claim belongs to. Not unique: a policy with several claims appears once per claim. |
| `ClaimAmount` | float (currency) | 1.00–4,075,400.56 (mean 2,278.5, median 1,172.0) | Cost of one claim. No zero or negative values. Right-skewed with a long tail: **Anomaly 4**. |

No missing values in either column. Total `ClaimAmount` across all 26,639 claims: **€60,697,930.68**.

## Known anomalies (documented, not yet corrected — that is Phase 3)

These are genuine properties of the downloaded files, confirmed by code in `src/auto_pricing/validation.py` and reproducible by running `notebooks/01_data_understanding.ipynb`. Recording them here — rather than silently fixing them — is what Phase 2 asks for: a decision about *how* to handle each one belongs in Phase 3, made deliberately and written down, not made implicitly inside a loading function.

1. **`Exposure` exceeds 1.0 for 1,224 policies** (max 2.01). Exposure is meant to be a fraction of a year, so it should not exceed 1. A likely cause is multi-year policy records collapsed into a single row; will need a documented capping or exclusion rule in Phase 3.

2. **`ClaimNb` disagrees with the actual severity row count for 9,123 policies.** Some policies report `ClaimNb=1` with zero matching severity rows, and vice versa. A few policies report severity-implausible `ClaimNb` values relative to their exposure — e.g. `IDpol=2241683` reports `ClaimNb=16` over an exposure of just 0.33 years (4 months). This exact pattern is documented in scikit-learn's own tutorial on this dataset, which caps `ClaimNb` for modelling; we will make and record the same kind of decision in Phase 3 rather than assume the raw count is correct.

3. **6 policies referenced in the severity table do not exist in the frequency table at all**, yet together they account for **195 claim rows** and **€788,714.18** in total claim value:

   | IDpol | claims | total amount |
   |---|---|---|
   | 2220367 | 24 | €84,441.63 |
   | 2227533 | 25 | €64,093.76 |
   | 2262511 | 66 | €151,800.59 |
   | 2277846 | 23 | €73,440.72 |
   | 2282134 | 36 | €330,920.05 |
   | 2286775 | 21 | €84,017.43 |

   Without a matching policy record, these claims have no exposure, no rating factors, and cannot be attributed to any modelled risk — they cannot be used by a policy-level model as-is. This was initially undercounted as "6 claim rows" from a quick manual check that only compared *unique* IDs; writing `find_orphan_claims()` as a real function and testing it caught that these 6 policies are unusually claim-heavy, which is a more important fact than "6 rows are missing."

4. **`ClaimAmount` has a long right tail.** The top claim (€4,075,400.56) is roughly 3,500× the median (€1,172). This is plausible for French motor third-party liability (bodily injury claims can be very large) but will need explicit large-loss sensitivity analysis in Phase 6 severity modelling, rather than being treated as an error to delete.

5. **`VehGas` values contain literal embedded quote characters** (`'Diesel'`, `'Regular'` — the quotes are part of the string, not just how Python displays it). Cosmetic, but must be stripped before one-hot encoding in Phase 4, or the model will see it as intended.

6. **`VehAge` has a maximum of 100 years**, which is not a plausible age for a registered vehicle. Needs the same kind of documented treatment (cap, investigate, or exclude) as Anomaly 1 in Phase 3.

## What Phase 2 deliberately does *not* do

- Does not drop, cap, or impute any value.
- Does not decide which of the two `ClaimNb` sources (frequency table vs. severity row count) is "correct" — Phase 3 will make that call explicitly.
- Does not join the two tables into a modelling table — that also happens in Phase 3/4, after cleaning rules are set.
