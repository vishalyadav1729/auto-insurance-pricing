# Feature Dictionary — `data/processed/model_table.parquet`

One row per policy (678,013 rows), built by `scripts/build_model_table.py` from
`frequency_clean.parquet` + `policy_splits.parquet`. Contains every raw column
alongside its engineered counterpart, so a later modelling phase can choose
either (e.g. compare a linear `BonusMalus` term against the banded version)
without re-deriving anything.

**Deliberately not one-hot encoded.** Phase 5's GLMs (statsmodels formulas,
e.g. `C(DrivAgeBand)`) and Phase 8's gradient boosting each need categorical
inputs in a different final shape. Baking in one fixed encoding here would
presuppose which model wins the eventual champion/challenger comparison —
each modelling phase encodes these columns however it actually needs to.

## Identifiers and target

| Column | Type | Role |
|---|---|---|
| `IDpol` | int | Policy identifier. **Not a feature** — never feed this into a model (a beginner mistake the plan explicitly calls out: it carries no predictive information, only a record-keeping ID that happens to correlate with nothing meaningful). |
| `split` | category | `train` / `validation` / `test`, from `policy_splits.parquet`. Filter on this, never re-split. |
| `ClaimNb` | int | The frequency modelling target (capped at 4, per `cleaning_policy.md` rule 2a). |
| `Exposure` | float | The offset for frequency modelling (`log(Exposure)` in a Poisson GLM) — capped at 1.0, per rule 1. |

## Raw columns (kept unchanged from `frequency_clean.parquet`)

`Area`, `VehPower`, `VehAge`, `DrivAge`, `BonusMalus`, `VehBrand`, `VehGas`,
`Density`, `Region` — see `reports/data_dictionary.md` for their original
definitions and ranges.

## Engineered columns

| Column | Type | Derived from | Rule | Fit-on-train? |
|---|---|---|---|---|
| `DrivAgeBand` | category | `DrivAge` | Fixed bins `[18,23,30,40,50,60,70,101)` | No — fixed constants |
| `VehAgeBand` | category | `VehAge` | Fixed bins `[0,1,3,6,10,15,20,101)` | No — fixed constants |
| `BonusMalusBand` | category | `BonusMalus` | Fixed bins `[50,51,60,80,100,130,231)` | No — fixed constants |
| `LogDensity` | float | `Density` | `log(Density)` | No — deterministic transform |
| `AreaOrdinal` | int (0–5) | `Area` | `A→0, B→1, ..., F→5` | No — fixed mapping |
| `VehGasBinary` | int (0/1) | `VehGas` | `Diesel→0, Regular→1` | No — fixed mapping |
| `RegionGrouped` | category | `Region` | Categories below 2,000 training policy-years → `"Other"` | **Yes — fit on `split == "train"` only** |
| `VehBrandGrouped` | category | `VehBrand` | Same rule, same threshold | **Yes — fit on `split == "train"` only** |

## Rare-category grouping — exact result on the real data

Threshold: **2,000 training-split policy-years**, chosen from a real gap in the
training data (Region jumps from 1,687 to 2,224 policy-years across that
line; VehBrand jumps from 1,601 to 4,745) — not an arbitrary round number.
The fitted rule, persisted in `artifacts/preprocessors/rare_category_maps.joblib`:

| Column | Categories before | Categories after | Grouped into `"Other"` |
|---|---|---|---|
| `Region` | 22 | 17 | `R21, R42, R43, R74, R83, R94` |
| `VehBrand` | 11 | 11 (relabelled, none merged further) | `B14` |

These are exactly the segments Phase 3 flagged by name (`Region R43`,
`VehBrand B14`, plus the other thin regions from the original exposure
profiling) — the threshold recovers findings already made, rather than
inventing new ones.

**Verified applied identically across all three splits** (not just fit
correctly on train): every row with `Region == "R43"` maps to
`RegionGrouped == "Other"` in train, validation, *and* test alike, and the
`"Other"` share is close across all three (train 3.11%, validation 3.07%,
test 3.00%) — consistent with one fixed rule applied everywhere, not three
different ones.

## Split integrity (cross-checked against Phase 4 step 1)

| Split | Policies | Share | Claim rate |
|---|---|---|---|
| train | 474,609 | 70.0% | 5.0235% |
| validation | 101,702 | 15.0% | 5.0235% |
| test | 101,702 | 15.0% | 5.0235% |

Matches `scripts/split_data.py`'s output exactly — this table did not
introduce any drift from the original split.
