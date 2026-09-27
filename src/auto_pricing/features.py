"""Cleaning and feature-engineering transformations.

Two kinds of function live here, and they carry different risk of leakage:

1. Cleaning rules (reports/cleaning_policy.md) and fixed-edge feature
   transforms (age/BonusMalus banding, log-Density, Area's ordinal
   mapping) use constants decided in advance, not statistics computed
   from data. They are safe to apply identically to train, validation,
   and test.
2. Rare-category grouping (fit_common_categories/apply_common_categories)
   computes a threshold FROM data (training exposure per category), so it
   follows an explicit fit/apply split, mirroring scikit-learn's
   fit/transform convention: fit only ever sees the training split,
   apply is what gets called on validation/test.

Every function takes a DataFrame or Series in and returns a *new* one
(never mutates its input), so each transform can be tested and audited
independently of the others.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from auto_pricing.eda import bin_numeric

# Thresholds match reports/cleaning_policy.md exactly; see that document
# for the evidence behind each number.
EXPOSURE_CAP = 1.0
CLAIM_NB_CAP = 4


def clip_exposure(freq: pd.DataFrame, cap: float = EXPOSURE_CAP) -> pd.DataFrame:
    """Cap Exposure at `cap` (cleaning_policy.md, rule 1).

    Exposure is a fraction of a policy-year and cannot exceed 1 by
    definition. Values above it are treated as a reporting artifact around
    policy-year boundaries rather than a distinct risk group: the
    over-exposed policies have a *lower* mean ClaimNb than the portfolio
    average, so there is no evidence they belong to a different population.
    """
    out = freq.copy()
    out["Exposure"] = out["Exposure"].clip(upper=cap)
    return out


def cap_claim_nb(freq: pd.DataFrame, cap: int = CLAIM_NB_CAP) -> pd.DataFrame:
    """Cap ClaimNb at `cap` (cleaning_policy.md, rule 2a).

    Limits the leverage of a handful of implausible extreme values (e.g.
    16 claims against 0.33 years of exposure) on a Poisson GLM's
    log-likelihood, without removing the policy itself. Affects only 9
    policies out of 678,013.
    """
    out = freq.copy()
    out["ClaimNb"] = out["ClaimNb"].clip(upper=cap)
    return out


def clean_veh_gas(freq: pd.DataFrame) -> pd.DataFrame:
    """Strip stray embedded quote characters from VehGas (cleaning_policy.md, rule 5).

    Source values arrive as the literal strings "'Diesel'" and "'Regular'"
    (quotes included in the string content, an ARFF-parsing artifact) - not
    a real third category. This does not change how many distinct values
    the column has, only how they are spelled.
    """
    out = freq.copy()
    out["VehGas"] = out["VehGas"].str.strip("'")
    return out


def exclude_orphan_claims(freq: pd.DataFrame, sev: pd.DataFrame) -> pd.DataFrame:
    """Drop severity rows with no matching policy in freq (cleaning_policy.md, rule 3).

    These claims have no exposure or rating factors to join against, so a
    rating-factor model cannot use them regardless of any other decision
    made about the frequency/severity mismatch (rule 2b). Affects 195 claim
    rows (6 policies), representing ~1.3% of total claim value - excluded
    here, but that total is disclosed in the modelling report rather than
    silently dropped.
    """
    known_ids = set(freq["IDpol"])
    return sev[sev["IDpol"].isin(known_ids)].reset_index(drop=True)


def clean_frequency(freq: pd.DataFrame) -> pd.DataFrame:
    """Apply every frequency-table cleaning rule, in one documented place."""
    out = clip_exposure(freq)
    out = cap_claim_nb(out)
    out = clean_veh_gas(out)
    return out


def clean_severity(freq_cleaned: pd.DataFrame, sev: pd.DataFrame) -> pd.DataFrame:
    """Apply every severity-table cleaning rule (currently: orphan exclusion only)."""
    return exclude_orphan_claims(freq_cleaned, sev)


# --- Feature engineering (Phase 4) -----------------------------------------
#
# Bin edges below are the same fixed, domain-motivated cut points used for
# exploratory analysis in notebooks/02_data_cleaning_eda.ipynb (Phase 3,
# step 4) - not statistics fitted from data, so unlike the rare-category
# functions further down, they need no separate "fit" step and can be
# applied to train, validation, and test identically. Disclosure, not a
# claim of zero risk: they were chosen by looking at the full cleaned
# dataset before any split existed. A stricter standard would re-derive
# them from the training split only; that has not been done here because
# they are fixed thresholds, not distributional statistics (no mean,
# quantile, or min/max of the data is baked into them).

DRIVAGE_BINS = [18, 23, 30, 40, 50, 60, 70, 101]
DRIVAGE_LABELS = ["18-22", "23-29", "30-39", "40-49", "50-59", "60-69", "70+"]

VEHAGE_BINS = [0, 1, 3, 6, 10, 15, 20, 101]
VEHAGE_LABELS = ["0", "1-2", "3-5", "6-9", "10-14", "15-19", "20+"]

BONUSMALUS_BINS = [50, 51, 60, 80, 100, 130, 231]
BONUSMALUS_LABELS = ["50 (best)", "51-59", "60-79", "80-99", "100-129", "130+"]

# Area's frequency relationship (Phase 3, step 4) was smooth and monotonic
# A->F, consistent with treating it as an ordered scale rather than an
# unordered category.
AREA_ORDER = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}


def add_age_and_bonusmalus_bands(freq: pd.DataFrame) -> pd.DataFrame:
    """Add DrivAgeBand/VehAgeBand/BonusMalusBand columns.

    Phase 3 found DrivAge and VehAge are clearly non-linear (a young-driver
    spike and an unexpected new-vehicle spike respectively), so a single
    linear coefficient would badly misfit both; BonusMalus's extreme band
    has too little exposure for its point estimate to be fully trusted on
    its own. Banding is the documented response to both findings.
    """
    out = freq.copy()
    out["DrivAgeBand"] = bin_numeric(out["DrivAge"], DRIVAGE_BINS, DRIVAGE_LABELS)
    out["VehAgeBand"] = bin_numeric(out["VehAge"], VEHAGE_BINS, VEHAGE_LABELS)
    out["BonusMalusBand"] = bin_numeric(out["BonusMalus"], BONUSMALUS_BINS, BONUSMALUS_LABELS)
    return out


def log_density(freq: pd.DataFrame) -> pd.DataFrame:
    """Add a LogDensity column (Phase 3: Density is heavily right-skewed).

    A plain log is safe here - Density's minimum observed value is 1, so
    there are no zeros or negatives that would need a shifted log instead.
    """
    out = freq.copy()
    out["LogDensity"] = np.log(out["Density"])
    return out


def encode_area_ordinal(freq: pd.DataFrame) -> pd.DataFrame:
    """Map Area's ordered categories (A-F) to integers 0-5, per AREA_ORDER."""
    out = freq.copy()
    out["AreaOrdinal"] = out["Area"].map(AREA_ORDER)
    return out


def encode_vehgas_binary(freq: pd.DataFrame) -> pd.DataFrame:
    """Encode VehGas as 0 (Diesel) / 1 (Regular). Fixed mapping, no fit needed."""
    out = freq.copy()
    out["VehGasBinary"] = (out["VehGas"] == "Regular").astype(int)
    return out


def fit_common_categories(
    train_df: pd.DataFrame, category_col: str, min_exposure: float
) -> set[str]:
    """Determine which categories of `category_col` have at least
    `min_exposure` total Exposure in the TRAINING split.

    Must be fit on the training split only: using validation/test data to
    decide which categories are "common enough to trust" would leak
    information about the held-out data into a choice that shapes the
    training features. This exists for the specific low-exposure segments
    Phase 3 flagged (Region R43, VehBrand B14, and others), so their noisy
    individual estimates are pooled into a single "Other" category rather
    than given their own unstable coefficient.
    """
    exposure_by_cat = train_df.groupby(category_col)["Exposure"].sum()
    return set(exposure_by_cat[exposure_by_cat >= min_exposure].index)


def apply_common_categories(
    series: pd.Series, common_categories: set[str], other_label: str = "Other"
) -> pd.Series:
    """Map any category not in `common_categories` to `other_label`.

    This also covers a category never seen during training at all (not
    just ones seen but judged too rare) - the same mechanism that fitting
    only on train uses to handle low-credibility segments is what would
    also protect a deployed model from an input category it has never
    encountered (the Phase 10 Streamlit app's concern), for free.
    """
    return series.where(series.isin(common_categories), other_label)


# Default rare-category threshold: 2,000 training-split policy-years. Chosen
# by inspecting the real gaps in train-only exposure (not an arbitrary round
# number) - it falls cleanly between R74 (1,687) and R23 (2,224) for Region,
# and even more clearly between B14 (1,601) and B13 (4,745) for VehBrand,
# and recovers exactly the specific low-exposure segments Phase 3 already
# flagged by name (Region R43/R42/R21/R94/R83/R74, VehBrand B14) rather than
# an arbitrary, differently-shaped set.
DEFAULT_MIN_EXPOSURE = 2000.0


def build_model_table(
    freq: pd.DataFrame, splits: pd.DataFrame, min_exposure: float = DEFAULT_MIN_EXPOSURE
) -> tuple[pd.DataFrame, dict[str, set[str]]]:
    """Assemble the Phase 4 modelling table: freq joined with its split
    assignment, with every feature-engineering decision from this module
    applied. Deliberately does NOT one-hot encode anything: Phase 5's GLMs
    (statsmodels formulas) and Phase 8's boosting models each need
    categorical inputs in a different final shape, so this table keeps
    engineered categoricals as plain labelled columns (bands, grouped
    regions/brands) and lets each modelling phase encode them however that
    model actually requires - baking in one fixed encoding here would be
    presumptuous and could conflict with a later modelling choice.

    Rare-category grouping for Region and VehBrand is fit using ONLY the
    rows where split == "train" (fit_common_categories), then applied via
    apply_common_categories to every row regardless of split - this is
    what keeps the encoding identical and leak-free across train,
    validation, and test.

    Returns (model_table, category_maps). Persist category_maps (e.g. via
    joblib) alongside the model table so the exact same rule can be
    replayed later on brand-new data (Phase 7 evaluation, the Streamlit app).
    """
    merged = freq.merge(splits, on="IDpol", how="inner")
    if len(merged) != len(freq):
        raise ValueError("every policy in freq must have a split assignment")

    merged = add_age_and_bonusmalus_bands(merged)
    merged = log_density(merged)
    merged = encode_area_ordinal(merged)
    merged = encode_vehgas_binary(merged)

    train_rows = merged[merged["split"] == "train"]
    category_maps = {
        "Region": fit_common_categories(train_rows, "Region", min_exposure),
        "VehBrand": fit_common_categories(train_rows, "VehBrand", min_exposure),
    }
    merged["RegionGrouped"] = apply_common_categories(merged["Region"], category_maps["Region"])
    merged["VehBrandGrouped"] = apply_common_categories(
        merged["VehBrand"], category_maps["VehBrand"]
    )

    return merged, category_maps
