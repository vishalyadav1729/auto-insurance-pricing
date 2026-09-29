"""Unit tests for auto_pricing.features.

Each test targets exactly one rule from reports/cleaning_policy.md, using
small hand-built DataFrames so the expected result can be checked by eye.
"""

import pandas as pd

from auto_pricing.features import (
    add_age_and_bonusmalus_bands,
    apply_common_categories,
    build_model_table,
    cap_claim_nb,
    clean_frequency,
    clean_severity,
    clean_veh_gas,
    clip_exposure,
    encode_area_ordinal,
    encode_vehgas_binary,
    engineer_single_policy,
    exclude_orphan_claims,
    fit_common_categories,
    log_density,
)


def test_clip_exposure_caps_values_above_one():
    df = pd.DataFrame({"Exposure": [0.5, 1.0, 1.5, 2.01]})
    out = clip_exposure(df)
    assert out["Exposure"].tolist() == [0.5, 1.0, 1.0, 1.0]


def test_clip_exposure_does_not_mutate_input():
    df = pd.DataFrame({"Exposure": [1.5]})
    clip_exposure(df)
    assert df["Exposure"].iloc[0] == 1.5


def test_cap_claim_nb_caps_values_above_four():
    df = pd.DataFrame({"ClaimNb": [0, 1, 4, 5, 16]})
    out = cap_claim_nb(df)
    assert out["ClaimNb"].tolist() == [0, 1, 4, 4, 4]


def test_clean_veh_gas_strips_quote_characters():
    df = pd.DataFrame({"VehGas": ["'Diesel'", "'Regular'"]})
    out = clean_veh_gas(df)
    assert out["VehGas"].tolist() == ["Diesel", "Regular"]


def test_exclude_orphan_claims_drops_unmatched_ids():
    freq = pd.DataFrame({"IDpol": [1, 2, 3]})
    sev = pd.DataFrame({"IDpol": [1, 2, 99], "ClaimAmount": [10.0, 20.0, 30.0]})
    out = exclude_orphan_claims(freq, sev)
    assert list(out["IDpol"]) == [1, 2]


def test_exclude_orphan_claims_keeps_all_when_no_orphans():
    freq = pd.DataFrame({"IDpol": [1, 2]})
    sev = pd.DataFrame({"IDpol": [1, 2], "ClaimAmount": [10.0, 20.0]})
    out = exclude_orphan_claims(freq, sev)
    assert len(out) == 2


def test_clean_frequency_applies_all_rules_together():
    df = pd.DataFrame(
        {
            "IDpol": [1, 2],
            "ClaimNb": [1, 16],
            "Exposure": [0.5, 2.01],
            "VehGas": ["'Diesel'", "'Regular'"],
        }
    )
    out = clean_frequency(df)
    assert out["Exposure"].tolist() == [0.5, 1.0]
    assert out["ClaimNb"].tolist() == [1, 4]
    assert out["VehGas"].tolist() == ["Diesel", "Regular"]


def test_clean_severity_removes_orphans_using_cleaned_freq():
    freq = pd.DataFrame(
        {
            "IDpol": [1, 2],
            "ClaimNb": [0, 1],
            "Exposure": [0.5, 1.0],
            "VehGas": ["'Diesel'", "'Regular'"],
        }
    )
    freq_clean = clean_frequency(freq)
    sev = pd.DataFrame({"IDpol": [1, 99], "ClaimAmount": [10.0, 20.0]})
    out = clean_severity(freq_clean, sev)
    assert list(out["IDpol"]) == [1]


def test_add_age_and_bonusmalus_bands_assigns_expected_labels():
    df = pd.DataFrame({"DrivAge": [18, 45], "VehAge": [0, 12], "BonusMalus": [50, 140]})
    out = add_age_and_bonusmalus_bands(df)
    assert out["DrivAgeBand"].astype(str).tolist() == ["18-22", "40-49"]
    assert out["VehAgeBand"].astype(str).tolist() == ["0", "10-14"]
    assert out["BonusMalusBand"].astype(str).tolist() == ["50 (best)", "130+"]


def test_log_density_matches_expected_values():
    import numpy as np

    df = pd.DataFrame({"Density": [1, 100]})
    out = log_density(df)
    assert out["LogDensity"].iloc[0] == 0.0  # log(1) == 0
    assert abs(out["LogDensity"].iloc[1] - np.log(100)) < 1e-9


def test_encode_area_ordinal_maps_a_through_f():
    df = pd.DataFrame({"Area": ["A", "C", "F"]})
    out = encode_area_ordinal(df)
    assert out["AreaOrdinal"].tolist() == [0, 2, 5]


def test_fit_common_categories_excludes_low_exposure_categories():
    train = pd.DataFrame(
        {"Region": ["R1", "R1", "R2", "R3"], "Exposure": [500.0, 500.0, 10.0, 1000.0]}
    )
    # R1 total = 1000, R2 total = 10, R3 total = 1000
    common = fit_common_categories(train, "Region", min_exposure=100.0)
    assert common == {"R1", "R3"}


def test_apply_common_categories_maps_rare_and_unseen_to_other():
    common = {"R1", "R3"}
    series = pd.Series(["R1", "R2", "R3", "R99"])  # R2 rare, R99 never seen at fit time
    out = apply_common_categories(series, common)
    assert out.tolist() == ["R1", "Other", "R3", "Other"]


def test_encode_vehgas_binary():
    df = pd.DataFrame({"VehGas": ["Diesel", "Regular"]})
    out = encode_vehgas_binary(df)
    assert out["VehGasBinary"].tolist() == [0, 1]


def _synthetic_freq_for_model_table():
    n = 6
    return pd.DataFrame(
        {
            "IDpol": [1, 2, 3, 4, 5, 6],
            "ClaimNb": [0] * n,
            "Exposure": [1000.0, 1000.0, 1000.0, 100.0, 5000.0, 5000.0],
            "Region": ["X", "X", "X", "Y", "Y", "Y"],
            "VehBrand": ["B1"] * n,
            "VehGas": ["Regular"] * n,
            "Area": ["A"] * n,
            "VehPower": [5] * n,
            "VehAge": [5] * n,
            "DrivAge": [40] * n,
            "BonusMalus": [50] * n,
            "Density": [100] * n,
        }
    )


def test_build_model_table_fits_rare_categories_using_train_only():
    freq = _synthetic_freq_for_model_table()
    # Region X: all 3 policies in train, total exposure 3000 -> common.
    # Region Y: only 1 policy (IDpol=4, exposure=100) is in train -> rare,
    # even though Y's *combined* exposure across all splits is 10,100
    # (100 + 5000 + 5000), which would look common if the fit incorrectly
    # used validation/test rows too.
    splits = pd.DataFrame(
        {"IDpol": [1, 2, 3, 4, 5, 6], "split": ["train", "train", "train", "train", "validation", "test"]}
    )

    model_table, category_maps = build_model_table(freq, splits, min_exposure=2000.0)

    assert category_maps["Region"] == {"X"}
    y_rows = model_table[model_table["Region"] == "Y"]
    assert (y_rows["RegionGrouped"] == "Other").all()


def test_build_model_table_raises_if_split_assignment_missing():
    freq = _synthetic_freq_for_model_table()
    splits = pd.DataFrame({"IDpol": [1, 2, 3], "split": ["train", "train", "train"]})  # missing 4,5,6
    try:
        build_model_table(freq, splits)
        assert False, "expected ValueError when a policy has no split assignment"
    except ValueError:
        pass


def test_build_model_table_adds_expected_engineered_columns():
    freq = _synthetic_freq_for_model_table()
    splits = pd.DataFrame({"IDpol": [1, 2, 3, 4, 5, 6], "split": ["train"] * 6})
    model_table, _ = build_model_table(freq, splits, min_exposure=2000.0)
    for col in [
        "DrivAgeBand", "VehAgeBand", "BonusMalusBand", "LogDensity",
        "AreaOrdinal", "VehGasBinary", "RegionGrouped", "VehBrandGrouped",
    ]:
        assert col in model_table.columns


def _sample_raw_policy():
    return {
        "Area": "D", "VehPower": 7, "VehAge": 2, "DrivAge": 35, "BonusMalus": 50,
        "VehBrand": "B1", "VehGas": "Diesel", "Density": 1500, "Region": "R24",
        "Exposure": 1.0,
    }


def test_engineer_single_policy_adds_every_engineered_column():
    category_maps = {"Region": {"R24"}, "VehBrand": {"B1"}}
    row = engineer_single_policy(_sample_raw_policy(), category_maps)
    assert len(row) == 1
    for col in [
        "DrivAgeBand", "VehAgeBand", "BonusMalusBand", "LogDensity",
        "AreaOrdinal", "VehGasBinary", "RegionGrouped", "VehBrandGrouped",
    ]:
        assert col in row.columns
    assert row["DrivAgeBand"].iloc[0] == "30-39"
    assert row["RegionGrouped"].iloc[0] == "R24"
    assert row["VehBrandGrouped"].iloc[0] == "B1"


def test_engineer_single_policy_pools_a_category_never_seen_as_common():
    # The exact mechanism a deployed app needs for free (apply_common_categories'
    # own docstring): a Region/VehBrand this policy reports that ISN'T in
    # category_maps (whether genuinely rare or simply never seen during
    # training) must fall back to "Other", not raise or silently create a
    # new category the trained model has no coefficient for.
    category_maps = {"Region": {"R24"}, "VehBrand": {"B1"}}
    raw = dict(_sample_raw_policy(), Region="R99", VehBrand="B99")
    row = engineer_single_policy(raw, category_maps)
    assert row["RegionGrouped"].iloc[0] == "Other"
    assert row["VehBrandGrouped"].iloc[0] == "Other"


def test_engineer_single_policy_declares_the_full_category_list():
    # Matches apply_common_categories' own documented requirement: the
    # returned column must be a Categorical with every trained category
    # declared, even though a single row can only ever show one of them -
    # otherwise a formula-based model builds the wrong-width design matrix
    # for this row (the exact bug predict_from_artifact's tests guard
    # against on the modelling side; this is the feature-engineering side
    # of the same requirement).
    category_maps = {"Region": {"R24", "R11", "R82"}, "VehBrand": {"B1", "B2"}}
    row = engineer_single_policy(_sample_raw_policy(), category_maps)
    assert set(row["RegionGrouped"].cat.categories) == {"R24", "R11", "R82", "Other"}
    assert set(row["VehBrandGrouped"].cat.categories) == {"B1", "B2", "Other"}
