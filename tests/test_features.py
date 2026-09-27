"""Unit tests for auto_pricing.features.

Each test targets exactly one rule from reports/cleaning_policy.md, using
small hand-built DataFrames so the expected result can be checked by eye.
"""

import pandas as pd

from auto_pricing.features import (
    add_age_and_bonusmalus_bands,
    apply_common_categories,
    cap_claim_nb,
    clean_frequency,
    clean_severity,
    clean_veh_gas,
    clip_exposure,
    encode_area_ordinal,
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
