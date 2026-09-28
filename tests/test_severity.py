"""Unit tests for auto_pricing.severity.

Uses small hand-built / synthetic data so these run fast while still
exercising the real statsmodels Gamma GLM fit/predict mechanics.
"""

import numpy as np
import pandas as pd

from auto_pricing.severity import baseline_severity, build_severity_table, fit_gamma_glm, predict_severity


def _synthetic_severity_train(n=300, seed=0):
    rng = np.random.default_rng(seed)
    group = rng.choice(["A", "B"], n)
    mean_cost = np.where(group == "A", 1000.0, 2000.0)
    # Gamma-distributed costs with a realistic shape (right-skewed, positive)
    shape = 2.0
    claim_amount = rng.gamma(shape=shape, scale=mean_cost / shape)
    return pd.DataFrame({"ClaimAmount": claim_amount, "group": group})


def test_build_severity_table_joins_policy_features_and_split():
    sev = pd.DataFrame({"IDpol": [1, 2, 2], "ClaimAmount": [100.0, 200.0, 300.0]})
    model_table = pd.DataFrame(
        {
            "IDpol": [1, 2, 3],
            "split": ["train", "validation", "train"],
            "Area": ["A", "B", "C"],
            "VehPower": [5, 6, 7],
            "VehAge": [1, 2, 3],
            "DrivAge": [30, 40, 50],
            "BonusMalus": [50, 60, 70],
            "VehBrand": ["B1", "B2", "B3"],
            "VehGas": ["Diesel", "Regular", "Diesel"],
            "Density": [100, 200, 300],
            "Region": ["R1", "R2", "R3"],
            "DrivAgeBand": ["a", "b", "c"],
            "VehAgeBand": ["a", "b", "c"],
            "BonusMalusBand": ["a", "b", "c"],
            "LogDensity": [1.0, 2.0, 3.0],
            "AreaOrdinal": [0, 1, 2],
            "VehGasBinary": [0, 1, 0],
            "RegionGrouped": ["R1", "R2", "R3"],
            "VehBrandGrouped": ["B1", "B2", "B3"],
        }
    )
    out = build_severity_table(sev, model_table)
    assert len(out) == 3  # one row per claim, not per policy
    assert out["split"].tolist() == ["train", "validation", "validation"]
    assert out.loc[out["IDpol"] == 2, "Area"].unique().tolist() == ["B"]


def test_build_severity_table_raises_if_a_claim_has_no_matching_policy():
    sev = pd.DataFrame({"IDpol": [1, 99], "ClaimAmount": [100.0, 200.0]})
    model_table = pd.DataFrame(
        {
            "IDpol": [1],
            "split": ["train"],
            "Area": ["A"],
            "VehPower": [5],
            "VehAge": [1],
            "DrivAge": [30],
            "BonusMalus": [50],
            "VehBrand": ["B1"],
            "VehGas": ["Diesel"],
            "Density": [100],
            "Region": ["R1"],
            "DrivAgeBand": ["a"],
            "VehAgeBand": ["a"],
            "BonusMalusBand": ["a"],
            "LogDensity": [1.0],
            "AreaOrdinal": [0],
            "VehGasBinary": [0],
            "RegionGrouped": ["R1"],
            "VehBrandGrouped": ["B1"],
        }
    )
    try:
        build_severity_table(sev, model_table)
        assert False, "expected ValueError for an unmatched claim"
    except ValueError:
        pass


def test_baseline_severity_is_the_mean():
    df = pd.DataFrame({"ClaimAmount": [100.0, 200.0, 300.0]})
    assert baseline_severity(df) == 200.0


def test_fit_gamma_glm_converges_and_predicts_positive_values():
    train = _synthetic_severity_train()
    result = fit_gamma_glm(train, formula="ClaimAmount ~ C(group)")
    assert result.converged

    pred = predict_severity(result, train)
    assert (pred > 0).all()  # Gamma/log-link predictions must always be positive


def test_fit_gamma_glm_recovers_the_group_difference():
    # Data was generated with group B costing 2x group A on average - the
    # fitted model should recover roughly that ratio, not just converge.
    train = _synthetic_severity_train(n=2000)
    result = fit_gamma_glm(train, formula="ClaimAmount ~ C(group)")
    relativity_b_vs_a = np.exp(result.params["C(group)[T.B]"])
    assert 1.5 < relativity_b_vs_a < 2.5
