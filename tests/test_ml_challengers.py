"""Unit tests for auto_pricing.ml_challengers."""

import numpy as np
import pandas as pd

from auto_pricing.ml_challengers import fit_frequency_boosting, predict_frequency_boosting


def _synthetic_train(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    exposure = rng.uniform(0.1, 1.0, n)
    drivage_band = pd.Categorical(
        rng.choice(["18-22", "30-39", "70+"], n), categories=["18-22", "30-39", "70+"]
    )
    # A genuinely non-linear relationship: 18-22 and 70+ both high risk,
    # 30-39 low risk (a "U-shape" no single linear term could capture) -
    # exactly the kind of pattern gradient boosting is supposed to find.
    true_rate = np.select(
        [drivage_band == "18-22", drivage_band == "30-39", drivage_band == "70+"],
        [0.30, 0.05, 0.25],
    )
    claim_nb = rng.poisson(true_rate * exposure)

    train = pd.DataFrame(
        {
            "ClaimNb": claim_nb,
            "ClaimNbFromSev": claim_nb,  # same, for the paid-frequency-target test
            "Exposure": exposure,
            "DrivAgeBand": drivage_band,
            "VehAgeBand": pd.Categorical(["1-2"] * n, categories=["1-2"]),
            "BonusMalusBand": pd.Categorical(["50 (best)"] * n, categories=["50 (best)"]),
            "AreaOrdinal": np.zeros(n),
            "LogDensity": np.zeros(n),
            "VehGasBinary": np.zeros(n),
            "RegionGrouped": pd.Categorical(["R1"] * n, categories=["R1"]),
            "VehBrandGrouped": pd.Categorical(["B1"] * n, categories=["B1"]),
            "VehPower": np.full(n, 5.0),
        }
    )
    return train


def test_fit_frequency_boosting_runs_and_predicts_positive_rates():
    train = _synthetic_train()
    model = fit_frequency_boosting(train, max_iter=50)
    pred = predict_frequency_boosting(model, train)
    assert (pred >= 0).all()


def test_predict_frequency_boosting_reconciles_reasonably_with_observed():
    # Not an exact-match guarantee like a canonical-link GLM (gradient
    # boosting has no such property) - but a reasonably close portfolio
    # total is still the right sanity check before trusting predictions.
    train = _synthetic_train()
    model = fit_frequency_boosting(train, max_iter=200)
    pred = predict_frequency_boosting(model, train)

    predicted_total = pred.sum()
    observed_total = train["ClaimNb"].sum()
    assert abs(predicted_total / observed_total - 1) < 0.15  # within 15%


def test_frequency_boosting_recovers_a_nonlinear_ushape_pattern():
    # This is the whole point of trying gradient boosting: it should
    # correctly identify that both 18-22 AND 70+ are higher risk than
    # 30-39, a U-shape a single linear DrivAge coefficient could never
    # represent (this project deliberately used *banded* categories for
    # the GLMs for exactly this reason - confirming the ML challenger can
    # find the same non-linear shape independently is a real check, not
    # a given).
    train = _synthetic_train(n=8000)
    model = fit_frequency_boosting(train, max_iter=200)

    def predict_for_band(band: str) -> float:
        row = train.iloc[[0]].copy()
        row["DrivAgeBand"] = pd.Categorical([band], categories=train["DrivAgeBand"].cat.categories)
        return predict_frequency_boosting(model, row).iloc[0] / row["Exposure"].iloc[0]

    rate_18_22 = predict_for_band("18-22")
    rate_30_39 = predict_for_band("30-39")
    rate_70_plus = predict_for_band("70+")

    assert rate_18_22 > rate_30_39
    assert rate_70_plus > rate_30_39


def test_fit_frequency_boosting_supports_paid_frequency_target():
    # Mirrors Phase 7's frequency/severity mismatch resolution: the same
    # model-fitting function must support targeting ClaimNbFromSev (paid
    # claims), not just ClaimNb (reported claims), for a pure-premium
    # comparison later.
    train = _synthetic_train()
    model = fit_frequency_boosting(train, target_col="ClaimNbFromSev", max_iter=50)
    pred = predict_frequency_boosting(model, train)
    assert (pred >= 0).all()
