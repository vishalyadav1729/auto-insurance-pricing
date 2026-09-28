"""Unit tests for auto_pricing.frequency.

Uses a small synthetic dataset (not the real 474k-row training split) so
these run fast, while still exercising the real statsmodels fit/predict
mechanics - in particular, the exact bug predict_frequency exists to
prevent (statsmodels' own .predict() dropping the offset silently).
"""

import numpy as np
import pandas as pd

from auto_pricing.frequency import (
    baseline_frequency,
    deviance_dispersion_ratio,
    dispersion_ratio,
    fit_poisson_glm,
    predict_frequency,
)


def _synthetic_train(n=500, seed=0):
    rng = np.random.default_rng(seed)
    exposure = rng.uniform(0.05, 1.0, n)
    risk_group = rng.choice(["low", "high"], n)
    true_rate = np.where(risk_group == "high", 0.30, 0.05)
    claim_nb = rng.poisson(true_rate * exposure)
    return pd.DataFrame({"ClaimNb": claim_nb, "Exposure": exposure, "risk_group": risk_group})


def test_baseline_frequency_is_exposure_weighted():
    df = pd.DataFrame({"ClaimNb": [1, 0, 1], "Exposure": [1.0, 1.0, 0.5]})
    assert baseline_frequency(df) == 2 / 2.5


def test_fit_poisson_glm_converges_on_synthetic_data():
    train = _synthetic_train()
    result = fit_poisson_glm(train, formula="ClaimNb ~ C(risk_group)")
    assert result.converged


def test_predict_frequency_matches_fittedvalues_on_training_data():
    # This is the core guarantee of a canonical-link Poisson GLM: total
    # predicted claims on the data it was fit on must equal total observed
    # claims, exactly (up to floating point). If predict_frequency ever
    # regressed to dropping the offset again, this test would catch it
    # immediately via a wildly wrong total.
    train = _synthetic_train()
    result = fit_poisson_glm(train, formula="ClaimNb ~ C(risk_group)")
    predicted_total = predict_frequency(result, train).sum()
    observed_total = train["ClaimNb"].sum()
    assert abs(predicted_total - observed_total) < 1e-6


def test_predict_frequency_differs_from_naive_predict_when_offset_dropped():
    # Documents the exact bug predict_frequency exists to prevent: calling
    # statsmodels' raw .predict() without re-supplying the offset silently
    # assumes Exposure == 1 for every row, which is wrong whenever Exposure
    # actually varies.
    train = _synthetic_train()
    result = fit_poisson_glm(train, formula="ClaimNb ~ C(risk_group)")

    correct = predict_frequency(result, train)
    naive_wrong = result.predict(train)  # the mistake, on purpose, to prove the contrast

    assert not np.allclose(correct, naive_wrong)


def test_dispersion_ratio_is_positive_float():
    train = _synthetic_train()
    result = fit_poisson_glm(train, formula="ClaimNb ~ C(risk_group)")
    assert dispersion_ratio(result) > 0
    assert deviance_dispersion_ratio(result) > 0
