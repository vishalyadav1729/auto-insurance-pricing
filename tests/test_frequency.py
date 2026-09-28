"""Unit tests for auto_pricing.frequency.

Uses a small synthetic dataset (not the real 474k-row training split) so
these run fast, while still exercising the real statsmodels fit/predict
mechanics - in particular, the exact bug predict_frequency exists to
prevent (statsmodels' own .predict() dropping the offset silently).
"""

import joblib
import numpy as np
import pandas as pd

from auto_pricing.frequency import (
    baseline_frequency,
    deviance_dispersion_ratio,
    dispersion_ratio,
    fit_negative_binomial_glm,
    fit_poisson_glm,
    predict_frequency,
    predict_from_artifact,
    save_frequency_model,
)


def _synthetic_train(n=500, seed=0):
    rng = np.random.default_rng(seed)
    exposure = rng.uniform(0.05, 1.0, n)
    risk_group = rng.choice(["low", "high"], n)
    true_rate = np.where(risk_group == "high", 0.30, 0.05)
    claim_nb = rng.poisson(true_rate * exposure)
    return pd.DataFrame({"ClaimNb": claim_nb, "Exposure": exposure, "risk_group": risk_group})


def _synthetic_overdispersed_train(n=2000, seed=0):
    # Genuinely overdispersed: each row's TRUE rate is itself random (drawn
    # from a Gamma distribution around the group mean), then ClaimNb is a
    # Poisson draw given that row's own true rate - the exact Poisson-Gamma
    # mixture mechanism that produces a Negative Binomial distribution.
    rng = np.random.default_rng(seed)
    exposure = rng.uniform(0.05, 1.0, n)
    risk_group = rng.choice(["low", "high"], n)
    mean_rate = np.where(risk_group == "high", 0.30, 0.05)
    true_alpha = 1.0  # dispersion of the per-row true rate around the group mean
    shape = 1.0 / true_alpha
    row_rate = rng.gamma(shape=shape, scale=mean_rate * true_alpha)
    claim_nb = rng.poisson(row_rate * exposure)
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


def test_fit_negative_binomial_glm_converges_and_finds_real_dispersion():
    train = _synthetic_overdispersed_train()
    result = fit_negative_binomial_glm(train, formula="ClaimNb ~ C(risk_group)")
    assert result.mle_retvals.get("converged")
    # Data was constructed with genuine overdispersion (alpha=1.0 in the
    # generator), so the fitted alpha should be clearly above zero, not
    # collapse back to a Poisson-like near-zero estimate.
    assert result.params["alpha"] > 0.2


def test_predict_frequency_works_for_negative_binomial_results_too():
    # predict_frequency must generalize across model classes: statsmodels'
    # NegativeBinomial results object has the same offset-dropping trap as
    # the Poisson GLM's, confirmed while building this (a naive
    # results.predict(df) overstated total training claims on the real
    # data by a similar 2x+ factor for this model class too).
    train = _synthetic_overdispersed_train()
    result = fit_negative_binomial_glm(train, formula="ClaimNb ~ C(risk_group)")

    correct = predict_frequency(result, train)
    naive_wrong = result.predict(train)
    assert not np.allclose(correct, naive_wrong)


def test_save_frequency_model_produces_a_small_file(tmp_path):
    train = _synthetic_train()
    result = fit_poisson_glm(train, formula="ClaimNb ~ C(risk_group)")
    path = tmp_path / "model.joblib"
    save_frequency_model(result, formula="ClaimNb ~ C(risk_group)", model_type="poisson", path=path)
    # Should be a few KB (a handful of coefficients + scalars), nowhere
    # near the hundreds of MB a raw pickle of the results object produced.
    assert path.stat().st_size < 50_000


def test_predict_from_artifact_matches_predict_frequency_poisson():
    train = _synthetic_train()
    result = fit_poisson_glm(train, formula="ClaimNb ~ C(risk_group)")
    artifact = {
        "formula": "ClaimNb ~ C(risk_group)",
        "model_type": "poisson",
        "params": result.params,
    }
    reconstructed = predict_from_artifact(artifact, train)
    correct = predict_frequency(result, train)
    assert np.allclose(reconstructed, correct)


def test_predict_from_artifact_matches_predict_frequency_negative_binomial():
    train = _synthetic_overdispersed_train()
    result = fit_negative_binomial_glm(train, formula="ClaimNb ~ C(risk_group)")
    artifact = {
        "formula": "ClaimNb ~ C(risk_group)",
        "model_type": "negative_binomial",
        "params": result.params,
    }
    reconstructed = predict_from_artifact(artifact, train)
    correct = predict_frequency(result, train)
    assert np.allclose(reconstructed, correct)


def test_predict_from_artifact_works_on_a_single_row_with_declared_categories():
    # Mirrors the exact bug found while building this: a single new row
    # must still produce the correct design matrix width, which requires
    # the categorical column to have its FULL category list declared via
    # pd.Categorical(..., categories=...) - not just whatever happens to
    # be present in that one row (which, being one row, is only ever one
    # category).
    n = 200
    rng = np.random.default_rng(0)
    exposure = rng.uniform(0.1, 1.0, n)
    group = pd.Categorical(rng.choice(["A", "B", "C"], n), categories=["A", "B", "C"])
    claim_nb = rng.poisson(0.1 * exposure)
    train = pd.DataFrame({"ClaimNb": claim_nb, "Exposure": exposure, "group": group})

    result = fit_poisson_glm(train, formula="ClaimNb ~ C(group)")
    artifact = {"formula": "ClaimNb ~ C(group)", "model_type": "poisson", "params": result.params}

    single_row = train.iloc[[0]]  # only one category level present here
    reconstructed = predict_from_artifact(artifact, single_row)
    correct = predict_frequency(result, single_row)
    assert np.allclose(reconstructed, correct)


def test_save_and_reload_artifact_round_trips_via_joblib(tmp_path):
    train = _synthetic_train()
    result = fit_poisson_glm(train, formula="ClaimNb ~ C(risk_group)")
    path = tmp_path / "model.joblib"
    save_frequency_model(result, formula="ClaimNb ~ C(risk_group)", model_type="poisson", path=path)

    loaded = joblib.load(path)
    reconstructed = predict_from_artifact(loaded, train)
    correct = predict_frequency(result, train)
    assert np.allclose(reconstructed, correct)
    assert loaded["aic"] == result.aic
    assert loaded["converged"] is True
