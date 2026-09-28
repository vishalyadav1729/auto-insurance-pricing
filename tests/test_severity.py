"""Unit tests for auto_pricing.severity.

Uses small hand-built / synthetic data so these run fast while still
exercising the real statsmodels Gamma GLM fit/predict mechanics.
"""

import numpy as np
import pandas as pd

from auto_pricing.severity import (
    baseline_severity,
    build_severity_table,
    exclude_large_losses,
    fit_gamma_glm,
    fit_lognormal_model,
    large_loss_threshold,
    predict_lognormal_severity,
    predict_severity,
)


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


def _synthetic_lognormal_train(n=20_000, seed=0, mu=7.0, sigma=1.2):
    # log(Y) ~ Normal(mu, sigma^2) -> Y is lognormal with a KNOWN true
    # mean of exp(mu + sigma^2/2), which is what the smearing-corrected
    # prediction should recover - and NOT what naive exp(prediction)
    # recovers (that targets the median, exp(mu), instead).
    rng = np.random.default_rng(seed)
    log_claim = rng.normal(mu, sigma, n)
    claim_amount = np.exp(log_claim)
    return pd.DataFrame({"ClaimAmount": claim_amount})


def test_lognormal_smearing_correction_recovers_the_true_mean_better_than_naive():
    train = _synthetic_lognormal_train()
    true_mean = np.exp(7.0 + 1.2**2 / 2)  # analytic lognormal mean

    result, smearing_factor = fit_lognormal_model(train, formula="ClaimAmount ~ 1")

    naive_pred = np.exp(result.predict(train)).mean()  # the mistake, on purpose
    corrected_pred = predict_lognormal_severity(result, smearing_factor, train).mean()

    naive_error = abs(naive_pred - true_mean)
    corrected_error = abs(corrected_pred - true_mean)
    # Naive exponentiation is a systematic BIAS that doesn't shrink with
    # sample size (confirmed separately: ~50% error at n=3,000 and again
    # at n=100,000); the smearing correction converges toward the true
    # mean as sample size grows, which is the actual signature of a real
    # bias fix rather than a lucky draw. At n=20,000 the margins below are
    # comfortably inside what was empirically observed (corrected ~0.4%,
    # naive ~51%).
    assert corrected_error < naive_error
    assert corrected_error / true_mean < 0.02  # within 2% of the true mean
    assert naive_error / true_mean > 0.3  # naive is off by a large, persistent margin


def test_large_loss_threshold_matches_the_underlying_quantile_call():
    # Not asserting a hand-computed value here: pandas' default quantile
    # interpolation on a small sample is not simply "the Nth largest
    # value" (confirmed while writing this test - quantile(0.8) on
    # [100,200,300,400,10000] is 2320.0, not 10000.0, because of linear
    # interpolation between the two nearest ranks). What actually matters
    # is that the function asks for the correct quantile (1 - top_pct).
    train = pd.DataFrame({"ClaimAmount": [100.0, 200.0, 300.0, 400.0, 10000.0]})
    assert large_loss_threshold(train, top_pct=0.2) == train["ClaimAmount"].quantile(0.8)
    assert large_loss_threshold(train, top_pct=0.01) == train["ClaimAmount"].quantile(0.99)


def test_exclude_large_losses_drops_claims_at_or_above_threshold():
    df = pd.DataFrame({"ClaimAmount": [100.0, 500.0, 1000.0, 5000.0]})
    out = exclude_large_losses(df, threshold=1000.0)
    assert out["ClaimAmount"].tolist() == [100.0, 500.0]  # 1000.0 itself excluded too
