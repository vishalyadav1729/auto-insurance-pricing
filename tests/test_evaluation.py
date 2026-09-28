"""Unit tests for auto_pricing.evaluation.

Key thing to pin down: the deviance metric should genuinely prefer a
model whose predictions are closer to the truth, not just return some
number - and calibration_by_decile should correctly separate a
low-predicted-risk group from a high-predicted-risk one.
"""

import numpy as np
import pandas as pd
import pytest

from auto_pricing.evaluation import (
    calibration_by_decile,
    exposure_weighted_poisson_deviance,
    gamma_deviance,
    gini_index,
    lorenz_curve,
    observed_to_expected_ratio,
    pure_premium_calibration_by_decile,
    severity_calibration_by_decile,
    tweedie_deviance,
)


def test_deviance_is_near_zero_for_a_near_perfect_prediction():
    # Poisson deviance requires strictly positive predictions (it involves
    # log(y_pred)) - a real model's predictions, coming from exp(...), can
    # get arbitrarily close to zero but never exactly reach it, so "almost
    # observed" rather than exactly-equal-including-zeros is the realistic
    # version of "a great prediction" to test here.
    observed = pd.Series([1.0, 0.0, 2.0, 0.0])
    near_perfect = pd.Series([1.0001, 0.0001, 2.0001, 0.0001])
    exposure = pd.Series([1.0, 1.0, 1.0, 1.0])
    deviance = exposure_weighted_poisson_deviance(observed, near_perfect, exposure)
    assert deviance < 1e-3


def test_deviance_prefers_the_closer_prediction():
    observed = pd.Series([0.0, 1.0, 0.0, 2.0, 0.0, 1.0])
    exposure = pd.Series([1.0] * 6)
    good_prediction = pd.Series([0.1, 0.9, 0.2, 1.8, 0.1, 1.1])
    bad_prediction = pd.Series([1.0, 0.1, 1.0, 0.1, 1.0, 0.1])

    good_deviance = exposure_weighted_poisson_deviance(observed, good_prediction, exposure)
    bad_deviance = exposure_weighted_poisson_deviance(observed, bad_prediction, exposure)
    assert good_deviance < bad_deviance


def test_observed_to_expected_ratio_is_one_for_matching_totals():
    observed = pd.Series([1.0, 0.0, 2.0])
    predicted = pd.Series([0.5, 0.5, 2.0])
    assert observed_to_expected_ratio(observed, predicted) == 1.0


def test_observed_to_expected_ratio_flags_underprediction():
    observed = pd.Series([10.0])
    predicted = pd.Series([5.0])
    assert observed_to_expected_ratio(observed, predicted) == 2.0


def test_calibration_by_decile_separates_low_and_high_risk_groups():
    # 100 low-risk policies (predicted and observed both low), 100
    # high-risk policies (predicted and observed both high). With 2 bins,
    # decile 0 should end up capturing the low-risk group and the other
    # bin the high-risk group.
    n = 100
    observed = pd.Series([0.05] * n + [0.5] * n)
    predicted = pd.Series([0.05] * n + [0.5] * n)
    exposure = pd.Series([1.0] * (2 * n))

    table = calibration_by_decile(observed, predicted, exposure, n_bins=2)
    assert len(table) == 2
    low_bin = table.iloc[0]
    high_bin = table.iloc[1]
    assert low_bin["observed_rate"] < high_bin["observed_rate"]
    assert low_bin["n_policies"] == n
    assert high_bin["n_policies"] == n


def test_gamma_deviance_prefers_the_closer_prediction():
    observed = pd.Series([100.0, 500.0, 1200.0, 300.0, 2000.0])
    good_prediction = pd.Series([110.0, 480.0, 1250.0, 290.0, 1900.0])
    bad_prediction = pd.Series([1000.0, 50.0, 100.0, 3000.0, 200.0])

    good_dev = gamma_deviance(observed, good_prediction)
    bad_dev = gamma_deviance(observed, bad_prediction)
    assert good_dev < bad_dev


def test_severity_calibration_by_decile_separates_cheap_and_expensive_claims():
    n = 50
    observed = pd.Series([200.0] * n + [5000.0] * n)
    predicted = pd.Series([200.0] * n + [5000.0] * n)

    table = severity_calibration_by_decile(observed, predicted, n_bins=2)
    assert len(table) == 2
    assert table.iloc[0]["observed_mean"] < table.iloc[1]["observed_mean"]
    assert table.iloc[0]["n_claims"] == n
    assert table.iloc[1]["n_claims"] == n


def test_tweedie_deviance_prefers_the_closer_prediction():
    observed = pd.Series([0.0, 500.0, 0.0, 1200.0, 0.0])
    good_prediction = pd.Series([50.0, 480.0, 50.0, 1150.0, 50.0])
    bad_prediction = pd.Series([500.0, 50.0, 500.0, 50.0, 500.0])

    good_dev = tweedie_deviance(observed, good_prediction, power=1.5)
    bad_dev = tweedie_deviance(observed, bad_prediction, power=1.5)
    assert good_dev < bad_dev


def test_pure_premium_calibration_by_decile_separates_low_and_high_risk():
    n = 100
    observed = pd.Series([0.0] * (n - 5) + [1000.0] * 5 + [0.0] * (n - 5) + [5000.0] * 5)
    predicted = pd.Series([50.0] * n + [500.0] * n)
    exposure = pd.Series([1.0] * (2 * n))

    table = pure_premium_calibration_by_decile(observed, predicted, exposure, n_bins=2)
    assert len(table) == 2
    assert table.iloc[0]["observed_rate"] < table.iloc[1]["observed_rate"]
    assert table.iloc[0]["n_policies"] == n
    assert table.iloc[1]["n_policies"] == n


def test_lorenz_curve_is_diagonal_for_a_non_discriminating_model():
    # If every policy gets the SAME predicted amount, ranking by it is
    # arbitrary - the Lorenz curve should sit on the diagonal (cumulative
    # observed share tracks cumulative exposure share exactly), the
    # textbook "no discrimination at all" case.
    n = 200
    rng = np.random.default_rng(0)
    observed = pd.Series(rng.exponential(scale=100.0, size=n))
    predicted = pd.Series([1.0] * n)  # identical for everyone -> no real ranking
    exposure = pd.Series([1.0] * n)

    gini = gini_index(observed, predicted, exposure)
    assert abs(gini) < 0.15  # not exactly 0 due to residual random ordering, but small


def test_gini_index_high_for_a_model_that_concentrates_risk_correctly():
    # A perfectly monotonic ranking is NOT sufficient for a high Gini on
    # its own (confirmed while writing this test: uniformly-distributed
    # observed cost with perfect ranking gives Gini ~0.32, not the >0.9
    # first assumed here) - Gini also depends on how CONCENTRATED the
    # underlying quantity is. Realistic insurance loss is concentrated
    # (this project found the top 1% of claims hold 38-41% of value), so
    # that is the shape used here: most policies contribute near-zero
    # loss, a small top-ranked share accounts for nearly all of it.
    n_low, n_high = 900, 100
    predicted = pd.Series([1.0] * n_low + [100.0] * n_high)
    observed = pd.Series([0.0] * n_low + [1000.0] * n_high)
    exposure = pd.Series([1.0] * (n_low + n_high))

    gini = gini_index(observed, predicted, exposure)
    assert gini > 0.8


def test_gini_index_is_unchanged_by_scaling_predictions_but_calibration_is_not():
    # This is the exact point the plan warns about: Gini measures ranking,
    # not calibration. Scaling every prediction by a constant factor
    # cannot change their relative order (so Gini is identical), but it
    # very much changes whether the predictions are correctly CALIBRATED
    # (observed-to-expected ratio moves a lot).
    n = 200
    rng = np.random.default_rng(0)
    predicted = pd.Series(rng.uniform(1, 100, n))
    observed = predicted * 10 + rng.normal(0, 50, n).clip(min=-predicted * 10 + 1)
    exposure = pd.Series([1.0] * n)

    scaled_predicted = predicted * 3.0  # badly miscalibrated on purpose, same ranking

    gini_original = gini_index(observed, predicted, exposure)
    gini_scaled = gini_index(observed, scaled_predicted, exposure)
    assert gini_original == pytest.approx(gini_scaled, abs=1e-9)

    oe_original = observed_to_expected_ratio(observed, predicted)
    oe_scaled = observed_to_expected_ratio(observed, scaled_predicted)
    assert oe_original != pytest.approx(oe_scaled, rel=0.01)


def test_gini_index_requires_annualized_rate_not_raw_expected_loss():
    # A model predicting the SAME rate for every policy has zero real
    # discriminating power - its Gini should be close to 0. Confirmed on
    # this project's real data that ranking by raw expected loss
    # (rate x exposure) instead gives a strongly NEGATIVE Gini for exactly
    # this kind of non-discriminating model, because that ranking becomes
    # mechanically a ranking by Exposure - and a handful of very-short-
    # exposure policies with one large ("unlucky") claim then dominate the
    # low end of that ranking, even though Exposure carries no real risk
    # signal at all.
    n = 500
    rng = np.random.default_rng(0)
    exposure = pd.Series(rng.uniform(0.01, 1.0, n))
    observed = pd.Series(rng.exponential(scale=50.0, size=n))

    large_idx = rng.choice(n, size=5, replace=False)
    observed.iloc[large_idx] = 5000.0
    exposure.iloc[large_idx] = 0.02

    flat_rate = pd.Series([1.0] * n)  # identical for everyone - no real ranking info
    expected_loss = flat_rate * exposure  # what ranking by raw expected loss would use

    gini_by_rate = gini_index(observed, flat_rate, exposure)
    gini_by_expected_loss = gini_index(observed, expected_loss, exposure)

    assert abs(gini_by_rate) < 0.2  # correctly close to zero
    assert gini_by_expected_loss < gini_by_rate - 0.3  # misleadingly, substantially more negative
