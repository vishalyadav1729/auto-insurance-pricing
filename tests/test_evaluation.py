"""Unit tests for auto_pricing.evaluation.

Key thing to pin down: the deviance metric should genuinely prefer a
model whose predictions are closer to the truth, not just return some
number - and calibration_by_decile should correctly separate a
low-predicted-risk group from a high-predicted-risk one.
"""

import pandas as pd

from auto_pricing.evaluation import (
    calibration_by_decile,
    exposure_weighted_poisson_deviance,
    observed_to_expected_ratio,
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
