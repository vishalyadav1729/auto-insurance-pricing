"""Unit tests for auto_pricing.pure_premium."""

import numpy as np
import pandas as pd
import pytest

from auto_pricing.pure_premium import (
    PAID_FREQUENCY_FORMULA,
    build_pure_premium_table,
    fit_paid_frequency_glm,
    predict_annual_pure_premium,
    predict_expected_loss,
)


def test_paid_frequency_formula_targets_claimnbfromsev_not_claimnb():
    assert PAID_FREQUENCY_FORMULA.startswith("ClaimNbFromSev ~")
    assert "ClaimNb ~" not in PAID_FREQUENCY_FORMULA
    # everything after the target should be unchanged from FREQUENCY_FORMULA
    from auto_pricing.frequency import FREQUENCY_FORMULA

    assert PAID_FREQUENCY_FORMULA.split("~")[1] == FREQUENCY_FORMULA.split("~")[1]


def test_build_pure_premium_table_fills_zero_for_policies_without_paid_claims():
    freq = pd.DataFrame({"IDpol": [1, 2, 3]})  # unused by this function, interface symmetry only
    sev = pd.DataFrame({"IDpol": [2, 3, 3], "ClaimAmount": [500.0, 100.0, 200.0]})
    model_table = pd.DataFrame({"IDpol": [1, 2, 3], "ClaimNb": [0, 1, 2], "Exposure": [1.0, 1.0, 1.0]})

    out = build_pure_premium_table(freq, sev, model_table)

    row1 = out.loc[out["IDpol"] == 1].iloc[0]
    assert row1["ClaimNbFromSev"] == 0
    assert row1["ClaimAmountSum"] == 0.0

    row3 = out.loc[out["IDpol"] == 3].iloc[0]
    assert row3["ClaimNbFromSev"] == 2
    assert row3["ClaimAmountSum"] == 300.0


def test_fit_paid_frequency_glm_uses_claimnbfromsev_as_target():
    # Construct data where ClaimNb and ClaimNbFromSev tell DIFFERENT
    # stories by group - a model fit on the wrong target would recover
    # the wrong group's higher rate.
    rng = np.random.default_rng(0)
    n = 1000
    group = rng.choice(["A", "B"], n)
    exposure = np.full(n, 1.0)
    # Group A: high ClaimNb, low ClaimNbFromSev. Group B: the reverse.
    claim_nb = rng.poisson(np.where(group == "A", 0.5, 0.05))
    claim_nb_from_sev = rng.poisson(np.where(group == "A", 0.05, 0.5))
    train = pd.DataFrame(
        {
            "ClaimNb": claim_nb,
            "ClaimNbFromSev": claim_nb_from_sev,
            "Exposure": exposure,
            "group": group,
        }
    )

    result = fit_paid_frequency_glm(train, formula="ClaimNbFromSev ~ C(group)")
    # Group B should show HIGHER paid-claim frequency than group A (the
    # opposite of what a ClaimNb-based model would find for this data).
    relativity_b_vs_a = np.exp(result.params["C(group)[T.B]"])
    assert relativity_b_vs_a > 1.0


def test_predict_expected_loss_multiplies_elementwise():
    # freq_pred here is an expected COUNT over each policy's own exposure
    # (already offset-adjusted, per predict_frequency's contract).
    freq_pred = pd.Series([0.1, 0.2, 0.3])
    severity_pred = pd.Series([1000.0, 2000.0, 3000.0])
    out = predict_expected_loss(freq_pred, severity_pred)
    assert out.tolist() == pytest.approx([100.0, 400.0, 900.0])


def test_predict_expected_loss_raises_on_mismatched_index():
    freq_pred = pd.Series([0.1, 0.2], index=[0, 1])
    severity_pred = pd.Series([1000.0, 2000.0], index=[0, 2])
    with pytest.raises(ValueError):
        predict_expected_loss(freq_pred, severity_pred)


def test_predict_annual_pure_premium_undoes_the_exposure_offset():
    # A policy observed for only half a year (exposure=0.5) with an
    # expected count of 0.1 over that half-year implies an ANNUALIZED
    # rate of 0.2/year - predict_annual_pure_premium must divide by
    # exposure to recover that, not just multiply the raw count by
    # severity (which would understate the annual price for a
    # short-exposure policy).
    freq_pred = pd.Series([0.1])
    severity_pred = pd.Series([1000.0])
    exposure = pd.Series([0.5])

    annual = predict_annual_pure_premium(freq_pred, severity_pred, exposure)
    expected_loss = predict_expected_loss(freq_pred, severity_pred)

    assert annual.iloc[0] == pytest.approx(200.0)  # (0.1/0.5) * 1000
    assert annual.iloc[0] > expected_loss.iloc[0]  # annualized rate > raw half-year expected loss


def test_predict_annual_pure_premium_matches_expected_loss_at_full_exposure():
    # A full-year policy (exposure=1.0): the annualized rate and the
    # expected loss over its observed period are numerically identical -
    # this is the case where the two quantities are easy to conflate.
    freq_pred = pd.Series([0.1])
    severity_pred = pd.Series([1000.0])
    exposure = pd.Series([1.0])

    annual = predict_annual_pure_premium(freq_pred, severity_pred, exposure)
    expected_loss = predict_expected_loss(freq_pred, severity_pred)
    assert annual.iloc[0] == pytest.approx(expected_loss.iloc[0])
