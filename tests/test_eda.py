"""Unit tests for auto_pricing.eda.

The key thing to pin down: exposure_weighted_frequency and
naive_mean_frequency should genuinely disagree when a short-exposure
policy has a claim - that disagreement is the whole point of having both
functions.
"""

import matplotlib

matplotlib.use("Agg")  # headless backend so tests never try to open a window

import pandas as pd

from auto_pricing.eda import (
    bin_numeric,
    concentration_curve,
    concentration_table,
    exposure_weighted_frequency,
    exposure_weighted_pure_premium,
    naive_mean_frequency,
    naive_mean_pure_premium,
    plot_mean_vs_median,
    plot_segment_frequency,
    zero_claim_share,
)


def test_exposure_weighted_frequency_portfolio_level():
    df = pd.DataFrame({"ClaimNb": [1, 0, 1], "Exposure": [1.0, 1.0, 0.5]})
    # sum(ClaimNb)/sum(Exposure) = 2 / 2.5 = 0.8
    assert exposure_weighted_frequency(df) == 0.8


def test_exposure_weighted_frequency_by_group():
    df = pd.DataFrame(
        {
            "IDpol": [1, 2, 3, 4],
            "ClaimNb": [1, 0, 1, 0],
            "Exposure": [1.0, 1.0, 1.0, 1.0],
            "Area": ["A", "A", "B", "B"],
        }
    )
    out = exposure_weighted_frequency(df, by="Area").set_index("Area")
    assert out.loc["A", "frequency"] == 0.5
    assert out.loc["B", "frequency"] == 0.5
    assert out.loc["A", "n_policies"] == 2


def test_naive_and_weighted_disagree_on_short_exposure_claim():
    # One policy: tiny exposure, one claim -> naive mean is dominated by it.
    # Nine policies: full exposure, no claims.
    df = pd.DataFrame(
        {
            "ClaimNb": [1] + [0] * 9,
            "Exposure": [0.01] + [1.0] * 9,
        }
    )
    weighted = exposure_weighted_frequency(df)
    naive = naive_mean_frequency(df)
    # weighted: 1 claim over 9.01 exposure years ~ 0.111
    assert abs(weighted - (1 / 9.01)) < 1e-9
    # naive: mean of [100, 0, 0, ..., 0] = 10.0 -- wildly different
    assert naive == 10.0
    assert naive > weighted * 50  # demonstrates the scale of the distortion


def test_zero_claim_share():
    df = pd.DataFrame({"ClaimNb": [0, 0, 1, 2]})
    assert zero_claim_share(df) == 0.5


def test_bin_numeric_assigns_left_inclusive_bands():
    series = pd.Series([18, 22, 23, 29, 30])
    bands = bin_numeric(series, bins=[18, 23, 30, 101], labels=["18-22", "23-29", "30+"])
    assert bands.astype(str).tolist() == ["18-22", "18-22", "23-29", "23-29", "30+"]


def test_bin_numeric_includes_lowest_value():
    # include_lowest=True: the very first bin edge itself must not become NaN
    series = pd.Series([18])
    bands = bin_numeric(series, bins=[18, 23], labels=["18-22"])
    assert bands.astype(str).tolist() == ["18-22"]


def test_plot_segment_frequency_returns_axes_without_error():
    table = pd.DataFrame(
        {"Area": ["A", "B", "C"], "exposure": [100.0, 50.0, 5.0], "frequency": [0.05, 0.08, 0.20]}
    )
    ax = plot_segment_frequency(table, category_col="Area", title="test")
    assert ax is not None
    assert ax.get_title() == "test"


def test_plot_segment_frequency_supports_custom_line_ylabel_for_reuse():
    # Same chart shape is reused for pure premium, not just frequency.
    table = pd.DataFrame(
        {"Band": ["A", "B"], "exposure": [100.0, 50.0], "pure_premium": [150.0, 400.0]}
    )
    ax = plot_segment_frequency(
        table, category_col="Band", freq_col="pure_premium", line_ylabel="Pure premium (EUR)"
    )
    twin_ax = ax.figure.axes[-1]
    assert twin_ax.get_ylabel() == "Pure premium (EUR)"


def test_concentration_table_top_claim_dominates():
    # One claim of 100, four claims of 1 each -> total 104.
    # The single largest claim (20% of the 5 claims) should hold 100/104 of value.
    amounts = pd.Series([100, 1, 1, 1, 1])
    table = concentration_table(amounts, thresholds=[0.2])
    assert table.loc[0, "n_claims"] == 1
    assert abs(table.loc[0, "share_of_total_value"] - 100 / 104) < 1e-9


def test_concentration_table_all_equal_claims_is_proportional():
    # With identical claim amounts, the top X% of claims should hold ~X% of value.
    amounts = pd.Series([10.0] * 100)
    table = concentration_table(amounts, thresholds=[0.10, 0.50])
    assert abs(table.loc[0, "share_of_total_value"] - 0.10) < 1e-9
    assert abs(table.loc[1, "share_of_total_value"] - 0.50) < 1e-9


def test_concentration_curve_reaches_full_coverage():
    amounts = pd.Series([10, 20, 30, 40])
    curve = concentration_curve(amounts, points=4)
    assert curve["pct_claims"].iloc[-1] == 1.0
    assert curve["pct_value"].iloc[-1] == 1.0
    # curve should be non-decreasing (cumulative share can't fall)
    assert (curve["pct_value"].diff().dropna() >= 0).all()


def test_plot_mean_vs_median_returns_axes_without_error():
    table = pd.DataFrame({"Band": ["A", "B"], "mean": [5000.0, 1800.0], "median": [1200.0, 1150.0]})
    ax = plot_mean_vs_median(table, category_col="Band", title="test")
    assert ax is not None
    assert ax.get_title() == "test"


def test_exposure_weighted_pure_premium_portfolio_level():
    df = pd.DataFrame({"ClaimAmountSum": [1000.0, 0.0, 500.0], "Exposure": [1.0, 1.0, 0.5]})
    # sum(cost)/sum(exposure) = 1500 / 2.5 = 600
    assert exposure_weighted_pure_premium(df) == 600.0


def test_exposure_weighted_and_naive_pure_premium_disagree_on_short_exposure_claim():
    df = pd.DataFrame(
        {
            "ClaimAmountSum": [1000.0] + [0.0] * 9,
            "Exposure": [0.01] + [1.0] * 9,
        }
    )
    weighted = exposure_weighted_pure_premium(df)
    naive = naive_mean_pure_premium(df)
    assert abs(weighted - (1000 / 9.01)) < 1e-9
    assert naive == 10000.0  # 1000/0.01 averaged in with nine zeros
    assert naive > weighted * 50
