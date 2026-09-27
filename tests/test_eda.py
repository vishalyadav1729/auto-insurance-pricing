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
    exposure_weighted_frequency,
    naive_mean_frequency,
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
