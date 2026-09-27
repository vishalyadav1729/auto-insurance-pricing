"""Reusable summary statistics for exploratory analysis (Phase 3).

The central function here, `exposure_weighted_frequency`, exists because
the plan is explicit that the correct annual claim frequency for a group
of policies is

    sum(ClaimNb) / sum(Exposure)

and NOT the unweighted mean of each policy's own ClaimNb / Exposure. The
difference is not cosmetic: a policy with very little exposure and even
one claim produces an enormous implied annual rate (e.g. 1 claim over 3
days implies ~366 claims/year), and averaging those per-policy rates lets
a handful of short-exposure policies dominate the result. Weighting by
exposure instead treats a claim on a short-exposure policy for exactly
what it is - a small amount of evidence - rather than an extreme rate.

`naive_mean_frequency` is included only so the EDA notebook can show this
error happening on the real data, side by side with the correct figure -
it should never be used to report an actual result.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def exposure_weighted_frequency(
    df: pd.DataFrame, by: str | list[str] | None = None
) -> pd.DataFrame | float:
    """Correct annual claim frequency: sum(ClaimNb) / sum(Exposure).

    With `by=None`, returns a single portfolio-level float. With `by` set
    to a column name (or list of column names), returns one row per group
    with columns [*by, n_policies, claim_nb, exposure, frequency] - the
    shape needed for segment-level tariff analysis in Phase 3 step 4.
    """
    if by is None:
        return df["ClaimNb"].sum() / df["Exposure"].sum()

    grouped = df.groupby(by).agg(
        n_policies=("IDpol", "size"),
        claim_nb=("ClaimNb", "sum"),
        exposure=("Exposure", "sum"),
    )
    grouped["frequency"] = grouped["claim_nb"] / grouped["exposure"]
    return grouped.reset_index()


def naive_mean_frequency(df: pd.DataFrame) -> float:
    """WRONG on purpose: unweighted mean of each policy's ClaimNb / Exposure.

    Exists only to demonstrate, with real numbers, why this is a mistake
    (see notebooks/02_data_cleaning_eda.ipynb) - never use this to report
    an actual frequency figure.
    """
    return (df["ClaimNb"] / df["Exposure"]).mean()


def zero_claim_share(df: pd.DataFrame) -> float:
    """Fraction of policies with ClaimNb == 0."""
    return (df["ClaimNb"] == 0).mean()


def bin_numeric(
    series: pd.Series, bins: list[float], labels: list[str] | None = None
) -> pd.Series:
    """Thin, named wrapper around pd.cut (left-inclusive, right-exclusive bins).

    Exists so that binning choices for rating-factor EDA (e.g. DrivAge,
    BonusMalus bands) are made in one place and are unit-testable, rather
    than as inline pd.cut calls scattered across a notebook. These bins are
    for exploratory tabulation only - Phase 4 may choose different bins, or
    splines, once the EDA here has shown which factors are roughly linear
    and which are not.
    """
    return pd.cut(series, bins=bins, labels=labels, right=False, include_lowest=True)


def plot_segment_frequency(
    table: pd.DataFrame,
    category_col: str,
    freq_col: str = "frequency",
    exposure_col: str = "exposure",
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Bar chart of exposure volume with an overlaid line of claim frequency.

    `table` is expected to be the output of exposure_weighted_frequency(df,
    by=category_col) (or anything with the same column names). Exposure is
    drawn as bars on the left axis, frequency as a line on the right axis -
    deliberately shown together, not frequency alone, so a rate estimated
    from very little exposure is visibly flagged as less credible rather
    than looking just as solid as a rate backed by tens of thousands of
    policy-years.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    categories = table[category_col].astype(str)
    ax.bar(categories, table[exposure_col], color="#B0B0B0", label="Exposure (policy-years)")
    ax.set_ylabel("Exposure (policy-years)")
    ax.tick_params(axis="x", rotation=45)

    ax2 = ax.twinx()
    ax2.plot(categories, table[freq_col], color="#C44E52", marker="o", label="Frequency")
    ax2.set_ylabel("Claim frequency (claims / policy-year)")

    if title:
        ax.set_title(title)
    ax.figure.tight_layout()
    return ax
