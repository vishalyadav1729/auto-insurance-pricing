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
import numpy as np
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


def exposure_weighted_pure_premium(
    df: pd.DataFrame, amount_col: str = "ClaimAmountSum", by: str | list[str] | None = None
) -> pd.DataFrame | float:
    """Correct annual pure premium: sum(amount_col) / sum(Exposure).

    Mirrors exposure_weighted_frequency exactly, but on claim cost instead
    of claim count - the same reason applies: a policy's own cost/exposure
    ratio is not a meaningful rate on its own when exposure is small, so
    costs and exposures are summed separately, across the group, before
    dividing. Expects `df` to be the output of
    auto_pricing.data.build_policy_claim_table (i.e. one row per policy,
    with amount_col already filled with 0 for policies without a claim).
    """
    if by is None:
        return df[amount_col].sum() / df["Exposure"].sum()

    grouped = df.groupby(by).agg(
        n_policies=("IDpol", "size"),
        claim_amount=(amount_col, "sum"),
        exposure=("Exposure", "sum"),
    )
    grouped["pure_premium"] = grouped["claim_amount"] / grouped["exposure"]
    return grouped.reset_index()


def naive_mean_pure_premium(df: pd.DataFrame, amount_col: str = "ClaimAmountSum") -> float:
    """WRONG on purpose: unweighted mean of each policy's amount_col / Exposure.

    Exists only to demonstrate the same distortion naive_mean_frequency
    shows, at the pure-premium level - never use this to report an actual
    pure premium figure.
    """
    return (df[amount_col] / df["Exposure"]).mean()


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
    line_ylabel: str = "Claim frequency (claims / policy-year)",
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Bar chart of exposure volume with an overlaid line of a rate metric.

    `table` is expected to be the output of exposure_weighted_frequency(df,
    by=category_col) or exposure_weighted_pure_premium(df, by=category_col)
    (or anything with the same column shape) - `freq_col`/`line_ylabel` let
    this same chart be reused for pure premium, not just frequency. Exposure
    is drawn as bars on the left axis, the rate as a line on the right axis
    - deliberately shown together, not the rate alone, so a value estimated
    from very little exposure is visibly flagged as less credible rather
    than looking just as solid as one backed by tens of thousands of
    policy-years.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    categories = table[category_col].astype(str)
    ax.bar(categories, table[exposure_col], color="#B0B0B0", label="Exposure (policy-years)")
    ax.set_ylabel("Exposure (policy-years)")
    ax.tick_params(axis="x", rotation=45)

    ax2 = ax.twinx()
    ax2.plot(categories, table[freq_col], color="#C44E52", marker="o", label=line_ylabel)
    ax2.set_ylabel(line_ylabel)

    if title:
        ax.set_title(title)
    ax.figure.tight_layout()
    return ax


def concentration_table(
    amounts: pd.Series, thresholds: list[float] = (0.01, 0.05, 0.10, 0.25, 0.50)
) -> pd.DataFrame:
    """Share of total claim value held by the largest X% of claims.

    For a right-skewed cost distribution like ClaimAmount, this answers a
    different question than describe() does: not "what does a typical
    claim cost" but "how much of the total bill is driven by a small
    number of large claims." Returns one row per threshold with columns
    [top_pct, n_claims, share_of_total_value].
    """
    sorted_amounts = amounts.sort_values(ascending=False).reset_index(drop=True)
    n = len(sorted_amounts)
    total = sorted_amounts.sum()

    rows = []
    for pct in thresholds:
        k = max(1, int(np.ceil(n * pct)))
        share = sorted_amounts.iloc[:k].sum() / total
        rows.append({"top_pct": pct, "n_claims": k, "share_of_total_value": share})
    return pd.DataFrame(rows)


def concentration_curve(amounts: pd.Series, points: int = 100) -> pd.DataFrame:
    """Cumulative share of total claim value vs. cumulative share of claims,
    sorted from the largest claim down.

    Downsampled to `points` rows (default 100) so the resulting plot is
    light regardless of how many claims are in `amounts`. Returns columns
    [pct_claims, pct_value], both running from just above 0 to 1.0.
    """
    sorted_amounts = amounts.sort_values(ascending=False).reset_index(drop=True)
    n = len(sorted_amounts)
    total = sorted_amounts.sum()

    pct_claims = np.arange(1, n + 1) / n
    pct_value = sorted_amounts.cumsum() / total

    idx = np.linspace(0, n - 1, min(points, n)).astype(int)
    return pd.DataFrame({"pct_claims": pct_claims[idx], "pct_value": pct_value[idx]})


def plot_mean_vs_median(
    table: pd.DataFrame,
    category_col: str,
    mean_col: str = "mean",
    median_col: str = "median",
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Grouped bar chart comparing mean vs. median within each category.

    For a heavy-tailed quantity like claim severity, a large gap between
    the mean and median in a given segment is itself informative: it means
    the segment's average is being pulled around by a small number of
    large claims rather than reflecting a typical claim in that segment.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    categories = table[category_col].astype(str)
    x = np.arange(len(categories))
    width = 0.35

    ax.bar(x - width / 2, table[mean_col], width, color="#C44E52", label="Mean")
    ax.bar(x + width / 2, table[median_col], width, color="#4C72B0", label="Median")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45)
    ax.set_ylabel("Claim amount")
    ax.legend()

    if title:
        ax.set_title(title)
    ax.figure.tight_layout()
    return ax
