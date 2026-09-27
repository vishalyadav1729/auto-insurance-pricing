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
