"""Model evaluation metrics for count/rate models.

Built for Phase 5 (comparing frequency model candidates on the validation
split), written generically enough to be reused for Phase 7's pure premium
evaluation later - the same "compare exposure-weighted deviance on
held-out data" logic applies to both.

These metrics only ever take already-computed observed/predicted/exposure
values - never a fitted model object - so they work identically regardless
of which model produced the predictions (baseline, Poisson, regularized
Poisson, or Negative Binomial all produce a plain predicted claim count
per policy, and that's all these functions need).
"""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import mean_poisson_deviance


def exposure_weighted_poisson_deviance(
    observed_count: pd.Series, predicted_count: pd.Series, exposure: pd.Series
) -> float:
    """Exposure-weighted mean Poisson deviance, on the annualized-rate scale.

    Converts both observed and predicted claim counts to annual rates
    (dividing by Exposure) before computing deviance, and weights each
    policy's contribution by its Exposure - the same convention scikit-learn's
    own official GLM insurance tutorial uses (this project's methodological
    reference point, per the plan). Lower is better; 0 is a perfect fit.
    Comparable across model types that AIC cannot be used for (a
    regularized/penalized fit has no proper likelihood-based AIC), which is
    exactly why this, not AIC, is the metric used to pick a champion here.
    """
    observed_rate = observed_count / exposure
    predicted_rate = predicted_count / exposure
    return mean_poisson_deviance(observed_rate, predicted_rate, sample_weight=exposure)


def observed_to_expected_ratio(observed_count: pd.Series, predicted_count: pd.Series) -> float:
    """sum(observed) / sum(predicted): the single-number "did we get the
    portfolio total about right" check. 1.0 is exact; above 1 means the
    model under-predicts the portfolio total, below 1 means it over-predicts.
    """
    return observed_count.sum() / predicted_count.sum()


def calibration_by_decile(
    observed_count: pd.Series, predicted_count: pd.Series, exposure: pd.Series, n_bins: int = 10
) -> pd.DataFrame:
    """Bin policies into deciles of predicted risk, compare each decile's
    actual exposure-weighted frequency against its predicted one.

    A well-calibrated model should track closely across every decile, not
    just balance out on average - a model can have a perfect portfolio-level
    observed-to-expected ratio while being badly wrong for specific risk
    bands, and this is what catches that.
    """
    predicted_rate = predicted_count / exposure
    decile = pd.qcut(predicted_rate, q=n_bins, labels=False, duplicates="drop")

    df = pd.DataFrame(
        {
            "decile": decile,
            "observed_count": observed_count.to_numpy(),
            "predicted_count": predicted_count.to_numpy(),
            "exposure": exposure.to_numpy(),
        }
    )
    grouped = df.groupby("decile").agg(
        n_policies=("exposure", "size"),
        exposure=("exposure", "sum"),
        observed_count=("observed_count", "sum"),
        predicted_count=("predicted_count", "sum"),
    )
    grouped["observed_rate"] = grouped["observed_count"] / grouped["exposure"]
    grouped["predicted_rate"] = grouped["predicted_count"] / grouped["exposure"]
    return grouped.reset_index()
