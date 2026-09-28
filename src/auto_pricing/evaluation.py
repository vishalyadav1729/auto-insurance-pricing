"""Model evaluation metrics for count/rate models (Phase 5 frequency) and
severity models (Phase 6), written generically enough to be reused for
Phase 7's pure premium evaluation too.

These metrics only ever take already-computed observed/predicted values
(plus exposure, for the frequency-specific ones) - never a fitted model
object - so they work identically regardless of which model produced the
predictions. The frequency functions weight by Exposure (a policy's
opportunity to claim varies); the severity functions do not (each claim
is already one full, independent observation of cost).
"""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import mean_gamma_deviance, mean_poisson_deviance


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


def gamma_deviance(observed: pd.Series, predicted: pd.Series) -> float:
    """Mean Gamma deviance: the appropriate goodness-of-fit metric for a
    continuous, positive, right-skewed outcome like claim severity - NOT
    plain RMSE, which the plan explicitly warns against relying on here.
    An RMSE-based comparison would be dominated almost entirely by the
    largest few claims (confirmed in Phase 6 step 2: the top 1% of claims
    hold 41% of total value) and would say little about how well a model
    fits the typical claim. Thin, named wrapper around scikit-learn's
    implementation, for a consistent call site alongside
    exposure_weighted_poisson_deviance.
    """
    return mean_gamma_deviance(observed, predicted)


def severity_calibration_by_decile(
    observed: pd.Series, predicted: pd.Series, n_bins: int = 10
) -> pd.DataFrame:
    """Bin claims into deciles of PREDICTED severity, compare each
    decile's mean observed cost against its mean predicted cost.

    Unlike calibration_by_decile (frequency), there is no exposure to
    weight by here: each claim is already one full, independent
    observation of "cost given a claim occurred," not a partial-year
    policy record - so deciles are formed directly from predicted
    severity and aggregated by a simple mean, not an exposure-weighted one.
    """
    decile = pd.qcut(predicted, q=n_bins, labels=False, duplicates="drop")
    df = pd.DataFrame(
        {"decile": decile, "observed": observed.to_numpy(), "predicted": predicted.to_numpy()}
    )
    grouped = df.groupby("decile").agg(
        n_claims=("observed", "size"),
        observed_mean=("observed", "mean"),
        predicted_mean=("predicted", "mean"),
    )
    return grouped.reset_index()
