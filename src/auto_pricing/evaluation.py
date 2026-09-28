"""Model evaluation metrics for count/rate models (Phase 5 frequency),
severity models (Phase 6), and pure premium (Phase 7).

These metrics only ever take already-computed observed/predicted values
(plus exposure, where relevant) - never a fitted model object - so they
work identically regardless of which model produced the predictions. The
frequency and pure-premium functions weight by Exposure (a policy's
opportunity to claim, or to accumulate cost, varies); the severity
functions do not (each claim is already one full, independent observation
of cost).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_gamma_deviance, mean_poisson_deviance, mean_tweedie_deviance


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


def tweedie_deviance(observed: pd.Series, predicted: pd.Series, power: float) -> float:
    """Mean Tweedie deviance at a given power - the plan's primary metric
    for comparing pure-premium candidates.

    The `power` MUST be the same fixed value across every candidate being
    compared (baseline, frequency x severity, direct Tweedie at whatever
    power it was fit with) - the Tweedie deviance formula is a genuinely
    different function of (observed, predicted) at different power
    values, so "deviance at power=1.1" and "deviance at power=1.8" are not
    on a comparable scale, even for the exact same predictions. This
    module does not pick that shared power - see
    auto_pricing.pure_premium.TWEEDIE_POWER for the value used
    consistently across Phase 7's model comparison, and the reasoning
    behind it.
    """
    return mean_tweedie_deviance(observed, predicted, power=power)


def pure_premium_calibration_by_decile(
    observed_amount: pd.Series, predicted_amount: pd.Series, exposure: pd.Series, n_bins: int = 10
) -> pd.DataFrame:
    """Bin ALL policies (not just claimants) into deciles of predicted
    ANNUAL pure premium, compare each decile's exposure-weighted observed
    rate against its predicted rate.

    Unlike severity_calibration_by_decile, this covers the whole
    portfolio (a policy with no claim contributes observed_amount=0, not
    nothing) and weights by Exposure - the frequency-style convention,
    because pure premium is fundamentally a rate per unit of time-at-risk,
    the same reasoning as calibration_by_decile (frequency).
    """
    predicted_rate = predicted_amount / exposure
    decile = pd.qcut(predicted_rate, q=n_bins, labels=False, duplicates="drop")

    df = pd.DataFrame(
        {
            "decile": decile,
            "observed_amount": observed_amount.to_numpy(),
            "exposure": exposure.to_numpy(),
        }
    )
    grouped = df.groupby("decile").agg(
        n_policies=("exposure", "size"),
        exposure=("exposure", "sum"),
        observed_amount=("observed_amount", "sum"),
    )
    grouped["observed_rate"] = grouped["observed_amount"] / grouped["exposure"]

    # predicted_rate is already a per-exposure-unit rate; its exposure-weighted
    # mean within each decile is the fair predicted-rate summary for that bin.
    pred_df = pd.DataFrame({"decile": decile, "predicted_rate": predicted_rate, "exposure": exposure})
    pred_df["weighted"] = pred_df["predicted_rate"] * pred_df["exposure"]
    pred_grouped = pred_df.groupby("decile").agg(weighted=("weighted", "sum"), exposure=("exposure", "sum"))
    grouped["predicted_rate"] = pred_grouped["weighted"] / pred_grouped["exposure"]

    return grouped.reset_index()


def lorenz_curve(
    observed_amount: pd.Series, predicted_rate: pd.Series, exposure: pd.Series
) -> pd.DataFrame:
    """Cumulative share of observed claim cost against cumulative share of
    exposure, with policies ordered from LOWEST to HIGHEST predicted risk
    - the standard actuarial ranking diagnostic.

    Ranks by `predicted_rate` - the ANNUALIZED pure premium
    (auto_pricing.pure_premium.predict_annual_pure_premium's output), NOT
    raw expected loss over a policy's own exposure. This matters and was
    checked directly, not assumed: ranking a baseline model (a flat rate
    for every policy, so its "expected loss" is just rate x Exposure) by
    expected loss is mechanically the same as ranking by Exposure alone -
    and Exposure has ~0 real correlation with risk in this data (0.006).
    Because a handful of very-short-exposure policies have one large
    claim (the same "short policy + one big claim" pattern documented
    repeatedly since Phase 3), ranking by expected loss let those few
    policies dominate the low end of the curve, producing a strongly
    NEGATIVE Gini (-0.31) for a model that has zero genuine discriminating
    power - the opposite of the ~0 a non-discriminating model should show.
    Ranking the same baseline by its (constant, all-tied) annualized rate
    instead gives Gini ~ -0.02, correctly close to zero.

    A model with no discriminating power at all produces a curve close to
    the diagonal (cumulative loss share tracks cumulative exposure share);
    a model that ranks risk well pushes the curve below the diagonal
    (little cost from the lowest-ranked share, most of it concentrated in
    the highest-ranked share).
    """
    order = np.argsort(predicted_rate.to_numpy())
    ranked_exposure = exposure.to_numpy()[order]
    ranked_observed = observed_amount.to_numpy()[order]

    cum_exposure = np.cumsum(ranked_exposure)
    cum_exposure = cum_exposure / cum_exposure[-1]
    cum_observed = np.cumsum(ranked_observed)
    cum_observed = cum_observed / cum_observed[-1]

    return pd.DataFrame({"cum_exposure_share": cum_exposure, "cum_observed_share": cum_observed})


def gini_index(observed_amount: pd.Series, predicted_rate: pd.Series, exposure: pd.Series) -> float:
    """Normalized Gini index from the Lorenz curve above: 1 - 2 x (area
    under the curve). 0 means predictions rank risk no better than random
    order; close to 1 means near-perfect risk ranking (the highest-ranked
    share of exposure accounts for almost all the observed cost).

    `predicted_rate` must be an ANNUALIZED rate, not raw expected loss -
    see lorenz_curve's docstring for why that distinction produced a
    materially wrong (sign-flipped) Gini for a non-discriminating baseline
    model when this was checked on real data.

    IMPORTANT, stated explicitly because the plan calls this out as a
    common mistake: this measures RANKING quality, not CALIBRATION. A
    model whose predictions are scaled by any positive constant (e.g.
    accidentally doubled, or divided by 2) produces the EXACT SAME Gini
    index, because scaling every prediction by the same factor never
    changes their relative order - while its calibration (deviance,
    observed-to-expected ratio) would be badly wrong. Never use Gini as
    the only metric for selecting or validating a pricing model; see
    notebooks/05_pure_premium_evaluation.ipynb for a direct demonstration.
    """
    curve = lorenz_curve(observed_amount, predicted_rate, exposure)
    area_under_curve = np.trapezoid(curve["cum_observed_share"], curve["cum_exposure_share"])
    return 1 - 2 * area_under_curve
