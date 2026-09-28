"""Claim-frequency modelling (Phase 5): baseline, Poisson GLM, and a
Negative Binomial challenger (fit only once the Poisson diagnostics showed
material overdispersion - see fit_negative_binomial_glm's docstring).

The formula below uses every engineered feature from Phase 4
(reports/feature_dictionary.md): banded DrivAge/VehAge/BonusMalus (Phase 3
found both age effects non-linear), Area as an ordinal (its frequency
relationship was smooth and monotonic), log-Density, VehGas as a binary,
grouped Region/VehBrand (rare categories already pooled into "Other"), and
VehPower as a plain linear term (Phase 3 found no clear pattern there, so no
special treatment is applied).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

FREQUENCY_FORMULA = (
    "ClaimNb ~ C(DrivAgeBand) + C(VehAgeBand) + C(BonusMalusBand) "
    "+ AreaOrdinal + LogDensity + VehGasBinary "
    "+ C(RegionGrouped) + C(VehBrandGrouped) + VehPower"
)


def baseline_frequency(train_df: pd.DataFrame) -> float:
    """The simplest possible frequency model: one portfolio-wide rate
    (sum(ClaimNb) / sum(Exposure)) applied to every policy regardless of
    its characteristics. Exists as a floor any real model should beat -
    see notebooks/03_frequency_glm.ipynb for the comparison.
    """
    return train_df["ClaimNb"].sum() / train_df["Exposure"].sum()


def fit_poisson_glm(train_df: pd.DataFrame, formula: str = FREQUENCY_FORMULA):
    """Fit a Poisson GLM with log(Exposure) as an offset.

    Only ever call this with the TRAINING split - fitting on validation or
    test data would defeat the entire point of holding them out. Returns
    the fitted statsmodels GLMResults object.
    """
    model = smf.glm(
        formula=formula,
        data=train_df,
        family=sm.families.Poisson(),
        offset=np.log(train_df["Exposure"]),
    )
    return model.fit()


def predict_frequency(results, df: pd.DataFrame) -> pd.Series:
    """Predict expected CLAIM COUNT (not an annual rate) for each row of `df`.

    Always use this instead of calling `results.predict(df)` directly.
    statsmodels' own `.predict()` silently drops the offset unless it is
    passed again explicitly - no error, no warning, just wrong numbers for
    every row where Exposure != 1. Confirmed while building this: calling
    `results.predict(train)` with no offset overstated total training
    claims by more than 2x (54,154 vs the true 25,227) purely because it
    implicitly treated every policy as if it had a full year of exposure.
    This wrapper makes that mistake structurally impossible to repeat.
    """
    return results.predict(df, offset=np.log(df["Exposure"]))


def fit_negative_binomial_glm(train_df: pd.DataFrame, formula: str = FREQUENCY_FORMULA):
    """Fit a Negative Binomial regression (NB2) with log(Exposure) as an
    offset, jointly estimating alpha (the extra dispersion parameter)
    alongside every coefficient via maximum likelihood.

    Two things are required for this to converge reliably on this data,
    found by trial while building this - statsmodels' own default start
    values and optimizer do NOT converge here (alpha diverges to infinity,
    producing NaN log-likelihood, with no hard error raised):

    1. Start the coefficients from the already-fitted Poisson GLM's values
       (a Poisson fit is the alpha=0 special case of this model, so it is
       a very good starting point) plus a plain, un-logged starting guess
       for alpha - the model's internal parameterization expects raw
       alpha here, not log(alpha).
    2. Use method="newton" rather than the default optimizer. Verified
       stable: starting alpha at 0.01, 0.1, or 1.0 all converge to the
       identical alpha (0.8115), which is good evidence this is a genuine
       optimum, not an artifact of a lucky starting guess.

    Only ever call this with the TRAINING split.
    """
    poisson_result = fit_poisson_glm(train_df, formula=formula)
    model = smf.negativebinomial(formula=formula, data=train_df, offset=np.log(train_df["Exposure"]))
    start_params = np.append(poisson_result.params.to_numpy(), 1.0)
    return model.fit(start_params=start_params, method="newton", maxiter=100, disp=0)


def dispersion_ratio(results) -> float:
    """Pearson chi-square statistic divided by residual degrees of freedom.

    A Poisson distribution assumes Var(N) == E(N); this ratio should be
    close to 1 if that assumption holds for the fitted model. Materially
    above 1 is evidence of overdispersion (more spread in the data than
    the model's Poisson assumption allows for) - evidence to weigh when
    considering a Negative Binomial alternative, not an automatic trigger.
    """
    return results.pearson_chi2 / results.df_resid


def deviance_dispersion_ratio(results) -> float:
    """Deviance divided by residual degrees of freedom - a second, related
    overdispersion diagnostic.

    This statistic's chi-squared approximation requires the fitted mean
    (mu) to not be too small for most observations; in a low-frequency
    insurance portfolio (this one: median fitted mu ~0.05, essentially all
    policies under 1), that approximation is known to be unreliable, and
    this ratio can disagree sharply with dispersion_ratio() as a result.
    Compute both, but prefer dispersion_ratio() (Pearson-based) as the
    primary diagnostic in this kind of low-mean-count setting - see
    notebooks/03_frequency_glm.ipynb for the real numbers behind this call.
    """
    return results.deviance / results.df_resid


def save_frequency_model(
    results, formula: str, model_type: str, path: str | Path, power: float | None = None
) -> dict:
    """Persist a small, self-contained snapshot of a fitted frequency
    (or, via auto_pricing.pure_premium, Tweedie pure-premium) model.

    `power` is required when model_type == "tweedie" (the Tweedie
    distribution's variance-power parameter, needed to reconstruct
    predictions correctly - it is not stored anywhere else, since it is
    an argument to the model family, not a fitted parameter).

    Deliberately does NOT pickle the statsmodels results object directly.
    Confirmed while building this: doing so produced an 878MB file for the
    Poisson fit and 650MB for the Negative Binomial fit on this project's
    ~475k-row training split - for a model with only 48 parameters -
    because those objects retain internal arrays sized by the training row
    count. `results.remove_data()` was tried as a fix and rejected: it
    only reduced the file to 278MB (something was still retained), and it
    silently broke `.aic`/`.llf` on reload if they hadn't already been
    computed and cached before removal.

    Saves only what's actually needed: the formula, the model family, the
    fitted coefficients, and a handful of scalar diagnostics computed now
    (while the full results object is available) rather than left to be
    recomputed lazily from data that won't exist after loading.
    """
    artifact = {
        "formula": formula,
        "model_type": model_type,
        "params": results.params,
        "bse": results.bse,
        "pvalues": results.pvalues,
        "aic": float(results.aic),
        "llf": float(results.llf),
    }
    if model_type == "poisson":
        artifact["deviance"] = float(results.deviance)
        artifact["pearson_chi2"] = float(results.pearson_chi2)
        artifact["df_resid"] = float(results.df_resid)
        artifact["converged"] = bool(results.converged)
    elif model_type == "negative_binomial":
        artifact["converged"] = bool(results.mle_retvals.get("converged"))
    elif model_type == "tweedie":
        if power is None:
            raise ValueError("power is required when model_type == 'tweedie'")
        artifact["deviance"] = float(results.deviance)
        artifact["df_resid"] = float(results.df_resid)
        artifact["converged"] = bool(results.converged)
        artifact["power"] = float(power)
    else:
        raise ValueError(f"unknown model_type: {model_type!r}")

    joblib.dump(artifact, path)
    return artifact


def predict_from_artifact(artifact: dict, df: pd.DataFrame) -> pd.Series:
    """Predict expected claim count for `df` from a saved artifact
    (save_frequency_model's output) - no fitted results object needed.

    Rebuilds a fresh, UNFIT statsmodels model bound to `df`'s own rows
    (using the saved formula), purely for its design-matrix machinery,
    then applies the saved coefficients directly - `.predict(params)`
    works identically on an unfit model instance.

    This works correctly even for a single new row, PROVIDED every
    categorical column the formula references is a proper pandas
    Categorical with its full training-time category list already
    attached (see auto_pricing.features.apply_common_categories and
    bin_numeric). Confirmed while building this: a plain string/object
    categorical column instead produces a design matrix with the WRONG
    number of columns for any slice that doesn't happen to contain every
    category seen during training - including, critically, a single row,
    which by definition can only ever show one category per column. This
    is not a hypothetical edge case: it is exactly the situation a
    real-time pricing app (Phase 10) needs to handle correctly.
    """
    offset = np.log(df["Exposure"])
    if artifact["model_type"] == "poisson":
        model = smf.glm(formula=artifact["formula"], data=df, family=sm.families.Poisson(), offset=offset)
    elif artifact["model_type"] == "negative_binomial":
        model = smf.negativebinomial(formula=artifact["formula"], data=df, offset=offset)
    elif artifact["model_type"] == "tweedie":
        family = sm.families.Tweedie(var_power=artifact["power"], link=sm.families.links.Log())
        model = smf.glm(formula=artifact["formula"], data=df, family=family, offset=offset)
    else:
        raise ValueError(f"unknown model_type: {artifact['model_type']!r}")
    return model.predict(artifact["params"])
