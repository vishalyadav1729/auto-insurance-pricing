"""Claim-frequency modelling (Phase 5): baseline and Poisson GLM.

The formula below uses every engineered feature from Phase 4
(reports/feature_dictionary.md): banded DrivAge/VehAge/BonusMalus (Phase 3
found both age effects non-linear), Area as an ordinal (its frequency
relationship was smooth and monotonic), log-Density, VehGas as a binary,
grouped Region/VehBrand (rare categories already pooled into "Other"), and
VehPower as a plain linear term (Phase 3 found no clear pattern there, so no
special treatment is applied).
"""

from __future__ import annotations

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
