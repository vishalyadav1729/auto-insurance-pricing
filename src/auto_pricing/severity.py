"""Claim-severity modelling (Phase 6): baseline and Gamma GLM.

Severity is a different kind of question from frequency: not "how often
does a policy claim" but "given a claim happened, how much did it cost."
That conditioning matters mechanically, not just conceptually - a
policy with zero claims has no severity observation at all (not a
severity of zero), so this module works from the claim-level severity
table, not the policy-level frequency one.

Phase 3 found essentially no rating factor with a convincing univariate
effect on severity (every apparent pattern traced back to a handful of
large claims distorting a small sample, not a real relationship). The
same full feature set used for frequency is tried here anyway, for the
same reason a "kitchen sink" first pass was used there: let the evidence
(deviance, significance under proper standard errors) decide what
matters, rather than guessing a smaller formula in advance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

SEVERITY_FORMULA = (
    "ClaimAmount ~ C(DrivAgeBand) + C(VehAgeBand) + C(BonusMalusBand) "
    "+ AreaOrdinal + LogDensity + VehGasBinary "
    "+ C(RegionGrouped) + C(VehBrandGrouped) + VehPower"
)

# Columns carried over from the frequency model table onto each claim.
# Deliberately excludes ClaimNb/Exposure - those are frequency-specific
# (an exposure offset has no meaning for severity: a claim's cost doesn't
# depend on how long the policy was observed before it happened).
_POLICY_FEATURE_COLUMNS = [
    "IDpol",
    "split",
    "Area",
    "VehPower",
    "VehAge",
    "DrivAge",
    "BonusMalus",
    "VehBrand",
    "VehGas",
    "Density",
    "Region",
    "DrivAgeBand",
    "VehAgeBand",
    "BonusMalusBand",
    "LogDensity",
    "AreaOrdinal",
    "VehGasBinary",
    "RegionGrouped",
    "VehBrandGrouped",
]


def build_severity_table(sev: pd.DataFrame, model_table: pd.DataFrame) -> pd.DataFrame:
    """One row per CLAIM, joined to that claim's policy features and split.

    Uses the `split` column already assigned at the policy level (Phase 4
    step 1), so every claim inherits its own policy's split rather than
    being split independently - the claim-level version of the same
    leakage-prevention rule: a policy with multiple claims must not have
    some in train and others in test.
    """
    merged = sev.merge(model_table[_POLICY_FEATURE_COLUMNS], on="IDpol", how="left")
    if merged["split"].isna().any():
        raise ValueError("some claims have no matching policy in model_table")
    return merged


def baseline_severity(train_df: pd.DataFrame) -> float:
    """The simplest possible severity model: the training claims' own
    mean cost, applied to every claim regardless of its characteristics.
    Exists as a floor any real model should beat.
    """
    return train_df["ClaimAmount"].mean()


def fit_gamma_glm(train_df: pd.DataFrame, formula: str = SEVERITY_FORMULA):
    """Fit a Gamma GLM with a log link.

    No offset here - unlike frequency, a claim's cost has no mechanical
    relationship to the policy's exposure period, so there is nothing to
    force in the way ClaimNb's offset forces "twice the exposure, twice
    the expected count."

    Only ever call this with the TRAINING split.
    """
    model = smf.glm(formula=formula, data=train_df, family=sm.families.Gamma(link=sm.families.links.Log()))
    return model.fit()


def predict_severity(results, df: pd.DataFrame) -> pd.Series:
    """Predict expected claim cost for each row of `df`.

    A thin wrapper that exists for the same reason frequency.py's
    predict_frequency does: to have one tested, reusable call site rather
    than relying on every caller remembering the right invocation - see
    that module's docstring for the offset-dropping bug this pattern
    guarded against in Phase 5. Severity has no offset to forget, but
    routing all prediction through one function keeps that discipline
    consistent and makes it trivial to add a correction later if one
    turns out to be needed.
    """
    return results.predict(df)


def fit_lognormal_model(train_df: pd.DataFrame, formula: str = SEVERITY_FORMULA):
    """Fit OLS on log(ClaimAmount), as a sensitivity check against the
    Gamma GLM.

    Returns (results, smearing_factor). The smearing factor is REQUIRED
    to use these predictions correctly - see predict_lognormal_severity's
    docstring for why. Splitting fit and predict into two functions (like
    fit_gamma_glm/predict_severity) would make it too easy to call
    results.predict() directly and get the biased answer; bundling the
    correction factor into what this function returns keeps that mistake
    from being the path of least resistance.
    """
    log_formula = formula.replace("ClaimAmount", "np.log(ClaimAmount)", 1)
    result = smf.ols(formula=log_formula, data=train_df).fit()
    smearing_factor = np.exp(result.resid).mean()  # Duan's smearing estimator
    return result, smearing_factor


def predict_lognormal_severity(results, smearing_factor: float, df: pd.DataFrame) -> pd.Series:
    """Predict expected claim cost from a fitted lognormal model, WITH
    Duan's smearing correction applied.

    Never call `results.predict(df)` directly for this model and
    exponentiate it - that gives the predicted MEDIAN of claim cost, not
    its mean, because exp(E[log Y]) != E[Y] for a right-skewed variable
    (the mean of a skewed distribution always sits above its median).
    This is explicitly a mistake the plan warns against ("taking logs and
    fitting OLS without addressing retransformation"), and it is not a
    small effect here: confirmed while building this - on this project's
    real training data, the uncorrected version predicts only 40% of the
    true total claim cost; the smearing-corrected version predicts 99%.
    Duan's smearing factor (the mean of exp(residual) on the training
    data, computed once by fit_lognormal_model) is a standard,
    distribution-free correction - it does not assume the log-residuals
    are exactly normal, unlike the simpler analytic exp(sigma^2/2)
    correction that does.
    """
    return np.exp(results.predict(df)) * smearing_factor


def large_loss_threshold(train_df: pd.DataFrame, top_pct: float = 0.01) -> float:
    """The claim amount at the top_pct-th percentile of TRAINING claims -
    the threshold used to identify "large losses" for the Phase 6
    sensitivity analysis. Computed from the training split only, the same
    fit-on-train discipline used everywhere else in this project.
    """
    return train_df["ClaimAmount"].quantile(1 - top_pct)


def exclude_large_losses(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Drop claims at or above `threshold`.

    Used ONLY for the sensitivity analysis in
    notebooks/04_severity_glm.ipynb / reports/severity_large_loss_sensitivity.md
    - never for the primary severity model. cleaning_policy.md rule 4 is
    explicit that large claims are not capped or removed by default; this
    function exists to test how much the model's story WOULD change if
    they were, not to actually make that change.
    """
    return df[df["ClaimAmount"] < threshold].reset_index(drop=True)
