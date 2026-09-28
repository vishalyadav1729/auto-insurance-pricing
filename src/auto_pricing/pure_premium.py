"""Pure premium (Phase 7): combining frequency and severity into expected
annual claim cost, resolving the frequency/severity mismatch documented
since Phase 3.

Background: Phase 3 found that naively multiplying Phase 5's frequency
champion (fit on ClaimNb, i.e. REPORTED claims) by Phase 6's severity
champion (fit on paid claim amounts) overstates true pure premium by
36.3% - proven algebraically to equal sum(ClaimNb)/sum(ClaimNbFromSev),
because ClaimNb counts some claims that were reported but never resulted
in a payment, while severity (and the money that actually needs pricing
for) only concerns claims that were paid.

Resolution, checked against a real alternative before being adopted: a
flat portfolio-average correction factor was considered and rejected,
because the share of reported claims that get paid varies substantially
by rating factor (0.65 to 1.00 across BonusMalus bands; 0.89 down to 0.61
across DrivAge bands, on training data) - a single flat multiplier would
trade one biased number for a differently-shaped one. Instead, a SECOND
frequency model is fit here, using the exact same formula and machinery
as Phase 5's champion, but targeting ClaimNbFromSev (paid claims) instead
of ClaimNb (reported claims). This resolves the mismatch exactly, not
approximately: confirmed on real training data, predicted pure premium
(paid-frequency x severity) matches actual observed pure premium at a
ratio of 1.0000, versus the 1.363 (36.3% overstatement) found using the
reported-claims frequency.

Phase 5's champion (fit on ClaimNb) remains correct and valuable for its
own stated purpose - predicting REPORTED claim frequency (e.g. for
claims-department workload or reserving) - this module does not replace
it, it adds a second, purpose-specific model for pricing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from auto_pricing.data import aggregate_severity_by_policy
from auto_pricing.frequency import FREQUENCY_FORMULA, fit_poisson_glm

PAID_FREQUENCY_FORMULA = FREQUENCY_FORMULA.replace("ClaimNb ~", "ClaimNbFromSev ~", 1)

# A direct Tweedie model predicts total claim cost per policy in one step,
# without splitting frequency and severity at all - the plan's alternative
# approach to compare against the two-model pipeline above. Same feature
# formula, targeting ClaimAmountSum with the same log(Exposure) offset.
TWEEDIE_FORMULA = FREQUENCY_FORMULA.replace("ClaimNb ~", "ClaimAmountSum ~", 1)

# Chosen via a fixed-evaluation-power grid search on validation data (see
# notebooks/05_pure_premium_evaluation.ipynb, step 2): fitting powers
# 1.1/1.3/1.5/1.7 landed within ~1.6% of each other when scored on one
# shared reference power, with 1.3 marginally best (67.65) and 1.5 only
# 0.36% behind (67.89) - not a meaningful difference to chase further.
# 1.5 is the standard actuarial default for this kind of loss data and is
# used for both fitting and evaluation here for consistency.
TWEEDIE_POWER = 1.5


def build_pure_premium_table(
    freq: pd.DataFrame, sev: pd.DataFrame, model_table: pd.DataFrame
) -> pd.DataFrame:
    """One row per policy: every engineered feature (from model_table),
    plus ClaimNbFromSev and ClaimAmountSum (from the severity table),
    filled with 0 for policies with no paid claims.

    `freq` is accepted for interface symmetry with other builders in this
    project (auto_pricing.data.build_policy_claim_table) but is not
    itself used - every column it would provide is already present in
    `model_table`.
    """
    sev_by_policy = aggregate_severity_by_policy(sev)
    merged = model_table.merge(sev_by_policy, on="IDpol", how="left")
    merged["ClaimNbFromSev"] = merged["ClaimNbFromSev"].fillna(0).astype(int)
    merged["ClaimAmountSum"] = merged["ClaimAmountSum"].fillna(0.0)
    return merged


def fit_paid_frequency_glm(train_df: pd.DataFrame, formula: str = PAID_FREQUENCY_FORMULA):
    """Fit a Poisson GLM predicting PAID claim frequency (ClaimNbFromSev),
    not reported claim frequency (ClaimNb) - see this module's docstring
    for why that distinction matters specifically for pure premium.

    Only ever call this with the TRAINING split.
    """
    return fit_poisson_glm(train_df, formula=formula)


def _check_aligned_indices(paid_freq_pred: pd.Series, severity_pred: pd.Series) -> None:
    if not paid_freq_pred.index.equals(severity_pred.index):
        raise ValueError("paid_freq_pred and severity_pred must share the same index")


def predict_expected_loss(paid_freq_pred: pd.Series, severity_pred: pd.Series) -> pd.Series:
    """Expected claim cost over each policy's OWN observed exposure period
    (the plan's ExpectedLoss_i = e_i * lambda_i * mu_i).

    `paid_freq_pred` (from auto_pricing.frequency.predict_frequency) is
    already an expected COUNT over that policy's own exposure - it has
    the exposure offset baked in - so multiplying it directly by expected
    severity gives dollars for however long that policy was actually
    observed, NOT a per-year rate. See predict_annual_pure_premium for
    the annualized version; the plan explicitly warns against confusing
    the two (a 3-month policy's expected loss is not its annual price).
    """
    _check_aligned_indices(paid_freq_pred, severity_pred)
    return paid_freq_pred * severity_pred


def predict_annual_pure_premium(
    paid_freq_pred: pd.Series, severity_pred: pd.Series, exposure: pd.Series
) -> pd.Series:
    """Expected annual claim cost (the plan's PurePremium_i = lambda_i * mu_i)
    - a rate in currency per policy-year, comparable across policies
    regardless of how long each was actually observed.

    Divides paid_freq_pred by exposure first to undo the offset and
    recover the annualized frequency rate, then multiplies by severity.
    This is the number to use for pricing comparisons between policies;
    predict_expected_loss is the number to use for "how much did/will
    this specific policy's coverage period cost."
    """
    _check_aligned_indices(paid_freq_pred, severity_pred)
    if not paid_freq_pred.index.equals(exposure.index):
        raise ValueError("exposure must share the same index as the predictions")
    annualized_frequency = paid_freq_pred / exposure
    return annualized_frequency * severity_pred


def fit_tweedie_glm(train_df: pd.DataFrame, formula: str = TWEEDIE_FORMULA, power: float = TWEEDIE_POWER):
    """Fit a direct Tweedie GLM predicting total claim cost (ClaimAmountSum)
    in one step, with log(Exposure) as an offset - no frequency/severity
    split at all.

    A Tweedie distribution with 1 < power < 2 is a compound Poisson-Gamma
    mixture: it can represent a policy with exact zero cost (no claim) and
    a policy with a continuous positive cost (a claim happened) within one
    single distribution, which is exactly the shape of ClaimAmountSum
    across the whole portfolio (~95% zeros, a right-skewed positive tail
    otherwise).

    Predict with auto_pricing.frequency.predict_frequency, not
    `results.predict(df)` directly - confirmed while building this that
    this model class has the same offset-dropping trap already found in
    three other statsmodels result classes (Phase 5's Poisson GLM,
    Negative Binomial, and regularized/elastic-net results).

    Only ever call this with the TRAINING split.
    """
    model = smf.glm(
        formula=formula,
        data=train_df,
        family=sm.families.Tweedie(var_power=power, link=sm.families.links.Log()),
        offset=np.log(train_df["Exposure"]),
    )
    return model.fit()
