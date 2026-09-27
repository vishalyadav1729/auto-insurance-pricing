"""Cleaning transformations implementing reports/cleaning_policy.md.

Each function applies exactly one documented rule, takes a DataFrame in,
and returns a *new* DataFrame (never mutates its input) so the raw tables
loaded by auto_pricing.data stay untouched and every rule can be tested and
audited independently of the others. If a rule's threshold ever needs to
change, cleaning_policy.md should be updated first and this module second -
the module implements the policy, it does not set it.
"""

from __future__ import annotations

import pandas as pd

# Thresholds match reports/cleaning_policy.md exactly; see that document
# for the evidence behind each number.
EXPOSURE_CAP = 1.0
CLAIM_NB_CAP = 4


def clip_exposure(freq: pd.DataFrame, cap: float = EXPOSURE_CAP) -> pd.DataFrame:
    """Cap Exposure at `cap` (cleaning_policy.md, rule 1).

    Exposure is a fraction of a policy-year and cannot exceed 1 by
    definition. Values above it are treated as a reporting artifact around
    policy-year boundaries rather than a distinct risk group: the
    over-exposed policies have a *lower* mean ClaimNb than the portfolio
    average, so there is no evidence they belong to a different population.
    """
    out = freq.copy()
    out["Exposure"] = out["Exposure"].clip(upper=cap)
    return out


def cap_claim_nb(freq: pd.DataFrame, cap: int = CLAIM_NB_CAP) -> pd.DataFrame:
    """Cap ClaimNb at `cap` (cleaning_policy.md, rule 2a).

    Limits the leverage of a handful of implausible extreme values (e.g.
    16 claims against 0.33 years of exposure) on a Poisson GLM's
    log-likelihood, without removing the policy itself. Affects only 9
    policies out of 678,013.
    """
    out = freq.copy()
    out["ClaimNb"] = out["ClaimNb"].clip(upper=cap)
    return out


def clean_veh_gas(freq: pd.DataFrame) -> pd.DataFrame:
    """Strip stray embedded quote characters from VehGas (cleaning_policy.md, rule 5).

    Source values arrive as the literal strings "'Diesel'" and "'Regular'"
    (quotes included in the string content, an ARFF-parsing artifact) - not
    a real third category. This does not change how many distinct values
    the column has, only how they are spelled.
    """
    out = freq.copy()
    out["VehGas"] = out["VehGas"].str.strip("'")
    return out


def exclude_orphan_claims(freq: pd.DataFrame, sev: pd.DataFrame) -> pd.DataFrame:
    """Drop severity rows with no matching policy in freq (cleaning_policy.md, rule 3).

    These claims have no exposure or rating factors to join against, so a
    rating-factor model cannot use them regardless of any other decision
    made about the frequency/severity mismatch (rule 2b). Affects 195 claim
    rows (6 policies), representing ~1.3% of total claim value - excluded
    here, but that total is disclosed in the modelling report rather than
    silently dropped.
    """
    known_ids = set(freq["IDpol"])
    return sev[sev["IDpol"].isin(known_ids)].reset_index(drop=True)


def clean_frequency(freq: pd.DataFrame) -> pd.DataFrame:
    """Apply every frequency-table cleaning rule, in one documented place."""
    out = clip_exposure(freq)
    out = cap_claim_nb(out)
    out = clean_veh_gas(out)
    return out


def clean_severity(freq_cleaned: pd.DataFrame, sev: pd.DataFrame) -> pd.DataFrame:
    """Apply every severity-table cleaning rule (currently: orphan exclusion only)."""
    return exclude_orphan_claims(freq_cleaned, sev)
