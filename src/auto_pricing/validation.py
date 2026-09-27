"""Data-quality checks for the raw freMTPL2 tables.

These functions only *describe* problems (missingness, duplicate keys,
out-of-range values, mismatches between the frequency and severity tables).
None of them drop rows or change values - that is a modelling decision that
belongs in Phase 3, made deliberately and documented, not buried inside a
validation helper. Keeping detection and correction separate means the same
functions can be reused to confirm the cleaning step actually worked.
"""

from __future__ import annotations

import pandas as pd


def missing_value_report(df: pd.DataFrame) -> pd.Series:
    """Count of missing values per column (0 for columns with none)."""
    return df.isna().sum()


def duplicate_id_count(df: pd.DataFrame, id_col: str) -> int:
    """Number of rows whose id_col value repeats elsewhere in df.

    For freMTPL2freq, IDpol should be a policy-level primary key: a
    duplicate would mean the same policy was somehow recorded twice.
    """
    return int(df[id_col].duplicated().sum())


def numeric_range_report(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Min/max/mean and negative-value count for each numeric column.

    A quick way to spot impossible values (e.g. negative Exposure) without
    reading through a full .describe() table by eye.
    """
    rows = []
    for col in columns:
        series = df[col]
        rows.append(
            {
                "column": col,
                "min": series.min(),
                "max": series.max(),
                "mean": series.mean(),
                "n_negative": int((series < 0).sum()),
            }
        )
    return pd.DataFrame(rows).set_index("column")


def find_orphan_claims(freq: pd.DataFrame, sev: pd.DataFrame) -> pd.DataFrame:
    """Severity rows whose IDpol does not exist in the frequency table.

    These claims cannot be attributed to any known policy's exposure or
    rating factors, so they cannot be used by a policy-level model as-is.
    Returns the offending rows of `sev` (empty DataFrame if none found).
    """
    known_ids = set(freq["IDpol"])
    return sev[~sev["IDpol"].isin(known_ids)]


def reconcile_claim_counts(freq: pd.DataFrame, sev: pd.DataFrame) -> pd.DataFrame:
    """Compare each policy's reported ClaimNb against its actual severity rows.

    Returns one row per policy that appears in either table, with columns
    [IDpol, ClaimNb, ClaimNbFromSev], restricted to rows where the two
    disagree. ClaimNb is what the frequency table claims; ClaimNbFromSev is
    how many severity records actually exist for that policy. Historically
    (and confirmed on this download) freMTPL2 does not always agree between
    the two tables - this function is how that gets caught before a
    frequency or severity model is ever fit.
    """
    sev_counts = sev.groupby("IDpol").size().rename("ClaimNbFromSev")
    freq_indexed = freq.set_index("IDpol")["ClaimNb"]

    combined = pd.concat([freq_indexed, sev_counts], axis=1)
    combined["ClaimNb"] = combined["ClaimNb"].fillna(0)
    combined["ClaimNbFromSev"] = combined["ClaimNbFromSev"].fillna(0)

    mismatched = combined[combined["ClaimNb"] != combined["ClaimNbFromSev"]]
    return mismatched.reset_index()
