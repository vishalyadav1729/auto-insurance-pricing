"""Loading functions for the raw freMTPL2 tables.

Deliberately thin: this module only reads what scripts/download_data.py
already wrote to data/raw/ and does light, lossless type handling (e.g.
IDpol as a consistent integer key for joining). It does NOT clean, filter,
join, or otherwise alter the data - that is Phase 3's job, and keeping it
separate means notebooks/01_data_understanding.ipynb can inspect the data
exactly as it arrived from source.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_frequency(path: Path | None = None) -> pd.DataFrame:
    """Load the policy-level frequency table (freMTPL2freq).

    One row per policy. IDpol is cast to int64 (it arrives as float64 from
    OpenML) purely so it can be reliably compared/joined against the
    severity table's IDpol without float-equality pitfalls.
    """
    path = path or RAW_DIR / "freMTPL2freq.csv"
    df = pd.read_csv(path)
    df["IDpol"] = df["IDpol"].astype("int64")
    return df


def load_severity(path: Path | None = None) -> pd.DataFrame:
    """Load the claim-level severity table (freMTPL2sev).

    One row per claim; a policy with multiple claims appears multiple
    times. IDpol is already int64 in this file.
    """
    path = path or RAW_DIR / "freMTPL2sev.csv"
    return pd.read_csv(path)


def aggregate_severity_by_policy(sev: pd.DataFrame) -> pd.DataFrame:
    """Collapse claim-level severity rows to one row per policy.

    Returns columns [IDpol, ClaimNbFromSev, ClaimAmountSum]:
      - ClaimNbFromSev: how many severity records exist for that policy
        (the *actual* observed claim count, as opposed to ClaimNb in the
        frequency table, which - see reports/data_dictionary.md - does not
        always agree with it).
      - ClaimAmountSum: total claim cost for that policy.

    This is the shape needed later to reconcile against freMTPL2freq and,
    eventually, to build the policy-level modelling table in Phase 4.
    """
    return (
        sev.groupby("IDpol")
        .agg(ClaimNbFromSev=("ClaimAmount", "size"), ClaimAmountSum=("ClaimAmount", "sum"))
        .reset_index()
    )


def load_frequency_clean(path: Path | None = None) -> pd.DataFrame:
    """Load the cleaned policy-level table written by scripts/prepare_data.py.

    This has the cleaning_policy.md rules already applied (Exposure clipped
    to 1.0, ClaimNb capped at 4, VehGas quote characters stripped) - use
    this, not load_frequency(), for any exploratory analysis or modelling
    from Phase 3 onward.
    """
    path = path or PROCESSED_DIR / "frequency_clean.parquet"
    return pd.read_parquet(path)


def load_severity_clean(path: Path | None = None) -> pd.DataFrame:
    """Load the cleaned claim-level table written by scripts/prepare_data.py.

    Orphaned claims (no matching policy) have already been excluded - see
    cleaning_policy.md rule 3.
    """
    path = path or PROCESSED_DIR / "severity_clean.parquet"
    return pd.read_parquet(path)


def build_policy_claim_table(freq: pd.DataFrame, sev: pd.DataFrame) -> pd.DataFrame:
    """Left-join per-policy severity aggregates onto the frequency table.

    Every row of `freq` is kept; policies with no claims get
    ClaimNbFromSev=0 and ClaimAmountSum=0.0 rather than NaN. This is the
    policy-level view needed to compute empirical pure premium (Phase 3
    step 6) and, later, to evaluate a fitted frequency x severity model
    against actual claim cost (Phase 7).

    Deliberately keeps ClaimNb (from freq) and ClaimNbFromSev (from sev)
    as two separate columns rather than reconciling them here - see
    cleaning_policy.md rule 2b for why they are allowed to disagree.
    """
    sev_by_policy = aggregate_severity_by_policy(sev)
    merged = freq.merge(sev_by_policy, on="IDpol", how="left")
    merged["ClaimNbFromSev"] = merged["ClaimNbFromSev"].fillna(0).astype(int)
    merged["ClaimAmountSum"] = merged["ClaimAmountSum"].fillna(0.0)
    return merged
