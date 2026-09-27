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
