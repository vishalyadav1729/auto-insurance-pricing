"""Unit tests for auto_pricing.validation.

Uses small, hand-built DataFrames rather than the real 678k-row download,
so these run in milliseconds and pin down exactly what each function is
supposed to catch.
"""

import pandas as pd

from auto_pricing.validation import (
    duplicate_id_count,
    find_orphan_claims,
    missing_value_report,
    numeric_range_report,
    reconcile_claim_counts,
)


def test_missing_value_report_counts_nans():
    df = pd.DataFrame({"a": [1, None, 3], "b": [1, 2, 3]})
    report = missing_value_report(df)
    assert report["a"] == 1
    assert report["b"] == 0


def test_duplicate_id_count_detects_repeats():
    df = pd.DataFrame({"IDpol": [1, 2, 2, 3]})
    assert duplicate_id_count(df, "IDpol") == 1


def test_duplicate_id_count_zero_when_unique():
    df = pd.DataFrame({"IDpol": [1, 2, 3]})
    assert duplicate_id_count(df, "IDpol") == 0


def test_numeric_range_report_flags_negatives():
    df = pd.DataFrame({"Exposure": [-0.1, 0.5, 1.0]})
    report = numeric_range_report(df, ["Exposure"])
    assert report.loc["Exposure", "n_negative"] == 1
    assert report.loc["Exposure", "min"] == -0.1
    assert report.loc["Exposure", "max"] == 1.0


def test_find_orphan_claims_returns_unmatched_ids():
    freq = pd.DataFrame({"IDpol": [1, 2, 3]})
    sev = pd.DataFrame({"IDpol": [1, 2, 99], "ClaimAmount": [100.0, 200.0, 300.0]})
    orphans = find_orphan_claims(freq, sev)
    assert list(orphans["IDpol"]) == [99]


def test_find_orphan_claims_empty_when_all_match():
    freq = pd.DataFrame({"IDpol": [1, 2, 3]})
    sev = pd.DataFrame({"IDpol": [1, 2], "ClaimAmount": [100.0, 200.0]})
    orphans = find_orphan_claims(freq, sev)
    assert orphans.empty


def test_reconcile_claim_counts_finds_mismatch():
    # Policy 1 says ClaimNb=1 but has no severity rows (mismatch).
    # Policy 2 says ClaimNb=1 and has exactly one severity row (matches).
    # Policy 3 says ClaimNb=0 but has two severity rows (mismatch).
    freq = pd.DataFrame({"IDpol": [1, 2, 3], "ClaimNb": [1, 1, 0]})
    sev = pd.DataFrame(
        {
            "IDpol": [2, 3, 3],
            "ClaimAmount": [500.0, 100.0, 200.0],
        }
    )
    mismatches = reconcile_claim_counts(freq, sev)
    mismatched_ids = set(mismatches["IDpol"])
    assert mismatched_ids == {1, 3}
    assert 2 not in mismatched_ids


def test_reconcile_claim_counts_empty_when_fully_consistent():
    freq = pd.DataFrame({"IDpol": [1, 2], "ClaimNb": [0, 1]})
    sev = pd.DataFrame({"IDpol": [2], "ClaimAmount": [500.0]})
    mismatches = reconcile_claim_counts(freq, sev)
    assert mismatches.empty
