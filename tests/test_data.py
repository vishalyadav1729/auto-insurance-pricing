"""Unit tests for auto_pricing.data.

Only targets the pure-Python assembly logic (build_policy_claim_table);
load_frequency/load_severity/etc. just read files and aren't worth testing
against synthetic data.
"""

import pandas as pd

from auto_pricing.data import build_policy_claim_table


def test_build_policy_claim_table_fills_zero_for_policies_without_claims():
    freq = pd.DataFrame({"IDpol": [1, 2, 3], "ClaimNb": [0, 1, 2], "Exposure": [1.0, 1.0, 1.0]})
    sev = pd.DataFrame({"IDpol": [2, 3, 3], "ClaimAmount": [500.0, 100.0, 200.0]})

    out = build_policy_claim_table(freq, sev)

    assert len(out) == 3  # every freq row kept
    row1 = out.loc[out["IDpol"] == 1].iloc[0]
    assert row1["ClaimNbFromSev"] == 0
    assert row1["ClaimAmountSum"] == 0.0

    row3 = out.loc[out["IDpol"] == 3].iloc[0]
    assert row3["ClaimNbFromSev"] == 2
    assert row3["ClaimAmountSum"] == 300.0


def test_build_policy_claim_table_keeps_original_claimnb_untouched():
    # ClaimNb (from freq) and ClaimNbFromSev (from sev) may legitimately
    # disagree - cleaning_policy.md rule 2b - so both must survive intact.
    freq = pd.DataFrame({"IDpol": [1], "ClaimNb": [1], "Exposure": [1.0]})
    sev = pd.DataFrame({"IDpol": [], "ClaimAmount": []})

    out = build_policy_claim_table(freq, sev)

    assert out.loc[0, "ClaimNb"] == 1
    assert out.loc[0, "ClaimNbFromSev"] == 0
