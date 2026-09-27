"""Unit tests for auto_pricing.split.

These target the properties that actually matter for a valid split: every
policy assigned exactly once, no policy in two partitions, proportions
close to requested, and the claim rate preserved across partitions
(stratification doing its job).
"""

import pandas as pd

from auto_pricing.split import split_policies


def _synthetic_freq(n=1000, claim_rate=0.05, seed=0):
    import numpy as np

    rng = np.random.default_rng(seed)
    n_claims = int(n * claim_rate)
    claim_nb = np.array([1] * n_claims + [0] * (n - n_claims))
    rng.shuffle(claim_nb)
    return pd.DataFrame({"IDpol": range(n), "ClaimNb": claim_nb, "Exposure": 1.0})


def test_every_policy_assigned_exactly_once():
    freq = _synthetic_freq()
    result = split_policies(freq)
    assert len(result) == len(freq)
    assert result["IDpol"].duplicated().sum() == 0


def test_split_covers_all_original_ids_with_no_overlap():
    freq = _synthetic_freq()
    result = split_policies(freq)
    assert set(result["IDpol"]) == set(freq["IDpol"])

    train_ids = set(result.loc[result["split"] == "train", "IDpol"])
    val_ids = set(result.loc[result["split"] == "validation", "IDpol"])
    test_ids = set(result.loc[result["split"] == "test", "IDpol"])
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


def test_split_proportions_are_close_to_requested():
    freq = _synthetic_freq(n=10_000)
    result = split_policies(freq, train_size=0.70, val_size=0.15, test_size=0.15)
    shares = result["split"].value_counts(normalize=True)
    assert abs(shares["train"] - 0.70) < 0.01
    assert abs(shares["validation"] - 0.15) < 0.01
    assert abs(shares["test"] - 0.15) < 0.01


def test_stratification_preserves_claim_rate_across_splits():
    freq = _synthetic_freq(n=10_000, claim_rate=0.05)
    result = split_policies(freq)
    merged = result.merge(freq, on="IDpol")

    overall_rate = (freq["ClaimNb"] > 0).mean()
    for split_name in ["train", "validation", "test"]:
        subset = merged[merged["split"] == split_name]
        split_rate = (subset["ClaimNb"] > 0).mean()
        assert abs(split_rate - overall_rate) < 0.01


def test_invalid_proportions_raise():
    freq = _synthetic_freq()
    try:
        split_policies(freq, train_size=0.5, val_size=0.3, test_size=0.3)
        assert False, "expected ValueError for proportions not summing to 1.0"
    except ValueError:
        pass


def test_reproducible_with_same_random_state():
    freq = _synthetic_freq()
    result1 = split_policies(freq, random_state=42)
    result2 = split_policies(freq, random_state=42)
    pd.testing.assert_frame_equal(
        result1.sort_values("IDpol").reset_index(drop=True),
        result2.sort_values("IDpol").reset_index(drop=True),
    )
