"""Create the policy-level train/validation/test split.

Reads data/processed/frequency_clean.parquet, assigns every policy to
train/validation/test (70/15/15, stratified by claim occurrence), and
writes the assignment to data/processed/policy_splits.parquet.

This file is the single source of truth for which policy belongs to which
split. Any table derived from a policy - including the severity table -
must be split by joining against this file on IDpol, never by being
sampled on its own (see src/auto_pricing/split.py for why).

Usage:
    python scripts/split_data.py
"""

from __future__ import annotations

from pathlib import Path

from auto_pricing.data import load_frequency_clean, load_severity_clean
from auto_pricing.split import split_policies

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def main() -> None:
    freq = load_frequency_clean()
    sev = load_severity_clean()

    splits = split_policies(freq)

    out_path = PROCESSED_DIR / "policy_splits.parquet"
    splits.to_parquet(out_path, index=False)

    merged = splits.merge(freq[["IDpol", "ClaimNb", "Exposure"]], on="IDpol")
    sev_with_split = sev.merge(splits, on="IDpol", how="left")

    print("=== Split summary ===")
    for split_name in ["train", "validation", "test"]:
        subset = merged[merged["split"] == split_name]
        n_policies = len(subset)
        claim_rate = (subset["ClaimNb"] > 0).mean()
        exposure = subset["Exposure"].sum()
        n_claims = (sev_with_split["split"] == split_name).sum()
        print(
            f"  {split_name:<10} {n_policies:>7,} policies "
            f"({n_policies/len(merged):5.1%})  "
            f"claim rate {claim_rate:.4%}  "
            f"exposure {exposure:>10,.1f}  "
            f"claims {n_claims:>6,}"
        )

    print()
    print(f"overall claim rate: {(merged['ClaimNb'] > 0).mean():.4%}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
