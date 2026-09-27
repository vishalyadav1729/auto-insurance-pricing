"""Assemble the Phase 4 modelling table and persist the fitted preprocessing.

Reads data/processed/frequency_clean.parquet and policy_splits.parquet,
applies every Phase 4 feature-engineering decision (age/BonusMalus
banding, log-Density, Area ordinal, VehGas binary, and rare-category
grouping for Region/VehBrand - fit strictly on the training split), and
writes:

  - data/processed/model_table.parquet: one row per policy, every raw and
    engineered column, plus its split assignment.
  - artifacts/preprocessors/rare_category_maps.joblib: the fitted
    common-category sets, so the exact same rule can be replayed on new
    data later (Phase 7 evaluation, the Streamlit app) instead of being
    re-derived (which would silently change the rule over time).

Usage:
    python scripts/build_model_table.py
"""

from __future__ import annotations

from pathlib import Path

import joblib

from auto_pricing.data import load_frequency_clean
from auto_pricing.features import DEFAULT_MIN_EXPOSURE, build_model_table

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PREPROCESSORS_DIR = PROJECT_ROOT / "artifacts" / "preprocessors"


def main() -> None:
    import pandas as pd

    freq = load_frequency_clean()
    splits = pd.read_parquet(PROCESSED_DIR / "policy_splits.parquet")

    model_table, category_maps = build_model_table(freq, splits, min_exposure=DEFAULT_MIN_EXPOSURE)

    out_path = PROCESSED_DIR / "model_table.parquet"
    model_table.to_parquet(out_path, index=False)

    PREPROCESSORS_DIR.mkdir(parents=True, exist_ok=True)
    artifact_path = PREPROCESSORS_DIR / "rare_category_maps.joblib"
    joblib.dump(
        {"min_exposure": DEFAULT_MIN_EXPOSURE, "category_maps": category_maps},
        artifact_path,
    )

    print("=== Rare-category grouping (fit on training split only) ===")
    print(f"threshold: {DEFAULT_MIN_EXPOSURE:,.0f} training policy-years")
    print()
    for col in ["Region", "VehBrand"]:
        all_categories = set(model_table[col].unique())
        common = category_maps[col]
        grouped = sorted(all_categories - common)
        print(f"{col}: {len(all_categories)} categories -> {len(common) + 1} after grouping")
        print(f"  grouped into 'Other': {grouped}")
    print()

    print("=== Model table summary ===")
    print(f"shape: {model_table.shape}")
    print(model_table["split"].value_counts())
    print()
    print(f"wrote {out_path}")
    print(f"wrote {artifact_path}")


if __name__ == "__main__":
    main()
