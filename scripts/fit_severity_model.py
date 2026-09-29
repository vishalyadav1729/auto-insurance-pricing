"""Fit the Phase 6 severity champion (lognormal, Duan-smearing corrected)
on the training split and persist it for Phase 10's app.

Mirrors scripts/fit_frequency_models.py: saves a small artifact via
save_severity_model - NOT a raw pickle of the OLS results object, which
carries the same 878MB-style risk save_frequency_model's docstring
describes for frequency models.

Usage:
    python scripts/fit_severity_model.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from auto_pricing.data import load_severity_clean
from auto_pricing.severity import (
    SEVERITY_FORMULA,
    build_severity_table,
    fit_lognormal_model,
    save_severity_model,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"


def main() -> None:
    sev = load_severity_clean()
    model_table = pd.read_parquet(PROCESSED_DIR / "model_table.parquet")

    severity_table = build_severity_table(sev, model_table)
    train = severity_table[severity_table["split"] == "train"].copy()

    result, smearing_factor = fit_lognormal_model(train)
    print(f"lognormal severity model fit on {len(train):,} training claims")
    print(f"smearing factor: {smearing_factor:.4f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "severity_lognormal.joblib"
    save_severity_model(result, smearing_factor, SEVERITY_FORMULA, model_path)
    print(f"wrote {model_path}  ({model_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
