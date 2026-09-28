"""Build the Phase 7 pure-premium table and fit the paid-frequency model.

Reads frequency_clean, severity_clean, and model_table; builds the
policy-level pure-premium table (every engineered feature, plus
ClaimNbFromSev/ClaimAmountSum); fits the paid-claim frequency GLM on the
training split (src/auto_pricing/pure_premium.py - resolves the 36.3%
frequency/severity mismatch documented since Phase 3) and the direct
Tweedie GLM (the plan's alternative to the frequency x severity
pipeline); and persists both fitted models using the same lightweight
artifact format Phase 5 built (save_frequency_model works unchanged for
any of these - it only needs a fitted statsmodels GLM results object and
a formula string, regardless of what the target column is called).

Usage:
    python scripts/build_pure_premium_table.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from auto_pricing.data import load_frequency_clean, load_severity_clean
from auto_pricing.frequency import predict_frequency, save_frequency_model
from auto_pricing.pure_premium import (
    PAID_FREQUENCY_FORMULA,
    TWEEDIE_FORMULA,
    TWEEDIE_POWER,
    build_pure_premium_table,
    fit_paid_frequency_glm,
    fit_tweedie_glm,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"


def main() -> None:
    freq = load_frequency_clean()
    sev = load_severity_clean()
    model_table = pd.read_parquet(PROCESSED_DIR / "model_table.parquet")

    pp_table = build_pure_premium_table(freq, sev, model_table)
    train = pp_table[pp_table["split"] == "train"].copy()

    paid_freq_result = fit_paid_frequency_glm(train)
    print(f"paid-frequency GLM converged: {paid_freq_result.converged}")

    pred = predict_frequency(paid_freq_result, train)
    observed = train["ClaimNbFromSev"].sum()
    print(f"predicted total paid claims (train): {pred.sum():,.2f}")
    print(f"observed total paid claims (train):  {observed:,}")
    print(f"ratio: {pred.sum() / observed:.4f}")

    out_table_path = PROCESSED_DIR / "pure_premium_table.parquet"
    pp_table.to_parquet(out_table_path, index=False)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "paid_frequency_poisson.joblib"
    save_frequency_model(paid_freq_result, PAID_FREQUENCY_FORMULA, "poisson", model_path)

    print()
    print(f"fitting direct Tweedie GLM (power={TWEEDIE_POWER})...")
    tweedie_result = fit_tweedie_glm(train)
    print(f"Tweedie GLM converged: {tweedie_result.converged}")
    tweedie_pred = predict_frequency(tweedie_result, train)
    tweedie_observed = train["ClaimAmountSum"].sum()
    print(f"predicted total claim cost (train): {tweedie_pred.sum():,.2f}")
    print(f"observed total claim cost (train):  {tweedie_observed:,.2f}")
    print(f"ratio: {tweedie_pred.sum() / tweedie_observed:.4f}")

    tweedie_model_path = MODELS_DIR / "pure_premium_tweedie.joblib"
    save_frequency_model(
        tweedie_result, TWEEDIE_FORMULA, "tweedie", tweedie_model_path, power=TWEEDIE_POWER
    )

    print()
    print(f"wrote {out_table_path}")
    print(f"wrote {model_path}")
    print(f"wrote {tweedie_model_path}")


if __name__ == "__main__":
    main()
