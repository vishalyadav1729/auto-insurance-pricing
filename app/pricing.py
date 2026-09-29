"""Pricing logic for the RiskRate Streamlit app (Phase 10, step 1).

Deliberately separate from app.py: everything here is plain functions with
no Streamlit import, so it can be unit-tested with ordinary pytest exactly
like every other module in this project, rather than requiring a running
Streamlit session to verify.

Loads the GLM champion pipeline from Phase 7 (paid-frequency Poisson GLM x
lognormal severity GLM, the operational champion per
reports/ml_champion_challenger_recommendation.md) via the lightweight
joblib artifacts built by scripts/fit_frequency_models.py,
scripts/build_pure_premium_table.py, and scripts/fit_severity_model.py -
never by refitting on the full 678k-row dataset inside the app itself.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from auto_pricing.features import engineer_single_policy
from auto_pricing.frequency import predict_from_artifact
from auto_pricing.pure_premium import predict_annual_pure_premium
from auto_pricing.severity import predict_severity_from_artifact

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"
PREPROCESSORS_DIR = PROJECT_ROOT / "artifacts" / "preprocessors"

# The rating-factor categories actually observed in the training data
# (data/processed/model_table.parquet) - used to populate the app's input
# widgets with only values the champion pipeline has ever seen. B7-B9 are
# genuinely absent from this dataset (reports/data_dictionary.md), not an
# omission here.
REGION_OPTIONS = [
    "R11", "R21", "R22", "R23", "R24", "R25", "R26", "R31", "R41", "R42",
    "R43", "R52", "R53", "R54", "R72", "R73", "R74", "R82", "R83", "R91",
    "R93", "R94",
]
VEHBRAND_OPTIONS = ["B1", "B2", "B3", "B4", "B5", "B6", "B10", "B11", "B12", "B13", "B14"]
AREA_OPTIONS = ["A", "B", "C", "D", "E", "F"]
VEHGAS_OPTIONS = ["Diesel", "Regular"]


def load_artifacts() -> dict:
    """Load every artifact the champion pipeline needs, once.

    Raises FileNotFoundError with a clear message (rather than a bare
    joblib traceback) if the pipeline scripts haven't been run yet - the
    three scripts in this docstring's module-level comment must run first.
    """
    paths = {
        "frequency": MODELS_DIR / "paid_frequency_poisson.joblib",
        "severity": MODELS_DIR / "severity_lognormal.joblib",
        "category_maps": PREPROCESSORS_DIR / "rare_category_maps.joblib",
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing model artifact(s): "
            + ", ".join(missing)
            + ". Run scripts/build_model_table.py, scripts/build_pure_premium_table.py, "
            "and scripts/fit_severity_model.py first."
        )
    return {
        "frequency": joblib.load(paths["frequency"]),
        "severity": joblib.load(paths["severity"]),
        "category_maps": joblib.load(paths["category_maps"])["category_maps"],
    }


def price_policy(raw_policy: dict, artifacts: dict) -> dict:
    """Predict frequency, severity, and annual pure premium for one policy.

    `raw_policy` must supply Area/VehPower/VehAge/DrivAge/BonusMalus/
    VehBrand/VehGas/Density/Region - the same raw fields
    engineer_single_policy requires. Exposure is fixed at 1.0 internally
    (a full policy-year) regardless of what's passed in raw_policy, if
    anything - the app quotes an ANNUAL premium, and Phase 7's
    reconciliation report is explicit that expected loss over a partial
    exposure period and an annualized rate are different numbers that must
    not be conflated (predict_annual_pure_premium exists specifically to
    return the latter).

    Returns a dict with `annual_frequency` (predicted claim count per
    year), `severity` (predicted cost given a claim, EUR), and
    `annual_pure_premium` (EUR/year) - deliberately not a `total_premium`
    key, since this is a technical pure premium, not a final consumer
    price (reports/model_card.md's Intended Use section is explicit about
    this distinction).
    """
    policy_for_pricing = dict(raw_policy, Exposure=1.0)
    row = engineer_single_policy(policy_for_pricing, artifacts["category_maps"])

    freq_pred = pd.Series(predict_from_artifact(artifacts["frequency"], row), index=row.index)
    sev_pred = pd.Series(predict_severity_from_artifact(artifacts["severity"], row), index=row.index)
    annual_pure_premium = predict_annual_pure_premium(freq_pred, sev_pred, row["Exposure"])

    return {
        "annual_frequency": float(freq_pred.iloc[0]),
        "severity": float(sev_pred.iloc[0]),
        "annual_pure_premium": float(annual_pure_premium.iloc[0]),
    }
