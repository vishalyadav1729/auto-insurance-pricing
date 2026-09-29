"""Pricing and relativity logic for the RiskRate Streamlit app (Phase 10,
steps 1-2: the single-policy calculator and the model-exploration
dashboard both build on the functions here).

Deliberately separate from app.py and pages/*.py: everything here is plain
functions with no Streamlit import, so it can be unit-tested with ordinary
pytest exactly like every other module in this project, rather than
requiring a running Streamlit session to verify.

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

# The exact reference combination reports/severity_relativities.md states
# ("18-22 driver, brand-new vehicle, best BonusMalus, Area A, lowest
# density, Diesel, lowest VehPower, Region R11, VehBrand B1") - using this
# as the fixed baseline for every relativity sweep means the live numbers
# below can be checked directly against the published relativity tables,
# not just plausible on their own.
REFERENCE_POLICY = {
    "Area": "A", "VehPower": 4, "VehAge": 0, "DrivAge": 20, "BonusMalus": 50,
    "VehBrand": "B1", "VehGas": "Diesel", "Density": 1, "Region": "R11",
}

# One representative raw value per band, chosen to fall inside that band's
# fixed bin edges (features.py: DRIVAGE_BINS/VEHAGE_BINS/BONUSMALUS_BINS) -
# so sweeping this factor alone (reference policy otherwise unchanged)
# reproduces reports/frequency_relativities.md and
# reports/severity_relativities.md's own band-by-band tables exactly.
FACTOR_SWEEP_VALUES: dict[str, list[tuple[object, str]]] = {
    "BonusMalus": [
        (50, "50 (best)"), (55, "51-59"), (70, "60-79"),
        (90, "80-99"), (110, "100-129"), (150, "130+"),
    ],
    "VehAge": [
        (0, "0 (new)"), (1, "1-2"), (4, "3-5"), (7, "6-9"),
        (12, "10-14"), (17, "15-19"), (25, "20+"),
    ],
    "DrivAge": [
        (20, "18-22"), (26, "23-29"), (35, "30-39"), (45, "40-49"),
        (55, "50-59"), (65, "60-69"), (75, "70+"),
    ],
    "Region": [(r, r) for r in REGION_OPTIONS],
    "VehBrand": [(b, b) for b in VEHBRAND_OPTIONS],
    "Area": [(a, a) for a in AREA_OPTIONS],
    "VehGas": [(g, g) for g in VEHGAS_OPTIONS],
}


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


def compute_relativity_curve(
    factor: str, artifacts: dict, target: str = "frequency", reference_policy: dict | None = None
) -> pd.DataFrame:
    """Sweep one rating factor across FACTOR_SWEEP_VALUES' levels, holding
    every other factor at `reference_policy` (REFERENCE_POLICY by default),
    and return each level's predicted value plus its relativity to the
    first (reference) level.

    This computes relativities LIVE from the deployed model artifacts, the
    same way reports/severity_relativities.md's table was derived
    (exp(coefficient) against a reference category) - not a copy of it.
    With REFERENCE_POLICY as the baseline, `target="severity"` reproduces
    that report's numbers almost exactly (e.g. BonusMalus 130+ -> 1.663x
    here vs. 1.66x published; DrivAge 70+ -> 1.259x here vs. 1.26x
    published) - confirmed while building this, a useful live check that
    the persisted severity artifact hasn't drifted from what was reported.

    `target="frequency"` deliberately does NOT reproduce
    reports/frequency_relativities.md's numbers, and this is expected, not
    a bug: this app prices using the paid-frequency model (fit on
    ClaimNbFromSev, Phase 7 - the correct model for pricing), while
    frequency_relativities.md documents Phase 5's reported-claims champion
    (fit on ClaimNb) - two different fitted models by design (see
    reports/pure_premium_reconciliation.md). Confirmed directly while
    building this: BonusMalus 130+ comes out to ~9.0x here vs. 6.43x
    published, and VehAge's relativities are much flatter here than
    published - both explained by payment rate itself varying by these
    same factors (0.65-1.00 across BonusMalus bands, Phase 7), not by any
    error in either model.

    `target` is "frequency" or "severity" - pure premium isn't offered here
    because it mixes both models' relativities into a single number Phase
    5/6's separately-published relativity tables were never trying to
    represent.
    """
    if target not in {"frequency", "severity"}:
        raise ValueError(f"target must be 'frequency' or 'severity', got {target!r}")
    base_policy = reference_policy if reference_policy is not None else REFERENCE_POLICY

    rows = []
    for raw_value, label in FACTOR_SWEEP_VALUES[factor]:
        policy = dict(base_policy, **{factor: raw_value})
        result = price_policy(policy, artifacts)
        value = result["annual_frequency"] if target == "frequency" else result["severity"]
        rows.append({"level": label, "value": value})

    curve = pd.DataFrame(rows)
    curve["relativity"] = curve["value"] / curve["value"].iloc[0]
    return curve
