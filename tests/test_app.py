"""Tests for the Phase 10 Streamlit app.

Two layers, tested separately:

1. app/pricing.py - plain functions with no Streamlit dependency, tested
   directly like every other module in this project.
2. The UI itself (app/app.py as the router, app/pages/*.py as the pages),
   exercised end-to-end via Streamlit's own AppTest harness
   (streamlit.testing.v1), which actually runs the script and lets a
   widget be interacted with, without needing a real browser or a running
   server. Confirmed while building this: hitting a running `streamlit
   run` server with plain HTTP only fetches its static JS shell - the
   Python script itself only executes once a live session connects, so
   that approach would not have caught a real bug in app.py.

app.py is a thin router (st.navigation over pages/pricing_calculator.py
and pages/model_exploration.py) - confirmed while building this that the
older implicit `pages/` directory auto-detection (no st.navigation call at
all) does NOT work on Streamlit 1.64: PagesManager reported zero
discovered pages without an explicit st.navigation call in the
entrypoint, so that approach was abandoned in favor of the explicit one
used here.

Both require the real fitted artifacts (artifacts/models/,
artifacts/preprocessors/) to already exist - run scripts/build_model_table.py,
scripts/build_pure_premium_table.py, and scripts/fit_severity_model.py
first, exactly as the rest of this project's pipeline requires.
"""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_PRESENT = (
    (PROJECT_ROOT / "artifacts" / "models" / "paid_frequency_poisson.joblib").exists()
    and (PROJECT_ROOT / "artifacts" / "models" / "severity_lognormal.joblib").exists()
    and (PROJECT_ROOT / "artifacts" / "preprocessors" / "rare_category_maps.joblib").exists()
)

pytestmark = pytest.mark.skipif(
    not ARTIFACTS_PRESENT,
    reason="requires fitted artifacts; run the data pipeline scripts first (see README)",
)


def test_load_artifacts_returns_all_three_pieces():
    from app.pricing import load_artifacts

    artifacts = load_artifacts()
    assert set(artifacts.keys()) == {"frequency", "severity", "category_maps"}


def test_price_policy_is_higher_for_a_known_higher_risk_policy():
    from app.pricing import load_artifacts, price_policy

    artifacts = load_artifacts()
    base_policy = {
        "Area": "D", "VehPower": 7, "VehAge": 2, "DrivAge": 35, "BonusMalus": 50,
        "VehBrand": "B1", "VehGas": "Diesel", "Density": 1500, "Region": "R24",
    }
    # Worst BonusMalus band + youngest-driver band: both independently
    # validated as real, robust risk increases (reports/frequency_relativities.md,
    # reports/severity_relativities.md) - the pure premium must reflect that,
    # not just run without error.
    risky_policy = dict(base_policy, BonusMalus=150, DrivAge=20)

    base_result = price_policy(base_policy, artifacts)
    risky_result = price_policy(risky_policy, artifacts)

    assert risky_result["annual_pure_premium"] > base_result["annual_pure_premium"]
    assert risky_result["annual_frequency"] > base_result["annual_frequency"]


def test_price_policy_pools_a_never_seen_category_without_error():
    from app.pricing import load_artifacts, price_policy

    artifacts = load_artifacts()
    # R43/B14 are the exact low-exposure categories Phase 3/4 flagged and
    # pooled into "Other" during training - a real app must handle a
    # policyholder reporting one of these without raising.
    policy = {
        "Area": "D", "VehPower": 7, "VehAge": 2, "DrivAge": 35, "BonusMalus": 50,
        "VehBrand": "B14", "VehGas": "Diesel", "Density": 1500, "Region": "R43",
    }
    result = price_policy(policy, artifacts)
    assert result["annual_pure_premium"] > 0


def test_compute_relativity_curve_severity_matches_the_published_report():
    # reports/severity_relativities.md documents BonusMalus 130+ at 1.66x
    # (using the exact reference combination REFERENCE_POLICY mirrors).
    # This checks the LIVE artifact reproduces that, not a copy of the
    # report - a real drift-detection check, not just "runs without error."
    from app.pricing import load_artifacts, compute_relativity_curve

    artifacts = load_artifacts()
    curve = compute_relativity_curve("BonusMalus", artifacts, target="severity")
    worst_band_relativity = curve.loc[curve["level"] == "130+", "relativity"].iloc[0]
    assert worst_band_relativity == pytest.approx(1.66, abs=0.02)


def test_compute_relativity_curve_frequency_uses_the_paid_frequency_model():
    # Deliberately does NOT assert this matches frequency_relativities.md
    # (Phase 5's reported-claims model) - confirmed while building this
    # that it should NOT match, since the app correctly uses the
    # paid-frequency model (Phase 7) instead. What IS asserted: the curve
    # is still monotonically informative (worst BonusMalus band well above
    # the reference), regardless of the exact multiplier.
    from app.pricing import load_artifacts, compute_relativity_curve

    artifacts = load_artifacts()
    curve = compute_relativity_curve("BonusMalus", artifacts, target="frequency")
    assert curve["relativity"].iloc[0] == 1.0  # reference level
    assert curve["relativity"].iloc[-1] > 2.0  # worst band clearly higher risk


def test_compute_relativity_curve_rejects_an_invalid_target():
    from app.pricing import load_artifacts, compute_relativity_curve

    artifacts = load_artifacts()
    with pytest.raises(ValueError):
        compute_relativity_curve("BonusMalus", artifacts, target="pure_premium")


def test_app_runs_without_exceptions():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(PROJECT_ROOT / "app" / "app.py"))
    at.run(timeout=30)
    assert not at.exception


def test_app_calculate_button_produces_metrics_without_exceptions():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(PROJECT_ROOT / "app" / "app.py"))
    at.run(timeout=30)
    at.button[0].click().run(timeout=30)
    assert not at.exception
    assert len(at.metric) == 3


def test_app_switches_to_model_exploration_page_without_exceptions():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(PROJECT_ROOT / "app" / "app.py"))
    at.run(timeout=30)
    at.switch_page("pages/model_exploration.py")
    at.run(timeout=30)
    assert not at.exception
    assert len(at.dataframe) == 4  # relativity table + region O/E + DrivAge O/E + scorecard


def test_model_exploration_page_relativity_selectors_are_interactive():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(PROJECT_ROOT / "app" / "app.py"))
    at.run(timeout=30)
    at.switch_page("pages/model_exploration.py")
    at.run(timeout=30)
    at.selectbox[0].select("VehAge").run(timeout=30)
    assert not at.exception
