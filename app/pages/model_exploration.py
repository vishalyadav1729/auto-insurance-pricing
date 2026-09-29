"""RiskRate — model exploration dashboard (Phase 10, step 2).

Three views onto already-established findings from Phases 5-9: a live
relativity explorer (computed from the real deployed artifacts, not pasted
from a report), the known segment-calibration gaps, and the GLM-vs-boosted
challenger scorecard. Wired into the sidebar navigation by app.py's
st.navigation call; run via `streamlit run app/app.py`, not this file
directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pricing import FACTOR_SWEEP_VALUES, load_artifacts, compute_relativity_curve  # noqa: E402

st.title("Model Exploration")

try:
    artifacts = load_artifacts()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

st.header("1. Live relativity explorer")
st.caption(
    "Computed on the fly from the actual deployed model artifacts (not pasted from a "
    "report) by sweeping one rating factor across its bands, holding every other factor "
    "at the fixed reference policy `reports/severity_relativities.md` uses (18-22 "
    "driver, brand-new vehicle, best BonusMalus, Area A, lowest density, Diesel, lowest "
    "VehPower, Region R11, VehBrand B1)."
)

with st.container(border=True):
    target = st.radio("Relativity for:", ["severity", "frequency"], horizontal=True)
    factor = st.selectbox("Rating factor:", list(FACTOR_SWEEP_VALUES.keys()))

    curve = compute_relativity_curve(factor, artifacts, target=target)
    st.bar_chart(curve.set_index("level")["relativity"])
    st.dataframe(curve, hide_index=True, width="stretch")

    if target == "frequency":
        st.warning(
            "**These frequency relativities will not match `reports/frequency_relativities.md`, "
            "and that's expected, not a bug.** This app prices using the *paid-frequency* model "
            "(fit on paid claims, `ClaimNbFromSev` — the correct model for pricing, Phase 7). "
            "`frequency_relativities.md` documents Phase 5's separate *reported-claims* champion "
            "(fit on `ClaimNb`). The two differ because the share of reported claims that get "
            "paid itself varies by rating factor (0.65–1.00 across BonusMalus bands alone) — "
            "confirmed directly while building this page. See `reports/pure_premium_reconciliation.md`."
        )
    else:
        st.success(
            "Severity has only one model in this project (Phase 6's lognormal champion), so "
            "this live computation reproduces `reports/severity_relativities.md`'s published "
            "numbers almost exactly — e.g. BonusMalus 130+ comes out to ~1.66x here, matching "
            "the report, which is a useful live check that the deployed artifact hasn't drifted "
            "from what was reported."
        )

st.header("2. Known segment-calibration gaps")
st.caption(
    "Observed-to-expected ratio (1.0 = perfectly calibrated) by segment, computed on the "
    "validation split in Phase 7 (Region) and Phase 9 (DrivAge) — not recomputed here, "
    "since a real O/E check needs actual observed outcomes, not a single synthetic policy."
)

region_oe = pd.DataFrame(
    {
        "Region": ["R41", "R73", "R52", "R72", "R11", "R24 (largest)", "R82", "R93", "Other"],
        "Exposure (policy-years)": [1223, 1070, 3326, 2084, 4487, 15439, 6767, 5318, 1418],
        "O/E ratio": [0.46, 0.51, 0.53, 0.55, 0.73, 0.76, 0.95, 1.20, 1.83],
    }
)
drivage_oe = pd.DataFrame(
    {
        "DrivAge band": ["18-22", "23-29", "30-39", "40-49", "50-59", "60-69", "70+"],
        "Exposure (policy-years)": [969, 4732, 12315, 13105, 12019, 6030, 4469],
        "O/E ratio": [0.97, 0.91, 0.87, 0.69, 0.87, 0.57, 1.33],
    }
)

with st.container(border=True):
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("By Region")
        st.dataframe(region_oe, hide_index=True, width="stretch")
        st.caption("Range 0.46–1.83, not explained by segment size — `reports/pure_premium_segment_stability.md`.")
    with c2:
        st.subheader("By DrivAge band")
        st.dataframe(drivage_oe, hide_index=True, width="stretch")
        st.caption("Range 0.57–1.33; ages 40-69 over-predicted, 70+ under-predicted — `reports/fairness_analysis.md`.")

st.header("3. GLM champion vs. gradient-boosting challenger")
st.caption("Phase 8's scorecard — champion-challenger, not a single winner. Full detail in `reports/ml_champion_challenger_recommendation.md`.")

scorecard = pd.DataFrame(
    {
        "Criterion": [
            "Predictive deviance (combined)", "Portfolio calibration (O/E)",
            "Segment calibration", "Ranking (Gini)", "Interpretability",
            "Sensitivity to large claims", "Governance burden",
        ],
        "GLM (operational champion)": [
            "65.44", "0.843", "Comparable", "0.302",
            "Explicit relativities", "BonusMalus finding validated real", "Lower",
        ],
        "Boosted (challenger)": [
            "64.57 (-1.3%)", "1.011 (better)", "Worse in extreme case (Other: 2.54x vs 1.83x)",
            "0.328 (+8.9%)", "Requires PDP/permutation-importance tooling",
            "Fails to learn the same validated BonusMalus signal", "Higher",
        ],
    }
)
with st.container(border=True):
    st.dataframe(scorecard, hide_index=True, width="stretch")
    st.info(
        "**Recommendation: champion-challenger, not a single winner.** The GLM pipeline is the "
        "operational champion — comparable accuracy, no validated-signal loss, lower governance "
        "burden. The boosted pipeline is a challenger worth continued development for its real "
        "calibration and ranking advantage, ideally via a GLM-corrected residual-boosting "
        "approach that would prevent it from silently dropping a validated signal."
    )
