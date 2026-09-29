"""RiskRate — single-policy pure-premium calculator (Phase 10, step 1).

The GLM champion pipeline (paid-frequency Poisson GLM x lognormal severity
GLM, Phase 7) applied to one user-entered policy at a time. All modelling
logic lives in auto_pricing (src/) and app/pricing.py; this file is UI only.

Run with:
    streamlit run app/app.py
"""

from __future__ import annotations

import streamlit as st
from pricing import (
    AREA_OPTIONS,
    REGION_OPTIONS,
    VEHBRAND_OPTIONS,
    VEHGAS_OPTIONS,
    load_artifacts,
    price_policy,
)

st.set_page_config(page_title="RiskRate — Pure Premium Calculator", page_icon="🚗")

st.title("RiskRate: Pure Premium Calculator")
st.caption(
    "An educational demonstration of frequency x severity pricing on the freMTPL2 "
    "French motor third-party-liability dataset. **Not a real insurance quote** — "
    "see the Important disclosures section below before reading any number here as "
    "a price."
)

try:
    artifacts = load_artifacts()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

st.header("Policy details")

col1, col2 = st.columns(2)
with col1:
    area = st.selectbox("Area (A = rural, F = urban)", AREA_OPTIONS, index=AREA_OPTIONS.index("D"))
    veh_power = st.slider("Vehicle power class", min_value=4, max_value=15, value=7)
    veh_age = st.slider("Vehicle age (years)", min_value=0, max_value=30, value=2)
    driv_age = st.slider("Driver age (years)", min_value=18, max_value=90, value=35)
    bonus_malus = st.slider(
        "BonusMalus (50 = best/max discount, 100 = neutral, >100 = penalized)",
        min_value=50, max_value=230, value=50,
    )
with col2:
    veh_brand = st.selectbox("Vehicle brand code", VEHBRAND_OPTIONS)
    veh_gas = st.selectbox("Fuel type", VEHGAS_OPTIONS)
    density = st.number_input(
        "Population density of policyholder's commune (inhabitants/km²)",
        min_value=1, max_value=27000, value=1500,
    )
    region = st.selectbox("Region", REGION_OPTIONS, index=REGION_OPTIONS.index("R24"))

raw_policy = {
    "Area": area,
    "VehPower": veh_power,
    "VehAge": veh_age,
    "DrivAge": driv_age,
    "BonusMalus": bonus_malus,
    "VehBrand": veh_brand,
    "VehGas": veh_gas,
    "Density": density,
    "Region": region,
}

if st.button("Calculate pure premium", type="primary"):
    result = price_policy(raw_policy, artifacts)

    st.header("Result")
    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted annual claim frequency", f"{result['annual_frequency']:.4f}")
    m2.metric("Predicted severity (given a claim)", f"€{result['severity']:,.2f}")
    m3.metric("Annual pure premium", f"€{result['annual_pure_premium']:,.2f}")

    st.caption(
        "Pure premium = annual claim frequency × severity given a claim. This is the "
        "expected annual claim cost only — no expense loading, profit margin, or "
        "regulatory adjustment is applied (see disclosures below)."
    )

with st.expander("Important disclosures — read before interpreting any result above"):
    st.markdown(
        "- **This is a technical pure premium, not a final insurance price.** "
        "Expense loading, profit margin, reinsurance cost, and regulatory "
        "rate-filing constraints are entirely absent from this number.\n"
        "- **This model is not fit for any real underwriting, pricing, or coverage "
        "decision.** It was built on a public research dataset from a single, "
        "anonymized French insurer, from a period well over a decade old.\n"
        "- **Two rating factors used here have documented, unresolved calibration "
        "problems**: predictions vary by Region (observed/expected ratio "
        "0.46–1.83) and by driver age band (0.57–1.33) in ways not explained by "
        "sample size. See `reports/pure_premium_segment_stability.md` and "
        "`reports/fairness_analysis.md`.\n"
        "- **No fairness or proxy-variable audit could be performed on this "
        "dataset** — it contains no demographic fields (race, gender, income), so "
        "whether Region/Density/Area act as proxies for a protected characteristic "
        "cannot be verified either way. See `reports/fairness_analysis.md`.\n"
        "- Full governance documentation: `reports/model_card.md`, "
        "`reports/limitations.md`."
    )
