"""RiskRate — app entrypoint / page router (Phase 10).

Declares the multipage navigation explicitly via st.navigation, per current
Streamlit convention (the older implicit `pages/` directory auto-detection
was checked directly while building this and confirmed non-functional on
Streamlit 1.64 without this explicit call). Each page's own UI logic lives
in its own file under pages/; this file only wires them together.

Run with:
    streamlit run app/app.py
"""

from __future__ import annotations

import streamlit as st
from styling import apply_custom_styling

st.set_page_config(page_title="RiskRate — Pure Premium Pricing", page_icon="🚗")
apply_custom_styling()

calculator_page = st.Page(
    "pages/pricing_calculator.py", title="Pricing Calculator", icon="🚗", default=True
)
exploration_page = st.Page(
    "pages/model_exploration.py", title="Model Exploration", icon="📊"
)
governance_page = st.Page(
    "pages/governance.py", title="Governance", icon="📋"
)

pg = st.navigation([calculator_page, exploration_page, governance_page])
pg.run()
