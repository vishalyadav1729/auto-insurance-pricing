"""Shared visual styling for the RiskRate app.

Called once, from app.py, before st.navigation's pg.run() - the router's
own top-level code re-executes on every rerun regardless of which page is
selected (confirmed via Streamlit's own st.navigation docs: "Streamlit
executes the entrypoint file with every app rerun"), so injecting CSS here
applies it consistently across all three pages without repeating the call
in each page file.
"""

from __future__ import annotations

import streamlit as st


def apply_custom_styling() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        h1 { font-weight: 700; letter-spacing: -0.02em; }
        h2, h3 { font-weight: 600; }

        /* Card-style bordered containers (st.container(border=True)) */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 14px;
        }

        /* Metric cards */
        div[data-testid="stMetric"] {
            background-color: #1E293B;
            border: 1px solid #334155;
            border-radius: 14px;
            padding: 18px 20px;
        }
        div[data-testid="stMetricLabel"] {
            font-weight: 500;
            opacity: 0.85;
        }
        div[data-testid="stMetricValue"] {
            color: #2DD4BF;
        }

        /* Buttons */
        .stButton button {
            border-radius: 8px;
            font-weight: 600;
            padding: 0.55rem 1.4rem;
        }

        /* Sidebar nav */
        section[data-testid="stSidebar"] {
            border-right: 1px solid #1E293B;
        }

        /* Dataframes */
        div[data-testid="stDataFrame"] {
            border-radius: 10px;
            overflow: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
