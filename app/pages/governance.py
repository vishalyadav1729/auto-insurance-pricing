"""RiskRate — governance page (Phase 10, step 3).

Renders the project's actual governance documents (reports/model_card.md,
reports/fairness_analysis.md, reports/limitations.md) directly from disk,
rather than a hand-written summary embedded in this file. Deliberate
choice: a summary would drift out of sync the next time any of those
reports is updated (as happened twice to model_card.md itself, in Phase 9
steps 1-2); reading the file at render time cannot drift, by construction.
Run via the router: streamlit run app/app.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"

st.title("Governance")
st.warning(
    "**RiskRate is an educational demonstration, not a real pricing system.** "
    "It must not be used for real underwriting, pricing, or coverage decisions. "
    "See *Prohibited use* in the Model Card tab below."
)

DOCUMENTS = {
    "Model Card": "model_card.md",
    "Fairness Analysis": "fairness_analysis.md",
    "Limitations": "limitations.md",
}


def read_report_or_error_message(path: Path, project_root: Path) -> str:
    """Return a report file's full text, or a clear error string if it's
    missing - a plain function (no Streamlit calls) so the missing-file
    branch can be unit-tested directly without needing AppTest or actually
    deleting a real project report to exercise it.
    """
    if path.exists():
        return path.read_text()
    return f"`{path.relative_to(project_root)}` not found."


tabs = st.tabs(list(DOCUMENTS.keys()))
for tab, filename in zip(tabs, DOCUMENTS.values()):
    with tab:
        content = read_report_or_error_message(REPORTS_DIR / filename, PROJECT_ROOT)
        st.markdown(content)
