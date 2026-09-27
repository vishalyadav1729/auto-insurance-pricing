"""Smoke test: confirms the auto_pricing package is importable and correctly
installed (editable) from anywhere, not just from inside a notebook cell.

This is the Phase 1 reproducibility check: if this test fails, notebooks,
scripts, and the Streamlit app will all fail to `import auto_pricing` too.
"""

import auto_pricing


def test_package_imports():
    assert auto_pricing is not None


def test_package_has_version():
    assert hasattr(auto_pricing, "__version__")
