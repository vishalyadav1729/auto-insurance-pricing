# Automobile Insurance Pricing Project

This project builds an end-to-end automobile insurance pricing analysis using public motor insurance claims data.

The goal is to estimate the expected claim cost for an insurance policy by modelling:

1. claim frequency: the expected number of claims per exposure period;
2. claim severity: the expected cost conditional on a claim occurring;
3. pure premium: the expected claim cost, obtained by combining frequency and severity.

The project is designed to demonstrate actuarial pricing judgement, reproducible data analysis, SQL-based data preparation, statistical modelling, model validation and business communication.

## Planned Workflow

1. Load policy exposure and claims data.
2. Store the data in a relational database.
3. Use SQL to create a modelling table.
4. Conduct data-quality and leakage checks.
5. Model claim frequency using a Poisson or negative-binomial GLM.
6. Model claim severity using a Gamma or lognormal model.
7. Combine frequency and severity into an expected loss-cost estimate.
8. Compare the GLM approach with a tree-based machine-learning model.
9. Evaluate calibration, lift and out-of-sample performance.
10. Present results in an executive summary.

## Tools

- Python
- Jupyter Notebook
- SQLite
- SQL
- pandas
- statsmodels
- scikit-learn
- Git and GitHub

## Repository Structure

```text
auto-insurance-pricing/
├── src/auto_pricing/   # reusable package (installed with `pip install -e .`)
├── scripts/            # download_data.py, prepare_data.py, train/evaluate scripts
├── notebooks/          # analysis notebooks; no production logic lives here
├── configs/            # data/model configuration files
├── data/               # raw/interim/processed data (git-ignored; reproducible via scripts/)
├── artifacts/          # fitted models and preprocessors (git-ignored except metadata/)
├── outputs/            # figures, tables, metrics (regenerable, git-ignored)
├── reports/            # data dictionary, modeling report, model card, limitations
├── tests/              # pytest tests for the package
└── app/                # Streamlit application (added in a later phase)
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"      # installs the auto_pricing package + pytest/ruff
pytest                        # sanity check: package imports correctly
```

## Status

**Phase 1 complete:** reproducible environment, package skeleton, and import test.
Next: Phase 2, downloading and validating the freMTPL2 frequency/severity data.