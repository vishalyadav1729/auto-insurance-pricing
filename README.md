# RiskRate: An Interpretable Personal Auto Insurance Pricing Engine

**Live app: [riskrate.streamlit.app](https://riskrate.streamlit.app/)**

An end-to-end actuarial pricing project built from raw claims data to a deployed pricing
app: frequency and severity GLMs, a machine-learning challenger comparison, model
governance and fairness analysis, and an interactive Streamlit application — all on the
public **freMTPL2** French motor third-party-liability dataset (678,013 policies, 26,639
claims).

The goal throughout was not just to fit a model, but to make and defend the judgment calls
a working actuary or pricing analyst actually has to make: which data anomalies to fix and
how, which model family to trust and why, when a machine-learning challenger's better
aggregate metric should *not* win, and what to disclose about a model's limitations rather
than paper over. Every number below is from an executed notebook or test, verified against
independent computation before being written down — see `docs/BUILD_LOG.md` for the
phase-by-phase account, including every bug found and fixed along the way.

## Key results

- **Portfolio pure premium: €167.18/policy-year** (exposure-weighted; naive per-policy
  averaging overstates it by 2.3–2.6×) — the headline number the whole pricing engine is
  built to estimate per-policy, not just on average.
- **`BonusMalus` (prior claims history) is the strongest, cleanest risk signal in the
  data** — a driver at the worst score claims **6.4× as often** and, when they do claim,
  **1.66× as expensively** as a driver at the best score. Independently reconfirmed by a
  completely different model family (gradient boosting, Phase 8).
- **A genuinely surprising finding, checked rather than dismissed**: brand-new vehicles are
  the *highest*-risk vehicle-age group (3–5× the claim frequency of any other age band),
  not the safest — reproduced independently by a second model family.
- **A real data problem, found and resolved, not glossed over**: naively multiplying
  frequency by severity overstated true pure premium by **36.3%**, traced to `ClaimNb`
  counting some claims that were reported but never paid. Fixed with a second,
  purpose-built frequency model targeting *paid* claims — not a bolted-on correction
  factor, which was tested and rejected because the effect varies by segment.
- **Machine learning was evaluated honestly, not favored by default**: gradient boosting
  narrowly beat the GLM on every aggregate metric, but interpretation revealed it silently
  failed to learn the one severity signal (`BonusMalus`) independently validated as real —
  a concrete governance risk this project observed directly. Recommendation:
  champion-challenger, not a single winner (`reports/ml_champion_challenger_recommendation.md`).
- **Limitations disclosed, not hidden**: segment-level calibration is measurably uneven by
  Region (O/E 0.46–1.83) and by driver age (O/E 0.57–1.33), neither explained by sample
  size alone; the dataset has no demographic fields at all, so no fairness/proxy-variable
  audit is actually possible on it — stated plainly rather than implied as checked
  (`reports/limitations.md`, `reports/fairness_analysis.md`).
- **Independently re-derived in SQL, not just pandas**: the raw data was loaded into S3
  and queried via AWS Athena (`CREATE EXTERNAL TABLE`, no Glue Crawlers/Jobs) to re-compute
  the portfolio frequency, pure premium, and orphan-claim figures above from scratch in a
  completely different engine — every number matched exactly. Total cost: under a tenth of
  a cent (`reports/sql_cross_validation.md`).
- **The model card's monitoring recommendation is implemented, not just written down**: a
  scheduled AWS Lambda publishes observed/expected ratio for the 6 worst-known segments to
  CloudWatch, with alarms tuned to the already-known-bad values — all 6 fired exactly as
  designed once the first evaluation period completed (`reports/monitoring_architecture.md`,
  `docs/adr/0001-production-monitoring-architecture.md`). Cost: $0/year.

## The pricing app

**Live: [riskrate.streamlit.app](https://riskrate.streamlit.app/)**

A three-page Streamlit application sits on top of the modelling pipeline — none of this
project's findings live only in markdown:

1. **Pricing Calculator** — prices one user-entered policy through the GLM champion
   pipeline in real time.
2. **Model Exploration** — a live relativity explorer computed from the actual deployed
   model artifacts (not pasted report tables), plus the known calibration gaps and the
   GLM-vs-ML scorecard.
3. **Governance** — the model card, fairness analysis, and limitations documents, rendered
   directly from `reports/` so they can never drift out of sync with the app.

```bash
streamlit run app/app.py
```

*(To run it locally instead, see Setup below.)*

## What this demonstrates

Actuarial and statistical judgment (GLM family selection under overdispersion, exposure
offsets, Duan smearing correction, large-loss sensitivity analysis), rigorous ML practice
(strict train/validation/test discipline — validation for all selection, test touched
exactly once per pipeline to confirm), honest model evaluation (a scorecard across
deviance, calibration, ranking, interpretability, and governance burden — not picking a
winner from one metric), responsible-AI awareness (a model card, a fairness analysis that
states what it cannot verify, a consolidated limitations document), shipping a real,
tested, interactive application on top of it all, and cloud/infrastructure literacy (S3,
Athena, Lambda, EventBridge, CloudWatch; an ADR-driven design process; cost-conscious
architecture decisions — e.g. skipping Glue Crawlers because they bill per DPU-hour where
a plain `CREATE EXTERNAL TABLE` doesn't, and curating monitored segments to stay inside
CloudWatch's free-tier metric allotment rather than publishing everything indiscriminately).

## Repository structure

```text
auto-insurance-pricing/
├── src/auto_pricing/   # reusable package (installed with `pip install -e .`)
├── scripts/            # download_data.py, prepare_data.py, and the rest of the pipeline
├── notebooks/          # analysis notebooks, one per modelling phase; no production logic here
├── data/               # raw/processed data (git-ignored; reproducible via scripts/)
├── artifacts/          # fitted models and preprocessors (git-ignored, regenerable)
├── reports/            # every modelling decision, relativity table, and governance document
├── tests/              # pytest suite (125 tests) covering src/auto_pricing, app/, and lambda/
├── app/                # the three-page Streamlit application (Phase 10)
├── sql/                # Athena DDL and cross-validation queries (AWS S3 + Athena)
├── lambda/             # monitoring Lambda + IAM/dashboard definitions (AWS Lambda + CloudWatch)
└── docs/               # BUILD_LOG.md (build history) and adr/ (architecture decision records)
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"           # installs the auto_pricing package + pytest/ruff
pip install -r requirements.txt   # installs streamlit and the rest of the pipeline's dependencies
pytest                             # sanity check: 125 tests should pass
```

### Running the pricing app

```bash
python scripts/download_data.py
python scripts/prepare_data.py
python scripts/split_data.py
python scripts/build_model_table.py
python scripts/build_pure_premium_table.py
python scripts/fit_severity_model.py
streamlit run app/app.py
```

## Tools

Python, pandas, statsmodels, scikit-learn, Streamlit, Jupyter, pytest, Git and GitHub,
AWS (S3, Athena, Lambda, EventBridge, CloudWatch).

## Further reading

- `docs/BUILD_LOG.md` — the detailed, phase-by-phase build history: every modelling
  decision, every bug found and fixed, and the reasoning behind each.
- `docs/INTERVIEW_PREP.md` — likely interview questions about this project, answered with
  the actual verified numbers and reasoning behind each modelling decision.
- `docs/adr/` — architecture decision records for infrastructure choices (e.g. why
  Lambda+CloudWatch over Glue or QuickSight for production monitoring).
- `reports/` — twenty-one reports covering data cleaning, frequency/severity/pure-premium
  modelling, the ML challenger comparison, model governance, the AWS SQL cross-validation,
  and the AWS monitoring architecture.
- `sql/` — the Athena DDL and cross-validation queries behind `reports/sql_cross_validation.md`.
- `lambda/monitoring/` — the monitoring Lambda, its IAM/dashboard definitions, and
  `deploy.sh` (the exact AWS CLI commands used to build it), behind
  `reports/monitoring_architecture.md`.

## License

[MIT](LICENSE)
- `notebooks/` — the executed analysis notebooks each report is drawn from.
