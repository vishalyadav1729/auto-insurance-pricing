# SQL Cross-Validation via AWS Athena

This project's README originally promised "SQL-based data preparation," from an earlier,
simpler plan that the 11-phase blueprint superseded — every table in this project has, in
fact, been built with pandas, never SQL (a gap identified and disclosed directly when
asked about it). This closes that gap, but not by bolting SQL onto the existing pipeline
for its own sake: the raw freMTPL2 data was loaded into Amazon S3 and queried via AWS
Athena to **independently re-derive several of this project's own published numbers**,
using a completely different engine (Presto/Trino SQL, not pandas) as a genuine correctness
check — the same "verify independently, don't just trust one method" discipline this
project has applied since Phase 2.

## Setup

- **S3**: raw `freMTPL2freq.csv` (678,013 rows) and `freMTPL2sev.csv` (26,639 rows)
  uploaded unchanged to `s3://riskrate-auto-pricing-data/raw/freq/` and `.../raw/sev/`.
- **Athena**: two external tables (`riskrate.freq_raw`, `riskrate.sev_raw`) defined via
  `CREATE EXTERNAL TABLE` (`sql/create_tables.sql`) — **no Glue Crawlers or Glue ETL jobs**,
  both of which bill per DPU-hour. Athena's DDL registers table metadata in the Glue Data
  Catalog automatically, which is the free part of Glue; it's specifically the Crawlers/Jobs
  that cost money, and neither was used.
- Row counts verified immediately after table creation: `SELECT COUNT(*)` returned exactly
  678,013 and 26,639 — confirming the SerDe parses every row correctly before trusting any
  further query against these tables.

## Results: every cross-check matched exactly

| Check | Established value (pandas, Phase 2/3) | SQL result (Athena) | Match |
|---|---|---|---|
| Portfolio frequency (cleaned, exposure-weighted) | 0.1006 | 0.100614 | Yes |
| Portfolio pure premium (exposure-weighted) | €167.18/policy-year | €167.176 | Yes |
| Total claim value, all 26,639 claims | €60,697,930.68 | €60,697,930.68 | Exact |
| Orphan claim total (6 policies, 195 rows) | €788,714.18 | €788,714.18 | Exact |
| Orphan claim row count | 195 | 195 | Exact |
| Orphan policy count | 6 | 6 | Exact |

The pure-premium query's join also confirms the orphan-exclusion mechanism works
correctly *by construction*, not by an explicit filter: joining `freq_clean` (LEFT JOIN)
to policy-level severity naturally excludes the 6 orphan policies, since they have no
matching `idpol` in `freq_raw` — the resulting total claim amount (€59,909,216.50) equals
the grand total minus exactly the orphan total (€60,697,930.68 − €788,714.18 =
€59,909,216.50), confirming the join behaves as intended without a separate check.

A fourth query aggregates frequency by `BonusMalus` band (the exact bin edges from
`features.py`'s `BONUSMALUS_BINS`) directly in SQL:

| BonusMalus band | Policies | Exposure (policy-years) | Frequency |
|---|---|---|---|
| 50 (best) | 384,156 | 225,152.65 | 0.0800 |
| 51-59 | 77,083 | 39,809.63 | 0.0954 |
| 60-79 | 118,032 | 53,777.10 | 0.1287 |
| 80-99 | 71,418 | 29,809.49 | 0.1431 |
| 100-129 | 26,539 | 9,447.27 | 0.3071 |
| 130+ | 785 | 363.96 | 0.4451 |

Cleanly monotonically increasing, consistent with Phase 3's original univariate finding
and Phase 5's GLM relativity — **but this is deliberately not presented as reproducing
the GLM's relativity table**. A naive per-band exposure-weighted rate (what SQL computes
well) and a multivariate regression coefficient (what a GLM computes, holding every other
factor constant) are different quantities that happen to tell the same qualitative story
here; conflating them would be a genuine mistake, not a shortcut. Refitting the GLM itself
in SQL was never attempted, deliberately — that's a statistical estimation problem, not an
aggregation, and `statsmodels` remains the right tool for it.

## Cost

Every query run while building this — 2 `CREATE EXTERNAL TABLE` statements, 1
`CREATE DATABASE`, and 6 analysis queries — scanned a combined **~163MB**. At Athena's
$5/TB-scanned pricing, that's approximately **$0.0008 total** — under a tenth of a cent.
S3 storage for the ~34MB of raw data (well within the 5GB free tier) adds effectively
nothing on top. No Glue Crawlers, Glue ETL jobs, or QuickSight were used — the three AWS
services identified in advance as the ones that could actually cost real money for this
kind of project.

## Files

- `sql/create_tables.sql` — database and external table DDL.
- `sql/cross_validation_queries.sql` — the four queries behind the results above.
