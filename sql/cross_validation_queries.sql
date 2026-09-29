-- RiskRate: SQL cross-validation of Phase 2/3's key pandas-computed findings.
--
-- Each query below replicates a specific, already-published pandas result
-- using only Athena SQL against the raw CSVs in S3 - independent
-- verification via a completely different execution engine (Presto/Trino,
-- not pandas), not a re-run of the same code. Results and the side-by-side
-- comparison are in reports/sql_cross_validation.md.
--
-- What this does NOT attempt: refitting the Poisson/lognormal GLMs in SQL.
-- Those are genuine statistical models (maximum-likelihood estimation),
-- not aggregations - SQL is the right tool for data preparation and
-- descriptive cross-checks, not for replacing statsmodels.

-- 1. Portfolio-level exposure-weighted frequency, applying Phase 3's exact
--    cleaning rules (Exposure clipped to 1.0, ClaimNb capped at 4).
--    Expected (reports/, README.md): 0.1006
WITH freq_clean AS (
    SELECT
        idpol,
        LEAST(claimnb, 4) AS claimnb_capped,
        LEAST(exposure, 1.0) AS exposure_clipped
    FROM riskrate.freq_raw
)
SELECT
    SUM(claimnb_capped) AS total_claims,
    SUM(exposure_clipped) AS total_exposure,
    SUM(claimnb_capped) * 1.0 / SUM(exposure_clipped) AS portfolio_frequency
FROM freq_clean;

-- 2. Portfolio-level exposure-weighted pure premium. Joins cleaned frequency
--    to severity aggregated by policy; orphan claims (6 policies, 195 rows,
--    present in sev_raw but absent from freq_raw - cleaning_policy.md rule
--    3) are excluded automatically by the LEFT JOIN direction, not by an
--    explicit filter.
--    Expected (reports/, README.md): EUR 167.18/policy-year
WITH freq_clean AS (
    SELECT
        idpol,
        LEAST(claimnb, 4) AS claimnb_capped,
        LEAST(exposure, 1.0) AS exposure_clipped
    FROM riskrate.freq_raw
),
sev_by_policy AS (
    SELECT idpol, SUM(claimamount) AS claim_amount_sum
    FROM riskrate.sev_raw
    GROUP BY idpol
)
SELECT
    SUM(COALESCE(s.claim_amount_sum, 0)) AS total_claim_amount,
    SUM(f.exposure_clipped) AS total_exposure,
    SUM(COALESCE(s.claim_amount_sum, 0)) * 1.0 / SUM(f.exposure_clipped) AS portfolio_pure_premium
FROM freq_clean f
LEFT JOIN sev_by_policy s ON f.idpol = s.idpol;

-- 3. Naive (unweighted-model) exposure-weighted frequency by BonusMalus
--    band, using the exact bin edges from features.py's BONUSMALUS_BINS.
--    Not a reproduction of the GLM's multivariate relativity (a different
--    kind of computation - regression, not aggregation) - this is the same
--    kind of univariate EDA cross-check Phase 3 did before any model was
--    fit, confirming the same monotonically increasing pattern in SQL.
SELECT
  CASE
    WHEN bonusmalus = 50 THEN '50 (best)'
    WHEN bonusmalus BETWEEN 51 AND 59 THEN '51-59'
    WHEN bonusmalus BETWEEN 60 AND 79 THEN '60-79'
    WHEN bonusmalus BETWEEN 80 AND 99 THEN '80-99'
    WHEN bonusmalus BETWEEN 100 AND 129 THEN '100-129'
    ELSE '130+'
  END AS bonusmalus_band,
  MIN(bonusmalus) AS band_sort_key,
  COUNT(*) AS n_policies,
  SUM(LEAST(exposure,1.0)) AS exposure_years,
  SUM(LEAST(claimnb,4)) * 1.0 / SUM(LEAST(exposure,1.0)) AS band_frequency
FROM riskrate.freq_raw
GROUP BY CASE
    WHEN bonusmalus = 50 THEN '50 (best)'
    WHEN bonusmalus BETWEEN 51 AND 59 THEN '51-59'
    WHEN bonusmalus BETWEEN 60 AND 79 THEN '60-79'
    WHEN bonusmalus BETWEEN 80 AND 99 THEN '80-99'
    WHEN bonusmalus BETWEEN 100 AND 129 THEN '100-129'
    ELSE '130+'
  END
ORDER BY band_sort_key;

-- 4. Direct verification of the total claim value and the orphan-claim
--    figures from cleaning_policy.md rule 3, independent of query 2's join.
--    Expected: total_all_claims EUR 60,697,930.68; orphan_claim_total
--    EUR 788,714.18; orphan_claim_rows 195; orphan_policies 6.
SELECT
    SUM(claimamount) AS total_all_claims,
    SUM(CASE WHEN f.idpol IS NULL THEN s.claimamount ELSE 0 END) AS orphan_claim_total,
    COUNT(CASE WHEN f.idpol IS NULL THEN 1 END) AS orphan_claim_rows,
    COUNT(DISTINCT CASE WHEN f.idpol IS NULL THEN s.idpol END) AS orphan_policies
FROM riskrate.sev_raw s
LEFT JOIN riskrate.freq_raw f ON s.idpol = f.idpol;
