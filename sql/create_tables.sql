-- RiskRate: Athena external tables over the raw freMTPL2 data in S3.
--
-- Deliberately skips AWS Glue Crawlers and Glue ETL jobs (both bill per
-- DPU-hour, real money) - CREATE EXTERNAL TABLE registers table metadata
-- directly in the Glue Data Catalog for free (the catalog itself has a
-- generous perpetual free tier; it's specifically Crawlers/Jobs that cost).
--
-- Uses the LazySimpleSerDe (plain ROW FORMAT DELIMITED), not OpenCSVSerDe,
-- because none of these columns contain embedded commas - VehGas's stray
-- single-quote characters ('Diesel'/'Regular', a known source-data artifact,
-- see reports/data_dictionary.md Anomaly 3) are literal data, not CSV
-- quoting, so the simple delimiter SerDe parses them correctly as-is while
-- still giving native numeric column types (OpenCSVSerDe would return
-- every column as a string, requiring a CAST on every numeric operation).

CREATE DATABASE IF NOT EXISTS riskrate;

CREATE EXTERNAL TABLE IF NOT EXISTS riskrate.freq_raw (
  idpol double,
  claimnb int,
  exposure double,
  area string,
  vehpower int,
  vehage int,
  drivage int,
  bonusmalus int,
  vehbrand string,
  vehgas string,
  density int,
  region string
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION 's3://riskrate-auto-pricing-data/raw/freq/'
TBLPROPERTIES ('skip.header.line.count'='1');

CREATE EXTERNAL TABLE IF NOT EXISTS riskrate.sev_raw (
  idpol double,
  claimamount double
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION 's3://riskrate-auto-pricing-data/raw/sev/'
TBLPROPERTIES ('skip.header.line.count'='1');
