"""Apply the documented cleaning policy to the raw freMTPL2 tables.

Reads data/raw/, applies every rule in reports/cleaning_policy.md via
auto_pricing.features, and writes the results to data/processed/ as
Parquet (typed, faster to reload than re-parsing CSV every time).

This script does NOT join the two tables into a single modelling table -
that happens after the exploratory analysis (Phase 3 steps 3-6) has a
chance to inform feature-engineering choices, and is a separate, later
script.

Usage:
    python scripts/prepare_data.py
"""

from __future__ import annotations

from pathlib import Path

from auto_pricing.data import load_frequency, load_severity
from auto_pricing.features import clean_frequency, clean_severity

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def main() -> None:
    freq = load_frequency()
    sev = load_severity()

    freq_clean = clean_frequency(freq)
    sev_clean = clean_severity(freq_clean, sev)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    freq_out = PROCESSED_DIR / "frequency_clean.parquet"
    sev_out = PROCESSED_DIR / "severity_clean.parquet"
    freq_clean.to_parquet(freq_out, index=False)
    sev_clean.to_parquet(sev_out, index=False)

    n_exposure_clipped = (freq["Exposure"] > 1.0).sum()
    n_claimnb_capped = (freq["ClaimNb"] > 4).sum()
    n_orphans_excluded = len(sev) - len(sev_clean)
    excluded_value = sev.loc[~sev["IDpol"].isin(freq["IDpol"]), "ClaimAmount"].sum()

    print("=== Cleaning summary (see reports/cleaning_policy.md for rationale) ===")
    print(f"frequency: {len(freq):,} rows in -> {len(freq_clean):,} rows out (no policies removed)")
    print(f"  rule 1  Exposure clipped to 1.0 for {n_exposure_clipped:,} policies")
    print(f"  rule 2a ClaimNb capped to 4 for {n_claimnb_capped:,} policies")
    print(f"  rule 5  VehGas quote characters stripped for all {len(freq_clean):,} rows")
    print()
    print(f"severity: {len(sev):,} rows in -> {len(sev_clean):,} rows out")
    print(f"  rule 3  excluded {n_orphans_excluded:,} orphaned claim rows "
          f"(EUR{excluded_value:,.2f} in claim value, disclosed not discarded)")
    print()
    print(f"wrote {freq_out}")
    print(f"wrote {sev_out}")


if __name__ == "__main__":
    main()
