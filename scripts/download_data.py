"""Download the freMTPL2 French Motor Third-Party Liability datasets.

Fetches the frequency (policy-level) and severity (claim-level) tables from
OpenML via scikit-learn, and writes them to data/raw/ as CSV so the rest of
the pipeline works from local files instead of re-downloading every time.

    freMTPL2freq (OpenML data_id=41214): ~678k policy records with exposure,
        claim counts, and rating factors (driver age, vehicle age/power,
        bonus-malus, vehicle brand, fuel type, area, density, region).
    freMTPL2sev  (OpenML data_id=41215): ~26k individual claim amounts,
        joined to policies via IDpol.

Raw files are written untouched (no cleaning, filtering, or joining here —
that happens in Phase 3) so data/raw/ always reflects the original source.

Usage:
    python scripts/download_data.py [--force]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sklearn.datasets import fetch_openml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

DATASETS = {
    "freMTPL2freq": 41214,
    "freMTPL2sev": 41215,
}


def download_dataset(name: str, data_id: int, force: bool = False) -> Path:
    """Fetch one OpenML dataset and write it to data/raw/<name>.csv."""
    out_path = RAW_DIR / f"{name}.csv"
    if out_path.exists() and not force:
        print(f"[skip] {out_path} already exists (use --force to re-download)")
        return out_path

    print(f"[fetch] {name} (OpenML data_id={data_id}) ...")
    bunch = fetch_openml(data_id=data_id, as_frame=True)
    df = bunch.frame

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[done] wrote {out_path}  shape={df.shape}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download even if the raw CSV already exists",
    )
    args = parser.parse_args()

    for name, data_id in DATASETS.items():
        download_dataset(name, data_id, force=args.force)


if __name__ == "__main__":
    main()
