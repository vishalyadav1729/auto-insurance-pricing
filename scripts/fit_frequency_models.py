"""Fit the Phase 5 frequency models on the training split and persist them.

Fits the Poisson GLM and the Negative Binomial challenger (see
src/auto_pricing/frequency.py for the modelling decisions and the
convergence fix NB required) and saves a small artifact for each to
artifacts/models/ via save_frequency_model - NOT a raw pickle of the
statsmodels results object, which was tried first and produced 878MB/
650MB files for a 48-parameter model. See save_frequency_model's
docstring for why, and predict_from_artifact for how predictions are
reproduced from the small saved artifact alone.

Usage:
    python scripts/fit_frequency_models.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from auto_pricing.frequency import (
    FREQUENCY_FORMULA,
    baseline_frequency,
    dispersion_ratio,
    fit_negative_binomial_glm,
    fit_poisson_glm,
    save_frequency_model,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"


def main() -> None:
    model_table = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "model_table.parquet")
    train = model_table[model_table["split"] == "train"].copy()

    baseline = baseline_frequency(train)
    print(f"baseline frequency: {baseline:.5f}")

    print("fitting Poisson GLM...")
    poisson_result = fit_poisson_glm(train)
    print(f"  converged={poisson_result.converged}  AIC={poisson_result.aic:,.1f}  "
          f"dispersion={dispersion_ratio(poisson_result):.3f}")

    print("fitting Negative Binomial GLM...")
    nb_result = fit_negative_binomial_glm(train)
    print(f"  converged={nb_result.mle_retvals.get('converged')}  AIC={nb_result.aic:,.1f}  "
          f"alpha={nb_result.params['alpha']:.4f} (p={nb_result.pvalues['alpha']:.2e})")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    poisson_path = MODELS_DIR / "frequency_poisson.joblib"
    nb_path = MODELS_DIR / "frequency_negative_binomial.joblib"
    baseline_path = MODELS_DIR / "frequency_baseline.joblib"

    save_frequency_model(poisson_result, FREQUENCY_FORMULA, "poisson", poisson_path)
    save_frequency_model(nb_result, FREQUENCY_FORMULA, "negative_binomial", nb_path)
    joblib.dump({"baseline_frequency": baseline}, baseline_path)

    print()
    for path in [poisson_path, nb_path, baseline_path]:
        print(f"wrote {path}  ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
