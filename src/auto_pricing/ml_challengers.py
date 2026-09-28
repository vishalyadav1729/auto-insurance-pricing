"""Machine-learning challengers (Phase 8): gradient boosting compared
against the GLM champions from Phases 5-7.

The plan is explicit that the goal is NOT to prove machine learning is
superior - it's to check whether any incremental predictive value
justifies the added complexity, on a genuinely fair comparison: same
splits, same target definitions, same exposure handling, tuned (not
left at library defaults) using only training/validation data.

`HistGradientBoostingRegressor` has no native "offset" parameter the way
the GLMs in this project do (frequency.py, pure_premium.py). The
standard, correct workaround - the same one used in scikit-learn's own
official insurance-pricing tutorial, this project's methodological
reference point - is to predict the ANNUALIZED RATE directly and pass
`sample_weight=Exposure` to `.fit()`, which weights each policy's
contribution to the loss by its time-at-risk, the same role the offset
plays in a GLM. Verified on synthetic data before trusting it on the
real portfolio: predicted total (rate x exposure, summed) landed within
1% of the observed total.

Uses the engineered categorical columns (DrivAgeBand, VehAgeBand,
BonusMalusBand, RegionGrouped, VehBrandGrouped) NATIVELY, via
`categorical_features="from_dtype"` - these are already proper pandas
Categoricals (Phase 5's fix for single-row prediction), so no separate
one-hot encoding step is needed, unlike the GLMs' patsy-based formulas.
"""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

# Same rating factors as FREQUENCY_FORMULA (frequency.py) / SEVERITY_FORMULA
# (severity.py), as plain column names instead of a patsy formula string -
# sklearn estimators take a feature matrix, not a formula.
ML_FEATURE_COLUMNS = [
    "DrivAgeBand",
    "VehAgeBand",
    "BonusMalusBand",
    "AreaOrdinal",
    "LogDensity",
    "VehGasBinary",
    "RegionGrouped",
    "VehBrandGrouped",
    "VehPower",
]


def fit_frequency_boosting(
    train_df: pd.DataFrame,
    target_col: str = "ClaimNb",
    max_leaf_nodes: int = 31,
    learning_rate: float = 0.1,
    max_iter: int = 300,
    random_state: int = 42,
) -> HistGradientBoostingRegressor:
    """Fit a Poisson-loss gradient boosting model for claim frequency.

    `target_col` defaults to "ClaimNb" (reported claims, matching Phase
    5's GLM champion, for a directly comparable challenger) - pass
    "ClaimNbFromSev" to train the paid-frequency variant needed for a
    pure-premium comparison (Phase 7's distinction between the two
    targets applies here too).

    `early_stopping` uses an internal validation carve-out from
    `train_df` (scikit-learn's own mechanism for choosing the number of
    boosting iterations) - this is separate from and does not touch this
    project's own train/validation/test split; the OFFICIAL validation
    split is still reserved entirely for comparing this model's tuned
    hyperparameters against the GLM champions and other ML configurations.

    Only ever call this with the TRAINING split.
    """
    X = train_df[ML_FEATURE_COLUMNS]
    y_rate = train_df[target_col] / train_df["Exposure"]

    model = HistGradientBoostingRegressor(
        loss="poisson",
        categorical_features="from_dtype",
        max_leaf_nodes=max_leaf_nodes,
        learning_rate=learning_rate,
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=random_state,
    )
    model.fit(X, y_rate, sample_weight=train_df["Exposure"])
    return model


def predict_frequency_boosting(model: HistGradientBoostingRegressor, df: pd.DataFrame) -> pd.Series:
    """Predict expected claim COUNT (not rate) for each row of `df`,
    matching auto_pricing.frequency.predict_frequency's contract exactly
    so both can be passed to the same evaluation functions
    (auto_pricing.evaluation) without any special-casing.

    The model itself predicts an annualized RATE (it never saw Exposure
    as a feature, only as a sample weight) - multiplying by Exposure here
    is what recovers the expected count over each policy's own exposure,
    the gradient-boosting equivalent of a GLM's offset-adjusted predict().
    """
    predicted_rate = model.predict(df[ML_FEATURE_COLUMNS])
    return pd.Series(predicted_rate, index=df.index) * df["Exposure"]
