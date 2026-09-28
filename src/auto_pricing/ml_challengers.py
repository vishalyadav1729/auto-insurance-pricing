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
from sklearn.inspection import partial_dependence, permutation_importance

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


def fit_severity_boosting(
    train_df: pd.DataFrame,
    max_leaf_nodes: int = 31,
    learning_rate: float = 0.1,
    max_iter: int = 300,
    random_state: int = 42,
) -> HistGradientBoostingRegressor:
    """Fit a Gamma-loss gradient boosting model for claim severity.

    No sample_weight/offset here - matches auto_pricing.severity's Gamma
    and lognormal GLMs: each claim is already one full, independent
    observation of cost, not something that needs exposure-weighting the
    way a policy's claim count does.

    Only ever call this with the TRAINING split of the CLAIM-LEVEL
    severity table (auto_pricing.severity.build_severity_table), not the
    policy-level frequency table.
    """
    X = train_df[ML_FEATURE_COLUMNS]
    y = train_df["ClaimAmount"]

    model = HistGradientBoostingRegressor(
        loss="gamma",
        categorical_features="from_dtype",
        max_leaf_nodes=max_leaf_nodes,
        learning_rate=learning_rate,
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=random_state,
    )
    model.fit(X, y)
    return model


def predict_severity_boosting(model: HistGradientBoostingRegressor, df: pd.DataFrame) -> pd.Series:
    """Predict expected claim cost for each row of `df`, matching
    auto_pricing.severity.predict_severity's contract exactly.
    """
    return pd.Series(model.predict(df[ML_FEATURE_COLUMNS]), index=df.index)


def compute_permutation_importance(
    model: HistGradientBoostingRegressor,
    X: pd.DataFrame,
    y: pd.Series,
    scoring,
    sample_weight: pd.Series | None = None,
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Permutation importance for a fitted boosting model, returned as a
    DataFrame sorted from most to least important.

    Must be called with `X`/`y` from the VALIDATION split, not training -
    the plan is explicit that this should be computed out-of-sample, and
    computing it on training data would just measure which features the
    model overfit to, not which ones actually help it predict new data.

    `scoring` should be a scikit-learn scorer built with
    `greater_is_better=False` for a loss metric (e.g.
    `make_scorer(mean_poisson_deviance, greater_is_better=False)`) -
    otherwise the sign of every importance value is backwards, since
    permutation_importance always interprets a larger score as better.
    """
    result = permutation_importance(
        model,
        X,
        y,
        sample_weight=sample_weight,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )
    return pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False, ignore_index=True)


def compute_partial_dependence(
    model: HistGradientBoostingRegressor, X: pd.DataFrame, feature: str
) -> pd.DataFrame:
    """Partial dependence of `model`'s prediction on a single feature,
    holding all others at their observed distribution.

    Always uses `method="brute"` - confirmed while building this that the
    default `method="recursion"` cannot handle the raw categorical values
    this project's engineered columns use (it requires numeric-encoded
    categories internally and raises a clear error otherwise); "brute"
    re-predicts directly and works correctly with the categorical dtype
    columns fit via `categorical_features="from_dtype"`.

    A plain, unweighted average across the given rows is standard for
    partial dependence, but interpret with the same care the plan asks
    for elsewhere: it can evaluate unrealistic combinations when
    predictors are correlated, and it never establishes causation.
    """
    result = partial_dependence(model, X, features=[feature], method="brute")
    return pd.DataFrame({feature: result["grid_values"][0], "partial_dependence": result["average"][0]})
