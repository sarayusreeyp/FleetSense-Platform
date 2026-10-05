from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import joblib
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)

from fleetsense.core.exceptions.data import DataValidationError
from fleetsense.core.exceptions.ml import ModelTrainingError

DEFAULT_N_ESTIMATORS: Final[int] = 100
DEFAULT_RANDOM_STATE: Final[int] = 42
DEFAULT_N_JOBS: Final[int] = -1


def train_model(
    features: pd.DataFrame,
    target: pd.Series,
    algorithm: str = "RandomForestRegressor",
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    random_state: int = DEFAULT_RANDOM_STATE,
    learning_rate: float = 0.1,
    max_depth: int | None = None,
    categorical_features: list[str] | None = None,
    subsample_size: int | None = None,
) -> RegressorMixin:
    """
    Train a FleetSense regression model.

    Supports both RandomForestRegressor and memory-efficient HistGradientBoostingRegressor.
    """
    _validate_training_data(features=features, target=target)

    if n_estimators <= 0:
        raise DataValidationError(
            message="n_estimators must be greater than zero."
        )

    # Subsample if dataset exceeds specified sample limit
    X = features
    y = target
    if subsample_size is not None and len(X) > subsample_size:
        sample_indices = X.sample(n=subsample_size, random_state=random_state).index
        X = X.loc[sample_indices]
        y = y.loc[sample_indices]

    try:
        if algorithm == "HistGradientBoostingRegressor":
            cat_mask = None
            if categorical_features:
                cat_mask = [col in categorical_features for col in X.columns]

            model = HistGradientBoostingRegressor(
                max_iter=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=random_state,
                categorical_features=cat_mask,
            )
        elif algorithm == "RandomForestRegressor":
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=DEFAULT_N_JOBS,
            )
        else:
            raise ValueError(f"Unsupported regression algorithm: '{algorithm}'")

        model.fit(X, y)
        return model

    except Exception as exc:
        if isinstance(exc, (DataValidationError, ValueError)):
            raise
        raise ModelTrainingError(
            message=f"Model training failed with {algorithm}: {exc}",
            details={"algorithm": algorithm, "error": str(exc)},
        ) from exc


def save_model(model: Any, filepath: Path | str) -> Path:
    """Serialize and save a trained model to disk."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(filepath: Path | str) -> Any:
    """Load a serialized model from disk."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


def _validate_training_data(
    features: pd.DataFrame,
    target: pd.Series,
) -> None:
    """Validate data before model training."""
    if not isinstance(features, pd.DataFrame):
        raise DataValidationError(message="Features must be a pandas DataFrame.")

    if not isinstance(target, pd.Series):
        raise DataValidationError(message="Target must be a pandas Series.")

    if features.empty:
        raise DataValidationError(
            message="Cannot train on an empty feature dataframe."
        )

    if target.empty:
        raise DataValidationError(
            message="Cannot train on an empty target series."
        )

    if len(features) != len(target):
        raise DataValidationError(
            message="Features and target must contain the same number of rows."
        )

    if features.isna().any().any():
        raise DataValidationError(
            message="Training features contain null values."
        )

    if target.isna().any():
        raise DataValidationError(
            message="Training target contains null values."
        )

    if not all(
        pd.api.types.is_numeric_dtype(dtype) for dtype in features.dtypes
    ):
        raise DataValidationError(
            message="All training features must be numeric."
        )