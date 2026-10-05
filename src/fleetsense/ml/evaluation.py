from typing import Any

import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from fleetsense.core.exceptions.data import DataValidationError


def evaluate_model(
    model: RegressorMixin,
    features: pd.DataFrame,
    target: pd.Series,
) -> dict[str, float]:
    """
    Evaluate a trained regression model.

    Returns:
        Dictionary containing MAE, RMSE, and R2.
    """

    _validate_evaluation_data(
        model=model,
        features=features,
        target=target,
    )

    predictions = model.predict(features)

    mae = mean_absolute_error(
        target,
        predictions,
    )

    mse = mean_squared_error(
        target,
        predictions,
    )

    rmse = mse ** 0.5

    r2 = r2_score(
        target,
        predictions,
    )

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }


def _validate_evaluation_data(
    model: Any,
    features: pd.DataFrame,
    target: pd.Series,
) -> None:
    """Validate inputs before model evaluation."""

    if not hasattr(model, "predict"):
        raise DataValidationError(
            message="Model must provide a predict method."
        )

    if not isinstance(features, pd.DataFrame):
        raise DataValidationError(
            message="Features must be a pandas DataFrame."
        )

    if not isinstance(target, pd.Series):
        raise DataValidationError(
            message="Target must be a pandas Series."
        )

    if features.empty:
        raise DataValidationError(
            message="Cannot evaluate with empty features."
        )

    if target.empty:
        raise DataValidationError(
            message="Cannot evaluate with empty target."
        )

    if len(features) != len(target):
        raise DataValidationError(
            message=(
                "Features and target must contain "
                "the same number of rows."
            )
        )

    if features.isna().any().any():
        raise DataValidationError(
            message="Evaluation features contain null values."
        )

    if target.isna().any():
        raise DataValidationError(
            message="Evaluation target contains null values."
        )