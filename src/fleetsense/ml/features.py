from typing import Final

import numpy as np
import pandas as pd

from fleetsense.core.exceptions.data import DataValidationError
from fleetsense.core.utils.data_ops import normalize_datetime_columns as _normalize_datetime_columns

TARGET_COLUMN: Final[str] = "trip_duration_minutes"

REQUIRED_COLUMNS: Final[set[str]] = {
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "PULocationID",
    "DOLocationID",
}



def create_features(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Create model features and regression target from TLC data.

    The target is trip duration in minutes.

    Only information available at pickup time is retained
    in the feature dataframe.
    """
    dataframe = _normalize_datetime_columns(dataframe)
    _validate_input(dataframe)

    data = dataframe.copy()

    data[
        "tpep_pickup_datetime"
    ] = pd.to_datetime(
        data["tpep_pickup_datetime"],
        errors="coerce",
    )

    data[
        "tpep_dropoff_datetime"
    ] = pd.to_datetime(
        data["tpep_dropoff_datetime"],
        errors="coerce",
    )

    if (
        data["tpep_pickup_datetime"].isna().any()
        or data["tpep_dropoff_datetime"].isna().any()
    ):
        raise DataValidationError(
            message="Invalid pickup or drop-off timestamps."
        )

    data[TARGET_COLUMN] = (
        (
            data["tpep_dropoff_datetime"]
            - data["tpep_pickup_datetime"]
        ).dt.total_seconds()
        / 60.0
    )

    _validate_target(data[TARGET_COLUMN])

    # Filter out anomalous trip durations and non-physical speeds
    valid_duration = (data[TARGET_COLUMN] >= 1.0) & (data[TARGET_COLUMN] <= 180.0)
    mph = data["trip_distance"] / (data[TARGET_COLUMN] / 60.0)
    valid_speed = (mph <= 75.0) & ((data[TARGET_COLUMN] <= 15.0) | (mph >= 1.0))
    data = data[valid_duration & valid_speed].copy()

    pickup = data["tpep_pickup_datetime"]

    data["pickup_hour"] = pickup.dt.hour
    data["pickup_day_of_week"] = pickup.dt.dayofweek
    data["pickup_month"] = pickup.dt.month
    data["pickup_day"] = pickup.dt.day
    data["is_weekend"] = (
        pickup.dt.dayofweek >= 5
    ).astype(int)

    # Cyclical hour encoding and route interactions
    data["pickup_hour_sin"] = np.sin(2 * np.pi * data["pickup_hour"] / 24.0)
    data["pickup_hour_cos"] = np.cos(2 * np.pi * data["pickup_hour"] / 24.0)
    data["is_rush_hour"] = (
        (data["pickup_day_of_week"] < 5)
        & (data["pickup_hour"].isin([7, 8, 9, 16, 17, 18, 19]))
    ).astype(int)
    # High-cardinality zone pair: represented as category for tree methods that support categorical
    data["PU_DO_pair"] = (
        data["PULocationID"].astype(str) + "_" + data["DOLocationID"].astype(str)
    ).astype("category")

    feature_columns = [
        "VendorID",
        "passenger_count",
        "trip_distance",
        "PULocationID",
        "DOLocationID",
        "pickup_hour",
        "pickup_day_of_week",
        "pickup_month",
        "pickup_day",
        "is_weekend",
        "pickup_hour_sin",
        "pickup_hour_cos",
        "is_rush_hour",
        "PU_DO_pair",
    ]

    available_features = [
        column
        for column in feature_columns
        if column in data.columns
    ]

    X = data[available_features].copy()
    y = data[TARGET_COLUMN].copy()

    _validate_features(X)

    return X, y


def _validate_input(
    dataframe: pd.DataFrame,
) -> None:
    """Validate the input dataframe structure."""

    if not isinstance(dataframe, pd.DataFrame):
        raise DataValidationError(
            message="Input must be a pandas DataFrame."
        )

    if dataframe.empty:
        raise DataValidationError(
            message="Cannot create features from empty data."
        )

    df = _normalize_datetime_columns(dataframe)
    missing_columns = (
        REQUIRED_COLUMNS
        - set(df.columns)
    )

    if missing_columns:
        raise DataValidationError(
            message=(
                "Missing required feature columns: "
                f"{sorted(missing_columns)}"
            )
        )


def _validate_target(
    target: pd.Series,
) -> None:
    """Validate the generated regression target."""

    if target.isna().any():
        raise DataValidationError(
            message="Trip duration contains null values."
        )

    if (target <= 0).any():
        raise DataValidationError(
            message=(
                "Trip duration must be greater than zero."
            )
        )

    # Maximum allowed trip duration is 3 hours (180 mins)
    if (target > 180.0).any():
        raise DataValidationError(
            message=(
                "Trip duration exceeds the maximum "
                "allowed duration of 180 minutes."
            )
        )


def _validate_features(
    features: pd.DataFrame,
) -> None:
    """Validate the final feature matrix."""

    if features.empty:
        raise DataValidationError(
            message="Feature matrix is empty."
        )

    if features.isna().any().any():
        raise DataValidationError(
            message="Feature matrix contains null values."
        )

    if (
        "tpep_dropoff_datetime"
        in features.columns
    ):
        raise DataValidationError(
            message=(
                "Drop-off timestamp detected in "
                "feature matrix. Possible data leakage."
            )
        )
