from typing import Final

import pandas as pd

from fleetsense.core.exceptions.data import DataValidationError


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

    pickup = data["tpep_pickup_datetime"]

    data["pickup_hour"] = pickup.dt.hour
    data["pickup_day_of_week"] = pickup.dt.dayofweek
    data["pickup_month"] = pickup.dt.month
    data["pickup_day"] = pickup.dt.day
    data["is_weekend"] = (
        pickup.dt.dayofweek >= 5
    ).astype(int)

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

    missing_columns = (
        REQUIRED_COLUMNS
        - set(dataframe.columns)
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

    # A 24-hour taxi trip is treated as invalid data.
    if (target > 24 * 60).any():
        raise DataValidationError(
            message=(
                "Trip duration exceeds the maximum "
                "allowed duration of 24 hours."
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