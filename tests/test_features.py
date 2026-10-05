import pandas as pd
import pytest

from fleetsense.core.exceptions.data import DataValidationError
from fleetsense.ml.features import (
    TARGET_COLUMN,
    create_features,
)


def sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "VendorID": [1, 2, 1],
            "tpep_pickup_datetime": [
                "2026-01-01 08:00:00",
                "2026-01-02 18:30:00",
                "2026-01-03 12:00:00",
            ],
            "tpep_dropoff_datetime": [
                "2026-01-01 08:20:00",
                "2026-01-02 19:00:00",
                "2026-01-03 12:45:00",
            ],
            "passenger_count": [1, 2, 1],
            "trip_distance": [2.5, 5.0, 3.0],
            "PULocationID": [100, 200, 300],
            "DOLocationID": [101, 201, 301],
        }
    )


def test_create_features_returns_x_and_y():
    X, y = create_features(
        sample_dataframe()
    )

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)

    assert len(X) == 3
    assert len(y) == 3


def test_trip_duration_is_correct():
    X, y = create_features(
        sample_dataframe()
    )

    assert y.tolist() == [
        20.0,
        30.0,
        45.0,
    ]


def test_pickup_features_are_created():
    X, y = create_features(
        sample_dataframe()
    )

    assert "pickup_hour" in X.columns
    assert "pickup_day_of_week" in X.columns
    assert "pickup_month" in X.columns
    assert "pickup_day" in X.columns
    assert "is_weekend" in X.columns


def test_dropoff_information_is_not_in_features():
    X, y = create_features(
        sample_dataframe()
    )

    assert (
        "tpep_dropoff_datetime"
        not in X.columns
    )

    assert TARGET_COLUMN not in X.columns


def test_weekend_feature_is_correct():
    X, y = create_features(
        sample_dataframe()
    )

    # 2026-01-03 is Saturday.
    assert X.iloc[2]["is_weekend"] == 1

    # 2026-01-01 is Thursday.
    assert X.iloc[0]["is_weekend"] == 0


def test_missing_required_column_fails():
    dataframe = sample_dataframe().drop(
        columns=["trip_distance"]
    )

    with pytest.raises(DataValidationError):
        create_features(dataframe)


def test_negative_duration_fails():
    dataframe = sample_dataframe()

    dataframe.loc[
        0,
        "tpep_dropoff_datetime",
    ] = "2026-01-01 07:00:00"

    with pytest.raises(DataValidationError):
        create_features(dataframe)


def test_excessive_duration_fails():
    dataframe = sample_dataframe()

    dataframe.loc[
        0,
        "tpep_dropoff_datetime",
    ] = "2026-01-02 09:00:01"

    with pytest.raises(DataValidationError):
        create_features(dataframe)