from pathlib import Path
import pandas as pd
import pytest

from fleetsense.core.exceptions.data import DataValidationError
from fleetsense.etl.validation import (
    clean_and_filter_tlc_dataframe,
    validate_tlc_dataframe,
    validate_tlc_file,
)


def valid_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "VendorID": [1, 2],
            "tpep_pickup_datetime": [
                "2026-01-01 10:00:00",
                "2026-01-01 11:00:00",
            ],
            "tpep_dropoff_datetime": [
                "2026-01-01 10:30:00",
                "2026-01-01 11:20:00",
            ],
            "passenger_count": [1, 2],
            "trip_distance": [2.5, 5.0],
            "PULocationID": [100, 200],
            "DOLocationID": [101, 201],
            "payment_type": [1, 1],
            "fare_amount": [10.0, 20.0],
            "total_amount": [12.0, 24.0],
        }
    )


def test_valid_dataframe_passes():
    dataframe = valid_dataframe()
    result = validate_tlc_dataframe(dataframe)
    assert result.equals(dataframe)


def test_empty_dataframe_fails():
    with pytest.raises(DataValidationError):
        validate_tlc_dataframe(pd.DataFrame())


def test_missing_required_column_fails():
    dataframe = valid_dataframe().drop(columns=["trip_distance"])
    with pytest.raises(DataValidationError):
        validate_tlc_dataframe(dataframe)


def test_negative_trip_distance_fails():
    dataframe = valid_dataframe()
    dataframe.loc[0, "trip_distance"] = -5
    with pytest.raises(DataValidationError):
        validate_tlc_dataframe(dataframe)


def test_invalid_passenger_count_fails():
    dataframe = valid_dataframe()
    dataframe.loc[0, "passenger_count"] = 20
    with pytest.raises(DataValidationError):
        validate_tlc_dataframe(dataframe)


def test_invalid_timestamps_fail():
    dataframe = valid_dataframe()
    dataframe.loc[0, "tpep_dropoff_datetime"] = "2026-01-01 09:00:00"
    with pytest.raises(DataValidationError):
        validate_tlc_dataframe(dataframe)


def test_excessive_nulls_fail():
    dataframe = valid_dataframe()
    dataframe["fare_amount"] = None
    with pytest.raises(DataValidationError):
        validate_tlc_dataframe(dataframe)


# --- New Tests for Production Data Cleaning & Filtering ---


def test_clean_and_filter_removes_anomalies():
    noisy_df = pd.DataFrame(
        {
            "VendorID": [1, 1, 1, 1, 1],
            "tpep_pickup_datetime": [
                "2026-01-01 10:00:00",  # 1. Valid row
                "2026-01-01 10:00:00",  # 2. Dist <= 0
                "2026-01-01 10:00:00",  # 3. Passenger = 0
                "2026-01-01 10:00:00",  # 4. Dropoff <= Pickup
                "2026-01-01 10:00:00",  # 5. Negative fare
            ],
            "tpep_dropoff_datetime": [
                "2026-01-01 10:20:00",
                "2026-01-01 10:20:00",
                "2026-01-01 10:20:00",
                "2026-01-01 09:50:00",
                "2026-01-01 10:20:00",
            ],
            "passenger_count": [1, 1, 0, 1, 1],
            "trip_distance": [2.5, 0.0, 3.0, 4.0, 2.0],
            "PULocationID": [100, 100, 100, 100, 100],
            "DOLocationID": [101, 101, 101, 101, 101],
            "payment_type": [1, 1, 1, 1, 1],
            "fare_amount": [15.0, 15.0, 15.0, 15.0, -5.0],
            "total_amount": [18.0, 18.0, 18.0, 18.0, -5.0],
        }
    )

    clean_df, report = clean_and_filter_tlc_dataframe(
        noisy_df, max_allowed_loss_pct=90.0
    )

    # Only row 0 should survive
    assert len(clean_df) == 1
    assert report["initial_rows"] == 5
    assert report["cleaned_rows"] == 1
    assert report["dropped_rows"] == 4
    assert report["loss_percentage"] == 80.0


def test_clean_mode_in_validate_tlc_file(tmp_path: Path):
    df = valid_dataframe()
    # Add one noisy row
    noisy_row = df.iloc[[0]].copy()
    noisy_row["trip_distance"] = 0.0
    combined = pd.concat([df, noisy_row], ignore_index=True)

    test_file = tmp_path / "test_tlc.parquet"
    combined.to_parquet(test_file)

    # In clean mode, it cleans anomalies and returns valid rows
    clean_result = validate_tlc_file(test_file, clean=True, max_allowed_loss_pct=50.0)
    assert len(clean_result) == 2