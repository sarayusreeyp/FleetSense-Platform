from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from fleetsense.core.exceptions.data import (
    DataValidationError,
    MissingColumnError,
)

REQUIRED_COLUMNS: set[str] = {
    "VendorID",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "PULocationID",
    "DOLocationID",
    "payment_type",
    "fare_amount",
    "total_amount",
}

MAX_NULL_PERCENTAGE: float = 50.0

NON_NEGATIVE_COLUMNS: set[str] = {
    "passenger_count",
    "trip_distance",
    "fare_amount",
    "extra",
    "mta_tax",
    "tip_amount",
    "tolls_amount",
    "improvement_surcharge",
    "total_amount",
    "congestion_surcharge",
    "Airport_fee",
}


def _validate_input_structure(dataframe: pd.DataFrame) -> None:
    if not isinstance(dataframe, pd.DataFrame):
        raise DataValidationError(message="Input must be a pandas DataFrame.")
    if dataframe.empty:
        raise DataValidationError(message="TLC dataset is empty.")


def _validate_required_columns(dataframe: pd.DataFrame) -> None:
    missing_columns = REQUIRED_COLUMNS - set(dataframe.columns)
    if missing_columns:
        raise DataValidationError(
            message=f"Missing required TLC columns: {sorted(missing_columns)}"
        )


def _validate_nulls(dataframe: pd.DataFrame) -> None:
    null_percentages = dataframe.isna().mean() * 100
    invalid_columns = {
        col: round(float(pct), 2)
        for col, pct in null_percentages.items()
        if pct > MAX_NULL_PERCENTAGE
    }
    if invalid_columns:
        raise DataValidationError(
            message=(
                f"Columns exceed the maximum allowed null percentage ({MAX_NULL_PERCENTAGE}%): "
                f"{invalid_columns}"
            )
        )


def _validate_datetime_columns(dataframe: pd.DataFrame) -> None:
    for column in ("tpep_pickup_datetime", "tpep_dropoff_datetime"):
        converted = pd.to_datetime(dataframe[column], errors="coerce")
        invalid_count = int(converted.isna().sum())
        if invalid_count > 0:
            raise DataValidationError(
                message=f"Column '{column}' contains {invalid_count} invalid datetime values."
            )


def _validate_numeric_columns(dataframe: pd.DataFrame) -> None:
    for column in ("passenger_count", "trip_distance", "fare_amount", "total_amount"):
        converted = pd.to_numeric(dataframe[column], errors="coerce")
        invalid_count = int(converted.isna().sum())
        if invalid_count > 0:
            raise DataValidationError(
                message=f"Column '{column}' contains {invalid_count} non-numeric values."
            )


def _validate_non_negative_values(dataframe: pd.DataFrame) -> None:
    for column in NON_NEGATIVE_COLUMNS:
        if column not in dataframe.columns:
            continue
        numeric_values = pd.to_numeric(dataframe[column], errors="coerce")
        negative_count = int((numeric_values < 0).sum())
        if negative_count > 0:
            raise DataValidationError(
                message=f"Column '{column}' contains {negative_count} negative values."
            )


def _validate_passenger_count(dataframe: pd.DataFrame) -> None:
    values = pd.to_numeric(dataframe["passenger_count"], errors="coerce")
    invalid_count = int(((values <= 0) | (values > 9)).sum())
    if invalid_count > 0:
        raise DataValidationError(
            message=f"Invalid passenger counts detected: {invalid_count} rows."
        )


def _validate_trip_distance(dataframe: pd.DataFrame) -> None:
    values = pd.to_numeric(dataframe["trip_distance"], errors="coerce")
    invalid_count = int((values <= 0).sum())
    if invalid_count > 0:
        raise DataValidationError(
            message=f"Trip distance must be greater than zero. Invalid rows: {invalid_count}."
        )


def _validate_trip_timestamps(dataframe: pd.DataFrame) -> None:
    pickup = pd.to_datetime(dataframe["tpep_pickup_datetime"], errors="coerce")
    dropoff = pd.to_datetime(dataframe["tpep_dropoff_datetime"], errors="coerce")
    invalid_count = int((dropoff <= pickup).sum())
    if invalid_count > 0:
        raise DataValidationError(
            message=f"Drop-off timestamp must be after pickup timestamp. Invalid rows: {invalid_count}."
        )


def clean_and_filter_tlc_dataframe(
    dataframe: pd.DataFrame,
    max_allowed_loss_pct: float = 35.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Clean and filter raw TLC data, removing physical anomalies and sensor errors.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Raw TLC dataframe.
    max_allowed_loss_pct : float
        Maximum allowed percentage of dropped rows before raising DataValidationError.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any]]
        Cleaned dataframe and detailed data quality report.
    """
    _validate_input_structure(dataframe)
    _validate_required_columns(dataframe)
    _validate_nulls(dataframe)

    initial_count = len(dataframe)
    df = dataframe.copy()

    # 1. Parse timestamps
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce")
    df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"], errors="coerce")
    valid_timestamps = df["tpep_pickup_datetime"].notna() & df["tpep_dropoff_datetime"].notna()

    # 2. Chronological sequence and duration limits (1 minute to 24 hours)
    duration_minutes = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60.0
    valid_duration = (duration_minutes >= 1.0) & (duration_minutes <= 1440.0)

    # 3. Distance bounds (0.01 to 150 miles)
    trip_dist = pd.to_numeric(df["trip_distance"], errors="coerce")
    valid_distance = (trip_dist > 0.0) & (trip_dist <= 150.0)

    # 4. Passenger count (1 to 9)
    pass_count = pd.to_numeric(df["passenger_count"], errors="coerce")
    valid_passengers = (pass_count >= 1) & (pass_count <= 9)

    # 5. Positive monetary amounts
    fare = pd.to_numeric(df["fare_amount"], errors="coerce")
    total = pd.to_numeric(df["total_amount"], errors="coerce")
    valid_monetary = (fare > 0.0) & (total > 0.0)

    # Combined cleaning mask
    keep_mask = (
        valid_timestamps
        & valid_duration
        & valid_distance
        & valid_passengers
        & valid_monetary
    )

    cleaned_df = df[keep_mask].copy()
    cleaned_count = len(cleaned_df)
    dropped_count = initial_count - cleaned_count
    loss_pct = (dropped_count / initial_count) * 100.0 if initial_count > 0 else 0.0

    report: dict[str, Any] = {
        "initial_rows": initial_count,
        "cleaned_rows": cleaned_count,
        "dropped_rows": dropped_count,
        "loss_percentage": round(loss_pct, 2),
        "dropped_details": {
            "invalid_timestamps": int((~valid_timestamps).sum()),
            "invalid_duration": int((~valid_duration).sum()),
            "invalid_distance": int((~valid_distance).sum()),
            "invalid_passengers": int((~valid_passengers).sum()),
            "invalid_monetary": int((~valid_monetary).sum()),
        },
    }

    if loss_pct > max_allowed_loss_pct:
        raise DataValidationError(
            message=(
                f"Data quality audit failed: {loss_pct:.2f}% of rows dropped, "
                f"exceeding threshold of {max_allowed_loss_pct}%."
            )
        )

    return cleaned_df, report


def validate_tlc_dataframe(
    dataframe: pd.DataFrame,
    clean: bool = False,
    max_allowed_loss_pct: float = 35.0,
) -> pd.DataFrame:
    """
    Validate a TLC trip dataset.

    When clean=True, applies anomaly filtering and returns cleaned data.
    When clean=False, strictly asserts all rows satisfy data quality rules.
    """
    _validate_input_structure(dataframe)
    _validate_required_columns(dataframe)
    _validate_nulls(dataframe)

    if clean:
        cleaned_df, _ = clean_and_filter_tlc_dataframe(
            dataframe, max_allowed_loss_pct=max_allowed_loss_pct
        )
        return cleaned_df

    _validate_datetime_columns(dataframe)
    _validate_numeric_columns(dataframe)
    _validate_non_negative_values(dataframe)
    _validate_passenger_count(dataframe)
    _validate_trip_distance(dataframe)
    _validate_trip_timestamps(dataframe)

    return dataframe


def validate_tlc_file(
    file_path: Path | str,
    clean: bool = False,
    max_allowed_loss_pct: float = 35.0,
) -> pd.DataFrame:
    """
    Read and validate/clean one TLC Parquet file.
    """
    path = Path(file_path)

    if not path.exists():
        raise DataValidationError(message=f"Data file does not exist: {path}")

    if path.suffix.lower() != ".parquet":
        raise DataValidationError(
            message=f"Expected a Parquet file, received: {path.suffix}"
        )

    try:
        dataframe = pd.read_parquet(path)
    except Exception as exc:
        raise DataValidationError(
            message=f"Unable to read Parquet file: {path}"
        ) from exc

    return validate_tlc_dataframe(
        dataframe,
        clean=clean,
        max_allowed_loss_pct=max_allowed_loss_pct,
    )