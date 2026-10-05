import pytest

from fleetsense.core.exceptions import (
    APIConnectionError,
    DataValidationError,
    FleetSenseError,
    ModelTrainingError,
    StorageConnectionError,
    StorageFileNotFoundError,
)


def test_base_fleetsense_error():
    err = FleetSenseError(
        message="Something went wrong",
        error_code="FS-999",
        details={"file": "sample.csv"},
    )
    assert err.message == "Something went wrong"
    assert err.error_code == "FS-999"
    assert err.details == {"file": "sample.csv"}
    assert "[FS-999] Something went wrong" in str(err)
    assert "'file': 'sample.csv'" in str(err)


def test_api_exceptions_inherit_base():
    err = APIConnectionError("Cannot reach TLC server")
    assert isinstance(err, FleetSenseError)
    assert err.error_code == "API-001"
    assert "Cannot reach TLC server" in str(err)


def test_data_exceptions_inherit_base():
    err = DataValidationError("Invalid parquet schema")
    assert isinstance(err, FleetSenseError)
    assert err.error_code == "DATA-001"


def test_ml_exceptions_inherit_base():
    err = ModelTrainingError("Convergence failed")
    assert isinstance(err, FleetSenseError)
    assert err.error_code == "ML-001"


def test_storage_exceptions_inherit_base():
    err = StorageFileNotFoundError("Missing yellow_tripdata_2026-01.parquet")
    assert isinstance(err, FleetSenseError)
    assert err.error_code == "STORAGE-002"

    conn_err = StorageConnectionError("Azure container unreachable")
    assert isinstance(conn_err, FleetSenseError)
    assert conn_err.error_code == "STORAGE-003"