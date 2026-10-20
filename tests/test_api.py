import pytest
from starlette.testclient import TestClient

from fleetsense.api.main import app
from fleetsense.api.service import PredictionService


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert data["service"] == "FleetSense ML Platform"


def test_health_endpoint_healthy(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("healthy", "degraded")
    assert data["model_loaded"] is True
    assert data["app_name"] == "FleetSense"
    assert "model_source" in data
    assert "is_dummy_model" in data


def test_health_endpoint_unloaded_model(client: TestClient):
    """Gap 23 test: If model is not loaded, health check must report degraded or unavailable."""
    original_service = getattr(app.state, "prediction_service", None)
    try:
        # Mock an uninitialized/broken service
        app.state.prediction_service = None
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unavailable"
        assert data["model_loaded"] is False
    finally:
        app.state.prediction_service = original_service


def test_predict_single_trip(client: TestClient):
    payload = {
        "VendorID": 1,
        "pickup_datetime": "2026-10-05T14:30:00",
        "passenger_count": 2,
        "trip_distance": 3.8,
        "PULocationID": 161,
        "DOLocationID": 236,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "predicted_duration_minutes" in data
    assert data["predicted_duration_minutes"] >= 1.0
    assert "estimated_dropoff_datetime" in data
    assert data["model_algorithm"] is not None
    assert "model_source" in data


def test_predict_batch_trips(client: TestClient):
    batch_payload = {
        "trips": [
            {
                "VendorID": 1,
                "pickup_datetime": "2026-10-05T08:00:00",
                "passenger_count": 1,
                "trip_distance": 2.0,
                "PULocationID": 100,
                "DOLocationID": 101,
            },
            {
                "VendorID": 2,
                "pickup_datetime": "2026-10-05T18:30:00",
                "passenger_count": 3,
                "trip_distance": 6.5,
                "PULocationID": 142,
                "DOLocationID": 237,
            },
        ]
    }

    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["total_trips"] == 2
    assert len(data["predictions"]) == 2
    for pred in data["predictions"]:
        assert pred["predicted_duration_minutes"] >= 1.0


def test_predict_invalid_distance(client: TestClient):
    invalid_payload = {
        "VendorID": 1,
        "pickup_datetime": "2026-10-05T14:30:00",
        "passenger_count": 1,
        "trip_distance": -2.5,  # Invalid: distance must be > 0
        "PULocationID": 161,
        "DOLocationID": 236,
    }

    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422  # Pydantic validation error


def test_predict_distance_exceeding_max(client: TestClient):
    """Gap 24 test: Distance > 100.0 miles must be rejected."""
    invalid_payload = {
        "VendorID": 1,
        "pickup_datetime": "2026-10-05T14:30:00",
        "passenger_count": 1,
        "trip_distance": 150.0,  # Invalid: distance exceeds 100 miles
        "PULocationID": 161,
        "DOLocationID": 236,
    }

    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422


def test_predict_invalid_passenger_count(client: TestClient):
    invalid_payload = {
        "VendorID": 1,
        "pickup_datetime": "2026-10-05T14:30:00",
        "passenger_count": 50,  # Invalid: passenger_count must be <= 9
        "trip_distance": 3.0,
        "PULocationID": 161,
        "DOLocationID": 236,
    }

    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422


def test_predict_invalid_pickup_datetime(client: TestClient):
    """Verify timestamps outside realistic bounds are rejected with 422."""
    payload_ancient = {
        "VendorID": 1,
        "pickup_datetime": "1999-01-01T00:00:00",
        "passenger_count": 1,
        "trip_distance": 3.0,
        "PULocationID": 161,
        "DOLocationID": 236,
    }
    response = client.post("/predict", json=payload_ancient)
    assert response.status_code == 422


def test_prediction_service_remote_storage_load(tmp_path, monkeypatch):
    """Verify PredictionService downloads model from storage backend when local disk has no model."""
    from unittest.mock import MagicMock

    import joblib
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingRegressor

    model = HistGradientBoostingRegressor(max_iter=5).fit(
        pd.DataFrame({
            "VendorID": [1, 2],
            "passenger_count": [1, 2],
            "trip_distance": [1.0, 2.0],
            "PULocationID": [100, 101],
            "DOLocationID": [101, 102],
            "pickup_hour": [9, 10],
            "pickup_day_of_week": [1, 2],
            "pickup_month": [1, 1],
            "pickup_day": [1, 2],
            "is_weekend": [0, 0],
        }),
        pd.Series([10.0, 15.0]),
    )

    mock_storage = MagicMock()
    mock_storage.list_files.return_value = ["latest_model.joblib"]
    mock_storage.download_to_file.side_effect = lambda category, filename, destination_path: joblib.dump(model, destination_path)

    # Ensure local search dirs don't find other models
    empty_dir = tmp_path / "empty_models"
    empty_dir.mkdir()
    monkeypatch.setattr("fleetsense.api.service.MODELS_DIR", empty_dir)
    monkeypatch.setattr("fleetsense.api.service.ARTIFACTS_DIR", empty_dir)
    monkeypatch.setattr("fleetsense.storage.factory.get_storage", lambda: mock_storage)
    monkeypatch.setattr("fleetsense.api.service.MLFLOW_AVAILABLE", False)

    service = PredictionService(model_path=None)
    assert service.model is not None
    assert "storage:" in service.model_source
