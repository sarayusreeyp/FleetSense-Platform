from datetime import datetime
import pytest
from starlette.testclient import TestClient

from fleetsense.api.main import app


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


def test_health_endpoint(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["app_name"] == "FleetSense"


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
