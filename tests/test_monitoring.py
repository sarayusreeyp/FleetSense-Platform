import pytest
from starlette.testclient import TestClient

from fleetsense.api.main import app
from fleetsense.monitoring import (
    MODEL_LOAD_STATUS,
    PREDICTIONS_TOTAL,
    get_latest_metrics,
    record_batch_predictions,
    record_prediction,
    set_model_status,
)


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_metric_helpers():
    """Verify that Prometheus tracking helper functions observe values correctly."""
    initial_count = PREDICTIONS_TOTAL.labels(model_version="test_v1", status="success")._value.get()
    
    record_prediction(
        predicted_duration_minutes=15.5,
        trip_distance=4.2,
        model_version="test_v1",
        status="success",
    )
    
    new_count = PREDICTIONS_TOTAL.labels(model_version="test_v1", status="success")._value.get()
    assert new_count == initial_count + 1

    # Batch recording
    record_batch_predictions(
        durations=[10.0, 20.0],
        distances=[2.5, 5.0],
        model_version="test_v1",
        status="success",
    )
    batch_count = PREDICTIONS_TOTAL.labels(model_version="test_v1", status="success")._value.get()
    assert batch_count == new_count + 2

    # Model status gauge
    set_model_status(loaded=True, model_name="TestModel")
    assert MODEL_LOAD_STATUS.labels(model_name="TestModel")._value.get() == 1.0

    set_model_status(loaded=False, model_name="TestModel")
    assert MODEL_LOAD_STATUS.labels(model_name="TestModel")._value.get() == 0.0


def test_get_latest_metrics():
    """Verify get_latest_metrics returns bytes and correct content-type header."""
    metrics_data, media_type = get_latest_metrics()
    assert isinstance(metrics_data, bytes)
    assert len(metrics_data) > 0
    assert "text/plain" in media_type or "openmetrics" in media_type


def test_metrics_endpoint_accessible(client: TestClient):
    """Verify GET /metrics endpoint returns 200 and Prometheus telemetry."""
    response = client.get("/metrics")
    assert response.status_code == 200
    content = response.text

    assert "fleetsense_http_requests_total" in content
    assert "fleetsense_http_request_duration_seconds" in content
    assert "fleetsense_model_loaded" in content


def test_metrics_updated_on_predict(client: TestClient):
    """Verify that invoking /predict updates HTTP and prediction metrics."""
    payload = {
        "VendorID": 1,
        "pickup_datetime": "2026-10-05T14:30:00",
        "passenger_count": 2,
        "trip_distance": 3.8,
        "PULocationID": 161,
        "DOLocationID": 236,
    }

    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200

    metrics_resp = client.get("/metrics")
    assert metrics_resp.status_code == 200
    text = metrics_resp.text

    assert "fleetsense_predictions_total" in text
    assert "fleetsense_predicted_duration_minutes" in text
    assert "fleetsense_incoming_trip_distance_miles" in text
