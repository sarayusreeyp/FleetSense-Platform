import collections
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
from starlette.testclient import TestClient

from fleetsense.api.main import InMemoryRateLimiter, _get_client_ip, app
from fleetsense.etl.validation import clean_and_filter_tlc_dataframe, validate_tlc_dataframe
from fleetsense.ml.features import create_features
from fleetsense.ml.tracking import MLflowTracker
from fleetsense.monitoring.drift import load_reference_distribution


def test_rate_limiter_evicts_stale_ips():
    """Verify InMemoryRateLimiter removes expired IPs to prevent unbounded memory growth."""
    limiter = InMemoryRateLimiter(requests_per_minute=2)
    # Add an IP with an old timestamp
    limiter._history["1.2.3.4"] = limiter._history.get("1.2.3.4", [])
    import collections
    import time
    limiter._history["1.2.3.4"] = collections.deque([time.time() - 70.0])
    limiter._last_cleanup = time.time() - 65.0  # trigger cleanup

    assert limiter.check("5.6.7.8") is True
    # The stale IP 1.2.3.4 should have been pruned
    assert "1.2.3.4" not in limiter._history


def test_get_client_ip_with_x_forwarded_for():
    """Verify _get_client_ip properly parses X-Forwarded-For headers from reverse proxies."""
    mock_request = MagicMock()
    mock_request.headers.get.side_effect = lambda h: "203.0.113.195, 70.41.3.18, 150.172.238.178" if h == "X-Forwarded-For" else None
    ip = _get_client_ip(mock_request)
    assert ip == "203.0.113.195"


def test_batch_prediction_max_length():
    """Verify BatchPredictionRequest rejects payloads exceeding 1000 trips."""
    client = TestClient(app)
    # Create 1001 valid trip payloads
    trip = {
        "VendorID": 1,
        "pickup_datetime": "2026-10-05T14:30:00",
        "passenger_count": 1,
        "trip_distance": 2.5,
        "PULocationID": 161,
        "DOLocationID": 236,
    }
    oversized_batch = {"trips": [trip] * 1001}
    response = client.post("/predict/batch", json=oversized_batch)
    assert response.status_code == 422  # Unprocessable Entity from Pydantic


def test_green_taxi_column_normalization():
    """Verify that Green taxi datasets with lpep_* columns are accepted and processed."""
    green_df = pd.DataFrame(
        {
            "VendorID": [2],
            "lpep_pickup_datetime": ["2026-02-01 10:00:00"],
            "lpep_dropoff_datetime": ["2026-02-01 10:25:00"],
            "passenger_count": [1],
            "trip_distance": [3.2],
            "PULocationID": [75],
            "DOLocationID": [238],
            "payment_type": [1],
            "fare_amount": [14.0],
            "total_amount": [17.5],
        }
    )
    validated = validate_tlc_dataframe(green_df)
    assert "tpep_pickup_datetime" in validated.columns

    cleaned, report = clean_and_filter_tlc_dataframe(green_df)
    assert report["cleaned_rows"] == 1

    X, y = create_features(green_df)
    assert len(X) == 1
    assert y.iloc[0] == 25.0


def test_champion_challenger_comparison_lower_is_better():
    """Verify that when higher_is_better=False, a worse model (higher MAE) does NOT beat champion."""
    tracker = MLflowTracker()
    tracker.enabled = True
    tracker.active_run = MagicMock()

    mock_client = MagicMock()
    mock_champion = MagicMock()
    mock_champion.run_id = "champ_run_123"
    mock_champion.version = "1"
    mock_client.get_model_version_by_alias.return_value = mock_champion

    mock_run = MagicMock()
    mock_run.data.metrics.get.return_value = 3.0  # Champion val_mae = 3.0
    mock_client.get_run.return_value = mock_run

    with patch("mlflow.sklearn.log_model") as mock_log_model, \
         patch("mlflow.tracking.MlflowClient", return_value=mock_client):
        mock_info = MagicMock()
        mock_info.registered_model_version = "2"
        mock_info.model_uri = "models:/test/2"
        mock_log_model.return_value = mock_info

        # Challenger has val_mae = 3.10 (worse than 3.0)
        tracker.log_and_register_model(
            model=MagicMock(),
            registered_model_name="test_model",
            alias="champion",
            challenger_metric=3.10,
            metric_name="val_mae",
            higher_is_better=False,
            tolerance=0.0,
        )

        # Must NOT be promoted to champion; should be tagged as challenger
        mock_client.set_registered_model_alias.assert_called_with(
            name="test_model",
            alias="challenger",
            version="2",
        )


def test_drift_reference_distribution_cached():
    """Verify reference distribution is cached in memory across calls."""
    b1, f1 = load_reference_distribution()
    b2, f2 = load_reference_distribution()
    assert b1 == b2
    assert f1 == f2


def test_storage_list_files_ignores_hidden_and_temp(tmp_path):
    """Verify LocalStorage.list_files ignores .gitkeep, hidden files and .tmp files."""
    from fleetsense.storage.local import LocalStorage
    storage = LocalStorage(root_dir=tmp_path)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / ".gitkeep").write_text("")
    (raw_dir / ".DS_Store").write_text("")
    (raw_dir / "tripdata.parquet.tmp").write_text("")
    (raw_dir / "valid_tripdata.parquet").write_text("dummy")

    files = storage.list_files("raw")
    assert files == ["valid_tripdata.parquet"]


def test_predict_returns_503_when_model_is_unloaded():
    """Verify /predict returns HTTP 503 Service Unavailable when model is not loaded."""
    client = TestClient(app)
    mock_service = MagicMock()
    mock_service.model = None

    orig_service = getattr(app.state, "prediction_service", None)
    try:
        app.state.prediction_service = mock_service
        payload = {
            "VendorID": 1,
            "pickup_datetime": "2026-10-05T14:30:00",
            "passenger_count": 1,
            "trip_distance": 2.5,
            "PULocationID": 161,
            "DOLocationID": 236,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 503
        assert "Inference model is not currently loaded" in response.json()["detail"]

        batch_response = client.post("/predict/batch", json={"trips": [payload]})
        assert batch_response.status_code == 503
    finally:
        app.state.prediction_service = orig_service


def test_model_training_error_accepts_details():
    """Verify ModelTrainingError and other ML exceptions accept details dictionary without TypeError."""
    from fleetsense.core.exceptions.ml import (
        FeatureEngineeringError,
        ModelInferenceError,
        ModelTrainingError,
    )

    err = ModelTrainingError("Training crashed", details={"algo": "HGBR", "error": "sample"})
    assert err.message == "Training crashed"
    assert err.error_code == "ML-001"
    assert err.details == {"algo": "HGBR", "error": "sample"}

    fe_err = FeatureEngineeringError("Feature failed", details={"step": "scaling"})
    assert fe_err.details == {"step": "scaling"}

    mi_err = ModelInferenceError("Inference failed", details={"row": 5})
    assert mi_err.details == {"row": 5}


def test_models_dir_exported_in_constants():
    """Verify MODELS_DIR is properly exposed in fleetsense.core.constants."""
    import fleetsense.core.constants as const
    assert hasattr(const, "MODELS_DIR")
    assert "MODELS_DIR" in const.__all__
    assert const.MODELS_DIR.name == "models"


def test_drift_throttle_at_full_window():
    """Verify drift detection throttles evaluation when window is full."""
    from fleetsense.monitoring import drift

    orig_window = drift._distance_window
    orig_counter = drift._sample_counter
    try:
        drift._distance_window = collections.deque(maxlen=drift._MIN_WINDOW_SIZE * 5)
        drift._sample_counter = 0

        # Pre-fill window past maxlen
        for _ in range(600):
            drift._distance_window.append(2.0)
            drift._sample_counter += 1

        with patch("fleetsense.monitoring.drift._compute_psi", return_value=0.05) as mock_psi:
            # 601 % 10 != 0 -> should NOT evaluate psi
            drift.check_and_alert_drift(2.5)
            mock_psi.assert_not_called()

            # Advance to multiple of 10 -> should evaluate psi
            for _ in range(8):
                drift.check_and_alert_drift(2.5)
            mock_psi.assert_not_called()

            # Now sample_counter is multiple of 10
            drift.check_and_alert_drift(2.5)
            assert mock_psi.call_count == 1
    finally:
        drift._distance_window = orig_window
        drift._sample_counter = orig_counter


def test_processed_and_features_state_management(tmp_path):
    """Verify ProcessedMetadata and FeaturesMetadata record and query state properly."""
    from fleetsense.etl.metadata import FeaturesMetadata, ProcessedMetadata

    proc_path = tmp_path / "processed_state.json"
    proc_meta = ProcessedMetadata(metadata_path=proc_path)
    assert not proc_meta.is_processed("yellow_tripdata_2026-01.parquet")

    proc_meta.mark_processed_file(
        filename="yellow_tripdata_2026-01.parquet",
        dataset="yellow",
        report={"cleaned_rows": 100, "loss_percentage": 5.0},
        processed_path="data/processed/yellow_tripdata_2026-01.parquet",
    )
    assert proc_meta.is_processed("yellow_tripdata_2026-01.parquet")

    feat_path = tmp_path / "features_state.json"
    feat_meta = FeaturesMetadata(metadata_path=feat_path)
    assert not feat_meta.is_feature_generated("yellow_tripdata_2026-01.parquet")

    feat_meta.mark_feature_generated(
        filename="yellow_tripdata_2026-01.parquet",
        dataset="yellow",
        row_count=100,
        feature_columns=["trip_distance", "VendorID"],
        features_path="data/features/yellow_tripdata_2026-01.parquet",
    )
    assert feat_meta.is_feature_generated("yellow_tripdata_2026-01.parquet")


def test_manage_storage_champion_artifacts_archives_previous_champion():
    """Verify that when a new champion arrives, previous champion is archived and deleted."""
    from fleetsense.workflows.train import _manage_storage_champion_artifacts

    mock_storage = MagicMock()
    mock_storage.list_files.return_value = ["old_champion.joblib"]

    with patch("fleetsense.workflows.train.get_storage", return_value=mock_storage), \
         patch.dict("os.environ", {"FLEETSENSE_STORAGE_BACKEND": "azure"}):

        _manage_storage_champion_artifacts(
            model_name="new_model.joblib",
            local_model_path=Path("models/new_model.joblib"),
            latest_model_path=Path("models/latest_model.joblib"),
            is_champion=True,
        )

        mock_storage.container_client.get_blob_client.assert_called()
        mock_storage.delete.assert_called_with("models/champion", "old_champion.joblib")
        mock_storage.save_file.assert_any_call(
            Path("models/new_model.joblib"),
            category="models/champion",
            filename="trip_duration_model.joblib",
        )




