from __future__ import annotations

import os
import threading
from datetime import timedelta
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from fleetsense.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionResponse,
    TripFeaturesPayload,
)
from fleetsense.core.config import settings
from fleetsense.core.constants.paths import ARTIFACTS_DIR, DATA_DIR, MODELS_DIR
from fleetsense.core.logger import get_logger

logger = get_logger(__name__)

try:
    import mlflow
    import mlflow.pyfunc
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class PredictionService:
    """
    Manages model loading, real-time feature extraction, and inference.

    Model resolution order:
      1. MLflow Model Registry URI (if remote MLFLOW_TRACKING_URI is provided)
      2. Remote Cloud Storage (Azure Blob Storage 'models/champion/...')
      3. Explicit model_path argument
      4. Local ARTIFACTS_DIR / DATA_DIR disk artifacts
      5. Fallback: warn loudly and raise RuntimeError in production
    """

    #: Set to True only during unit tests to permit the lightweight dummy fallback.
    _allow_dummy_fallback: bool = False

    def __init__(self, model_path: Path | str | None = None) -> None:
        self.model = None
        self.algorithm = settings.model.algorithm
        self.model_version = settings.app.version
        resolved_path = model_path or os.getenv("FLEETSENSE_MODEL_ARTIFACT_PATH")
        self.model_path = Path(resolved_path) if resolved_path else None
        self.model_source: str = "unloaded"
        self.mlflow_run_id: str | None = None
        self.mlflow_model_version: str | None = None
        self._lock = threading.Lock()
        self._load_or_initialize_model()

    def _load_or_initialize_model(self) -> None:
        """Load trained model artifact from MLflow Model Registry or local artifacts."""

        # 1. Attempt loading directly from MLflow Model Registry
        mlflow_model_uri = os.getenv("MLFLOW_MODEL_URI")
        candidate_uris: list[str] = []
        if mlflow_model_uri:
            candidate_uris.append(mlflow_model_uri)
        else:
            model_name = settings.model.name
            alias = os.getenv("MLFLOW_MODEL_ALIAS", "champion")
            # Support both modern alias syntax (@champion) and stage/version
            if not alias.startswith("@") and alias in ("champion", "challenger"):
                candidate_uris.append(f"models:/{model_name}@{alias}")
            candidate_uris.append(f"models:/{model_name}/{alias}")
            candidate_uris.append(f"models:/{model_name}/latest")

        if MLFLOW_AVAILABLE:
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or settings.model.mlflow_tracking_uri
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            for uri in candidate_uris:
                try:
                    logger.info(
                        "Attempting to load model from MLflow Model Registry: %s (tracking_uri=%s)...",
                        uri,
                        tracking_uri,
                    )
                    self.model = mlflow.sklearn.load_model(uri)
                    self.model_source = f"mlflow:{uri}"
                    # Extract MLflow lineage metadata
                    try:
                        from mlflow.tracking import MlflowClient
                        client = MlflowClient(tracking_uri=tracking_uri)
                        model_name = settings.model.name
                        alias = os.getenv("MLFLOW_MODEL_ALIAS", "champion")
                        try:
                            mv = client.get_model_version_by_alias(model_name, alias)
                            self.mlflow_run_id = mv.run_id
                            self.mlflow_model_version = mv.version
                        except Exception:
                            pass
                    except Exception:
                        pass
                    logger.info("Successfully loaded model from MLflow Model Registry: %s", uri)
                    return
                except Exception as exc:
                    logger.info("Could not load from MLflow Model Registry (%s): %s", uri, exc)

        # 2. Check Azure Blob Storage for remote Champion model
        try:
            from fleetsense.storage.factory import get_storage
            active_storage = get_storage()
            remote_model_filename = os.getenv("FLEETSENSE_REMOTE_MODEL_FILENAME", "trip_duration_model.joblib")
            remote_category = os.getenv("FLEETSENSE_REMOTE_MODEL_CATEGORY", "models/champion")
            
            # Check if this remote model exists in storage
            if hasattr(active_storage, "download_to_file") and active_storage.exists(f"{remote_category}/{remote_model_filename}"):
                cache_dir = MODELS_DIR / "cache"
                cache_dir.mkdir(parents=True, exist_ok=True)
                local_cache_path = cache_dir / remote_model_filename
                
                logger.info(
                    "Downloading champion model from %s/%s to local cache %s...",
                    remote_category,
                    remote_model_filename,
                    local_cache_path,
                )
                active_storage.download_to_file(
                    category=remote_category,
                    filename=remote_model_filename,
                    destination_path=local_cache_path,
                )
                self.model = joblib.load(local_cache_path)
                self.model_path = local_cache_path
                self.model_source = f"storage:{remote_category}/{remote_model_filename}"
                logger.info("Successfully loaded champion model from cloud storage (%s)", self.model_source)
                return
        except Exception as exc:
            logger.info("Could not load model from cloud storage: %s", exc)

        # 3. Check explicit path (passed in or from FLEETSENSE_MODEL_ARTIFACT_PATH)
        if self.model_path and self.model_path.exists():
            try:
                self.model = joblib.load(self.model_path)
                self.model_source = f"local:{self.model_path}"
                logger.info("Loaded model from explicit path %s", self.model_path)
                return
            except Exception as exc:
                logger.warning("Failed to load model from %s: %s", self.model_path, exc)

        # 3. Search local directories (prefer latest_model.joblib or most recently modified)
        search_dirs = [MODELS_DIR, ARTIFACTS_DIR, DATA_DIR / "models"]
        for search_dir in search_dirs:
            if search_dir.exists():
                latest_pointer = search_dir / "latest_model.joblib"
                if latest_pointer.exists():
                    try:
                        self.model = joblib.load(latest_pointer)
                        self.model_path = latest_pointer
                        self.model_source = f"local:{latest_pointer}"
                        logger.info("Loaded model from latest pointer %s", latest_pointer)
                        return
                    except Exception as exc:
                        logger.warning("Failed loading %s: %s", latest_pointer, exc)

                model_files = sorted(
                    [p for p in search_dir.glob("*.joblib") if p.name != "latest_model.joblib"],
                    key=lambda p: p.stat().st_mtime,
                )
                if model_files:
                    latest_model = model_files[-1]
                    try:
                        self.model = joblib.load(latest_model)
                        self.model_path = latest_model
                        self.model_source = f"local:{latest_model}"
                        logger.info("Loaded model from local dir %s", latest_model)
                        return
                    except Exception as exc:
                        logger.warning("Failed loading %s: %s", latest_model, exc)

        # 4. Check remote storage backend (e.g. Azure Blob Storage) for models
        try:
            from fleetsense.storage.factory import get_storage
            storage = get_storage()
            remote_models = sorted(storage.list_files("models"))
            if remote_models:
                MODELS_DIR.mkdir(parents=True, exist_ok=True)
                target_model_file = "latest_model.joblib" if "latest_model.joblib" in remote_models else remote_models[-1]
                cached_model_path = MODELS_DIR / target_model_file
                logger.info("Downloading model '%s' from storage backend to %s...", target_model_file, cached_model_path)
                storage.download_to_file(
                    category="models",
                    filename=target_model_file,
                    destination_path=cached_model_path,
                )
                self.model = joblib.load(cached_model_path)
                self.model_path = cached_model_path
                self.model_source = f"storage:{target_model_file}"
                logger.info("Successfully loaded model from remote storage: %s", target_model_file)
                return
        except Exception as storage_exc:
            logger.debug("Storage backend model loading skipped/failed: %s", storage_exc)

        # 5. Dummy fallback — ONLY allowed in test environments
        if self._allow_dummy_fallback or os.getenv("FLEETSENSE_APP_ENV") in ("testing", "test"):
            logger.warning(
                "No real model found. Fitting a MINIMAL dummy baseline. "
                "This is only acceptable in testing environments."
            )
            baseline = HistGradientBoostingRegressor(
                max_iter=20,
                random_state=settings.model.random_state,
            )
            X_dummy = pd.DataFrame(
                {
                    "VendorID": [1, 2, 1, 2],
                    "passenger_count": [1, 1, 2, 2],
                    "trip_distance": [1.5, 3.0, 5.0, 8.0],
                    "PULocationID": [161, 236, 142, 237],
                    "DOLocationID": [236, 161, 237, 142],
                    "pickup_hour": [9, 14, 18, 22],
                    "pickup_day_of_week": [1, 2, 4, 5],
                    "pickup_month": [10, 10, 10, 10],
                    "pickup_day": [5, 5, 5, 5],
                    "is_weekend": [0, 0, 0, 1],
                }
            )
            y_dummy = pd.Series([10.0, 18.0, 28.0, 40.0])
            baseline.fit(X_dummy, y_dummy)
            self.model = baseline
            self.model_source = "dummy:baseline"
            return

        # 5. No model found and strict mode enabled — fail loudly
        if os.getenv("STRICT_MODEL_LOAD", "false").lower() == "true":
            raise RuntimeError(
                "FleetSense API failed to start: no trained model could be loaded.\n"
                "Checked locations:\n"
                f"  - MLflow Model Registry: {mlflow_model_uri}\n"
                f"  - Local ARTIFACTS_DIR: {ARTIFACTS_DIR}\n"
                f"  - Local DATA_DIR/models: {DATA_DIR / 'models'}\n\n"
                "Train and register a model first by running: python -m fleetsense.workflows.train\n"
                "Or configure MLFLOW_TRACKING_URI to point to your MLflow server."
            )

        # Graceful startup without model — API stays alive, reports health as 'unavailable'
        logger.warning(
            "FleetSense API initialized without a trained model artifact. "
            "Endpoints /predict and /predict/batch will return errors until a model is deployed. "
            "Health status is 'unavailable'."
        )
        self.model = None
        self.model_source = "unloaded"

    @property
    def is_dummy(self) -> bool:
        """True if the service is running on a dummy model (not a real trained artifact)."""
        return self.model_source.startswith("dummy:")

    def reload_model(self) -> dict[str, str]:
        """Thread-safe hot-reload of the model from MLflow."""
        with self._lock:
            old_source = self.model_source
            old_version = self.mlflow_model_version
            self.model = None
            self.model_source = "unloaded"
            self.mlflow_run_id = None
            self.mlflow_model_version = None
            self._load_or_initialize_model()
            return {
                "previous_source": old_source,
                "previous_version": old_version or "unknown",
                "new_source": self.model_source,
                "new_version": self.mlflow_model_version or "unknown",
                "status": "reloaded" if self.model is not None else "failed",
            }

    def extract_features(
        self,
        trips: list[TripFeaturesPayload],
    ) -> pd.DataFrame:
        """
        Transform API trip payloads into aligned feature vectors dynamically
        matching the expected feature schema of the loaded model.
        """
        records = []
        for trip in trips:
            dt = trip.pickup_datetime
            records.append(
                {
                    "VendorID": trip.VendorID,
                    "passenger_count": trip.passenger_count,
                    "trip_distance": trip.trip_distance,
                    "PULocationID": trip.PULocationID,
                    "DOLocationID": trip.DOLocationID,
                    "pickup_hour": dt.hour,
                    "pickup_day_of_week": dt.weekday(),
                    "pickup_month": dt.month,
                    "pickup_day": dt.day,
                    "is_weekend": 1 if dt.weekday() >= 5 else 0,
                }
            )

        features_df = pd.DataFrame(records)

        # Dynamic feature alignment to the loaded model
        expected_features = (
            list(self.model.feature_names_in_)
            if hasattr(self.model, "feature_names_in_")
            else settings.model.features
        )

        for col in expected_features:
            if col not in features_df.columns:
                features_df[col] = 0

        return features_df[expected_features]

    def predict_trip(self, trip: TripFeaturesPayload) -> PredictionResponse:
        """Predict duration for a single trip."""
        if self.model is None:
            raise RuntimeError("Inference model is not currently loaded on the server.")

        X = self.extract_features([trip])
        pred_duration = float(self.model.predict(X)[0])
        pred_duration = max(1.0, round(pred_duration, 2))

        estimated_dropoff = trip.pickup_datetime + timedelta(
            minutes=pred_duration
        )

        return PredictionResponse(
            predicted_duration_minutes=pred_duration,
            pickup_datetime=trip.pickup_datetime,
            estimated_dropoff_datetime=estimated_dropoff,
            model_version=self.model_version,
            model_algorithm=self.algorithm,
            model_source=self.model_source,
            mlflow_run_id=self.mlflow_run_id,
        )

    def predict_batch(
        self,
        request: BatchPredictionRequest,
    ) -> BatchPredictionResponse:
        """Predict durations for multiple trips in a vectorized pass."""
        if self.model is None:
            raise RuntimeError("Inference model is not currently loaded on the server.")

        X = self.extract_features(request.trips)
        raw_predictions = self.model.predict(X)

        responses = []
        for trip, raw_pred in zip(request.trips, raw_predictions):
            duration = max(1.0, round(float(raw_pred), 2))
            estimated_dropoff = trip.pickup_datetime + timedelta(
                minutes=duration
            )
            responses.append(
                PredictionResponse(
                    predicted_duration_minutes=duration,
                    pickup_datetime=trip.pickup_datetime,
                    estimated_dropoff_datetime=estimated_dropoff,
                    model_version=self.model_version,
                    model_algorithm=self.algorithm,
                    model_source=self.model_source,
                    mlflow_run_id=self.mlflow_run_id,
                )
            )

        return BatchPredictionResponse(
            predictions=responses,
            total_trips=len(responses),
        )
