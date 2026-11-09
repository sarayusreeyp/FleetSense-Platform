from __future__ import annotations

import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
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
            # Quick check if MLflow server is reachable before blocking startup with retries
            mlflow_reachable = False
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
                if tracking_uri.startswith(("http://", "https://")):
                    try:
                        import urllib.request
                        urllib.request.urlopen(f"{tracking_uri.rstrip('/')}/health", timeout=1.5)
                        mlflow_reachable = True
                    except Exception:
                        logger.warning("MLflow tracking server at %s is not reachable; bypassing registry.", tracking_uri)
                else:
                    # SQLite or PostgreSQL database backend (MLflow handles connection directly)
                    mlflow_reachable = True

            if mlflow_reachable:
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

        # 2. Check Remote Cloud Storage Backend (Azure Blob Storage / S3)
        try:
            from fleetsense.storage.factory import get_storage
            active_storage = get_storage()
            configured_cat = os.getenv("FLEETSENSE_REMOTE_MODEL_CATEGORY", "models/champion")
            target_filename = os.getenv("FLEETSENSE_REMOTE_MODEL_FILENAME", "trip_duration_model.joblib")

            candidate_categories = [configured_cat]
            for cat_candidate in ("champion", "models/champion", "models"):
                if cat_candidate not in candidate_categories:
                    candidate_categories.append(cat_candidate)

            found_cat = None
            found_file = None

            # Check direct file existence
            for cat in candidate_categories:
                for fname in (target_filename, "trip_duration_model.joblib", "latest_model.joblib"):
                    if hasattr(active_storage, "exists") and active_storage.exists(cat, fname):
                        found_cat = cat
                        found_file = fname
                        break
                if found_cat:
                    break

            # If not directly matched, query category file listing
            if not found_cat and hasattr(active_storage, "list_files"):
                for cat in candidate_categories:
                    try:
                        r_files = active_storage.list_files(cat)
                        if r_files:
                            found_cat = cat
                            found_file = "latest_model.joblib" if "latest_model.joblib" in r_files else r_files[-1]
                            break
                    except Exception:
                        pass

            if found_cat and found_file and hasattr(active_storage, "download_to_file"):
                MODELS_DIR.mkdir(parents=True, exist_ok=True)
                cached_model_path = MODELS_DIR / found_file
                if not cached_model_path.exists():
                    logger.info("Downloading model '%s' from %s to %s...", found_file, found_cat, cached_model_path)
                    active_storage.download_to_file(
                        category=found_cat,
                        filename=found_file,
                        destination_path=cached_model_path,
                    )
                else:
                    logger.info("Using cached model '%s' from %s", found_file, cached_model_path)
                self.model = joblib.load(cached_model_path)
                self.model_path = cached_model_path
                self.model_source = f"storage:{found_cat}/{found_file}"
                logger.info("Successfully loaded champion model from cloud storage (%s)", self.model_source)
                return
        except Exception as storage_exc:
            logger.info("Cloud storage model check skipped/failed: %s", storage_exc)

        # 3. Check explicit path (passed in or from FLEETSENSE_MODEL_ARTIFACT_PATH)
        if self.model_path and self.model_path.exists():
            try:
                self.model = joblib.load(self.model_path)
                self.model_source = f"local:{self.model_path}"
                logger.info("Loaded model from explicit path %s", self.model_path)
                return
            except Exception as exc:
                logger.warning("Failed to load model from %s: %s", self.model_path, exc)

        # 4. Search local directories (prefer latest_model.joblib or most recently modified)
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
            # If timezone is provided, convert to Eastern Time (America/New_York) where TLC data originates
            if dt.tzinfo is not None:
                try:
                    from zoneinfo import ZoneInfo
                    dt = dt.astimezone(ZoneInfo("America/New_York"))
                except Exception:
                    # Fallback to standard Eastern Time offset (UTC-5)
                    dt = dt.astimezone(timezone(timedelta(hours=-5)))

            hour = dt.hour
            dow = dt.weekday()
            records.append(
                {
                    "VendorID": trip.VendorID,
                    "passenger_count": trip.passenger_count,
                    "trip_distance": trip.trip_distance,
                    "PULocationID": trip.PULocationID,
                    "DOLocationID": trip.DOLocationID,
                    "pickup_hour": hour,
                    "pickup_day_of_week": dow,
                    "pickup_month": dt.month,
                    "pickup_day": dt.day,
                    "is_weekend": 1 if dow >= 5 else 0,
                    "pickup_hour_sin": float(np.sin(2 * np.pi * hour / 24.0)),
                    "pickup_hour_cos": float(np.cos(2 * np.pi * hour / 24.0)),
                    "is_rush_hour": 1 if (dow < 5 and hour in (7, 8, 9, 16, 17, 18, 19)) else 0,
                    "PU_DO_pair": f"{trip.PULocationID}_{trip.DOLocationID}",
                }
            )

        features_df = pd.DataFrame(records)

        # Cast all configured categorical features to categorical dtype (matching XGBoost training schema)
        cat_cols = getattr(settings.model, "categorical_features", []) or ["VendorID", "PULocationID", "DOLocationID", "PU_DO_pair"]
        for col in cat_cols:
            if col in features_df.columns:
                features_df[col] = features_df[col].astype("category")

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
