from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

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
from fleetsense.core.constants.paths import ARTIFACTS_DIR, DATA_DIR
from fleetsense.core.logger import get_logger

logger = get_logger(__name__)


class PredictionService:
    """
    Manages model loading, real-time feature extraction, and inference.
    """

    def __init__(self, model_path: Path | str | None = None) -> None:
        self.model = None
        self.algorithm = settings.model.algorithm
        self.model_version = settings.app.version
        self.model_path = Path(model_path) if model_path else None
        self._load_or_initialize_model()

    def _load_or_initialize_model(self) -> None:
        """Load trained model artifact or initialize fallback model."""
        # 1. Check explicit path
        if self.model_path and self.model_path.exists():
            try:
                self.model = joblib.load(self.model_path)
                logger.info("Loaded model from %s", self.model_path)
                return
            except Exception as exc:
                logger.warning("Failed to load model from %s: %s", self.model_path, exc)

        # 2. Search ARTIFACTS_DIR and DATA_DIR / 'models'
        search_dirs = [ARTIFACTS_DIR, DATA_DIR / "models"]
        for search_dir in search_dirs:
            if search_dir.exists():
                model_files = list(search_dir.glob("*.joblib"))
                if model_files:
                    latest_model = sorted(model_files)[-1]
                    try:
                        self.model = joblib.load(latest_model)
                        self.model_path = latest_model
                        logger.info("Loaded model from %s", latest_model)
                        return
                    except Exception as exc:
                        logger.warning("Failed loading %s: %s", latest_model, exc)

        # 3. Fallback: lightweight baseline model if no serialized weights exist yet
        logger.info("No saved model found. Fitting default baseline regressor.")
        baseline = HistGradientBoostingRegressor(
            max_iter=20,
            random_state=settings.model.random_state,
        )
        # Synthetic baseline fit on NYC taxi boundaries
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
        )

    def predict_batch(
        self,
        request: BatchPredictionRequest,
    ) -> BatchPredictionResponse:
        """Predict durations for multiple trips in a vectorized pass."""
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
                )
            )

        return BatchPredictionResponse(
            predictions=responses,
            total_trips=len(responses),
        )
