from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Trip distance bounds — SINGLE SOURCE OF TRUTH
# Must match training-side filter in ingestion/validation pipeline:
#   - gt=0.0  (no zero-distance trips)
#   - le=100.0 (TLC data quality cap; >100 miles is almost certainly bad GPS data)
# ---------------------------------------------------------------------------
TRIP_DISTANCE_MIN: float = 0.0       # exclusive
TRIP_DISTANCE_MAX: float = 100.0     # inclusive


class TripFeaturesPayload(BaseModel):
    """Payload representing a single taxi trip at pickup time."""

    VendorID: int = Field(
        default=1,
        ge=1,
        le=2,
        description="TLC vendor ID (1=Creative Mobile Technologies, 2=VeriFone Inc.)",
        examples=[1],
    )
    pickup_datetime: datetime = Field(
        ...,
        description="Trip pickup timestamp (ISO 8601)",
        examples=["2026-10-05T14:30:00"],
    )
    passenger_count: int = Field(
        default=1,
        ge=1,
        le=9,
        description="Number of passengers",
        examples=[1],
    )
    trip_distance: float = Field(
        ...,
        gt=TRIP_DISTANCE_MIN,
        le=TRIP_DISTANCE_MAX,
        description=f"Trip distance in miles (must be > {TRIP_DISTANCE_MIN} and <= {TRIP_DISTANCE_MAX})",
        examples=[3.5],
    )
    PULocationID: int = Field(
        default=161,
        ge=1,
        le=265,
        description="Pickup TLC Taxi Zone ID (1-265)",
        examples=[161],
    )
    DOLocationID: int = Field(
        default=236,
        ge=1,
        le=265,
        description="Dropoff TLC Taxi Zone ID (1-265)",
        examples=[236],
    )


class PredictionResponse(BaseModel):
    """Predicted trip duration and estimated dropoff time."""

    predicted_duration_minutes: float = Field(
        ..., description="Predicted trip duration in minutes"
    )
    pickup_datetime: datetime = Field(
        ..., description="Trip pickup timestamp"
    )
    estimated_dropoff_datetime: datetime = Field(
        ..., description="Estimated arrival timestamp"
    )
    model_version: str = Field(
        default="0.1.0", description="Model version"
    )
    model_algorithm: str = Field(
        default="HistGradientBoostingRegressor",
        description="Algorithm used for prediction",
    )
    model_source: str = Field(
        default="unknown",
        description="Where the model was loaded from (local path, azure blob, or dummy)",
    )


class BatchPredictionRequest(BaseModel):
    """Request payload containing multiple trips for bulk inference."""

    trips: list[TripFeaturesPayload] = Field(
        ..., min_length=1, description="List of trip records"
    )


class BatchPredictionResponse(BaseModel):
    """Batch prediction results."""

    predictions: list[PredictionResponse] = Field(
        ..., description="List of predicted durations"
    )
    total_trips: int = Field(..., description="Total trips evaluated")


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str = Field(default="healthy")
    app_name: str
    environment: str
    version: str
    model_loaded: bool
    model_algorithm: str | None = None
    model_source: str | None = Field(
        default=None,
        description="Where the active model was loaded from",
    )
    is_dummy_model: bool = Field(
        default=False,
        description="True if a dummy baseline is running (not a real trained artifact)",
    )
