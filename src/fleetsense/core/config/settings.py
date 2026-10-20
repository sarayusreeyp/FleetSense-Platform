from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    name: str = "FleetSense"
    version: str = "0.1.0"
    environment: str = "development"


class DataConfig(BaseModel):
    raw_data_path: str = "data/raw"
    processed_data_path: str = "data/processed"
    artifacts_path: str = "artifacts"


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    timeout: int = 30
    retry_attempts: int = 3


class LoggingConfig(BaseModel):
    level: str = "INFO"


class StorageSettings(BaseModel):
    backend: str = "local"
    local_data_dir: str = "data"
    azure_container_name: str = "fleetsense-data"
    azure_connection_string: str = ""


class IngestionConfig(BaseModel):
    dataset: str = "yellow"
    start_year: int = 2026
    start_month: int = 1


class QualityGateConfig(BaseModel):
    """Gap 14: Model promotion thresholds loaded from model.yaml."""

    max_val_mae: Optional[float] = Field(
        default=8.0,
        description="Maximum allowed validation MAE (minutes). None = disabled.",
    )
    min_val_r2: Optional[float] = Field(
        default=0.50,
        description="Minimum required validation R². None = disabled.",
    )


class ModelConfig(BaseModel):
    name: str = "trip_duration_regressor"
    algorithm: str = "XGBRegressor"
    target_column: str = "trip_duration_minutes"
    max_duration_minutes: float = 1440.0
    features: list[str] = Field(
        default_factory=lambda: [
            "VendorID",
            "passenger_count",
            "trip_distance",
            "PULocationID",
            "DOLocationID",
            "pickup_hour",
            "pickup_day_of_week",
            "pickup_month",
            "pickup_day",
            "is_weekend",
        ]
    )
    categorical_features: list[str] = Field(
        default_factory=lambda: [
            "VendorID",
            "PULocationID",
            "DOLocationID",
        ]
    )
    random_state: int = 42
    n_estimators: int = 100
    learning_rate: float = 0.1
    max_depth: int | None = 15
    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    mlflow_tracking_uri: str = "postgresql://mlflow:mlflow@localhost:5432/mlflow"
    mlflow_experiment_name: str = "fleetsense-trip-duration"

    # Gap 14: Quality gate thresholds — read from model.yaml quality_gate section
    quality_gate: QualityGateConfig = Field(default_factory=QualityGateConfig)

    # Gap 24: Trip distance validation bounds (mirrors schemas.py TRIP_DISTANCE_MAX)
    trip_distance_min: float = 0.0
    trip_distance_max: float = 100.0

    # Real-time Drift Detection PSI Thresholds
    drift_warning_threshold: float = 0.10
    drift_alert_threshold: float = 0.20

    @property
    def quality_gate_max_mae(self) -> float | None:
        """Convenience accessor for the quality gate MAE threshold."""
        return self.quality_gate.max_val_mae

    @property
    def quality_gate_min_r2(self) -> float | None:
        """Convenience accessor for the quality gate R² threshold."""
        return self.quality_gate.min_val_r2


class Settings(BaseModel):
    app: AppConfig
    data: DataConfig
    api: APIConfig
    logging: LoggingConfig
    storage: StorageSettings
    ingestion: IngestionConfig
    model: ModelConfig = Field(default_factory=ModelConfig)
