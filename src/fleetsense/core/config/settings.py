from __future__ import annotations

from typing import Any
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


class ModelConfig(BaseModel):
    name: str = "trip_duration_regressor"
    algorithm: str = "HistGradientBoostingRegressor"
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
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment_name: str = "fleetsense-trip-duration"


class Settings(BaseModel):
    app: AppConfig
    data: DataConfig
    api: APIConfig
    logging: LoggingConfig
    storage: StorageSettings
    ingestion: IngestionConfig
    model: ModelConfig = Field(default_factory=ModelConfig)