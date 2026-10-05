from pathlib import Path
import pytest

from fleetsense.core.config import load_settings, settings
from fleetsense.core.constants import (
    ARTIFACTS_DIR,
    CONFIG_DIR,
    DATA_DIR,
    FEATURES_DATA_DIR,
    LOG_DIR,
    METADATA_DIR,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    TLC_RAW_DIR,
    TMP_DATA_DIR,
)


def test_settings_loaded():
    assert settings.app.name == "FleetSense"
    assert settings.api.timeout == 30
    assert settings.logging.level == "INFO"
    assert settings.storage.backend in {"local", "azure"}
    assert settings.ingestion.dataset == "yellow"
    assert settings.ingestion.start_year == 2026
    assert settings.ingestion.start_month == 1
    assert settings.model.algorithm == "HistGradientBoostingRegressor"
    assert settings.model.target_column == "trip_duration_minutes"


def test_load_settings_missing_file(tmp_path: Path):
    missing_file = tmp_path / "non_existent.yaml"
    with pytest.raises(FileNotFoundError):
        load_settings(config_path=missing_file)


def test_load_settings_empty_file(tmp_path: Path):
    empty_file = tmp_path / "empty.yaml"
    empty_file.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Configuration file is empty"):
        load_settings(config_path=empty_file)


def test_load_settings_env_var(tmp_path: Path, monkeypatch):
    custom_config = tmp_path / "custom.yaml"
    custom_config.write_text(
        """
app:
  name: CustomApp
  version: "0.2.0"
  environment: testing
data:
  raw_data_path: test/raw
  processed_data_path: test/proc
  artifacts_path: test/art
api:
  timeout: 10
  retry_attempts: 1
logging:
  level: DEBUG
storage:
  backend: local
  local_data_dir: test_data
ingestion:
  dataset: green
  start_year: 2025
  start_month: 2
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("FLEETSENSE_CONFIG_PATH", str(custom_config))
    custom_settings = load_settings()
    assert custom_settings.app.name == "CustomApp"
    assert custom_settings.ingestion.dataset == "green"


def test_constants_paths():
    assert (PROJECT_ROOT / "pyproject.toml").exists()
    assert CONFIG_DIR == PROJECT_ROOT / "configs"
    assert DATA_DIR == PROJECT_ROOT / "data"
    assert RAW_DATA_DIR == DATA_DIR / "raw"
    assert TLC_RAW_DIR == RAW_DATA_DIR
    assert PROCESSED_DATA_DIR == DATA_DIR / "processed"
    assert FEATURES_DATA_DIR == DATA_DIR / "features"
    assert TMP_DATA_DIR == DATA_DIR / "tmp"
    assert METADATA_DIR == PROJECT_ROOT / "metadata"
    assert LOG_DIR == PROJECT_ROOT / "logs"
    assert ARTIFACTS_DIR == PROJECT_ROOT / "artifacts"