import os
from pathlib import Path

# Project root directory: configurable via FLEETSENSE_PROJECT_ROOT for container/pip installs
_env_root = os.getenv("FLEETSENSE_PROJECT_ROOT")
PROJECT_ROOT = Path(_env_root).resolve() if _env_root else Path(__file__).resolve().parents[4]

# Configuration & metadata
CONFIG_DIR = Path(os.getenv("FLEETSENSE_CONFIG_DIR", PROJECT_ROOT / "configs"))
METADATA_DIR = Path(os.getenv("FLEETSENSE_METADATA_DIR", PROJECT_ROOT / "metadata"))

# Data directories
DATA_DIR = Path(os.getenv("FLEETSENSE_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DATA_DIR = DATA_DIR / "raw"
TLC_RAW_DIR = RAW_DATA_DIR
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DATA_DIR = DATA_DIR / "features"
TMP_DATA_DIR = DATA_DIR / "tmp"

# Artifacts & Logging
ARTIFACTS_DIR = Path(os.getenv("FLEETSENSE_ARTIFACTS_DIR", PROJECT_ROOT / "artifacts"))
MODELS_DIR = ARTIFACTS_DIR / "models"
LOG_DIR = Path(os.getenv("FLEETSENSE_LOG_DIR", PROJECT_ROOT / "logs"))
