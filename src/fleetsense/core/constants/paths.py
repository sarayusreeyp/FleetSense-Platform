from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[4]

# Configuration & metadata
CONFIG_DIR = PROJECT_ROOT / "configs"
METADATA_DIR = PROJECT_ROOT / "metadata"

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
TLC_RAW_DIR = RAW_DATA_DIR
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DATA_DIR = DATA_DIR / "features"
TMP_DATA_DIR = DATA_DIR / "tmp"

# Artifacts & Logging
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
LOG_DIR = PROJECT_ROOT / "logs"