from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from fleetsense.core.config.settings import Settings
from fleetsense.core.constants.paths import CONFIG_DIR

DEFAULT_CONFIG_PATH = CONFIG_DIR / "config.yaml"
DEFAULT_MODEL_CONFIG_PATH = CONFIG_DIR / "model.yaml"

CONFIG_PATH = DEFAULT_CONFIG_PATH


def resolve_config_path(config_path: Path | str | None = None) -> Path:
    if config_path is not None:
        return Path(config_path)

    env_path = os.getenv("FLEETSENSE_CONFIG_PATH")
    if env_path:
        return Path(env_path)

    return DEFAULT_CONFIG_PATH


def load_settings(
    config_path: Path | str | None = None,
    model_config_path: Path | str | None = None,
) -> Settings:
    target_path = resolve_config_path(config_path)

    if not target_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {target_path}")

    with target_path.open("r", encoding="utf-8") as file:
        config_data: dict[str, Any] | None = yaml.safe_load(file)

    if config_data is None:
        raise ValueError(f"Configuration file is empty: {target_path}")

    target_model_path = (
        Path(model_config_path)
        if model_config_path is not None
        else DEFAULT_MODEL_CONFIG_PATH
    )

    if target_model_path.exists() and target_model_path.stat().st_size > 0:
        with target_model_path.open("r", encoding="utf-8") as m_file:
            model_data = yaml.safe_load(m_file)
            if isinstance(model_data, dict):
                if "model" in model_data:
                    config_data["model"] = model_data["model"]
                else:
                    config_data["model"] = model_data

    return Settings(**config_data)