from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

try:
    from dotenv import load_dotenv

    load_dotenv()
    env_mode = os.getenv("ENVIRONMENT") or os.getenv("FLEETSENSE_APP_ENV")
    if env_mode:
        load_dotenv(f".env.{env_mode}")
except ImportError:
    pass

from fleetsense.core.config.settings import Settings
from fleetsense.core.constants.paths import CONFIG_DIR

DEFAULT_CONFIG_PATH = CONFIG_DIR / "config.yaml"
DEFAULT_MODEL_CONFIG_PATH = CONFIG_DIR / "model.yaml"

CONFIG_PATH = DEFAULT_CONFIG_PATH



def _apply_env_overrides(config_data: dict[str, Any]) -> None:
    """Apply hierarchical environment variable overrides.

    Variables prefixed with ``FLEETSENSE__`` are mapped into the config
    dict using ``__`` as the nesting separator.  For example::

        FLEETSENSE__API__PORT=9000  ->  config_data['api']['port'] = 9000
        FLEETSENSE__STORAGE__BACKEND=azure  ->  config_data['storage']['backend'] = 'azure'

    Leaf values are coerced to int/float/bool when possible so Pydantic
    receives the expected types.
    """
    prefix = "FLEETSENSE__"
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        parts = [p.lower() for p in key[len(prefix):].split("__")]
        if len(parts) < 2:
            continue
        target = config_data
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        # Coerce leaf value to appropriate Python type
        leaf = parts[-1]
        target[leaf] = _coerce_value(value)


def _coerce_value(value: str) -> Any:
    """Best-effort coercion of string env values to Python types."""
    if value.lower() in ("true", "yes", "1"):
        return True
    if value.lower() in ("false", "no", "0"):
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


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

    # Allow environment variables to override storage backend and azure settings
    storage_dict = config_data.get("storage", {})
    env_backend = os.getenv("FLEETSENSE_STORAGE_BACKEND") or os.getenv("STORAGE_BACKEND")
    if env_backend:
        storage_dict["backend"] = env_backend
    env_azure_conn = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if env_azure_conn is not None:
        storage_dict["azure_connection_string"] = env_azure_conn
    env_azure_container = os.getenv("AZURE_STORAGE_CONTAINER_NAME") or os.getenv("AZURE_CONTAINER_NAME")
    if env_azure_container:
        storage_dict["azure_container_name"] = env_azure_container
    config_data["storage"] = storage_dict

    # Allow environment variables to override app environment
    env_app_env = os.getenv("FLEETSENSE_APP_ENV") or os.getenv("ENVIRONMENT")
    if env_app_env:
        app_dict = config_data.get("app", {})
        app_dict["environment"] = env_app_env
        config_data["app"] = app_dict

    _apply_env_overrides(config_data)

    return Settings(**config_data)

