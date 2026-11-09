from __future__ import annotations

import os
from typing import Any

from fleetsense.core.constants.paths import LOG_DIR


def get_logging_config() -> dict[str, Any]:
    """Dynamically generate logging configuration dictionary."""
    log_level = (os.getenv("LOG_LEVEL") or os.getenv("FLEETSENSE_LOG_LEVEL") or "INFO").upper()

    log_format = (
        os.getenv("LOG_FORMAT")
        or os.getenv("FLEETSENSE_LOG_FORMAT")
        or ("json" if os.getenv("FLEETSENSE_APP_ENV") in ("production", "staging") else "standard")
    ).lower()
    console_formatter = "json" if log_format == "json" else "standard"

    file_handler_enabled = False
    log_file = LOG_DIR / "application.log"
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with log_file.open("a", encoding="utf-8"):
            pass
        file_handler_enabled = True
    except Exception:
        file_handler_enabled = False

    handlers: dict[str, dict[str, Any]] = {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": console_formatter,
            "level": log_level,
        },
    }

    active_root_handlers = ["console"]

    if file_handler_enabled:
        handlers["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(log_file),
            "formatter": "standard",
            "level": "DEBUG",
            "encoding": "utf-8",
            "maxBytes": 10 * 1024 * 1024,
            "backupCount": 5,
        }
        active_root_handlers.append("file")

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
            },
            "json": {
                "format": (
                    '{"time": "%(asctime)s", "level": "%(levelname)s", '
                    '"logger": "%(name)s", "message": "%(message)s"}'
                )
            },
        },
        "handlers": handlers,
        "root": {
            "handlers": active_root_handlers,
            "level": "DEBUG" if log_level == "DEBUG" else log_level,
        },
    }


# Backwards compatibility reference for external modules and tests
LOG_FILE = LOG_DIR / "application.log"


class _LazyLoggingConfig(dict):
    """Lazy mapping proxy to preserve backward compatibility for LOGGING_CONFIG access."""

    def __getitem__(self, item: Any) -> Any:
        return get_logging_config()[item]

    def get(self, item: Any, default: Any = None) -> Any:
        return get_logging_config().get(item, default)

    def __contains__(self, item: Any) -> bool:
        return item in get_logging_config()


LOGGING_CONFIG = _LazyLoggingConfig()

