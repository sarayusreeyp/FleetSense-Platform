from __future__ import annotations

import os

from fleetsense.core.constants.paths import LOG_DIR

_log_level = os.getenv("LOG_LEVEL") or os.getenv("FLEETSENSE_LOG_LEVEL") or "INFO"
_log_level = _log_level.upper()

# Gap 2: Support structured JSON console logs for production log aggregators
_log_format = (
    os.getenv("LOG_FORMAT")
    or os.getenv("FLEETSENSE_LOG_FORMAT")
    or ("json" if os.getenv("FLEETSENSE_APP_ENV") in ("production", "staging") else "standard")
).lower()
_console_formatter = "json" if _log_format == "json" else "standard"

# Gap 4: Graceful fallback when log directory is not writable
_file_handler_enabled = False
try:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE = LOG_DIR / "application.log"
    # Test file writability
    with LOG_FILE.open("a", encoding="utf-8") as _test_f:
        pass
    _file_handler_enabled = True
except Exception:
    LOG_FILE = LOG_DIR / "application.log"
    _file_handler_enabled = False

_handlers: dict[str, dict] = {
    "console": {
        "class": "logging.StreamHandler",
        "formatter": _console_formatter,
        "level": _log_level,
    },
}

_active_root_handlers = ["console"]

if _file_handler_enabled:
    _handlers["file"] = {
        "class": "logging.handlers.RotatingFileHandler",
        "filename": str(LOG_FILE),
        "formatter": "standard",
        "level": "DEBUG",
        "encoding": "utf-8",
        "maxBytes": 10 * 1024 * 1024,
        "backupCount": 5,
    }
    _active_root_handlers.append("file")

LOGGING_CONFIG = {
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
    "handlers": _handlers,
    "root": {
        "handlers": _active_root_handlers,
        "level": "DEBUG" if _log_level == "DEBUG" else _log_level,
    },
}

