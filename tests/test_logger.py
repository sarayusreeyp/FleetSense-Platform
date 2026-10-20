import logging

from fleetsense.core.logger import get_logger
from fleetsense.core.logger.config import LOG_FILE


def test_logger_instance():
    logger = get_logger("fleetsense.test")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "fleetsense.test"


def test_logger_writes_to_file():
    logger = get_logger("fleetsense.test_write")
    test_msg = "FLEETSENSE_LOGGER_UNIT_TEST_MARKER"
    logger.info(test_msg)

    assert LOG_FILE.exists()
    content = LOG_FILE.read_text(encoding="utf-8")
    assert test_msg in content


def test_logger_json_formatting_option(monkeypatch):
    """Verify LOGGING_CONFIG uses json formatter when LOG_FORMAT=json."""
    import importlib
    monkeypatch.setenv("LOG_FORMAT", "json")
    import fleetsense.core.logger.config as log_cfg
    importlib.reload(log_cfg)

    assert log_cfg.LOGGING_CONFIG["handlers"]["console"]["formatter"] == "json"

    # Reset
    monkeypatch.delenv("LOG_FORMAT", raising=False)
    importlib.reload(log_cfg)


def test_logger_non_writable_directory_fallback(tmp_path, monkeypatch):
    """Verify that if log file cannot be created, file handler is omitted without crashing."""
    import importlib
    from unittest.mock import patch

    # Mock open on log file to simulate PermissionError
    with patch("pathlib.Path.open", side_effect=PermissionError("Read-only filesystem")):
        import fleetsense.core.logger.config as log_cfg
        importlib.reload(log_cfg)

        assert "file" not in log_cfg.LOGGING_CONFIG["root"]["handlers"]
        assert "console" in log_cfg.LOGGING_CONFIG["root"]["handlers"]

    # Reload cleanly
    import fleetsense.core.logger.config as log_cfg
    importlib.reload(log_cfg)

