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