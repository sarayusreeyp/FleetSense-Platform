"""
Logger Factory.

Provides a centralized logger for the FleetSense application.
"""

import logging
import logging.config

from fleetsense.core.logger.config import get_logging_config

_is_configured = False


def setup_logging() -> None:
    """Initialize logging configuration once."""
    global _is_configured
    if not _is_configured:
        logging.config.dictConfig(get_logging_config())
        _is_configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Return a configured logger.

    Parameters
    ----------
    name : str
        Module name (__name__)

    Returns
    -------
    logging.Logger
    """
    setup_logging()
    return logging.getLogger(name)
