"""
Logger Factory.

Provides a centralized logger for the FleetSense application.
"""

import logging
import logging.config

from fleetsense.core.logger.config import LOGGING_CONFIG

# Configure logging only once.
logging.config.dictConfig(LOGGING_CONFIG)


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
    return logging.getLogger(name)