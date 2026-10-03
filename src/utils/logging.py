"""
Centralized structured logging configuration.
"""

import logging
import os
import sys
from typing import Optional


def setup_logging(level: Optional[str] = None) -> None:
    """
    Configures the root application logger with formatting and appropriate stream handlers.
    """
    log_level_name = level or os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    log_format = "%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Avoid duplicate handlers if setup is called multiple times
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(fmt=log_format, datefmt=date_format)
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

    root_logger.setLevel(log_level)


def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger instance with the given module name.
    """
    if not logging.getLogger().handlers:
        setup_logging()
    return logging.getLogger(name)
