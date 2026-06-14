"""
utils/logger.py
---------------
Centralized logging configuration for the entire project.
Produces structured, timestamped logs to both console and file.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path


# Ensure logs directory exists
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

_configured = False


def _configure_root_logger():
    """Configure root logger once at import time."""
    global _configured
    if _configured:
        return
    _configured = True

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    numeric_level = getattr(logging, log_level, logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(numeric_level)

    # File handler with rotation (max 10MB, keep 5 backups)
    file_handler = RotatingFileHandler(
        LOG_DIR / "app.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)  # Always log DEBUG to file

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


_configure_root_logger()


def get_logger(name: str) -> logging.Logger:
    """Get a named logger. Use __name__ in each module."""
    return logging.getLogger(name)
