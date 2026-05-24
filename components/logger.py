"""
logger.py — Centralised logging setup for Aadi.

Sets up a logger that writes to both the console and a rotating log file.
All other modules import `get_logger(__name__)` to get a named logger.
"""

import logging
import os
from logging.handlers import RotatingFileHandler


def setup_logger(log_file: str = "logs/aadi.log", level: str = "INFO") -> logging.Logger:
    """
    Create and configure the root 'aadi' logger.

    Parameters
    ----------
    log_file : str
        Path to the log file (created automatically if missing).
    level : str
        Logging level string, e.g. "DEBUG", "INFO", "WARNING".

    Returns
    -------
    logging.Logger
        The configured root logger for the project.
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    log_level = getattr(logging, level.upper(), logging.INFO)

    # Root logger for the whole project
    logger = logging.getLogger("aadi")
    logger.setLevel(log_level)

    # Avoid adding duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # --- Console handler ---
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # --- Rotating file handler (10 MB max, keep 3 backups) ---
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Return a child logger under the 'aadi' namespace.

    Parameters
    ----------
    name : str
        Typically ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
        Named child logger.
    """
    return logging.getLogger(f"aadi.{name}")
