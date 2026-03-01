"""Structured logging with rotating file handler.

All trading system components share this logger configuration.
Logs go to both console and data/logs/trading.log with rotation.
"""

import logging
import os
from logging.handlers import RotatingFileHandler

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "logs")
LOG_FILE = os.path.join(LOG_DIR, "trading.log")
MAX_LOG_BYTES = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 3


def setup_logger(name: str = "trading", level: int = logging.INFO) -> logging.Logger:
    """Configure and return the trading system logger.

    Handlers are only attached to the root 'trading' logger.
    Child loggers (e.g., 'trading.main') propagate up to it.
    """
    # Ensure root trading logger is configured first
    root = logging.getLogger("trading")
    if not root.handlers:
        root.setLevel(level)

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console handler
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        root.addHandler(console)

        # File handler (rotating)
        os.makedirs(LOG_DIR, exist_ok=True)
        file_handler = RotatingFileHandler(
            LOG_FILE, maxBytes=MAX_LOG_BYTES, backupCount=BACKUP_COUNT
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # Return the requested (possibly child) logger
    return logging.getLogger(name)


# Module-level convenience — importable as `from core.logger import log`
log = setup_logger()
