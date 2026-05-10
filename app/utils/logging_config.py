"""
app/utils/logging_config.py
───────────────────────────
Configures structured logging for the whole application.
Call setup_logging() once at startup (done in main.py).
"""

import logging
import sys
from app.config import settings


def setup_logging() -> None:
    """Configure root logger with a clean, readable format."""
    log_level = logging.DEBUG if settings.DEBUG else logging.INFO

    fmt = "%(asctime)s | %(levelname)-8s | %(name)-35s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=log_level,
        format=fmt,
        datefmt=datefmt,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Quieten noisy third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("apscheduler").setLevel(logging.WARNING)
    logging.getLogger("firebase_admin").setLevel(logging.WARNING)

    logging.getLogger("app").setLevel(log_level)
