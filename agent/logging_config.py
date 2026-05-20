"""Centralized logging configuration for the Native-SWE agent.

Call ``setup_logging()`` once at process startup (both the LangGraph graph
entrypoint and the FastAPI webapp import this).  Every logger under the
``agent`` namespace — and the root logger for third-party libraries — will
emit to **both** the console *and* a rotating log file.

The log file lives at ``LOG_DIR / LOG_FILENAME`` (defaults to ``logs/native_swe.log``
relative to the repo root).  Rotation happens at 10 MB with 5 backups kept.

Environment variables
---------------------
``LOG_LEVEL``   – root log level (default ``INFO``).
``LOG_DIR``     – directory for the log file (default ``logs``).
``LOG_FILENAME``– log file name (default ``native_swe.log``).
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

_CONFIGURED = False

# ── Tunables ────────────────────────────────────────────────────────────────
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
LOG_DIR = os.environ.get("LOG_DIR", "logs")
LOG_FILENAME = os.environ.get("LOG_FILENAME", "native_swe.log")

# 10 MB per file, keep 5 rotated backups
_MAX_BYTES = 10 * 1024 * 1024
_BACKUP_COUNT = 5

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging() -> None:
    """Configure root logger with console + rotating file handlers.

    Safe to call multiple times — subsequent calls are no-ops.
    """
    global _CONFIGURED  # noqa: PLW0603
    if _CONFIGURED:
        return
    _CONFIGURED = True

    # Ensure log directory exists
    log_dir = Path(LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / LOG_FILENAME

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # ── Console handler ─────────────────────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # ── File handler (rotating) ─────────────────────────────────────────
    file_handler = RotatingFileHandler(
        str(log_path),
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    # ── Root logger ─────────────────────────────────────────────────────
    root_logger = logging.getLogger()
    root_logger.setLevel(LOG_LEVEL)

    # Remove any pre-existing handlers to avoid duplicate output
    root_logger.handlers.clear()

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Quieten noisy third-party loggers
    for noisy in ("httpx", "httpcore", "urllib3", "openai", "langchain", "langgraph"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        "Logging initialised — level=%s, file=%s", LOG_LEVEL, log_path.resolve()
    )
