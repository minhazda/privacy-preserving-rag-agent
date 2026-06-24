"""Structured JSON logging via structlog.

A single ``configure_logging`` call wires stdlib logging to emit one JSON
object per line — friendly to Docker, Loki, and CloudWatch ingestion.
"""

from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(level: str = "INFO") -> None:
    """Configure process-wide structured JSON logging.

    Args:
        level: Root log level name (e.g. ``"INFO"``, ``"DEBUG"``).
    """
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level.upper())
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level.upper())),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound structured logger for ``name``."""
    return structlog.get_logger(name)
