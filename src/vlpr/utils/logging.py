"""Logging configuration."""

import logging
import os


def configure_logging() -> None:
    """Configure consistent logs for scripts and services."""
    level_name = os.getenv("VLPR_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
