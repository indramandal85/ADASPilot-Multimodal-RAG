"""Small standard-library logging setup used by the application."""

from __future__ import annotations

import logging

from adaspilot.config import Settings


def configure_logging(settings: Settings) -> None:
    """Configure readable logs without introducing a logging framework."""

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=getattr(logging, settings.log_level),
        force=True,
    )
