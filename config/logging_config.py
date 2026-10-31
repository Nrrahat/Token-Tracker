"""Centralized logging setup."""

import logging.config

from config.settings import Settings, get_settings


def configure_logging(settings: Settings | None = None) -> None:
    """Configure consistent console logging for application modules."""
    resolved_settings = settings or get_settings()
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "standard",
                    "stream": "ext://sys.stdout",
                }
            },
            "root": {
                "handlers": ["console"],
                "level": resolved_settings.log_level.upper(),
            },
        }
    )