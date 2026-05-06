import logging
import logging.config
import os
from pathlib import Path

from utils.config import LoggingConfig


def setup_logging(config: LoggingConfig) -> None:
    log_file = Path(config.file)
    if log_file.parent and not log_file.parent.exists():
        os.makedirs(log_file.parent, exist_ok=True)

    handlers = {
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": config.level,
            "formatter": "standard",
            "filename": str(log_file),
            "maxBytes": 5 * 1024 * 1024,
            "backupCount": 3,
            "encoding": "utf-8",
        }
    }

    root_handlers = ["file"]
    if config.console:
        handlers["console"] = {
            "class": "logging.StreamHandler",
            "level": config.level,
            "formatter": "standard",
        }
        root_handlers.append("console")

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
                }
            },
            "handlers": handlers,
            "root": {"level": config.level, "handlers": root_handlers},
        }
    )
