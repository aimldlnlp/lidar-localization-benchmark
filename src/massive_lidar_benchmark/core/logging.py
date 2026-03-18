"""Logging configuration."""

from __future__ import annotations

import logging
from pathlib import Path

from massive_lidar_benchmark.core.io import ensure_dir


def configure_logging(log_path: str | Path | None = None, level: int = logging.INFO) -> None:
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(level)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(stream_handler)

    if log_path is not None:
        path = Path(log_path)
        ensure_dir(path.parent)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

