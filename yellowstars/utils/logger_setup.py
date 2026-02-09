"""
Logging configuration using Loguru.

Sets up structured logging with:
- Console output with colors
- File rotation (daily)
- JSON structured logs
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def setup_logging(
    level: str = "INFO",
    log_dir: str = "logs",
    console: bool = True,
    file: bool = True,
) -> None:
    """Configure the logging system.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_dir: Directory for log files.
        console: Enable console output.
        file: Enable file output.
    """
    # Remove default handler
    logger.remove()

    # Console handler with rich formatting
    if console:
        logger.add(
            sys.stderr,
            level=level,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>"
            ),
            colorize=True,
        )

    # File handler with rotation
    if file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        # Main log file (rotated daily, kept 30 days)
        logger.add(
            str(log_path / "yellowstars_{time:YYYY-MM-DD}.log"),
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
            rotation="00:00",
            retention="30 days",
            compression="gz",
        )

        # Error-only log
        logger.add(
            str(log_path / "errors_{time:YYYY-MM-DD}.log"),
            level="ERROR",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
            rotation="00:00",
            retention="90 days",
        )

    logger.info(f"Logging initialized: level={level}")
