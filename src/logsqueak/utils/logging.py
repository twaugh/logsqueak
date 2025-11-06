"""Structured logging setup for Logsqueak."""

import structlog
from pathlib import Path
from typing import Any
import sys


def configure_logging() -> None:
    """
    Configure structlog for JSON logging to ~/.cache/logsqueak/logs/logsqueak.log.

    Log levels:
    - DEBUG: LLM response chunks, detailed state changes
    - INFO: User actions, phase transitions, LLM requests
    - WARNING: Partial failures, retry attempts
    - ERROR: Operation failures, validation errors
    """
    # Ensure log directory exists
    log_dir = Path.home() / ".cache" / "logsqueak" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "logsqueak.log"

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging_level="INFO"),
        context_class=dict,
        logger_factory=structlog.WriteLoggerFactory(file=open(log_file, "a")),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> Any:
    """
    Get a structured logger instance.

    Args:
        name: Logger name (typically __name__ of calling module)

    Returns:
        Structured logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("llm_request_started", phase="classification", model="gpt-4")
    """
    return structlog.get_logger(name)
