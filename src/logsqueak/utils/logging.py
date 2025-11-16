"""Structured logging setup for Logsqueak."""

import structlog
from pathlib import Path
from typing import Any
import os


def configure_logging() -> None:
    """
    Configure structlog for JSON logging to ~/.cache/logsqueak/logs/logsqueak.log.

    Log level can be controlled via LOGSQUEAK_LOG_LEVEL environment variable:
    - Set to "DEBUG" to see all LLM request/response details
    - Defaults to "INFO" if not set

    Log levels:
    - DEBUG: LLM request payloads, raw response lines, parsed JSON, detailed state changes
    - INFO: User actions, phase transitions, LLM request summary
    - WARNING: Partial failures, retry attempts
    - ERROR: Operation failures, validation errors

    Example:
        # Enable debug logging
        export LOGSQUEAK_LOG_LEVEL=DEBUG
        logsqueak extract

        # Or inline:
        LOGSQUEAK_LOG_LEVEL=DEBUG logsqueak extract

        # View logs with jq for readability:
        tail -f ~/.cache/logsqueak/logs/logsqueak.log | jq .
    """
    # Ensure log directory exists
    log_dir = Path.home() / ".cache" / "logsqueak" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "logsqueak.log"

    # Get log level from environment variable (default to INFO)
    log_level = os.environ.get("LOGSQUEAK_LOG_LEVEL", "INFO").upper()

    # Validate log level
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
    if log_level not in valid_levels:
        log_level = "INFO"

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
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
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
