"""Validation functions for wizard inputs and external systems."""

from dataclasses import dataclass
from typing import Any


@dataclass
class ValidationResult:
    """Result of a validation operation.

    Attributes:
        success: Whether validation passed
        error_message: Error details if failed (None if success)
        data: Additional data from validation (e.g., models list, disk space)
    """
    success: bool
    error_message: str | None = None
    data: Any | None = None
