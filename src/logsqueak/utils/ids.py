"""UUID generation utilities for Logsqueak."""

import uuid
from typing import Optional


# Namespace UUID for Logsqueak (generated once, fixed)
# This ensures deterministic UUID v5 generation across runs
LOGSQUEAK_NAMESPACE = uuid.UUID("32e497fc-abf0-4d71-8cff-e302eb3e2bb0")


def generate_deterministic_uuid(content: str, namespace: Optional[uuid.UUID] = None) -> str:
    """
    Generate deterministic UUID v5 from content string.

    Uses UUID v5 (SHA-1 based) to create reproducible UUIDs from content.
    Same content always generates the same UUID.

    Args:
        content: Content string to hash (typically block content or ID)
        namespace: UUID namespace (defaults to LOGSQUEAK_NAMESPACE)

    Returns:
        UUID string in standard format (e.g., "a1b2c3d4-5e6f-5a7b-8c9d-0e1f2a3b4c5d")

    Example:
        >>> generate_deterministic_uuid("My knowledge block content")
        "a1b2c3d4-5e6f-5a7b-8c9d-0e1f2a3b4c5d"
    """
    if namespace is None:
        namespace = LOGSQUEAK_NAMESPACE

    return str(uuid.uuid5(namespace, content))


def generate_random_uuid() -> str:
    """
    Generate random UUID v4.

    Used for new blocks that don't have existing IDs.

    Returns:
        UUID string in standard format

    Example:
        >>> generate_random_uuid()
        "f47ac10b-58cc-4372-a567-0e02b2c3d479"
    """
    return str(uuid.uuid4())
