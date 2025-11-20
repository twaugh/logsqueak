"""LLM ID Mapper - Bidirectional mapping between hybrid IDs and short IDs.

This utility reduces token usage in LLM prompts by mapping long hybrid IDs
(UUIDs or content hashes) to short sequential IDs (1, 2, 3, ...).

The mapping is scoped to each individual LLM request - short IDs are only
valid within that specific prompt and response.

Example:
    >>> mapper = LLMIDMapper()
    >>> short_id = mapper.add("65f3a8e0-1234-5678-9abc-def012345678")
    >>> short_id
    "1"
    >>> mapper.to_hybrid("1")
    "65f3a8e0-1234-5678-9abc-def012345678"
"""

from typing import Optional


class LLMIDMapper:
    """Maps between application hybrid IDs and LLM-friendly short IDs.

    Short IDs are sequential integers (1, 2, 3, ...) that are valid only
    for the scope of a single LLM request.

    Thread safety: Not thread-safe. Create one instance per LLM request.
    """

    def __init__(self):
        """Initialize empty mapping."""
        self._short_to_hybrid: dict[str, str] = {}
        self._hybrid_to_short: dict[str, str] = {}
        self._next_id: int = 1

    def add(self, hybrid_id: str) -> str:
        """Register a hybrid ID and return its short ID.

        If the hybrid ID was already registered, returns the existing
        short ID (idempotent).

        Args:
            hybrid_id: Application hybrid ID (UUID or content hash)

        Returns:
            Short ID string (e.g., "1", "2", "3")

        Example:
            >>> mapper = LLMIDMapper()
            >>> mapper.add("abc123")
            "1"
            >>> mapper.add("def456")
            "2"
            >>> mapper.add("abc123")  # Already registered
            "1"
        """
        if hybrid_id in self._hybrid_to_short:
            return self._hybrid_to_short[hybrid_id]

        short_id = str(self._next_id)
        self._next_id += 1

        self._short_to_hybrid[short_id] = hybrid_id
        self._hybrid_to_short[hybrid_id] = short_id

        return short_id

    def to_short(self, hybrid_id: str) -> str:
        """Convert hybrid ID to short ID.

        Args:
            hybrid_id: Application hybrid ID

        Returns:
            Short ID string

        Raises:
            KeyError: If hybrid_id was never registered via add()
        """
        return self._hybrid_to_short[hybrid_id]

    def to_hybrid(self, short_id: str) -> str:
        """Convert short ID to hybrid ID.

        Args:
            short_id: Short ID from LLM response

        Returns:
            Application hybrid ID

        Raises:
            KeyError: If short_id is invalid (never registered)
        """
        return self._short_to_hybrid[short_id]

    def try_to_hybrid(self, short_id: str) -> Optional[str]:
        """Safely convert short ID, returning None if invalid.

        Use this when parsing LLM responses to handle hallucinated IDs.

        Args:
            short_id: Short ID from LLM response

        Returns:
            Application hybrid ID, or None if short_id is invalid

        Example:
            >>> mapper = LLMIDMapper()
            >>> mapper.add("abc123")
            "1"
            >>> mapper.try_to_hybrid("1")
            "abc123"
            >>> mapper.try_to_hybrid("999")  # Invalid ID
            None
        """
        return self._short_to_hybrid.get(short_id)

    def __len__(self) -> int:
        """Return number of registered IDs.

        Example:
            >>> mapper = LLMIDMapper()
            >>> mapper.add("abc123")
            "1"
            >>> mapper.add("def456")
            "2"
            >>> len(mapper)
            2
        """
        return len(self._short_to_hybrid)
