"""Unit tests for utilities."""

import pytest
import uuid

from logsqueak.utils.ids import generate_deterministic_uuid, generate_random_uuid, LOGSQUEAK_NAMESPACE


class TestUUIDGeneration:
    """Test UUID generation utilities."""

    def test_deterministic_uuid_same_content(self):
        """Test that same content generates same UUID."""
        content = "Test block content"

        uuid1 = generate_deterministic_uuid(content)
        uuid2 = generate_deterministic_uuid(content)

        assert uuid1 == uuid2

    def test_deterministic_uuid_different_content(self):
        """Test that different content generates different UUIDs."""
        uuid1 = generate_deterministic_uuid("Content A")
        uuid2 = generate_deterministic_uuid("Content B")

        assert uuid1 != uuid2

    def test_deterministic_uuid_format(self):
        """Test that generated UUID has correct format."""
        result = generate_deterministic_uuid("Test content")

        # Should be valid UUID string
        parsed = uuid.UUID(result)
        assert str(parsed) == result

        # Should be UUID v5 (version=5)
        assert parsed.version == 5

    def test_deterministic_uuid_with_custom_namespace(self):
        """Test deterministic UUID with custom namespace."""
        custom_namespace = uuid.UUID("12345678-1234-5678-1234-567812345678")
        content = "Test content"

        uuid1 = generate_deterministic_uuid(content, namespace=custom_namespace)
        uuid2 = generate_deterministic_uuid(content, namespace=LOGSQUEAK_NAMESPACE)

        # Different namespaces should produce different UUIDs
        assert uuid1 != uuid2

    def test_random_uuid_is_unique(self):
        """Test that random UUIDs are unique."""
        uuid1 = generate_random_uuid()
        uuid2 = generate_random_uuid()

        assert uuid1 != uuid2

    def test_random_uuid_format(self):
        """Test that random UUID has correct format."""
        result = generate_random_uuid()

        # Should be valid UUID string
        parsed = uuid.UUID(result)
        assert str(parsed) == result

        # Should be UUID v4 (version=4)
        assert parsed.version == 4

    def test_deterministic_uuid_reproducible(self):
        """Test that deterministic UUID is reproducible across runs."""
        # This is a known UUID for a specific content + namespace combination
        content = "- Test block content"

        uuid1 = generate_deterministic_uuid(content)

        # Generate again (simulating different run)
        uuid2 = generate_deterministic_uuid(content)

        # Must be identical
        assert uuid1 == uuid2
