"""Unit tests for LLMIDMapper."""

import pytest
from logsqueak.utils.llm_id_mapper import LLMIDMapper


class TestLLMIDMapper:
    """Test LLM ID mapping functionality."""

    def test_init(self):
        """Test mapper initializes empty."""
        mapper = LLMIDMapper()
        assert len(mapper) == 0

    def test_add_single_id(self):
        """Test adding a single hybrid ID."""
        mapper = LLMIDMapper()
        short_id = mapper.add("abc123")

        assert short_id == "1"
        assert len(mapper) == 1

    def test_add_multiple_ids(self):
        """Test adding multiple hybrid IDs returns sequential short IDs."""
        mapper = LLMIDMapper()

        short1 = mapper.add("abc123")
        short2 = mapper.add("def456")
        short3 = mapper.add("ghi789")

        assert short1 == "1"
        assert short2 == "2"
        assert short3 == "3"
        assert len(mapper) == 3

    def test_add_idempotent(self):
        """Test adding same hybrid ID twice returns same short ID."""
        mapper = LLMIDMapper()

        short1 = mapper.add("abc123")
        short2 = mapper.add("abc123")  # Same hybrid ID

        assert short1 == "1"
        assert short2 == "1"  # Same short ID
        assert len(mapper) == 1  # Only one mapping

    def test_add_uuid_format(self):
        """Test adding UUID-format hybrid IDs."""
        mapper = LLMIDMapper()

        uuid1 = "65f3a8e0-1234-5678-9abc-def012345678"
        uuid2 = "12345678-abcd-efef-1234-567890abcdef"

        short1 = mapper.add(uuid1)
        short2 = mapper.add(uuid2)

        assert short1 == "1"
        assert short2 == "2"

    def test_add_content_hash_format(self):
        """Test adding content hash hybrid IDs."""
        mapper = LLMIDMapper()

        hash1 = "a" * 32  # 32 hex chars
        hash2 = "b" * 32

        short1 = mapper.add(hash1)
        short2 = mapper.add(hash2)

        assert short1 == "1"
        assert short2 == "2"

    def test_to_short_valid(self):
        """Test converting hybrid ID to short ID."""
        mapper = LLMIDMapper()
        mapper.add("abc123")
        mapper.add("def456")

        assert mapper.to_short("abc123") == "1"
        assert mapper.to_short("def456") == "2"

    def test_to_short_invalid_raises_keyerror(self):
        """Test converting unregistered hybrid ID raises KeyError."""
        mapper = LLMIDMapper()
        mapper.add("abc123")

        with pytest.raises(KeyError):
            mapper.to_short("not_registered")

    def test_to_hybrid_valid(self):
        """Test converting short ID to hybrid ID."""
        mapper = LLMIDMapper()
        mapper.add("abc123")
        mapper.add("def456")

        assert mapper.to_hybrid("1") == "abc123"
        assert mapper.to_hybrid("2") == "def456"

    def test_to_hybrid_invalid_raises_keyerror(self):
        """Test converting invalid short ID raises KeyError."""
        mapper = LLMIDMapper()
        mapper.add("abc123")

        with pytest.raises(KeyError):
            mapper.to_hybrid("999")

    def test_try_to_hybrid_valid(self):
        """Test safely converting valid short ID."""
        mapper = LLMIDMapper()
        mapper.add("abc123")
        mapper.add("def456")

        assert mapper.try_to_hybrid("1") == "abc123"
        assert mapper.try_to_hybrid("2") == "def456"

    def test_try_to_hybrid_invalid_returns_none(self):
        """Test safely converting invalid short ID returns None."""
        mapper = LLMIDMapper()
        mapper.add("abc123")

        assert mapper.try_to_hybrid("999") is None
        assert mapper.try_to_hybrid("abc") is None
        assert mapper.try_to_hybrid("") is None

    def test_try_to_hybrid_hallucinated_ids(self):
        """Test handling hallucinated IDs from LLM."""
        mapper = LLMIDMapper()

        # Register IDs 1-5
        for i in range(5):
            mapper.add(f"block_{i}")

        # Valid IDs
        assert mapper.try_to_hybrid("1") is not None
        assert mapper.try_to_hybrid("5") is not None

        # Hallucinated IDs (common LLM errors)
        assert mapper.try_to_hybrid("6") is None  # Next sequential
        assert mapper.try_to_hybrid("0") is None  # Off-by-one
        assert mapper.try_to_hybrid("100") is None  # Random high number
        assert mapper.try_to_hybrid("-1") is None  # Negative

    def test_bidirectional_mapping(self):
        """Test round-trip conversion."""
        mapper = LLMIDMapper()

        hybrid_ids = [
            "65f3a8e0-1234-5678-9abc-def012345678",
            "a" * 32,
            "12345678-abcd-efef-1234-567890abcdef",
        ]

        # Add all IDs
        short_ids = [mapper.add(h) for h in hybrid_ids]

        # Test round-trip: hybrid → short → hybrid
        for hybrid_id in hybrid_ids:
            short = mapper.to_short(hybrid_id)
            back = mapper.to_hybrid(short)
            assert back == hybrid_id

        # Test round-trip: short → hybrid → short
        for short_id in short_ids:
            hybrid = mapper.to_hybrid(short_id)
            back = mapper.to_short(hybrid)
            assert back == short_id

    def test_large_number_of_ids(self):
        """Test mapper handles many IDs efficiently."""
        mapper = LLMIDMapper()

        # Add 100 IDs
        hybrid_ids = [f"block_{i:03d}" for i in range(100)]
        for hybrid_id in hybrid_ids:
            mapper.add(hybrid_id)

        assert len(mapper) == 100

        # Verify all mappings work
        assert mapper.to_short("block_000") == "1"
        assert mapper.to_short("block_099") == "100"
        assert mapper.to_hybrid("1") == "block_000"
        assert mapper.to_hybrid("100") == "block_099"

    def test_ordering_preserved(self):
        """Test short IDs are assigned in registration order."""
        mapper = LLMIDMapper()

        # Add in specific order
        ids = ["zzz", "aaa", "mmm", "bbb"]
        short_ids = [mapper.add(id_) for id_ in ids]

        # Short IDs should be sequential regardless of ID values
        assert short_ids == ["1", "2", "3", "4"]

        # Verify mapping preserved order
        assert mapper.to_hybrid("1") == "zzz"
        assert mapper.to_hybrid("2") == "aaa"
        assert mapper.to_hybrid("3") == "mmm"
        assert mapper.to_hybrid("4") == "bbb"

    def test_empty_hybrid_id(self):
        """Test handling empty hybrid ID."""
        mapper = LLMIDMapper()

        # Empty string is technically valid (unusual but allowed)
        short = mapper.add("")
        assert short == "1"
        assert mapper.to_hybrid("1") == ""

    def test_duplicate_prevention(self):
        """Test duplicate hybrid IDs don't create duplicate short IDs."""
        mapper = LLMIDMapper()

        # Add same ID multiple times
        short1 = mapper.add("abc123")
        short2 = mapper.add("abc123")
        short3 = mapper.add("abc123")

        # All return same short ID
        assert short1 == short2 == short3 == "1"

        # Only one mapping exists
        assert len(mapper) == 1

    def test_isolated_mappers(self):
        """Test multiple mapper instances are independent."""
        mapper1 = LLMIDMapper()
        mapper2 = LLMIDMapper()

        # Same hybrid ID in different mappers
        short1 = mapper1.add("abc123")
        short2 = mapper2.add("abc123")

        # Both get "1" (independent counters)
        assert short1 == "1"
        assert short2 == "1"

        # But they're independent
        mapper1.add("def456")
        assert len(mapper1) == 2
        assert len(mapper2) == 1

    def test_str_integer_short_ids(self):
        """Test short IDs are string representations of integers."""
        mapper = LLMIDMapper()

        for i in range(10):
            short = mapper.add(f"block_{i}")
            assert short == str(i + 1)
            assert short.isdigit()
