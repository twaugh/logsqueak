"""Unit tests for full-context generation and hashing (M1.2)."""

import hashlib
from textwrap import dedent

import pytest

from logsqueak.logseq.context import (
    generate_chunks,
    generate_content_hash,
    generate_full_context,
)
from logsqueak.logseq.parser import LogseqBlock, LogseqOutline


class TestGenerateFullContext:
    """Tests for generate_full_context function."""

    def test_block_with_no_parents(self):
        """Test full context for root-level block."""
        block = LogseqBlock(content="Root content", indent_level=0)

        context = generate_full_context(block, [])

        assert context == "- Root content"

    def test_block_with_one_parent(self):
        """Test full context for child block."""
        parent = LogseqBlock(content="Parent", indent_level=0)
        child = LogseqBlock(content="Child", indent_level=1)

        context = generate_full_context(child, [parent])

        assert context == "- Parent\n  - Child"

    def test_block_with_multiple_parents(self):
        """Test full context for deeply nested block."""
        root = LogseqBlock(content="Root", indent_level=0)
        child = LogseqBlock(content="Child", indent_level=1)
        grandchild = LogseqBlock(content="Grandchild", indent_level=2)

        context = generate_full_context(grandchild, [root, child])

        assert context == "- Root\n  - Child\n    - Grandchild"

    def test_parent_order_matters(self):
        """Test that parent order affects context."""
        block_a = LogseqBlock(content="A", indent_level=0)
        block_b = LogseqBlock(content="B", indent_level=0)
        block_c = LogseqBlock(content="C", indent_level=0)

        context_abc = generate_full_context(block_c, [block_a, block_b])
        context_bac = generate_full_context(block_c, [block_b, block_a])

        assert context_abc == "- A\n  - B\n    - C"
        assert context_bac == "- B\n  - A\n    - C"
        assert context_abc != context_bac


class TestGenerateContentHash:
    """Tests for generate_content_hash function."""

    def test_hash_is_deterministic(self):
        """Test that same content produces same hash."""
        content = "Test content"

        hash1 = generate_content_hash(content)
        hash2 = generate_content_hash(content)

        assert hash1 == hash2

    def test_hash_is_md5(self):
        """Test that hash matches expected MD5."""
        content = "Test content"

        hash_result = generate_content_hash(content)
        expected = hashlib.md5(content.encode()).hexdigest()

        assert hash_result == expected

    def test_different_content_different_hash(self):
        """Test that different content produces different hash."""
        hash1 = generate_content_hash("Content A")
        hash2 = generate_content_hash("Content B")

        assert hash1 != hash2

    def test_hash_is_hexadecimal_string(self):
        """Test that hash is 32-character hex string."""
        hash_result = generate_content_hash("Test")

        assert len(hash_result) == 32
        assert all(c in "0123456789abcdef" for c in hash_result)


class TestGenerateChunks:
    """Tests for generate_chunks function."""

    def test_single_root_block(self):
        """Test chunking single root block."""
        markdown = "- Root block"
        outline = LogseqOutline.parse(markdown)

        chunks = generate_chunks(outline)

        assert len(chunks) == 1
        block, context, hybrid_id = chunks[0]
        assert block.content == "Root block"
        assert context == "- Root block"
        # Should be hash since no id:: property
        assert len(hybrid_id) == 32  # MD5 hex length

    def test_multiple_root_blocks(self):
        """Test chunking multiple root blocks."""
        markdown = dedent(
            """\
            - Block 1
            - Block 2
            - Block 3"""
        )
        outline = LogseqOutline.parse(markdown)

        chunks = generate_chunks(outline)

        assert len(chunks) == 3
        assert chunks[0][1] == "- Block 1"
        assert chunks[1][1] == "- Block 2"
        assert chunks[2][1] == "- Block 3"

    def test_nested_blocks(self):
        """Test chunking nested block structure."""
        markdown = dedent(
            """\
            - Parent
              - Child
                - Grandchild"""
        )
        outline = LogseqOutline.parse(markdown)

        chunks = generate_chunks(outline)

        assert len(chunks) == 3
        # Parent
        assert chunks[0][1] == "- Parent"
        # Child has parent context
        assert chunks[1][1] == "- Parent\n  - Child"
        # Grandchild has full context
        assert chunks[2][1] == "- Parent\n  - Child\n    - Grandchild"

    def test_hybrid_id_uses_explicit_id_property(self):
        """Test that blocks with id:: use that as hybrid ID."""
        markdown = dedent(
            """\
            - Block with ID
              id:: explicit-id-123"""
        )
        outline = LogseqOutline.parse(markdown)

        chunks = generate_chunks(outline)

        assert len(chunks) == 1
        block, context, hybrid_id = chunks[0]
        assert hybrid_id == "explicit-id-123"  # Uses id:: property

    def test_hybrid_id_uses_hash_without_id_property(self):
        """Test that blocks without id:: use content hash."""
        markdown = "- Block without ID"
        outline = LogseqOutline.parse(markdown)

        chunks = generate_chunks(outline)

        block, context, hybrid_id = chunks[0]
        expected_hash = generate_content_hash("- Block without ID")
        assert hybrid_id == expected_hash

    def test_sibling_blocks_different_hashes(self):
        """Test that sibling blocks with different content get different hashes."""
        markdown = dedent(
            """\
            - Block A
            - Block B"""
        )
        outline = LogseqOutline.parse(markdown)

        chunks = generate_chunks(outline)

        hash_a = chunks[0][2]
        hash_b = chunks[1][2]
        assert hash_a != hash_b

    def test_identical_content_different_locations(self):
        """Test that identical content in different locations gets different hashes."""
        markdown = dedent(
            """\
            - Parent A
              - Same content
            - Parent B
              - Same content"""
        )
        outline = LogseqOutline.parse(markdown)

        chunks = generate_chunks(outline)

        # Find the two "Same content" blocks
        same_content_chunks = [c for c in chunks if c[0].content == "Same content"]
        assert len(same_content_chunks) == 2

        # Different parents mean different full context
        assert same_content_chunks[0][1] == "- Parent A\n  - Same content"
        assert same_content_chunks[1][1] == "- Parent B\n  - Same content"

        # Different context means different hash
        assert same_content_chunks[0][2] != same_content_chunks[1][2]

    def test_mixed_explicit_and_hashed_ids(self):
        """Test chunking with mix of explicit IDs and hashed IDs."""
        markdown = dedent(
            """\
            - Block with ID
              id:: explicit-123
              - Child without ID
            - Block without ID"""
        )
        outline = LogseqOutline.parse(markdown)

        chunks = generate_chunks(outline)

        assert len(chunks) == 3
        # First block has explicit ID
        assert chunks[0][2] == "explicit-123"
        # Child has hash (parent context included)
        assert len(chunks[1][2]) == 32
        # Last block has hash
        assert len(chunks[2][2]) == 32

    def test_empty_outline(self):
        """Test chunking empty outline."""
        outline = LogseqOutline.parse("")

        chunks = generate_chunks(outline)

        assert len(chunks) == 0

    def test_complex_nested_structure(self):
        """Test chunking complex nested structure with multiple branches."""
        markdown = dedent(
            """\
            - Root 1
              id:: root-1
              - Child 1.1
              - Child 1.2
                - Grandchild 1.2.1
            - Root 2
              - Child 2.1"""
        )
        outline = LogseqOutline.parse(markdown)

        chunks = generate_chunks(outline)

        assert len(chunks) == 6
        # Root 1 has explicit ID
        assert chunks[0][2] == "root-1"
        # Child 1.1 has parent context
        assert chunks[1][1] == "- Root 1\n  - Child 1.1"
        # Grandchild has full ancestry
        assert chunks[3][1] == "- Root 1\n  - Child 1.2\n    - Grandchild 1.2.1"
