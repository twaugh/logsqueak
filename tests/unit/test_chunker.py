"""Unit tests for page chunking functionality."""

from pathlib import Path
from textwrap import dedent

import pytest

from logsqueak.logseq.parser import LogseqOutline
from logsqueak.rag.chunker import Chunk, chunk_page, chunk_page_file


class TestChunkDataclass:
    """Tests for Chunk dataclass."""

    def test_chunk_creation(self):
        """Test creating a Chunk instance."""
        chunk = Chunk(
            full_context_text="- Parent\n  - Child",
            hybrid_id="test-id-123",
            page_name="Test Page",
            block_content="Child",
            metadata={"page_name": "Test Page", "block_content": "Child"},
        )

        assert chunk.full_context_text == "- Parent\n  - Child"
        assert chunk.hybrid_id == "test-id-123"
        assert chunk.page_name == "Test Page"
        assert chunk.block_content == "Child"
        assert chunk.metadata["page_name"] == "Test Page"


class TestChunkPage:
    """Tests for chunk_page function."""

    def test_chunk_single_block(self):
        """Test chunking page with single block."""
        markdown = "- Single block"
        outline = LogseqOutline.parse(markdown)

        chunks = chunk_page(outline, "Test Page")

        assert len(chunks) == 1
        assert chunks[0].page_name == "Test Page"
        assert chunks[0].block_content == "Single block"
        assert chunks[0].full_context_text == "- Single block"

    def test_chunk_multiple_root_blocks(self):
        """Test chunking page with multiple root blocks."""
        markdown = dedent(
            """\
            - Block 1
            - Block 2
            - Block 3"""
        )
        outline = LogseqOutline.parse(markdown)

        chunks = chunk_page(outline, "Test Page")

        assert len(chunks) == 3
        assert chunks[0].block_content == "Block 1"
        assert chunks[1].block_content == "Block 2"
        assert chunks[2].block_content == "Block 3"

        # All chunks should have same page name
        for chunk in chunks:
            assert chunk.page_name == "Test Page"

    def test_chunk_nested_blocks(self):
        """Test chunking with nested block structure."""
        markdown = dedent(
            """\
            - Parent
              - Child
                - Grandchild"""
        )
        outline = LogseqOutline.parse(markdown)

        chunks = chunk_page(outline, "Nested Page")

        assert len(chunks) == 3

        # Parent block
        assert chunks[0].block_content == "Parent"
        assert chunks[0].full_context_text == "- Parent"

        # Child has parent context
        assert chunks[1].block_content == "Child"
        assert chunks[1].full_context_text == "- Parent\n  - Child"

        # Grandchild has full ancestry
        assert chunks[2].block_content == "Grandchild"
        assert chunks[2].full_context_text == "- Parent\n  - Child\n    - Grandchild"

    def test_chunk_with_explicit_ids(self):
        """Test chunking blocks with explicit id:: properties."""
        markdown = dedent(
            """\
            - Block with ID
              id:: explicit-id-123"""
        )
        outline = LogseqOutline.parse(markdown)

        chunks = chunk_page(outline, "ID Page")

        assert len(chunks) == 1
        # ID should be globally unique: page_name::hybrid_id
        assert chunks[0].hybrid_id == "ID Page::explicit-id-123"
        # Local ID stored in metadata
        assert chunks[0].metadata["local_hybrid_id"] == "explicit-id-123"

    def test_chunk_without_explicit_ids(self):
        """Test chunking blocks without id:: (uses content hash)."""
        markdown = "- Block without ID"
        outline = LogseqOutline.parse(markdown)

        chunks = chunk_page(outline, "Hash Page")

        assert len(chunks) == 1
        # Should be globally unique: page_name::hash
        # Format: "Hash Page::32_hex_chars"
        assert chunks[0].hybrid_id.startswith("Hash Page::")
        local_id = chunks[0].hybrid_id.split("::", 1)[1]
        assert len(local_id) == 32  # MD5 hash
        assert all(c in "0123456789abcdef" for c in local_id)

    def test_chunk_metadata_includes_page_name(self):
        """Test that metadata includes page name."""
        markdown = "- Test block"
        outline = LogseqOutline.parse(markdown)

        chunks = chunk_page(outline, "Metadata Test")

        assert chunks[0].metadata["page_name"] == "Metadata Test"

    def test_chunk_metadata_includes_block_content(self):
        """Test that metadata includes block content."""
        markdown = "- Content to test"
        outline = LogseqOutline.parse(markdown)

        chunks = chunk_page(outline, "Test")

        assert chunks[0].metadata["block_content"] == "Content to test"

    def test_chunk_metadata_includes_indent_level(self):
        """Test that metadata includes indent level."""
        markdown = dedent(
            """\
            - Root block
              - Indented block"""
        )
        outline = LogseqOutline.parse(markdown)

        chunks = chunk_page(outline, "Test")

        assert chunks[0].metadata["indent_level"] == 0
        assert chunks[1].metadata["indent_level"] == 1

    def test_chunk_empty_page(self):
        """Test chunking empty page."""
        outline = LogseqOutline.parse("")

        chunks = chunk_page(outline, "Empty")

        assert chunks == []

    def test_chunk_complex_page(self):
        """Test chunking complex page with multiple branches."""
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

        chunks = chunk_page(outline, "Complex Page")

        assert len(chunks) == 6

        # Verify page name on all chunks
        for chunk in chunks:
            assert chunk.page_name == "Complex Page"

        # Verify root block has explicit ID (globally unique)
        assert chunks[0].hybrid_id == "Complex Page::root-1"
        assert chunks[0].metadata["local_hybrid_id"] == "root-1"
        assert chunks[0].block_content == "Root 1"

        # Verify child context
        assert "Root 1" in chunks[1].full_context_text
        assert chunks[1].block_content == "Child 1.1"


class TestChunkPageFile:
    """Tests for chunk_page_file function."""

    def test_chunk_page_file(self, tmp_path):
        """Test chunking a page file."""
        page_file = tmp_path / "Test Page.md"
        page_file.write_text("- Block 1\n- Block 2", encoding="utf-8")

        chunks = chunk_page_file(page_file)

        assert len(chunks) == 2
        assert chunks[0].page_name == "Test Page"
        assert chunks[1].page_name == "Test Page"

    def test_chunk_page_file_with_nested_content(self, tmp_path):
        """Test chunking file with nested blocks."""
        page_file = tmp_path / "Nested.md"
        content = dedent(
            """\
            - Parent
              - Child"""
        )
        page_file.write_text(content, encoding="utf-8")

        chunks = chunk_page_file(page_file)

        assert len(chunks) == 2
        assert chunks[0].block_content == "Parent"
        assert chunks[1].block_content == "Child"
        assert chunks[1].full_context_text == "- Parent\n  - Child"

    def test_chunk_page_file_extracts_name_from_path(self, tmp_path):
        """Test that page name is extracted from filename."""
        page_file = tmp_path / "My Project.md"
        page_file.write_text("- Block", encoding="utf-8")

        chunks = chunk_page_file(page_file)

        assert chunks[0].page_name == "My Project"

    def test_chunk_page_file_not_found(self, tmp_path):
        """Test chunking nonexistent file raises error."""
        page_file = tmp_path / "nonexistent.md"

        with pytest.raises(FileNotFoundError):
            chunk_page_file(page_file)

    def test_chunk_page_file_with_properties(self, tmp_path):
        """Test chunking file with block properties."""
        page_file = tmp_path / "Props.md"
        content = dedent(
            """\
            - Block with props
              id:: prop-id-123
              tags:: test"""
        )
        page_file.write_text(content, encoding="utf-8")

        chunks = chunk_page_file(page_file)

        assert len(chunks) == 1
        # ID should be globally unique
        assert chunks[0].hybrid_id == "Props::prop-id-123"
        assert chunks[0].metadata["local_hybrid_id"] == "prop-id-123"
