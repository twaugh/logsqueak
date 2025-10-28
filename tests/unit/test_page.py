"""Unit tests for TargetPage and PageIndex models."""

from pathlib import Path
from textwrap import dedent

import pytest

from logsqueak.models.page import ConventionType, TargetPage, _detect_convention
from logsqueak.logseq.parser import LogseqOutline


class TestTargetPage:
    """Tests for TargetPage model."""

    def test_load_existing_page(self, tmp_path):
        """Test loading an existing page."""
        graph_path = tmp_path
        pages_dir = graph_path / "pages"
        pages_dir.mkdir()

        page_file = pages_dir / "Test Page.md"
        content = dedent(
            """\
            - ## Timeline
              - Q1 2025
            - ## Status
              - Active"""
        )
        page_file.write_text(content)

        page = TargetPage.load(graph_path, "Test Page")

        assert page is not None
        assert page.name == "Test Page"
        assert page.file_path == page_file
        assert len(page.outline.blocks) == 2

    def test_load_nonexistent_page(self, tmp_path):
        """Test loading nonexistent page returns None (FR-009)."""
        graph_path = tmp_path
        pages_dir = graph_path / "pages"
        pages_dir.mkdir()

        page = TargetPage.load(graph_path, "Nonexistent Page")

        assert page is None

    def test_find_section_found(self, tmp_path):
        """Test finding section that exists."""
        graph_path = tmp_path
        pages_dir = graph_path / "pages"
        pages_dir.mkdir()

        page_file = pages_dir / "Test Page.md"
        content = dedent(
            """\
            - ## Timeline
              - Q1 2025
            - ## Status
              - Active"""
        )
        page_file.write_text(content)

        page = TargetPage.load(graph_path, "Test Page")
        section = page.find_section("Timeline")

        assert section is not None
        assert "Timeline" in section.content[0]

    def test_find_section_not_found(self, tmp_path):
        """Test finding section that doesn't exist."""
        graph_path = tmp_path
        pages_dir = graph_path / "pages"
        pages_dir.mkdir()

        page_file = pages_dir / "Test Page.md"
        content = "- Item 1\n- Item 2"
        page_file.write_text(content)

        page = TargetPage.load(graph_path, "Test Page")
        section = page.find_section("Nonexistent")

        assert section is None

    def test_has_duplicate_true(self, tmp_path):
        """Test FR-017: Duplicate detection returns True."""
        graph_path = tmp_path
        pages_dir = graph_path / "pages"
        pages_dir.mkdir()

        page_file = pages_dir / "Test Page.md"
        content = "- Existing content here"
        page_file.write_text(content)

        page = TargetPage.load(graph_path, "Test Page")

        assert page.has_duplicate("Existing content here") is True

    def test_has_duplicate_false(self, tmp_path):
        """Test FR-017: Duplicate detection returns False."""
        graph_path = tmp_path
        pages_dir = graph_path / "pages"
        pages_dir.mkdir()

        page_file = pages_dir / "Test Page.md"
        content = "- Existing content"
        page_file.write_text(content)

        page = TargetPage.load(graph_path, "Test Page")

        assert page.has_duplicate("New unique content") is False


class TestDetectConvention:
    """Tests for convention detection."""

    def test_detect_plain_bullets(self):
        """Test detecting plain bullet convention."""
        markdown = dedent(
            """\
            - Section A
              - Item 1
            - Section B
              - Item 2"""
        )
        outline = LogseqOutline.parse(markdown)

        convention = _detect_convention(outline)

        assert convention == ConventionType.PLAIN_BULLETS

    def test_detect_heading_bullets(self):
        """Test detecting heading bullet convention."""
        markdown = dedent(
            """\
            - ## Section A
              - ## Nested heading
            - ## Section B
              - ## Another heading"""
        )
        outline = LogseqOutline.parse(markdown)

        convention = _detect_convention(outline)

        # Should detect as heading bullets (all bullets have headings)
        assert convention == ConventionType.HEADING_BULLETS

    def test_detect_mixed_convention(self):
        """Test detecting mixed convention."""
        markdown = dedent(
            """\
            - ## Section A
              - Plain item
            - Plain section
              - Another item"""
        )
        outline = LogseqOutline.parse(markdown)

        convention = _detect_convention(outline)

        assert convention == ConventionType.MIXED

    def test_detect_empty_outline(self):
        """Test detecting convention in empty outline."""
        outline = LogseqOutline.parse("")

        convention = _detect_convention(outline)

        # Empty outline defaults to plain bullets
        assert convention == ConventionType.PLAIN_BULLETS


class TestPageIndex:
    """Tests for PageIndex model.

    Note: Full PageIndex tests require sentence-transformers and are expensive.
    These tests focus on basic functionality without actual embedding.
    """

    def test_build_empty_graph(self, tmp_path):
        """Test building index from empty graph."""
        from logsqueak.models.page import PageIndex

        graph_path = tmp_path

        # PageIndex.build should handle missing pages directory gracefully
        index = PageIndex.build(graph_path)

        assert len(index.pages) == 0
        assert index.embeddings.size == 0

    def test_build_with_single_page(self, tmp_path):
        """Test building index with single page."""
        from logsqueak.models.page import PageIndex

        graph_path = tmp_path
        pages_dir = graph_path / "pages"
        pages_dir.mkdir()

        page_file = pages_dir / "Test Page.md"
        page_file.write_text("- Test content")

        index = PageIndex.build(graph_path)

        assert len(index.pages) == 1
        assert index.pages[0].name == "Test Page"

    @pytest.mark.slow
    def test_find_similar_empty_index(self, tmp_path):
        """Test find_similar with empty index."""
        from logsqueak.models.page import PageIndex

        graph_path = tmp_path

        index = PageIndex.build(graph_path)
        results = index.find_similar("test query")

        assert results == []

    @pytest.mark.slow
    def test_caching_mechanism(self, tmp_path):
        """Test that embedding caching works."""
        from logsqueak.models.page import PageIndex

        graph_path = tmp_path
        pages_dir = graph_path / "pages"
        pages_dir.mkdir()

        page_file = pages_dir / "Test Page.md"
        page_file.write_text("- Test content")

        cache_dir = tmp_path / "cache"

        # First build - should create cache
        index1 = PageIndex.build(graph_path, cache_dir=cache_dir)

        # Cache file should exist
        cache_file = cache_dir / "Test Page.pkl"
        assert cache_file.exists()

        # Second build - should use cache
        index2 = PageIndex.build(graph_path, cache_dir=cache_dir)

        assert len(index2.pages) == 1
