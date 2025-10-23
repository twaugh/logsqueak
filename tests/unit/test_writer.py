"""Unit tests for knowledge writer module.

Tests section finding, provenance links, child bullet addition,
and fallback logic for adding knowledge to pages.
"""

from datetime import date
from pathlib import Path
from unittest.mock import Mock

import pytest

from logsqueak.integration.writer import (
    _add_child_bullet,
    _add_provenance_link,
    _add_to_page_end,
    _find_section_recursive,
    _find_target_section,
    add_knowledge_to_page,
    write_page_safely,
)
from logsqueak.logseq.parser import LogseqBlock, LogseqOutline
from logsqueak.models.knowledge import ActionType, KnowledgeBlock
from logsqueak.models.page import TargetPage


class TestProvenanceLinks:
    """Test provenance link addition (T037)."""

    def test_add_provenance_link(self):
        """Test adding provenance link to content."""
        content = "Use PostgreSQL for database"
        source_date = date(2025, 1, 15)

        result = _add_provenance_link(content, source_date)

        assert result == "Use PostgreSQL for database [[2025-01-15]]"

    def test_provenance_link_format(self):
        """Test provenance link uses correct date format."""
        content = "Test content"
        source_date = date(2025, 10, 5)

        result = _add_provenance_link(content, source_date)

        assert "[[2025-10-05]]" in result

    def test_provenance_link_with_existing_links(self):
        """Test adding provenance to content with existing links."""
        content = "Related to [[Other Page]]"
        source_date = date(2025, 1, 15)

        result = _add_provenance_link(content, source_date)

        assert "[[Other Page]]" in result
        assert "[[2025-01-15]]" in result


class TestSectionFinding:
    """Test target section finding (T036)."""

    def test_find_target_section_none(self):
        """Test finding section when path is None (page root)."""
        outline = LogseqOutline(blocks=[], source_text="")

        result = _find_target_section(outline, None)

        assert result is None

    def test_find_target_section_empty_path(self):
        """Test finding section when path is empty list."""
        outline = LogseqOutline(blocks=[], source_text="")

        result = _find_target_section(outline, [])

        assert result is None

    def test_find_section_single_level(self):
        """Test finding section at single level."""
        blocks = [
            LogseqBlock(content="## Tech Stack", indent_level=0),
            LogseqBlock(content="## Database", indent_level=0),
        ]
        outline = LogseqOutline(blocks=blocks, source_text="")

        result = _find_target_section(outline, ["Tech Stack"])

        assert result is not None
        assert "Tech Stack" in result.content

    def test_find_section_nested(self):
        """Test finding nested section."""
        parent = LogseqBlock(content="## Tech Stack", indent_level=0)
        child = LogseqBlock(content="## Database", indent_level=1)
        parent.children = [child]

        outline = LogseqOutline(blocks=[parent], source_text="")

        result = _find_target_section(outline, ["Tech Stack", "Database"])

        assert result is not None
        assert "Database" in result.content

    def test_find_section_not_found(self):
        """Test section finding when section doesn't exist."""
        blocks = [LogseqBlock(content="## Existing Section", indent_level=0)]
        outline = LogseqOutline(blocks=blocks, source_text="")

        result = _find_target_section(outline, ["Nonexistent"])

        assert result is None

    def test_find_section_partial_match(self):
        """Test section finding with partial text match."""
        blocks = [LogseqBlock(content="Tech Stack Overview", indent_level=0)]
        outline = LogseqOutline(blocks=blocks, source_text="")

        result = _find_target_section(outline, ["Tech Stack"])

        assert result is not None


class TestChildBulletAddition:
    """Test child bullet addition (T038)."""

    def test_add_child_bullet(self):
        """Test adding child bullet to parent block."""
        parent = LogseqBlock(content="## Parent", indent_level=0, children=[])

        _add_child_bullet(parent, "Child content [[2025-01-15]]", ActionType.ADD_CHILD)

        assert len(parent.children) == 1
        assert parent.children[0].content == "Child content [[2025-01-15]]"
        assert parent.children[0].indent_level == 1

    def test_add_child_bullet_to_existing_children(self):
        """Test adding child bullet when parent already has children."""
        existing_child = LogseqBlock(content="Existing child", indent_level=1)
        parent = LogseqBlock(
            content="## Parent", indent_level=0, children=[existing_child]
        )

        _add_child_bullet(parent, "New child [[2025-01-15]]", ActionType.ADD_CHILD)

        assert len(parent.children) == 2
        assert parent.children[1].content == "New child [[2025-01-15]]"

    def test_child_bullet_indent_level(self):
        """Test that child bullet has correct indent level."""
        parent = LogseqBlock(content="Nested parent", indent_level=2, children=[])

        _add_child_bullet(parent, "Content [[2025-01-15]]", ActionType.ADD_CHILD)

        assert parent.children[0].indent_level == 3


class TestPageEndFallback:
    """Test fallback to page end (T039)."""

    def test_add_to_page_end_empty_page(self):
        """Test adding to end of empty page."""
        outline = LogseqOutline(blocks=[], source_text="")

        _add_to_page_end(outline, "New content [[2025-01-15]]")

        assert len(outline.blocks) == 1
        assert outline.blocks[0].content == "New content [[2025-01-15]]"
        assert outline.blocks[0].indent_level == 0

    def test_add_to_page_end_existing_content(self):
        """Test adding to end of page with existing content."""
        existing_block = LogseqBlock(content="Existing content", indent_level=0)
        outline = LogseqOutline(blocks=[existing_block], source_text="")

        _add_to_page_end(outline, "New content [[2025-01-15]]")

        assert len(outline.blocks) == 2
        assert outline.blocks[1].content == "New content [[2025-01-15]]"

    def test_add_to_page_end_root_level(self):
        """Test that fallback adds at root level."""
        outline = LogseqOutline(blocks=[], source_text="")

        _add_to_page_end(outline, "Content [[2025-01-15]]")

        assert outline.blocks[0].indent_level == 0


class TestKnowledgeAddition:
    """Test full knowledge addition workflow."""

    def test_add_knowledge_with_section(self):
        """Test adding knowledge to specific section."""
        parent = LogseqBlock(content="## Tech Stack", indent_level=0, children=[])
        outline = LogseqOutline(blocks=[parent], source_text="")

        target_page = Mock(spec=TargetPage)
        target_page.outline = outline

        knowledge = KnowledgeBlock(
            content="Use PostgreSQL",
            source_date=date(2025, 1, 15),
            confidence=0.9,
            target_page="Test",
            target_section=["Tech Stack"],
            suggested_action=ActionType.ADD_CHILD,
        )

        add_knowledge_to_page(target_page, knowledge)

        assert len(parent.children) == 1
        assert "Use PostgreSQL" in parent.children[0].content
        assert "[[2025-01-15]]" in parent.children[0].content

    def test_add_knowledge_fallback_to_page_end(self):
        """Test adding knowledge falls back to page end when section not found."""
        outline = LogseqOutline(blocks=[], source_text="")

        target_page = Mock(spec=TargetPage)
        target_page.outline = outline

        knowledge = KnowledgeBlock(
            content="Use PostgreSQL",
            source_date=date(2025, 1, 15),
            confidence=0.9,
            target_page="Test",
            target_section=["Nonexistent Section"],
            suggested_action=ActionType.ADD_CHILD,
        )

        add_knowledge_to_page(target_page, knowledge)

        # Should add to page end
        assert len(outline.blocks) == 1
        assert "Use PostgreSQL" in outline.blocks[0].content
        assert "[[2025-01-15]]" in outline.blocks[0].content

    def test_add_knowledge_to_page_root(self):
        """Test adding knowledge to page root (no section)."""
        outline = LogseqOutline(blocks=[], source_text="")

        target_page = Mock(spec=TargetPage)
        target_page.outline = outline

        knowledge = KnowledgeBlock(
            content="General note",
            source_date=date(2025, 1, 15),
            confidence=0.9,
            target_page="Test",
            target_section=None,
            suggested_action=ActionType.ADD_CHILD,
        )

        add_knowledge_to_page(target_page, knowledge)

        assert len(outline.blocks) == 1
        assert "General note" in outline.blocks[0].content


class TestSafeFileWriting:
    """Test safe file writing (T040)."""

    def test_write_page_safely(self, tmp_path):
        """Test writing page to disk safely."""
        page_file = tmp_path / "Test Page.md"
        outline = LogseqOutline(
            blocks=[LogseqBlock(content="Test content", indent_level=0)],
            source_text="- Test content",
        )

        target_page = Mock(spec=TargetPage)
        target_page.file_path = page_file
        target_page.outline = outline

        write_page_safely(target_page)

        assert page_file.exists()
        content = page_file.read_text()
        assert "Test content" in content

    def test_write_page_with_custom_path(self, tmp_path):
        """Test writing page to custom output path."""
        original_path = tmp_path / "original.md"
        custom_path = tmp_path / "custom.md"

        outline = LogseqOutline(
            blocks=[LogseqBlock(content="Content", indent_level=0)],
            source_text="- Content",
        )

        target_page = Mock(spec=TargetPage)
        target_page.file_path = original_path
        target_page.outline = outline

        write_page_safely(target_page, custom_path)

        assert custom_path.exists()
        assert not original_path.exists()

    def test_write_page_error_cleanup(self, tmp_path):
        """Test that temp file is cleaned up on error."""
        page_file = tmp_path / "test.md"
        outline = Mock(spec=LogseqOutline)
        outline.render.side_effect = Exception("Render failed")

        target_page = Mock(spec=TargetPage)
        target_page.file_path = page_file
        target_page.outline = outline

        with pytest.raises(IOError):
            write_page_safely(target_page)

        # Temp file should be cleaned up
        temp_file = page_file.with_suffix('.tmp')
        assert not temp_file.exists()
