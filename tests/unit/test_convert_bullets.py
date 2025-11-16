"""Unit tests for _convert_to_logseq_bullets method."""

import pytest
from unittest.mock import MagicMock
from logsqueak.tui.screens.content_editing import Phase2Screen
from logseq_outline.graph import GraphPaths


@pytest.fixture
def phase2_screen():
    """Create a Phase2Screen instance for testing."""
    graph_paths = MagicMock(spec=GraphPaths)
    screen = Phase2Screen(
        blocks=[],
        edited_content=[],
        journals={},
        graph_paths=graph_paths,
        llm_client=None,
        rag_search=None,
        auto_start_workers=False
    )
    return screen


def test_convert_to_logseq_bullets_with_spaces(phase2_screen):
    """Test conversion with space indentation."""
    text = "Parent\n  Child\n    Grandchild"
    expected = "- Parent\n  - Child\n    - Grandchild"

    result = phase2_screen._convert_to_logseq_bullets(text)
    assert result == expected


def test_convert_to_logseq_bullets_with_tabs(phase2_screen):
    """Test conversion with tab indentation."""
    text = "Parent\n\tChild\n\t\tGrandchild"
    expected = "- Parent\n\t- Child\n\t\t- Grandchild"

    result = phase2_screen._convert_to_logseq_bullets(text)
    assert result == expected


def test_convert_to_logseq_bullets_mixed_indentation(phase2_screen):
    """Test conversion with mixed tab and space indentation."""
    # First level: tabs, second level: spaces
    text = "Parent\n\tChild with tab\n  Child with spaces"
    expected = "- Parent\n\t- Child with tab\n  - Child with spaces"

    result = phase2_screen._convert_to_logseq_bullets(text)
    assert result == expected


def test_convert_to_logseq_bullets_already_has_bullets(phase2_screen):
    """Test conversion when bullets already exist."""
    text = "- Parent\n  - Child\n    - Grandchild"
    expected = "- Parent\n  - Child\n    - Grandchild"

    result = phase2_screen._convert_to_logseq_bullets(text)
    assert result == expected


def test_convert_to_logseq_bullets_empty_lines(phase2_screen):
    """Test conversion with empty lines."""
    text = "Parent\n\nChild\n  Grandchild"
    expected = "- Parent\n\n- Child\n  - Grandchild"

    result = phase2_screen._convert_to_logseq_bullets(text)
    assert result == expected
