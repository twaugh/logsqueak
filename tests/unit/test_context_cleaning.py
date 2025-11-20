"""Unit tests for context cleaning utilities."""

import pytest
from logsqueak.utils.context_cleaning import strip_id_property


def test_strip_id_property_removes_id_property():
    """Test that strip_id_property() removes id:: properties from context."""
    context = """- Example block
  id:: 65f3a8e0-1234-5678-9abc-def012345678
  - Child block"""

    result = strip_id_property(context)

    assert "id::" not in result
    assert "- Example block" in result
    assert "- Child block" in result


def test_strip_id_property_removes_multiple_id_properties():
    """Test that strip_id_property() removes all id:: properties."""
    context = """- Parent block
  id:: parent-id-123
  - Child block
    id:: child-id-456
    - Grandchild block
      id:: grandchild-id-789"""

    result = strip_id_property(context)

    assert "id::" not in result
    assert result.count("parent-id-123") == 0
    assert result.count("child-id-456") == 0
    assert result.count("grandchild-id-789") == 0
    assert "- Parent block" in result
    assert "- Child block" in result
    assert "- Grandchild block" in result


def test_strip_id_property_preserves_other_properties():
    """Test that strip_id_property() preserves non-id properties."""
    context = """- Example block
  id:: 65f3a8e0-1234-5678
  type:: best-practice
  tags:: #python, #testing
  - Child block"""

    result = strip_id_property(context)

    assert "id::" not in result
    assert "type:: best-practice" in result
    assert "tags:: #python, #testing" in result


def test_strip_id_property_handles_various_indentation():
    """Test that strip_id_property() handles different indentation levels."""
    context = """- Level 1
  id:: id1
  - Level 2
    id:: id2
    - Level 3
      id:: id3"""

    result = strip_id_property(context)

    assert "id::" not in result
    assert "- Level 1" in result
    assert "- Level 2" in result
    assert "- Level 3" in result


def test_strip_id_property_empty_context():
    """Test that strip_id_property() handles empty context."""
    result = strip_id_property("")
    assert result == ""


def test_strip_id_property_no_id_properties():
    """Test that strip_id_property() preserves content with no id:: properties."""
    context = """- Example block
  type:: note
  - Child block"""

    result = strip_id_property(context)

    assert result == context


def test_strip_id_property_with_hierarchical_context():
    """Test strip_id_property() with realistic hierarchical context from generate_chunks."""
    # Simulate what generate_chunks() returns
    context = """- DONE Review [[Python]] documentation for new features
  id:: task-123
  - [[Python]] 3.12 introduced some interesting improvements
    id:: c2153dfb48e2b495f600acf8e09800a3"""

    result = strip_id_property(context)

    # ID properties should be removed
    assert "id::" not in result
    assert "task-123" not in result
    assert "c2153dfb48e2b495f600acf8e09800a3" not in result

    # Content should be preserved
    assert "- DONE Review [[Python]] documentation for new features" in result
    assert "- [[Python]] 3.12 introduced some interesting improvements" in result
