"""Unit tests for Logseq parser and renderer."""

from textwrap import dedent

import pytest

from logsqueak.logseq.parser import LogseqBlock, LogseqOutline


class TestLogseqBlock:
    """Tests for LogseqBlock model."""

    def test_create_block(self):
        """Test creating a basic block."""
        block = LogseqBlock(content="Test content", indent_level=0)

        assert block.content == "Test content"
        assert block.indent_level == 0
        assert block.properties == {}
        assert block.children == []

    def test_add_child_to_end(self):
        """Test adding child block to end."""
        parent = LogseqBlock(content="Parent", indent_level=0)
        child = parent.add_child("Child content")

        assert len(parent.children) == 1
        assert parent.children[0].content == "Child content"
        assert parent.children[0].indent_level == 1  # One level deeper

    def test_add_multiple_children(self):
        """Test adding multiple children."""
        parent = LogseqBlock(content="Parent", indent_level=0)
        parent.add_child("Child 1")
        parent.add_child("Child 2")
        parent.add_child("Child 3")

        assert len(parent.children) == 3
        assert parent.children[0].content == "Child 1"
        assert parent.children[1].content == "Child 2"
        assert parent.children[2].content == "Child 3"

    def test_add_child_at_position(self):
        """Test adding child at specific position."""
        parent = LogseqBlock(content="Parent", indent_level=0)
        parent.add_child("Child 1")
        parent.add_child("Child 3")
        parent.add_child("Child 2", position=1)  # Insert in middle

        assert len(parent.children) == 3
        assert parent.children[0].content == "Child 1"
        assert parent.children[1].content == "Child 2"
        assert parent.children[2].content == "Child 3"

    def test_nested_children(self):
        """Test creating deeply nested structure."""
        root = LogseqBlock(content="Root", indent_level=0)
        child = root.add_child("Child")
        grandchild = child.add_child("Grandchild")

        assert grandchild.indent_level == 2
        assert root.children[0].children[0].content == "Grandchild"


class TestLogseqOutline:
    """Tests for LogseqOutline parsing and rendering."""

    def test_parse_empty_markdown(self):
        """Test parsing empty markdown."""
        outline = LogseqOutline.parse("")

        assert outline.blocks == []
        assert outline.source_text == ""

    def test_parse_single_bullet(self):
        """Test parsing single bullet."""
        markdown = "- Single item"
        outline = LogseqOutline.parse(markdown)

        assert len(outline.blocks) == 1
        assert outline.blocks[0].content == "Single item"
        assert outline.blocks[0].indent_level == 0

    def test_parse_multiple_bullets(self):
        """Test parsing multiple root-level bullets."""
        markdown = dedent(
            """\
            - First item
            - Second item
            - Third item"""
        )
        outline = LogseqOutline.parse(markdown)

        assert len(outline.blocks) == 3
        assert outline.blocks[0].content == "First item"
        assert outline.blocks[1].content == "Second item"
        assert outline.blocks[2].content == "Third item"

    def test_parse_nested_bullets(self):
        """Test parsing nested bullet structure."""
        markdown = dedent(
            """\
            - Parent
              - Child
                - Grandchild"""
        )
        outline = LogseqOutline.parse(markdown)

        assert len(outline.blocks) == 1
        assert outline.blocks[0].content == "Parent"
        assert len(outline.blocks[0].children) == 1
        assert outline.blocks[0].children[0].content == "Child"
        assert len(outline.blocks[0].children[0].children) == 1
        assert outline.blocks[0].children[0].children[0].content == "Grandchild"

    def test_parse_mixed_indentation(self):
        """Test parsing mixed indentation levels."""
        markdown = dedent(
            """\
            - Level 0
              - Level 1
            - Another level 0
              - Level 1
                - Level 2"""
        )
        outline = LogseqOutline.parse(markdown)

        assert len(outline.blocks) == 2
        assert outline.blocks[0].content == "Level 0"
        assert len(outline.blocks[0].children) == 1
        assert outline.blocks[1].content == "Another level 0"
        assert len(outline.blocks[1].children) == 1
        assert len(outline.blocks[1].children[0].children) == 1

    def test_parse_with_page_links(self):
        """Test parsing bullets with Logseq page links."""
        markdown = "- Working on [[Project X]]"
        outline = LogseqOutline.parse(markdown)

        assert len(outline.blocks) == 1
        assert outline.blocks[0].content == "Working on [[Project X]]"

    def test_parse_with_markdown_headings(self):
        """Test parsing bullets with markdown headings."""
        markdown = "- ## Section Heading"
        outline = LogseqOutline.parse(markdown)

        assert len(outline.blocks) == 1
        assert outline.blocks[0].content == "## Section Heading"

    def test_render_empty_outline(self):
        """Test rendering empty outline."""
        outline = LogseqOutline(blocks=[], source_text="")
        result = outline.render()

        assert result == ""

    def test_render_single_bullet(self):
        """Test rendering single bullet."""
        block = LogseqBlock(content="Single item", indent_level=0)
        outline = LogseqOutline(blocks=[block], source_text="")
        result = outline.render()

        assert result == "- Single item"

    def test_render_multiple_bullets(self):
        """Test rendering multiple bullets."""
        blocks = [
            LogseqBlock(content="First", indent_level=0),
            LogseqBlock(content="Second", indent_level=0),
            LogseqBlock(content="Third", indent_level=0),
        ]
        outline = LogseqOutline(blocks=blocks, source_text="")
        result = outline.render()

        expected = dedent(
            """\
            - First
            - Second
            - Third"""
        )
        assert result == expected

    def test_render_nested_structure(self):
        """Test rendering nested structure."""
        parent = LogseqBlock(content="Parent", indent_level=0)
        parent.add_child("Child")
        parent.children[0].add_child("Grandchild")

        outline = LogseqOutline(blocks=[parent], source_text="")
        result = outline.render()

        expected = dedent(
            """\
            - Parent
              - Child
                - Grandchild"""
        )
        assert result == expected

    def test_find_heading_found(self):
        """Test finding heading that exists."""
        markdown = dedent(
            """\
            - Introduction
            - ## Timeline
              - Q1 goals
            - Summary"""
        )
        outline = LogseqOutline.parse(markdown)

        found = outline.find_heading("Timeline")

        assert found is not None
        assert found.content == "## Timeline"

    def test_find_heading_case_insensitive(self):
        """Test finding heading is case-insensitive."""
        markdown = "- ## Project Timeline"
        outline = LogseqOutline.parse(markdown)

        found = outline.find_heading("timeline")

        assert found is not None
        assert "Timeline" in found.content

    def test_find_heading_not_found(self):
        """Test finding heading that doesn't exist."""
        markdown = "- Item 1\n- Item 2"
        outline = LogseqOutline.parse(markdown)

        found = outline.find_heading("Nonexistent")

        assert found is None

    def test_find_heading_in_nested_structure(self):
        """Test finding heading in nested blocks."""
        markdown = dedent(
            """\
            - Parent
              - ## Timeline
                - Child item"""
        )
        outline = LogseqOutline.parse(markdown)

        found = outline.find_heading("Timeline")

        assert found is not None
        assert found.content == "## Timeline"

    def test_preserve_original_lines(self):
        """Test that _original_lines is preserved for round-trip."""
        markdown = "- Original line with specific formatting"
        outline = LogseqOutline.parse(markdown)

        assert outline.blocks[0]._original_lines == [markdown]

    def test_skip_non_bullet_lines(self):
        """Test that non-bullet lines are skipped."""
        markdown = dedent(
            """\
            Some text
            - Bullet 1
            More text
            - Bullet 2"""
        )
        outline = LogseqOutline.parse(markdown)

        # Should only parse the bullet lines
        assert len(outline.blocks) == 2
        assert outline.blocks[0].content == "Bullet 1"
        assert outline.blocks[1].content == "Bullet 2"

    def test_skip_empty_lines(self):
        """Test that empty lines are skipped."""
        markdown = dedent(
            """\
            - Item 1

            - Item 2

            - Item 3"""
        )
        outline = LogseqOutline.parse(markdown)

        assert len(outline.blocks) == 3


class TestBlockProperties:
    """Tests for parsing block properties (M1.1)."""

    def test_parse_block_with_id_property(self):
        """Test parsing block with id:: property."""
        markdown = dedent(
            """\
            - Block with ID
              id:: 65f3a8e0-1234-5678-9abc-def012345678"""
        )
        outline = LogseqOutline.parse(markdown)

        assert len(outline.blocks) == 1
        block = outline.blocks[0]
        assert block.content == "Block with ID"
        assert block.properties == {"id": "65f3a8e0-1234-5678-9abc-def012345678"}
        assert block.block_id == "65f3a8e0-1234-5678-9abc-def012345678"

    def test_parse_block_without_id_property(self):
        """Test parsing block without id:: property."""
        markdown = "- Block without ID"
        outline = LogseqOutline.parse(markdown)

        block = outline.blocks[0]
        assert block.properties == {}
        assert block.block_id is None

    def test_parse_multiple_properties(self):
        """Test parsing block with multiple properties."""
        markdown = dedent(
            """\
            - Block with properties
              id:: abc123
              tags:: important, urgent
              priority:: high"""
        )
        outline = LogseqOutline.parse(markdown)

        block = outline.blocks[0]
        assert block.properties == {
            "id": "abc123",
            "tags": "important, urgent",
            "priority": "high",
        }
        assert block.block_id == "abc123"

    def test_property_order_preserved(self):
        """Test that property insertion order is preserved."""
        markdown = dedent(
            """\
            - Block
              zebra:: last
              alpha:: first
              middle:: second"""
        )
        outline = LogseqOutline.parse(markdown)

        block = outline.blocks[0]
        # Properties should be in the order they appear
        keys = list(block.properties.keys())
        assert keys == ["zebra", "alpha", "middle"]

    def test_parse_property_with_double_colon_in_value(self):
        """Test parsing property where value contains ::."""
        markdown = dedent(
            """\
            - Block
              url:: https://example.com/path::with::colons"""
        )
        outline = LogseqOutline.parse(markdown)

        block = outline.blocks[0]
        assert block.properties == {"url": "https://example.com/path::with::colons"}

    def test_parse_nested_blocks_with_properties(self):
        """Test parsing nested blocks with properties."""
        markdown = dedent(
            """\
            - Parent block
              id:: parent-id
              - Child block
                id:: child-id"""
        )
        outline = LogseqOutline.parse(markdown)

        parent = outline.blocks[0]
        assert parent.block_id == "parent-id"
        assert len(parent.children) == 1

        child = parent.children[0]
        assert child.block_id == "child-id"

    def test_properties_with_mixed_content(self):
        """Test block with properties mixed with other continuation lines."""
        markdown = dedent(
            """\
            - Block with mixed content
              id:: test-id
              Some text continuation
              tags:: test
              More text"""
        )
        outline = LogseqOutline.parse(markdown)

        block = outline.blocks[0]
        # Should parse properties correctly
        assert block.properties == {
            "id": "test-id",
            "tags": "test",
        }
        assert block.block_id == "test-id"
        # All continuation lines should be preserved
        assert len(block.continuation_lines) == 4
