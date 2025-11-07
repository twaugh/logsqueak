"""Integration tests for TargetPagePreview block rendering and tracking.

Tests the full rendering pipeline including:
- Block rendering with wrapping
- Proper indentation (bullet and hanging indent)
- Block ID tracking
"""

import pytest
from logseq_outline.parser import LogseqOutline, LogseqBlock
from logseq_outline.context import generate_full_context, generate_content_hash
from logsqueak.tui.widgets.target_page_preview import (
    render_block_content,
    render_outline_with_tracking,
)


def get_block_id(block: LogseqBlock, parents: list[LogseqBlock] = None) -> str:
    """Helper to get block ID (explicit or generated hash)."""
    if block.block_id:
        return block.block_id
    full_context = generate_full_context(block, parents or [])
    return generate_content_hash(full_context)


class TestRenderBlockContent:
    """Test rendering individual blocks with word wrapping."""

    def test_render_simple_block_no_wrapping(self):
        """Test rendering a simple block that fits in width."""
        block = LogseqBlock(
            content=["Short content"],
            indent_level=0,
        )

        lines = render_block_content(block, max_width=80)

        # Should produce one line with bullet
        assert len(lines) == 1
        assert "- Short content" in lines[0].plain

    def test_render_block_with_property_no_wrapping(self):
        """Test rendering block with property, both fitting in width."""
        block = LogseqBlock(
            content=["Block content", "key:: value"],
            indent_level=0,
        )

        lines = render_block_content(block, max_width=80)

        # Should produce two lines: content + property
        assert len(lines) == 2
        assert "- Block content" in lines[0].plain
        assert "key: value" in lines[1].plain
        # Property should be indented (2 spaces for bullet width)
        assert lines[1].plain.startswith("  ")

    def test_render_block_with_wrapping(self):
        """Test block with content that wraps."""
        block = LogseqBlock(
            content=["This is a very long line that should definitely wrap when rendered in a narrow widget"],
            indent_level=0,
        )

        # Narrow width to force wrapping
        lines = render_block_content(block, max_width=40)

        # Should produce multiple lines
        assert len(lines) > 1

        # First line should have bullet
        assert lines[0].plain.startswith("- ")

        # Subsequent wrapped lines should have hanging indent (2 spaces)
        for wrapped_line in lines[1:]:
            assert wrapped_line.plain.startswith("  "), f"Line should start with hanging indent: {wrapped_line.plain!r}"

    def test_render_block_with_nested_indent(self):
        """Test block at deeper indent level."""
        block = LogseqBlock(
            content=["Child block content"],
            indent_level=1,  # One level deep
        )

        lines = render_block_content(block, max_width=80)

        # Should have base indent (2 spaces) + bullet
        assert lines[0].plain.startswith("  - ")

    def test_render_block_with_multi_line_content(self):
        """Test block with multiple content lines (continuation lines)."""
        block = LogseqBlock(
            content=[
                "First line of block",
                "Continuation line",
                "Another continuation",
            ],
            indent_level=0,
        )

        lines = render_block_content(block, max_width=80)

        # Should produce 3 lines
        assert len(lines) == 3

        # First line with bullet
        assert lines[0].plain.startswith("- First line")

        # Continuation lines with hanging indent
        assert lines[1].plain.startswith("  Continuation")
        assert lines[2].plain.startswith("  Another")

    def test_render_block_with_property_wrapping(self):
        """Test property line that wraps."""
        block = LogseqBlock(
            content=[
                "Block",
                "tags:: #programming, #python, #tutorial, #logseq, #knowledge-management, #productivity",
            ],
            indent_level=0,
        )

        lines = render_block_content(block, max_width=50)

        # Should wrap the property value
        assert len(lines) > 2  # At least: block line + property start + wrapped continuation

        # First line is block
        assert "- Block" in lines[0].plain

        # Second line is property start
        assert lines[1].plain.startswith("  tags:")

        # Subsequent lines are wrapped property value with hanging indent
        for wrapped_line in lines[2:]:
            # Property wrapping should maintain indent
            assert wrapped_line.plain.startswith("  ")

    def test_render_block_preserves_indentation(self):
        """Test that block content indentation is preserved."""
        block = LogseqBlock(
            content=["Content", "  nested:: value"],  # Property already has extra indent
            indent_level=0,
        )

        lines = render_block_content(block, max_width=80)

        # The property line should preserve its indentation
        assert "nested: value" in lines[1].plain


class TestRenderOutlineWithTracking:
    """Test rendering full outlines and tracking block IDs to line ranges."""

    def test_render_simple_outline(self):
        """Test rendering simple outline with tracking."""
        content = """- First block
- Second block"""

        outline = LogseqOutline.parse(content)
        rendered_lines, block_map = render_outline_with_tracking(outline, max_width=80)

        # Should produce 2 lines
        assert len(rendered_lines) == 2

        # Each block should map to one line
        assert len(block_map) == 2

        # Verify block IDs exist in map
        for block in outline.blocks:
            block_id = get_block_id(block, [])
            assert block_id in block_map

    def test_render_outline_with_children(self):
        """Test rendering outline with nested children."""
        content = """- Parent block
  - Child block
    - Grandchild block"""

        outline = LogseqOutline.parse(content)
        rendered_lines, block_map = render_outline_with_tracking(outline, max_width=80)

        # Should produce 3 lines
        assert len(rendered_lines) == 3

        # Should have 3 blocks in map
        assert len(block_map) == 3

        # Verify indentation
        assert rendered_lines[0].plain.startswith("- Parent")
        assert rendered_lines[1].plain.startswith("  - Child")
        assert rendered_lines[2].plain.startswith("    - Grandchild")

    def test_render_outline_with_properties(self):
        """Test rendering blocks with properties and tracking."""
        content = """- Block one
  property:: value
- Block two
  id:: 12345"""

        outline = LogseqOutline.parse(content)
        rendered_lines, block_map = render_outline_with_tracking(outline, max_width=80)

        # Should produce 4 lines (2 blocks x 2 lines each)
        assert len(rendered_lines) == 4

        # Block one should map to lines 0 and 1
        block_one = outline.blocks[0]
        block_one_id = get_block_id(block_one, [])
        assert 0 in block_map[block_one_id]
        assert 1 in block_map[block_one_id]

        # Block two should map to lines 2 and 3
        block_two = outline.blocks[1]
        block_two_id = get_block_id(block_two, [])
        assert block_two_id == "12345"  # Explicit ID
        assert 2 in block_map[block_two_id]
        assert 3 in block_map[block_two_id]

    def test_render_outline_with_wrapping(self):
        """Test tracking with word-wrapped content."""
        content = """- This is a very long block that will wrap when rendered in a narrow widget"""

        outline = LogseqOutline.parse(content)
        rendered_lines, block_map = render_outline_with_tracking(outline, max_width=40)

        # Should wrap into multiple lines
        assert len(rendered_lines) > 1

        # Block should map to ALL wrapped lines
        block = outline.blocks[0]
        block_id = get_block_id(block, [])
        line_numbers = block_map[block_id]

        # Should include all rendered lines
        assert len(line_numbers) == len(rendered_lines)
        assert line_numbers == list(range(len(rendered_lines)))

    def test_render_outline_complex_tracking(self):
        """Test complex outline with wrapping and properties."""
        content = """- Short parent
  long-property:: This is a long property value that will wrap
  - Short child
    id:: child-id
  - Another child with a very long description that will wrap when rendered"""

        outline = LogseqOutline.parse(content)
        rendered_lines, block_map = render_outline_with_tracking(outline, max_width=50)

        # Parent block should map to its content + property lines
        parent = outline.blocks[0]
        parent_id = get_block_id(parent, [])
        parent_lines = block_map[parent_id]

        # Should include at least 2 lines (content + property, possibly more with wrapping)
        assert len(parent_lines) >= 2

        # First child should have explicit ID
        first_child = outline.blocks[0].children[0]
        assert first_child.block_id == "child-id"
        assert "child-id" in block_map

        # Verify children don't overlap with parent in line numbers
        for i, child in enumerate(outline.blocks[0].children):
            child_id = get_block_id(child, [parent])
            child_lines = block_map[child_id]
            # Child lines should not overlap with parent lines
            assert not set(parent_lines).intersection(set(child_lines))

    def test_find_block_by_line_number(self):
        """Test finding which block corresponds to a line number."""
        content = """- Block A
  property:: value
- Block B
  - Child of B"""

        outline = LogseqOutline.parse(content)
        _, block_map = render_outline_with_tracking(outline, max_width=80)

        # Helper to find block ID by line number
        def find_block_at_line(line_num: int) -> str | None:
            for block_id, line_numbers in block_map.items():
                if line_num in line_numbers:
                    return block_id
            return None

        # Line 0 should be Block A
        block_at_0 = find_block_at_line(0)
        assert block_at_0 is not None

        # Line 1 should also be Block A (property line)
        block_at_1 = find_block_at_line(1)
        assert block_at_1 == block_at_0

        # Line 2 should be Block B
        block_at_2 = find_block_at_line(2)
        assert block_at_2 != block_at_0

        # Line 3 should be Child of B
        block_at_3 = find_block_at_line(3)
        assert block_at_3 not in (block_at_0, block_at_2)
