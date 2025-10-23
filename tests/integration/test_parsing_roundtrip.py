"""Integration tests for parser round-trip (parse → modify → render)."""

from textwrap import dedent

import pytest

from logsqueak.logseq.parser import LogseqOutline


class TestParsingRoundTrip:
    """Test that parsing and rendering preserves structure."""

    def test_simple_roundtrip(self):
        """Test simple parse and render produces identical output."""
        original = dedent(
            """\
            - First item
            - Second item
            - Third item"""
        )

        outline = LogseqOutline.parse(original)
        rendered = outline.render()

        assert rendered == original

    def test_nested_roundtrip(self):
        """Test nested structure round-trip."""
        original = dedent(
            """\
            - Parent 1
              - Child 1
                - Grandchild 1
            - Parent 2
              - Child 2"""
        )

        outline = LogseqOutline.parse(original)
        rendered = outline.render()

        assert rendered == original

    def test_complex_structure_roundtrip(self):
        """Test complex structure with multiple levels."""
        original = dedent(
            """\
            - ## Timeline
              - Q1 2025
                - January
                - February
                - March
              - Q2 2025
            - ## Team
              - Alice
              - Bob"""
        )

        outline = LogseqOutline.parse(original)
        rendered = outline.render()

        assert rendered == original

    def test_roundtrip_with_page_links(self):
        """Test round-trip preserves page links."""
        original = dedent(
            """\
            - Working on [[Project X]]
              - Meeting with [[Alice]]
            - Reviewed [[Product Y]] competitor"""
        )

        outline = LogseqOutline.parse(original)
        rendered = outline.render()

        assert rendered == original

    def test_modify_then_render(self):
        """Test that modifying structure and rendering works correctly."""
        original = dedent(
            """\
            - Parent
              - Existing child"""
        )

        outline = LogseqOutline.parse(original)

        # Add new child
        parent = outline.blocks[0]
        parent.add_child("New child")

        rendered = outline.render()

        # Should have original structure plus new child
        assert "Parent" in rendered
        assert "Existing child" in rendered
        assert "New child" in rendered

        # Verify structure
        lines = rendered.split("\n")
        assert lines[0] == "- Parent"
        assert lines[1] == "  - Existing child"
        assert lines[2] == "  - New child"

    def test_add_child_at_position(self):
        """Test adding child at specific position."""
        original = dedent(
            """\
            - Parent
              - Child 1
              - Child 3"""
        )

        outline = LogseqOutline.parse(original)

        # Insert child 2 in the middle
        parent = outline.blocks[0]
        parent.add_child("Child 2", position=1)

        rendered = outline.render()

        lines = rendered.split("\n")
        assert lines[0] == "- Parent"
        assert lines[1] == "  - Child 1"
        assert lines[2] == "  - Child 2"
        assert lines[3] == "  - Child 3"

    def test_deeply_nested_modification(self):
        """Test modifying deeply nested structure."""
        original = dedent(
            """\
            - Level 0
              - Level 1
                - Level 2"""
        )

        outline = LogseqOutline.parse(original)

        # Add to deepest level
        level2 = outline.blocks[0].children[0].children[0]
        level2.add_child("Level 3")

        rendered = outline.render()

        assert "Level 3" in rendered
        assert "      - Level 3" in rendered  # 3 levels deep (6 spaces)

    def test_multiple_root_blocks_with_modifications(self):
        """Test modifying multiple root blocks."""
        original = dedent(
            """\
            - Root 1
              - Child 1
            - Root 2
              - Child 2"""
        )

        outline = LogseqOutline.parse(original)

        # Add to both roots
        outline.blocks[0].add_child("New child 1")
        outline.blocks[1].add_child("New child 2")

        rendered = outline.render()

        assert "New child 1" in rendered
        assert "New child 2" in rendered

        # Verify both are properly indented
        lines = rendered.split("\n")
        assert "  - New child 1" in lines
        assert "  - New child 2" in lines

    def test_property_order_preservation(self):
        """Test that property order is preserved (FR-008)."""
        # Note: This is a basic test - full property parsing would be in Phase 3
        original = dedent(
            """\
            - Item with content
              - Nested item"""
        )

        outline = LogseqOutline.parse(original)

        # Properties dict should preserve insertion order (Python 3.7+)
        block = outline.blocks[0]
        block.properties["first"] = "value1"
        block.properties["second"] = "value2"
        block.properties["third"] = "value3"

        # Verify order is preserved
        keys = list(block.properties.keys())
        assert keys == ["first", "second", "third"]

    def test_empty_lines_handling(self):
        """Test that empty lines don't break parsing."""
        original = dedent(
            """\
            - Item 1

            - Item 2

            - Item 3"""
        )

        outline = LogseqOutline.parse(original)

        assert len(outline.blocks) == 3
        assert outline.blocks[0].content == "Item 1"
        assert outline.blocks[1].content == "Item 2"
        assert outline.blocks[2].content == "Item 3"

    def test_non_bullet_lines_ignored(self):
        """Test that non-bullet lines are skipped during parsing."""
        original = dedent(
            """\
            Some header text
            - Bullet 1
            Random text
            - Bullet 2"""
        )

        outline = LogseqOutline.parse(original)

        # Should only parse bullets
        assert len(outline.blocks) == 2
        assert outline.blocks[0].content == "Bullet 1"
        assert outline.blocks[1].content == "Bullet 2"

    def test_multiline_content_roundtrip(self):
        """Test that multi-line bullet content is preserved."""
        original = dedent(
            """\
            - This is a bullet with continuation
              that spans multiple lines
              and includes more text
            - Another bullet"""
        )

        outline = LogseqOutline.parse(original)
        rendered = outline.render()

        assert rendered == original

    def test_code_block_roundtrip(self):
        """Test that code blocks within bullets are preserved."""
        original = dedent(
            """\
            - Here is a code example:
              ```python
              def hello():
                  print("world")
              ```
            - Next item"""
        )

        outline = LogseqOutline.parse(original)
        rendered = outline.render()

        assert rendered == original

    def test_nested_code_block_with_tabs(self):
        """Test code blocks with tab indentation (real Logseq format)."""
        # Using actual tab characters like in quay-access-management.md
        original = "- Architecture diagram\n\t- Renderer\n\t\t- ```mermaid\n\t\t  graph LR\n\t\t      Node1\n\t\t  ```"

        outline = LogseqOutline.parse(original)
        rendered = outline.render()

        assert rendered == original

    def test_properties_on_continuation_lines(self):
        """Test that properties on continuation lines are preserved."""
        original = dedent(
            """\
            - Item with properties
              id:: 12345
              tags:: important, urgent
            - Next item"""
        )

        outline = LogseqOutline.parse(original)
        rendered = outline.render()

        assert rendered == original

    def test_complex_nested_multiline(self):
        """Test complex nested structure with multi-line content."""
        original = dedent(
            """\
            - Parent item
              - Child with code:
                ```bash
                echo "hello"
                ```
              - Another child
                with continuation text
            - Root item 2"""
        )

        outline = LogseqOutline.parse(original)
        rendered = outline.render()

        assert rendered == original
