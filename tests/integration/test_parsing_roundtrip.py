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


class TestIDPreservation:
    """Test that block IDs are preserved during round-trip (M1.5)."""

    def test_explicit_id_preserved_in_roundtrip(self):
        """Test that explicit id:: properties are preserved through parse-render cycle."""
        original = dedent(
            """\
            - Block with ID
              id:: 65f3a8e0-1234-5678-9abc-def012345678"""
        )

        outline = LogseqOutline.parse(original)
        rendered = outline.render()

        assert rendered == original
        # Verify the id was parsed correctly
        assert outline.blocks[0].block_id == "65f3a8e0-1234-5678-9abc-def012345678"

    def test_explicit_id_never_modified(self):
        """Test that existing id:: properties are never changed during modifications."""
        original = dedent(
            """\
            - Parent block
              id:: parent-id-123
              - Child block
                id:: child-id-456"""
        )

        outline = LogseqOutline.parse(original)

        # Store original IDs
        parent_id = outline.blocks[0].block_id
        child_id = outline.blocks[0].children[0].block_id

        # Add new child to parent
        outline.blocks[0].add_child("New child")

        # Render and re-parse
        rendered = outline.render()
        reparsed = LogseqOutline.parse(rendered)

        # Original IDs must be unchanged
        assert reparsed.blocks[0].block_id == parent_id
        assert reparsed.blocks[0].children[0].block_id == child_id
        # New child should not have an ID (only writer adds IDs)
        assert reparsed.blocks[0].children[1].block_id is None

    def test_multiple_blocks_with_ids_roundtrip(self):
        """Test multiple blocks with IDs preserve all IDs."""
        original = dedent(
            """\
            - Block 1
              id:: id-001
            - Block 2
              id:: id-002
            - Block 3
              id:: id-003"""
        )

        outline = LogseqOutline.parse(original)
        rendered = outline.render()

        assert rendered == original

        # Verify all IDs parsed
        assert outline.blocks[0].block_id == "id-001"
        assert outline.blocks[1].block_id == "id-002"
        assert outline.blocks[2].block_id == "id-003"

    def test_nested_blocks_preserve_ids_through_modification(self):
        """Test that nested blocks preserve IDs when parent is modified."""
        original = dedent(
            """\
            - Parent
              id:: parent-123
              - Child 1
                id:: child-1-456
              - Child 2
                id:: child-2-789"""
        )

        outline = LogseqOutline.parse(original)

        # Store original IDs
        original_ids = {
            "parent": outline.blocks[0].block_id,
            "child1": outline.blocks[0].children[0].block_id,
            "child2": outline.blocks[0].children[1].block_id,
        }

        # Add new child between existing children
        outline.blocks[0].add_child("New child", position=1)

        # Render and re-parse
        rendered = outline.render()
        reparsed = LogseqOutline.parse(rendered)

        # All original IDs must be preserved
        assert reparsed.blocks[0].block_id == original_ids["parent"]
        assert reparsed.blocks[0].children[0].block_id == original_ids["child1"]
        assert reparsed.blocks[0].children[2].block_id == original_ids["child2"]

        # New child at position 1 should not have ID
        assert reparsed.blocks[0].children[1].block_id is None
        assert reparsed.blocks[0].children[1].content == "New child"

    def test_id_property_order_preserved(self):
        """Test that id:: property maintains its position relative to other properties."""
        original = dedent(
            """\
            - Block with multiple properties
              id:: test-id-123
              tags:: important, urgent
              priority:: high"""
        )

        outline = LogseqOutline.parse(original)
        rendered = outline.render()

        assert rendered == original

        # Verify property order is preserved
        props = outline.blocks[0].properties
        keys = list(props.keys())
        assert keys == ["id", "tags", "priority"]

    def test_hybrid_id_consistency_across_roundtrip(self):
        """Test that blocks without explicit IDs get consistent hybrid IDs."""
        original = "- Block without explicit ID"

        outline = LogseqOutline.parse(original)
        block = outline.blocks[0]

        # Get hybrid ID (should be hash)
        hybrid_id_1 = block.get_hybrid_id()

        # Render and re-parse
        rendered = outline.render()
        reparsed = LogseqOutline.parse(rendered)
        reparsed_block = reparsed.blocks[0]

        # Get hybrid ID again
        hybrid_id_2 = reparsed_block.get_hybrid_id()

        # Should be the same hash
        assert hybrid_id_1 == hybrid_id_2
        # Should be MD5 hash format
        assert len(hybrid_id_1) == 32

    def test_find_block_by_id_after_roundtrip(self):
        """Test that blocks can be found by ID after render-parse cycle."""
        original = dedent(
            """\
            - Block A
              id:: findable-id
            - Block B
            - Block C"""
        )

        outline = LogseqOutline.parse(original)

        # Find by ID before round-trip
        found_before = outline.find_block_by_id("findable-id")
        assert found_before is not None
        assert found_before.content == "Block A"

        # Render and re-parse
        rendered = outline.render()
        reparsed = LogseqOutline.parse(rendered)

        # Find by ID after round-trip
        found_after = reparsed.find_block_by_id("findable-id")
        assert found_after is not None
        assert found_after.content == "Block A"
        assert found_after.block_id == "findable-id"

    def test_deep_nesting_preserves_all_ids(self):
        """Test that deeply nested structure preserves all IDs."""
        original = dedent(
            """\
            - Root
              id:: root-id
              - Level 1
                id:: level1-id
                - Level 2
                  id:: level2-id
                  - Level 3
                    id:: level3-id"""
        )

        outline = LogseqOutline.parse(original)
        rendered = outline.render()
        reparsed = LogseqOutline.parse(rendered)

        # Navigate down and verify all IDs preserved
        assert reparsed.blocks[0].block_id == "root-id"
        assert reparsed.blocks[0].children[0].block_id == "level1-id"
        assert reparsed.blocks[0].children[0].children[0].block_id == "level2-id"
        assert reparsed.blocks[0].children[0].children[0].children[0].block_id == "level3-id"
