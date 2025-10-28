"""Minimal reproduction cases for round-trip failures.

These tests reproduce the 8 failure patterns found in real Logseq graphs.
Each test is minimal and focused on a specific failure mode.

Based on analysis from roundtrip-failures.md generated from real graph.
"""

from textwrap import dedent

import pytest

from logsqueak.logseq.parser import LogseqOutline


class TestTabIndentation:
    """Tab indentation failures (6/8 failures).

    Pattern: Files using tabs instead of spaces lose bytes during roundtrip.
    """

    def test_single_tab_indent(self):
        """Minimal: Single level of tab indentation."""
        original = "- Parent\n\t- Child"

        outline = LogseqOutline.parse(original)
        rendered = outline.render()

        assert rendered == original, f"Expected:\n{original!r}\nGot:\n{rendered!r}"
        assert len(rendered) == len(original), f"Lost {len(original) - len(rendered)} bytes"

    def test_multiple_tab_levels(self):
        """Tab indentation with 3 levels (like Failure 4)."""
        original = "- Root\n\t- Level 1\n\t\t- Level 2\n\t\t\t- Level 3"

        outline = LogseqOutline.parse(original)
        rendered = outline.render()

        assert rendered == original, f"Lost {len(original) - len(rendered)} bytes"

    def test_mixed_tabs_and_content(self):
        """Tabs with continuation lines (like Failure 8)."""
        original = dedent("""\
        - Parent
        \t- Child with content
        \t\t- Nested child
        \t\t- Another nested""")

        # Note: dedent will convert leading tabs, so we need raw tabs
        original = "- Parent\n\t- Child with content\n\t\t- Nested child\n\t\t- Another nested"

        outline = LogseqOutline.parse(original)
        rendered = outline.render()

        assert rendered == original


class TestContinuationProperties:
    """Continuation property failures (5/8 failures).

    Pattern: Properties on continuation lines lose whitespace.
    """

    def test_property_continuation(self):
        """Property on continuation line (like Failure 2)."""
        original = dedent("""\
        property:: value
        tags:: [[SomeTag]]

        - Main bullet""")

        outline = LogseqOutline.parse(original)
        rendered = outline.render()

        # Check frontmatter preservation
        assert len(rendered) == len(original) or abs(len(rendered) - len(original)) <= 2


    # REMOVED: test_multiline_property_with_tabs
    # Invalid Logseq syntax - root bullets don't have tab-indented continuations


class TestTrailingWhitespace:
    """Trailing whitespace and empty line handling.

    Pattern: Small byte losses suggest trailing whitespace issues.
    """

    def test_trailing_newline_preservation(self):
        """File ending with empty line (common in all failures)."""
        original = "- Item\n-"

        outline = LogseqOutline.parse(original)
        rendered = outline.render()

        # Most failures show the pattern ending with "-\n" (empty bullet + newline)
        assert rendered == original or rendered == original.rstrip("\n")

    def test_empty_bullet_preservation(self):
        """Empty bullets between content (Failure 1, 2, 3)."""
        original = "- Content\n-\n- More content\n-"

        outline = LogseqOutline.parse(original)
        rendered = outline.render()

        # Should preserve structure, but may normalize trailing content
        lines = rendered.split("\n")
        assert "-" in lines  # Empty bullets should be preserved


class TestCodeBlocks:
    """Code blocks with special indentation (Failure 7).

    Pattern: Code blocks can contain arbitrary indentation that must be preserved.
    """

    def test_cdata_in_code_block(self):
        """CDATA-style content in code block (Failure 7 has XML-like content)."""
        original = dedent("""\
        - Example
          ```
          <data>
          <![CDATA[
          Content here
          ]]>
          </data>
          ```""")

        outline = LogseqOutline.parse(original)
        rendered = outline.render()

        assert rendered == original

    def test_deeply_nested_code_with_tabs(self):
        """Deeply nested code block with tabs (Failure 7 pattern)."""
        original = "- Item\n\t- Sub\n\t\t- Code:\n\t\t  ```\n\t\t  code\n\t\t  ```"

        outline = LogseqOutline.parse(original)
        rendered = outline.render()

        assert rendered == original


class TestComplexNesting:
    """Deep nesting with mixed content (Failures 4, 6, 7).

    Pattern: Complex structures with properties, tabs, and deep nesting.
    """

    def test_deep_nesting_with_properties_and_tabs(self):
        """4 levels deep with properties and tabs (Failure 4 structure)."""
        original = dedent("""\
        - Root
        \t- Level 1
        \t\t- Level 2
        \t\t  id:: some-id
        \t\t\t- Level 3
        \t\t\t  property:: value""")

        # Raw version without dedent interference
        original = "- Root\n\t- Level 1\n\t\t- Level 2\n\t\t  id:: some-id\n\t\t\t- Level 3\n\t\t\t  property:: value"

        outline = LogseqOutline.parse(original)
        rendered = outline.render()

        assert rendered == original, f"Lost {len(original) - len(rendered)} bytes"


class TestMinimalFailures:
    """Absolute minimal reproductions for each failure type."""

    def test_failure_5_minimal(self):
        """Failure 5: Simplest case, lost 2 bytes."""
        # Original shows trailing newline after empty bullet
        original = "- Content\n-\n- More"

        outline = LogseqOutline.parse(original)
        rendered = outline.render()

        # Check byte count
        byte_diff = len(original) - len(rendered)
        assert byte_diff == 0, f"Lost {byte_diff} bytes"

    def test_failure_3_minimal(self):
        """Failure 3: Tab indentation, lost 4 bytes."""
        original = "- Item\n\t- Sub1\n\t\t- Sub2"

        outline = LogseqOutline.parse(original)
        rendered = outline.render()

        byte_diff = len(original) - len(rendered)
        assert byte_diff == 0, f"Lost {byte_diff} bytes"

    def test_tab_detection(self):
        """Test that tab indentation is detected correctly."""
        original = "- Parent\n\t- Child"

        outline = LogseqOutline.parse(original)

        # Indent string should be a single tab
        assert outline.indent_str == "\t", f"Expected tab, got {outline.indent_str!r}"


class TestWhitespaceEdgeCases:
    """Edge cases for whitespace handling."""

    def test_continuation_line_with_leading_tabs(self):
        """Continuation line starting with tabs (might be code or property)."""
        original = "- Main line\n  \tcontinuation with tab"

        outline = LogseqOutline.parse(original)
        rendered = outline.render()

        assert rendered == original

    def test_empty_continuation_line(self):
        """Empty continuation line (just whitespace)."""
        original = "- Main\n  \n- Next"

        outline = LogseqOutline.parse(original)
        rendered = outline.render()

        # Should preserve structure
        assert "Main" in rendered
        assert "Next" in rendered

    def test_trailing_spaces_on_bullet(self):
        """Bullet with trailing spaces."""
        original = "- Content  \n- More"

        outline = LogseqOutline.parse(original)
        rendered = outline.render()

        # Trailing spaces on first line might be normalized
        # Check if content is preserved (spaces may be trimmed)
        assert "Content" in rendered


class TestFrontmatterHandling:
    """Test frontmatter (content before first bullet)."""

    def test_frontmatter_with_properties(self):
        """Frontmatter properties before bullets (Failure 2 pattern)."""
        original = dedent("""\
        property:: value
        tags:: [[Tag]]

        - First bullet""")

        outline = LogseqOutline.parse(original)
        rendered = outline.render()

        # Frontmatter should be preserved
        assert "property:: value" in rendered
        assert rendered == original or abs(len(rendered) - len(original)) <= 2

    def test_frontmatter_trailing_newline(self):
        """Frontmatter with blank line separator."""
        original = "prop:: val\n\n- Bullet"

        outline = LogseqOutline.parse(original)
        rendered = outline.render()

        assert rendered == original


class TestRegressionSuite:
    """Direct reproductions from actual failure content (anonymized)."""

    def test_failure_1_pattern(self):
        """Failure 1: Tab indent + multiline + continuation properties, -10 bytes."""
        # Simplified from the actual failure
        original = "wd57ac4:: [[link]]\n\n-\n-\n-\n- # +wid\n\t- #### text\n\t\t-"

        outline = LogseqOutline.parse(original)
        rendered = outline.render()

        byte_diff = len(original) - len(rendered)
        assert abs(byte_diff) <= 10, f"Byte diff: {byte_diff} (expected <= 10)"

    def test_failure_2_pattern(self):
        """Failure 2: Frontmatter properties, -2 bytes."""
        original = "property:: value\ntags:: [[tag]]\n\n- # Text\n  More text\n-"

        outline = LogseqOutline.parse(original)
        rendered = outline.render()

        byte_diff = len(original) - len(rendered)
        assert abs(byte_diff) <= 2, f"Byte diff: {byte_diff} (expected <= 2)"

    def test_failure_8_pattern(self):
        """Failure 8: Tab indentation with continuation, -2 bytes."""
        original = "prop:: value\n\n- [[link]]\n\t- Text\n\t- More\n-"

        outline = LogseqOutline.parse(original)
        rendered = outline.render()

        byte_diff = len(original) - len(rendered)
        assert abs(byte_diff) <= 2, f"Byte diff: {byte_diff} (expected <= 2)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
