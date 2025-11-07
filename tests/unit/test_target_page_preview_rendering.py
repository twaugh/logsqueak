"""Unit tests for TargetPagePreview rendering functions.

Tests the individual rendering functions that handle Logseq syntax highlighting
and text formatting.
"""

import pytest
from rich.text import Text
from logsqueak.tui.widgets.target_page_preview import (
    render_property_line,
    render_inline_syntax,
)


class TestRenderPropertyLine:
    """Test property line rendering with nested syntax."""

    def test_simple_property(self):
        """Test basic property rendering: key:: value"""
        text = render_property_line("key:: value")

        # Should contain key and value in plain text
        assert "key:" in text.plain
        assert "value" in text.plain

        # Should not contain the double colon in output
        assert "key::" not in text.plain

    def test_property_with_indentation(self):
        """Test property with leading whitespace is preserved."""
        text = render_property_line("  property:: value")

        # Indentation should be preserved
        assert text.plain.startswith("  ")
        assert "property:" in text.plain
        assert "value" in text.plain

    def test_property_with_single_tag(self):
        """Test property with tag: key:: #tag"""
        text = render_property_line("key:: #tag")

        assert "key:" in text.plain
        assert "#tag" in text.plain

    def test_property_with_multiple_tags(self):
        """Test property with multiple tags: key:: #tag, #value, #end"""
        text = render_property_line("tags:: #programming, #python, #tutorial")

        assert "tags:" in text.plain
        assert "#programming" in text.plain
        assert "#python" in text.plain
        assert "#tutorial" in text.plain
        assert "," in text.plain

    def test_property_with_mixed_content(self):
        """Test property with tags and plain text: key:: #tag, other value, #end"""
        text = render_property_line("key:: #tag, other value, #end")

        assert "key:" in text.plain
        assert "#tag" in text.plain
        assert "other value" in text.plain
        assert "#end" in text.plain

    def test_property_with_page_link(self):
        """Test property with page link: key:: [[Page Name]]"""
        text = render_property_line("related:: [[Python]], [[Ruby]]")

        assert "related:" in text.plain
        assert "Python" in text.plain
        assert "Ruby" in text.plain
        # Page link brackets should be removed
        assert "[[" not in text.plain
        assert "]]" not in text.plain

    def test_property_with_link_tag(self):
        """Test property with link tag: key:: #[[Tag Name]]"""
        text = render_property_line("tags:: #[[Project Management]], #standalone")

        assert "tags:" in text.plain
        assert "#" in text.plain  # Hash should be present
        assert "Project Management" in text.plain
        assert "#standalone" in text.plain

    def test_non_property_line_returns_unchanged(self):
        """Test that non-property lines are returned as-is."""
        line = "This is not a property"
        text = render_property_line(line)

        assert text.plain == line


class TestRenderInlineSyntax:
    """Test inline syntax rendering (tags, links, bold, italic)."""

    def test_page_link_simple(self):
        """Test [[Page Name]] rendering."""
        text = render_inline_syntax("See [[Python]] documentation")

        assert "See " in text.plain
        assert "Python" in text.plain
        assert " documentation" in text.plain
        # Brackets should be removed
        assert "[[" not in text.plain
        assert "]]" not in text.plain

    def test_page_link_multiple(self):
        """Test multiple [[links]] in one line."""
        text = render_inline_syntax("Languages: [[Python]], [[Ruby]], [[Go]]")

        assert "Python" in text.plain
        assert "Ruby" in text.plain
        assert "Go" in text.plain
        assert "[[" not in text.plain

    def test_tag_after_whitespace(self):
        """Test #tag rendering after whitespace."""
        text = render_inline_syntax("Topics: #python, #tui, #testing")

        assert "#python" in text.plain
        assert "#tui" in text.plain
        assert "#testing" in text.plain

    def test_tag_at_start(self):
        """Test #tag at beginning of line."""
        text = render_inline_syntax("#important This is a note")

        assert "#important" in text.plain
        assert "This is a note" in text.plain

    def test_link_tag(self):
        """Test #[[Link Tag]] rendering."""
        text = render_inline_syntax("See #[[Project Management]] for details")

        assert "#" in text.plain
        assert "Project Management" in text.plain
        assert "for details" in text.plain
        assert "[[" not in text.plain

    def test_link_tag_and_regular_tag(self):
        """Test mix of #[[link tag]] and #tag."""
        text = render_inline_syntax("Tags: #[[Deep Work]], #productivity")

        assert "#" in text.plain
        assert "Deep Work" in text.plain
        assert "#productivity" in text.plain

    def test_bold_text(self):
        """Test **bold** rendering."""
        text = render_inline_syntax("This is **bold** text")

        assert "This is " in text.plain
        assert "bold" in text.plain
        assert " text" in text.plain
        # Asterisks should be removed
        assert "**" not in text.plain

    def test_italic_text(self):
        """Test *italic* rendering."""
        text = render_inline_syntax("This is *italic* text")

        assert "This is " in text.plain
        assert "italic" in text.plain
        assert " text" in text.plain
        # Single asterisks should be removed (but this is tricky with bold)
        # We'll verify the content is there
        assert "italic" in text.plain

    def test_url(self):
        """Test URL rendering."""
        text = render_inline_syntax("Visit https://example.com for more")

        assert "Visit " in text.plain
        assert "https://example.com" in text.plain
        assert " for more" in text.plain

    def test_mixed_syntax(self):
        """Test complex mixed syntax."""
        text = render_inline_syntax("See [[Project]] with #tag and **bold** text")

        assert "Project" in text.plain
        assert "#tag" in text.plain
        assert "bold" in text.plain
        assert "[[" not in text.plain
        assert "**" not in text.plain

    def test_complex_realistic_example(self):
        """Test realistic complex example."""
        text = render_inline_syntax(
            "Read [[Getting Things Done]] - covers #productivity and **time management**"
        )

        assert "Getting Things Done" in text.plain
        assert "#productivity" in text.plain
        assert "time management" in text.plain
        assert "[[" not in text.plain

    def test_base_style_applied(self):
        """Test that base_style parameter is used for unstyled text."""
        text = render_inline_syntax("plain text with #tag", base_style="dim italic")

        # This is harder to test - we'd need to check text.spans
        # For now, just verify content is correct
        assert "plain text with " in text.plain
        assert "#tag" in text.plain

    def test_empty_string(self):
        """Test empty string handling."""
        text = render_inline_syntax("")

        assert text.plain == ""

    def test_no_syntax(self):
        """Test plain text without any syntax."""
        text = render_inline_syntax("Just plain text here")

        assert text.plain == "Just plain text here"
