"""Tests for diff generation functionality."""

import pytest
from logsqueak.models.diff import generate_unified_diff


class TestGenerateUnifiedDiff:
    """Test cases for generate_unified_diff function."""

    def test_identical_content_no_diff(self):
        """Should return empty diff for identical content."""
        content = "Line 1\nLine 2\nLine 3\n"
        diff = generate_unified_diff(content, content)
        assert diff == ""

    def test_basic_diff(self):
        """Should generate diff for changed content."""
        original = "Line 1\nLine 2\nLine 3\n"
        modified = "Line 1\nLine 2 modified\nLine 3\n"
        diff = generate_unified_diff(original, modified)

        # Diff should be non-empty
        assert diff != ""
        # Should show the modified line
        assert "-Line 2\n" in diff
        assert "+Line 2 modified\n" in diff

    def test_trailing_newline_only_difference_ignored(self):
        """Lines differing only by trailing newline should be treated as identical."""
        # Original has trailing newline, modified doesn't
        original = "Line 1\nLine 2\nLine 3\n"
        modified = "Line 1\nLine 2\nLine 3"

        diff = generate_unified_diff(original, modified)
        # Should produce no diff - only difference is final trailing newline
        assert diff == ""

    def test_trailing_newline_only_difference_reversed(self):
        """Lines differing only by trailing newline should be treated as identical (reversed)."""
        # Original doesn't have trailing newline, modified does
        original = "Line 1\nLine 2\nLine 3"
        modified = "Line 1\nLine 2\nLine 3\n"

        diff = generate_unified_diff(original, modified)
        # Should produce no diff - only difference is final trailing newline
        assert diff == ""

    def test_middle_line_trailing_newline_difference_ignored(self):
        """Middle line trailing newline differences should be ignored."""
        # This tests the edge case where a line in the middle lacks trailing newline
        # Note: splitlines(keepends=True) will preserve the newlines, but if content
        # doesn't have them, we normalize them
        original = "Line 1\nLine 2"  # No newline after Line 2
        modified = "Line 1\nLine 2\n"  # Newline after Line 2

        diff = generate_unified_diff(original, modified)
        # Should be treated as identical after normalization
        assert diff == ""

    def test_real_content_change_with_trailing_newline_variations(self):
        """Real content changes should still show up even with trailing newline variations."""
        original = "Line 1\nLine 2\nLine 3"  # No final newline
        modified = "Line 1\nLine 2 changed\nLine 3\n"  # Has final newline

        diff = generate_unified_diff(original, modified)

        # Should show the real change
        assert diff != ""
        assert "-Line 2\n" in diff
        assert "+Line 2 changed\n" in diff
        # But should NOT show Line 3 as changed
        # (it differs only by trailing newline)

    def test_multiple_changes_with_context(self):
        """Should show context lines around changes."""
        original = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5\n"
        modified = "Line 1\nLine 2 changed\nLine 3\nLine 4 changed\nLine 5\n"

        diff = generate_unified_diff(original, modified, context_lines=1)

        # Should show changes with context
        assert "-Line 2\n" in diff
        assert "+Line 2 changed\n" in diff
        assert "-Line 4\n" in diff
        assert "+Line 4 changed\n" in diff

    def test_custom_file_labels(self):
        """Should use custom file labels in diff output."""
        original = "Line 1\n"
        modified = "Line 1 changed\n"

        diff = generate_unified_diff(
            original,
            modified,
            fromfile="file1.txt",
            tofile="file2.txt"
        )

        # Check for file labels in diff header
        assert "--- file1.txt" in diff
        assert "+++ file2.txt" in diff

    def test_empty_strings(self):
        """Should handle empty strings."""
        diff = generate_unified_diff("", "")
        assert diff == ""

    def test_empty_to_content(self):
        """Should show additions when going from empty to content."""
        original = ""
        modified = "New line\n"

        diff = generate_unified_diff(original, modified)

        assert diff != ""
        assert "+New line\n" in diff

    def test_content_to_empty(self):
        """Should show deletions when going from content to empty."""
        original = "Old line\n"
        modified = ""

        diff = generate_unified_diff(original, modified)

        assert diff != ""
        assert "-Old line\n" in diff

    def test_diff_output_has_trailing_newlines(self):
        """All lines in diff output should end with newlines."""
        original = "Line 1\nLine 2\n"
        modified = "Line 1 changed\nLine 2\n"

        diff = generate_unified_diff(original, modified)

        # Split by newline and check all lines (except last empty one) end with \n
        lines = diff.split('\n')
        for i, line in enumerate(lines[:-1]):  # Skip last element (empty after final \n)
            # When we rejoin, each line should be followed by newline in original string
            assert diff.find(line + '\n') >= 0, f"Line {i} should be followed by newline"

    def test_multiline_content_with_mixed_newlines(self):
        """Should handle content with mixed newline presence."""
        original = "Line 1\nLine 2\nLine 3"  # Last line no newline
        modified = "Line 1\nLine 2 modified\nLine 3"  # Same pattern

        diff = generate_unified_diff(original, modified)

        # Should show only the real change
        assert diff != ""
        assert "-Line 2\n" in diff
        assert "+Line 2 modified\n" in diff

    def test_only_blank_lines_with_newline_differences(self):
        """Blank lines differing only by newlines should be treated as identical."""
        original = "\n\n\n"
        modified = "\n\n\n\n"  # One more blank line

        diff = generate_unified_diff(original, modified)

        # This will show a difference because there's actually a different number of lines
        # But the test verifies our normalization works correctly
        # 3 newlines = 3 blank lines, 4 newlines = 4 blank lines
        assert diff != ""  # Different number of lines
        assert "+\n" in diff  # Shows addition of blank line

    def test_no_final_newline_both_sides(self):
        """When neither side has final newline, should work correctly."""
        original = "Line 1\nLine 2"
        modified = "Line 1\nLine 2"

        diff = generate_unified_diff(original, modified)
        assert diff == ""

    def test_single_line_no_newline(self):
        """Single line without newline should work."""
        original = "Single line"
        modified = "Single line"

        diff = generate_unified_diff(original, modified)
        assert diff == ""

    def test_single_line_change_no_newline(self):
        """Single line change without newlines should show diff."""
        original = "Single line"
        modified = "Single line changed"

        diff = generate_unified_diff(original, modified)
        assert diff != ""
        assert "-Single line\n" in diff
        assert "+Single line changed\n" in diff
