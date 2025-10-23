"""Unit tests for JournalEntry model."""

from datetime import date
from pathlib import Path
from textwrap import dedent

import pytest

from logsqueak.models.journal import JournalEntry, _parse_date_from_filename


class TestParseDateFromFilename:
    """Tests for date parsing from Logseq journal filenames."""

    def test_valid_filename(self):
        """Test parsing valid journal filename."""
        result = _parse_date_from_filename("2025_01_15.md")
        assert result == date(2025, 1, 15)

    def test_different_date(self):
        """Test parsing different valid date."""
        result = _parse_date_from_filename("2024_12_31.md")
        assert result == date(2024, 12, 31)

    def test_invalid_format_fails(self):
        """Test that invalid format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid journal filename format"):
            _parse_date_from_filename("2025-01-15.md")

    def test_invalid_date_values(self):
        """Test that invalid date values raise ValueError."""
        with pytest.raises(ValueError, match="Invalid date"):
            _parse_date_from_filename("2025_13_01.md")  # Month 13

    def test_missing_extension(self):
        """Test parsing without .md extension."""
        result = _parse_date_from_filename("2025_01_15")
        assert result == date(2025, 1, 15)


class TestJournalEntry:
    """Tests for JournalEntry model."""

    def test_load_valid_journal(self, tmp_path):
        """Test loading valid journal entry."""
        journal_file = tmp_path / "2025_01_15.md"
        content = dedent(
            """\
            - worked on project
            - met with team
            - Discovered deadline slipping"""
        )
        journal_file.write_text(content)

        entry = JournalEntry.load(journal_file)

        assert entry.date == date(2025, 1, 15)
        assert entry.file_path == journal_file
        assert entry.raw_content == content
        assert entry.line_count == 3
        assert len(entry.outline.blocks) == 3

    def test_load_empty_journal(self, tmp_path):
        """Test loading empty journal file."""
        journal_file = tmp_path / "2025_01_15.md"
        journal_file.write_text("")

        entry = JournalEntry.load(journal_file)

        assert entry.date == date(2025, 1, 15)
        assert entry.raw_content == ""
        assert entry.line_count == 0  # Empty string splits to empty list
        assert len(entry.outline.blocks) == 0

    def test_load_journal_with_nested_bullets(self, tmp_path):
        """Test loading journal with nested structure."""
        journal_file = tmp_path / "2025_01_15.md"
        content = dedent(
            """\
            - Parent item
              - Child item
                - Grandchild item"""
        )
        journal_file.write_text(content)

        entry = JournalEntry.load(journal_file)

        assert entry.line_count == 3
        assert len(entry.outline.blocks) == 1  # One root block
        assert len(entry.outline.blocks[0].children) == 1  # One child
        assert len(entry.outline.blocks[0].children[0].children) == 1  # One grandchild

    def test_load_exceeds_2000_lines(self, tmp_path):
        """Test FR-019: 2000-line limit with truncation warning."""
        journal_file = tmp_path / "2025_01_15.md"

        # Create 2500 lines
        lines = [f"- Line {i}" for i in range(2500)]
        journal_file.write_text("\n".join(lines))

        entry = JournalEntry.load(journal_file)

        # Should truncate to 2000 lines
        assert entry.line_count == 2000
        assert "Line 1999" in entry.raw_content  # Last line (0-indexed)
        assert "Line 2000" not in entry.raw_content  # Should be truncated

    def test_load_nonexistent_file_fails(self):
        """Test that loading nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Journal file not found"):
            JournalEntry.load(Path("/nonexistent/journal.md"))

    def test_load_malformed_markdown(self, tmp_path):
        """Test FR-018: Malformed entry handling."""
        journal_file = tmp_path / "2025_01_15.md"
        # This will parse but might have unexpected structure
        content = "Not a bullet list\nJust plain text"
        journal_file.write_text(content)

        entry = JournalEntry.load(journal_file)

        # Should still load, but outline might be empty (no bullets)
        assert entry.raw_content == content
        assert len(entry.outline.blocks) == 0  # No bullets found

    def test_load_with_properties(self, tmp_path):
        """Test loading journal with Logseq properties."""
        journal_file = tmp_path / "2025_01_15.md"
        content = dedent(
            """\
            - Item with property
              property:: value
            - Another item"""
        )
        journal_file.write_text(content)

        entry = JournalEntry.load(journal_file)

        assert entry.line_count == 3
        # Parser should handle properties (tested separately)
        assert len(entry.outline.blocks) > 0
