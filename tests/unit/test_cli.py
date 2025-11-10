"""Unit tests for CLI module."""

import pytest
from datetime import date
from logsqueak.cli import parse_date_or_range


class TestParseDateOrRange:
    """Test date and date range parsing."""

    def test_parse_single_date(self):
        """Test parsing a single date string."""
        result = parse_date_or_range("2025-01-15")
        assert result == [date(2025, 1, 15)]

    def test_parse_date_range(self):
        """Test parsing a date range string."""
        result = parse_date_or_range("2025-01-10..2025-01-15")
        expected = [
            date(2025, 1, 10),
            date(2025, 1, 11),
            date(2025, 1, 12),
            date(2025, 1, 13),
            date(2025, 1, 14),
            date(2025, 1, 15),
        ]
        assert result == expected

    def test_parse_none_returns_today(self):
        """Test that None returns today's date."""
        result = parse_date_or_range(None)
        assert result == [date.today()]

    def test_parse_empty_string_returns_today(self):
        """Test that empty string returns today's date."""
        result = parse_date_or_range("")
        assert result == [date.today()]

    def test_invalid_date_format_raises_error(self):
        """Test that invalid date format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid date format"):
            parse_date_or_range("2025/01/15")

    def test_invalid_date_values_raise_error(self):
        """Test that invalid date values raise ValueError."""
        with pytest.raises(ValueError, match="Invalid date"):
            parse_date_or_range("2025-13-01")  # Invalid month

    def test_invalid_range_format_raises_error(self):
        """Test that invalid range format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid date range format"):
            parse_date_or_range("2025-01-10...2025-01-15")  # Three dots

    def test_reversed_range_raises_error(self):
        """Test that reversed date range raises ValueError."""
        with pytest.raises(ValueError, match="Start date must be before or equal to end date"):
            parse_date_or_range("2025-01-15..2025-01-10")

    def test_single_day_range(self):
        """Test that a range with same start and end date works."""
        result = parse_date_or_range("2025-01-15..2025-01-15")
        assert result == [date(2025, 1, 15)]

    def test_leap_year_date(self):
        """Test parsing a leap year date."""
        result = parse_date_or_range("2024-02-29")
        assert result == [date(2024, 2, 29)]

    def test_non_leap_year_feb_29_raises_error(self):
        """Test that Feb 29 in non-leap year raises ValueError."""
        with pytest.raises(ValueError, match="Invalid date"):
            parse_date_or_range("2025-02-29")

    def test_range_across_month_boundary(self):
        """Test date range across month boundary."""
        result = parse_date_or_range("2025-01-30..2025-02-02")
        expected = [
            date(2025, 1, 30),
            date(2025, 1, 31),
            date(2025, 2, 1),
            date(2025, 2, 2),
        ]
        assert result == expected

    def test_range_across_year_boundary(self):
        """Test date range across year boundary."""
        result = parse_date_or_range("2024-12-30..2025-01-02")
        expected = [
            date(2024, 12, 30),
            date(2024, 12, 31),
            date(2025, 1, 1),
            date(2025, 1, 2),
        ]
        assert result == expected
