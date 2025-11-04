"""Journal entry model for Logseq daily journals."""

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from logseq_outline import LogseqOutline

logger = logging.getLogger(__name__)


@dataclass
class JournalEntry:
    """Represents a daily Logseq journal file.

    Attributes:
        date: The journal date
        file_path: Absolute path to the journal markdown file
        raw_content: Complete file contents
        outline: Parsed outline structure
        line_count: Number of lines (for 2000-line limit enforcement)
    """

    date: date
    file_path: Path
    raw_content: str
    outline: "LogseqOutline"
    line_count: int

    @classmethod
    def load(cls, file_path: Path) -> "JournalEntry":
        """Load and parse journal entry from file.

        Enforces FR-019 2000-line limit with warning and truncation.

        Args:
            file_path: Path to journal markdown file

        Returns:
            Loaded JournalEntry instance

        Raises:
            FileNotFoundError: If journal file doesn't exist
            ValueError: If file is malformed (FR-018 - logged and re-raised)
        """
        from logseq_outline import LogseqOutline

        if not file_path.exists():
            raise FileNotFoundError(f"Journal file not found: {file_path}")

        try:
            raw_content = file_path.read_text()
            lines = raw_content.splitlines()

            # FR-019: Enforce 2000-line limit
            if len(lines) > 2000:
                logger.warning(
                    f"Journal entry exceeds 2000 lines ({len(lines)} lines), "
                    f"truncating to first 2000 lines: {file_path.name}"
                )
                raw_content = "\n".join(lines[:2000])
                lines = lines[:2000]

            # Parse the outline
            outline = LogseqOutline.parse(raw_content)

            # Extract date from filename (e.g., "2025_01_15.md" -> 2025-01-15)
            entry_date = _parse_date_from_filename(file_path.name)

            return cls(
                date=entry_date,
                file_path=file_path,
                raw_content=raw_content,
                outline=outline,
                line_count=len(lines),
            )

        except Exception as e:
            # FR-018: Log malformed entries
            logger.warning(f"Malformed markdown in {file_path}: {e}")
            raise ValueError(f"Failed to parse journal entry: {e}") from e


def _parse_date_from_filename(filename: str) -> date:
    """Parse date from Logseq journal filename.

    Logseq uses YYYY_MM_DD.md format.

    Args:
        filename: Journal filename (e.g., "2025_01_15.md")

    Returns:
        Parsed date

    Raises:
        ValueError: If filename doesn't match expected format
    """
    # Remove .md extension
    name = filename.replace(".md", "")

    # Split by underscore
    parts = name.split("_")
    if len(parts) != 3:
        raise ValueError(f"Invalid journal filename format: {filename}")

    try:
        year = int(parts[0])
        month = int(parts[1])
        day = int(parts[2])
        return date(year, month, day)
    except (ValueError, IndexError) as e:
        raise ValueError(f"Invalid date in filename {filename}: {e}") from e
