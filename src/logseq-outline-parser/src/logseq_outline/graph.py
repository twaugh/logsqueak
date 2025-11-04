"""Logseq graph path operations.

This module provides utilities for navigating Logseq graph directory structure.
"""

from pathlib import Path


class GraphPaths:
    """Utility class for Logseq graph path operations.

    Attributes:
        graph_path: Root path to Logseq graph directory
    """

    def __init__(self, graph_path: Path):
        """Initialize with graph root path.

        Args:
            graph_path: Path to Logseq graph directory

        Raises:
            ValueError: If graph_path doesn't exist or isn't a directory
        """
        if not graph_path.exists():
            raise ValueError(f"Graph path does not exist: {graph_path}")
        if not graph_path.is_dir():
            raise ValueError(f"Graph path is not a directory: {graph_path}")

        self.graph_path = graph_path

    @property
    def journals_dir(self) -> Path:
        """Get journals directory path.

        Returns:
            Path to journals/ directory
        """
        return self.graph_path / "journals"

    @property
    def pages_dir(self) -> Path:
        """Get pages directory path.

        Returns:
            Path to pages/ directory
        """
        return self.graph_path / "pages"

    def get_journal_path(self, date_str: str) -> Path:
        """Get path to specific journal file.

        Args:
            date_str: Date string in YYYY_MM_DD format (e.g., "2025_01_15")

        Returns:
            Path to journal file (e.g., journals/2025_01_15.md)
        """
        return self.journals_dir / f"{date_str}.md"

    def get_page_path(self, page_name: str) -> Path:
        """Get path to specific page file.

        Args:
            page_name: Page name (e.g., "Project X")

        Returns:
            Path to page file (e.g., pages/Project X.md)
        """
        return self.pages_dir / f"{page_name}.md"

    def journal_exists(self, date_str: str) -> bool:
        """Check if journal file exists for given date.

        Args:
            date_str: Date string in YYYY_MM_DD format

        Returns:
            True if journal file exists
        """
        return self.get_journal_path(date_str).exists()

    def page_exists(self, page_name: str) -> bool:
        """Check if page file exists.

        Args:
            page_name: Page name

        Returns:
            True if page file exists
        """
        return self.get_page_path(page_name).exists()

    def list_journals(self) -> list[Path]:
        """List all journal files in chronological order.

        Returns:
            List of journal file paths, sorted by date
        """
        if not self.journals_dir.exists():
            return []

        journal_files = list(self.journals_dir.glob("*.md"))
        return sorted(journal_files)

    def list_pages(self) -> list[Path]:
        """List all page files in alphabetical order.

        Returns:
            List of page file paths, sorted alphabetically
        """
        if not self.pages_dir.exists():
            return []

        page_files = list(self.pages_dir.glob("*.md"))
        return sorted(page_files)
