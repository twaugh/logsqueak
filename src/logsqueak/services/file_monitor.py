"""File modification monitoring for concurrent edit detection."""

from pathlib import Path
from typing import Dict


class FileMonitor:
    """
    Track file modification times to detect external changes.

    Used to detect when Logseq files are modified externally during TUI session
    and safely reload them before write operations.

    Example:
        >>> tracker = FileMonitor()
        >>> journal_path = Path("journal/2025-01-15.md")
        >>> tracker.record(journal_path)
        >>> # Later, before write:
        >>> if tracker.is_modified(journal_path):
        ...     # Reload file and revalidate
        ...     tracker.refresh(journal_path)
    """

    def __init__(self) -> None:
        """Initialize empty file tracker."""
        self._mtimes: Dict[Path, float] = {}

    def record(self, path: Path) -> None:
        """
        Record current modification time for a file.

        Args:
            path: File path to track

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        self._mtimes[path] = path.stat().st_mtime

    def is_modified(self, path: Path) -> bool:
        """
        Check if file has been modified since last record.

        Args:
            path: File path to check

        Returns:
            True if file modified or not yet tracked, False otherwise

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        current_mtime = path.stat().st_mtime
        if path not in self._mtimes:
            return True
        return current_mtime != self._mtimes[path]

    def refresh(self, path: Path) -> None:
        """
        Update recorded modification time after successful reload.

        Call this after reloading a file to track its new state.

        Args:
            path: File path to refresh

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        self._mtimes[path] = path.stat().st_mtime

    def check_and_reload(self, path: Path) -> bool:
        """
        Check if file is modified and needs reloading.

        Convenience method that combines is_modified check with common
        reload pattern.

        Args:
            path: File path to check

        Returns:
            True if file was modified and needs reload, False otherwise

        Example:
            >>> if tracker.check_and_reload(journal_path):
            ...     outline = LogseqOutline.parse(journal_path.read_text())
            ...     tracker.refresh(journal_path)
        """
        return self.is_modified(path)
