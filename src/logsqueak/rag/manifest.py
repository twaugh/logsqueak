"""Cache manifest for tracking indexed pages.

This module provides a manifest system to track which pages have been indexed
and their modification times (mtime). This enables incremental indexing by
detecting additions, updates, and deletions.
"""

import json
from pathlib import Path
from typing import Dict, Optional


class CacheManifest:
    """Manifest tracking indexed pages and their mtimes.

    The manifest is stored as JSON with format:
    {
        "page_name": mtime_timestamp,
        ...
    }
    """

    def __init__(self, manifest_path: Path):
        """Initialize cache manifest.

        Args:
            manifest_path: Path to manifest JSON file
        """
        self.manifest_path = manifest_path
        self.entries: Dict[str, float] = {}

        # Load existing manifest if present
        if manifest_path.exists():
            self.load()

    def load(self) -> None:
        """Load manifest from disk.

        Raises:
            ValueError: If manifest file is malformed
        """
        try:
            content = self.manifest_path.read_text(encoding="utf-8")
            self.entries = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Malformed manifest file: {e}") from e

    def save(self) -> None:
        """Save manifest to disk.

        Creates parent directories if needed.
        """
        # Create parent directory if needed
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

        # Write manifest atomically (write to temp, then rename)
        temp_path = self.manifest_path.with_suffix(".tmp")
        try:
            content = json.dumps(self.entries, indent=2, sort_keys=True)
            temp_path.write_text(content, encoding="utf-8")
            temp_path.replace(self.manifest_path)
        except Exception as e:
            # Clean up temp file on error
            if temp_path.exists():
                temp_path.unlink()
            raise IOError(f"Failed to save manifest: {e}") from e

    def get_mtime(self, page_name: str) -> Optional[float]:
        """Get cached mtime for a page.

        Args:
            page_name: Page name to look up

        Returns:
            Cached mtime timestamp, or None if not in manifest
        """
        return self.entries.get(page_name)

    def set_mtime(self, page_name: str, mtime: float) -> None:
        """Set mtime for a page.

        Args:
            page_name: Page name
            mtime: Modification time timestamp
        """
        self.entries[page_name] = mtime

    def remove(self, page_name: str) -> None:
        """Remove a page from the manifest.

        Args:
            page_name: Page name to remove
        """
        self.entries.pop(page_name, None)

    def has_page(self, page_name: str) -> bool:
        """Check if page is in manifest.

        Args:
            page_name: Page name to check

        Returns:
            True if page is in manifest
        """
        return page_name in self.entries

    def get_all_pages(self) -> list[str]:
        """Get list of all pages in manifest.

        Returns:
            List of page names
        """
        return list(self.entries.keys())

    def clear(self) -> None:
        """Clear all entries from manifest."""
        self.entries = {}
