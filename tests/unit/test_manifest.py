"""Unit tests for cache manifest system."""

from pathlib import Path

import pytest

from logsqueak.rag.manifest import CacheManifest


class TestCacheManifest:
    """Tests for CacheManifest class."""

    def test_initialization_new_manifest(self, tmp_path):
        """Test initializing manifest with nonexistent file."""
        manifest_path = tmp_path / "manifest.json"

        manifest = CacheManifest(manifest_path)

        assert manifest.manifest_path == manifest_path
        assert manifest.entries == {}

    def test_initialization_existing_manifest(self, tmp_path):
        """Test initializing manifest with existing file."""
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text('{"Page A": 1234.5, "Page B": 6789.0}', encoding="utf-8")

        manifest = CacheManifest(manifest_path)

        assert manifest.entries == {"Page A": 1234.5, "Page B": 6789.0}

    def test_load_malformed_manifest(self, tmp_path):
        """Test loading malformed manifest raises error."""
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text("not valid json", encoding="utf-8")

        with pytest.raises(ValueError, match="Malformed manifest file"):
            CacheManifest(manifest_path)

    def test_save_manifest(self, tmp_path):
        """Test saving manifest to disk."""
        manifest_path = tmp_path / "manifest.json"
        manifest = CacheManifest(manifest_path)

        manifest.set_mtime("Page A", 1234.5)
        manifest.set_mtime("Page B", 6789.0)
        manifest.save()

        # Verify file was created
        assert manifest_path.exists()

        # Verify content
        saved_manifest = CacheManifest(manifest_path)
        assert saved_manifest.entries == {"Page A": 1234.5, "Page B": 6789.0}

    def test_save_creates_parent_directory(self, tmp_path):
        """Test that save creates parent directories."""
        manifest_path = tmp_path / "nested" / "dir" / "manifest.json"
        manifest = CacheManifest(manifest_path)

        manifest.set_mtime("Page A", 1234.5)
        manifest.save()

        assert manifest_path.exists()
        assert manifest_path.parent.exists()

    def test_save_atomic_write(self, tmp_path):
        """Test that save uses atomic write (temp file + rename)."""
        manifest_path = tmp_path / "manifest.json"
        manifest = CacheManifest(manifest_path)

        manifest.set_mtime("Page A", 1234.5)
        manifest.save()

        # Temp file should not exist after successful save
        temp_path = manifest_path.with_suffix(".tmp")
        assert not temp_path.exists()

    def test_get_mtime_existing_page(self, tmp_path):
        """Test getting mtime for existing page."""
        manifest_path = tmp_path / "manifest.json"
        manifest = CacheManifest(manifest_path)

        manifest.set_mtime("Page A", 1234.5)

        assert manifest.get_mtime("Page A") == 1234.5

    def test_get_mtime_nonexistent_page(self, tmp_path):
        """Test getting mtime for nonexistent page returns None."""
        manifest_path = tmp_path / "manifest.json"
        manifest = CacheManifest(manifest_path)

        assert manifest.get_mtime("Nonexistent") is None

    def test_set_mtime(self, tmp_path):
        """Test setting mtime for a page."""
        manifest_path = tmp_path / "manifest.json"
        manifest = CacheManifest(manifest_path)

        manifest.set_mtime("Page A", 1234.5)

        assert manifest.entries["Page A"] == 1234.5

    def test_set_mtime_updates_existing(self, tmp_path):
        """Test that set_mtime updates existing entry."""
        manifest_path = tmp_path / "manifest.json"
        manifest = CacheManifest(manifest_path)

        manifest.set_mtime("Page A", 1234.5)
        manifest.set_mtime("Page A", 6789.0)

        assert manifest.entries["Page A"] == 6789.0

    def test_remove_existing_page(self, tmp_path):
        """Test removing existing page."""
        manifest_path = tmp_path / "manifest.json"
        manifest = CacheManifest(manifest_path)

        manifest.set_mtime("Page A", 1234.5)
        manifest.set_mtime("Page B", 6789.0)

        manifest.remove("Page A")

        assert "Page A" not in manifest.entries
        assert "Page B" in manifest.entries

    def test_remove_nonexistent_page(self, tmp_path):
        """Test removing nonexistent page does nothing."""
        manifest_path = tmp_path / "manifest.json"
        manifest = CacheManifest(manifest_path)

        # Should not raise
        manifest.remove("Nonexistent")

        assert manifest.entries == {}

    def test_has_page_existing(self, tmp_path):
        """Test has_page returns True for existing page."""
        manifest_path = tmp_path / "manifest.json"
        manifest = CacheManifest(manifest_path)

        manifest.set_mtime("Page A", 1234.5)

        assert manifest.has_page("Page A") is True

    def test_has_page_nonexistent(self, tmp_path):
        """Test has_page returns False for nonexistent page."""
        manifest_path = tmp_path / "manifest.json"
        manifest = CacheManifest(manifest_path)

        assert manifest.has_page("Nonexistent") is False

    def test_get_all_pages(self, tmp_path):
        """Test getting all page names."""
        manifest_path = tmp_path / "manifest.json"
        manifest = CacheManifest(manifest_path)

        manifest.set_mtime("Page A", 1234.5)
        manifest.set_mtime("Page B", 6789.0)
        manifest.set_mtime("Page C", 4567.0)

        pages = manifest.get_all_pages()

        assert set(pages) == {"Page A", "Page B", "Page C"}

    def test_get_all_pages_empty(self, tmp_path):
        """Test get_all_pages returns empty list for empty manifest."""
        manifest_path = tmp_path / "manifest.json"
        manifest = CacheManifest(manifest_path)

        assert manifest.get_all_pages() == []

    def test_clear(self, tmp_path):
        """Test clearing all entries."""
        manifest_path = tmp_path / "manifest.json"
        manifest = CacheManifest(manifest_path)

        manifest.set_mtime("Page A", 1234.5)
        manifest.set_mtime("Page B", 6789.0)

        manifest.clear()

        assert manifest.entries == {}
        assert manifest.get_all_pages() == []

    def test_persistence_across_instances(self, tmp_path):
        """Test that manifest persists across instances."""
        manifest_path = tmp_path / "manifest.json"

        # First instance
        manifest1 = CacheManifest(manifest_path)
        manifest1.set_mtime("Page A", 1234.5)
        manifest1.save()

        # Second instance (load from disk)
        manifest2 = CacheManifest(manifest_path)

        assert manifest2.get_mtime("Page A") == 1234.5

    def test_manifest_json_format(self, tmp_path):
        """Test that manifest is saved as valid JSON."""
        manifest_path = tmp_path / "manifest.json"
        manifest = CacheManifest(manifest_path)

        manifest.set_mtime("Page A", 1234.5)
        manifest.set_mtime("Page B", 6789.0)
        manifest.save()

        # Read and verify JSON format
        import json
        content = manifest_path.read_text(encoding="utf-8")
        data = json.loads(content)

        assert data == {"Page A": 1234.5, "Page B": 6789.0}

    def test_manifest_sorted_keys(self, tmp_path):
        """Test that manifest keys are sorted for readability."""
        manifest_path = tmp_path / "manifest.json"
        manifest = CacheManifest(manifest_path)

        # Add in non-alphabetical order
        manifest.set_mtime("Zebra", 3.0)
        manifest.set_mtime("Apple", 1.0)
        manifest.set_mtime("Banana", 2.0)
        manifest.save()

        # Read file content
        content = manifest_path.read_text(encoding="utf-8")

        # Keys should appear in alphabetical order
        lines = content.split("\n")
        # Find the lines with keys (skip braces)
        key_lines = [line for line in lines if '":' in line]
        assert '"Apple"' in key_lines[0]
        assert '"Banana"' in key_lines[1]
        assert '"Zebra"' in key_lines[2]
