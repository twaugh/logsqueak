"""Unit tests for atomic_write function."""

import pytest
import time
from pathlib import Path
from logsqueak.services.file_operations import atomic_write
from logsqueak.services.file_monitor import FileMonitor
from logsqueak.services.exceptions import FileModifiedError


class TestAtomicWrite:
    """Test atomic_write function with concurrent modification detection."""

    def test_atomic_write_creates_new_file(self, tmp_path):
        """Test that atomic_write creates a new file successfully."""
        target = tmp_path / "new_file.md"
        content = "# Test Content\n- Block 1\n- Block 2"

        atomic_write(target, content)

        assert target.exists()
        assert target.read_text() == content

    def test_atomic_write_overwrites_existing_file(self, tmp_path):
        """Test that atomic_write overwrites existing file."""
        target = tmp_path / "existing.md"
        target.write_text("Old content")

        new_content = "New content"
        atomic_write(target, new_content)

        assert target.read_text() == new_content

    def test_atomic_write_with_file_monitor(self, tmp_path):
        """Test atomic_write updates file monitor after successful write."""
        target = tmp_path / "monitored.md"
        target.write_text("Initial content")

        monitor = FileMonitor()
        monitor.record(target)

        new_content = "Updated content"
        atomic_write(target, new_content, monitor)

        assert target.read_text() == new_content
        # File monitor should be refreshed (not show as modified)
        assert not monitor.is_modified(target)

    def test_atomic_write_detects_early_modification(self, tmp_path):
        """Test that atomic_write detects file modification before write (early check)."""
        target = tmp_path / "file.md"
        target.write_text("Initial content")

        monitor = FileMonitor()
        monitor.record(target)

        # Small sleep to ensure mtime changes on filesystems with low precision (CI environments)
        time.sleep(0.01)

        # Simulate external modification
        target.write_text("Modified by external process")

        # atomic_write should detect early modification
        with pytest.raises(FileModifiedError, match="early check"):
            atomic_write(target, "New content", monitor)

    def test_atomic_write_detects_late_modification(self, tmp_path, monkeypatch):
        """Test that atomic_write detects file modification during write (late check)."""
        target = tmp_path / "file.md"
        target.write_text("Initial content")

        monitor = FileMonitor()
        monitor.record(target)

        # Track if write was attempted
        write_attempted = []

        # Patch Path.write_text to simulate concurrent modification
        original_write_text = Path.write_text

        def mock_write_text(self, *args, **kwargs):
            write_attempted.append(True)
            # Simulate external modification to target file during temp file write
            if self.name.startswith('.'):  # This is the temp file
                # Small sleep to ensure mtime changes on filesystems with low precision (CI environments)
                time.sleep(0.01)
                target.write_text("Modified during write")
            return original_write_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, 'write_text', mock_write_text)

        # atomic_write should detect late modification
        with pytest.raises(FileModifiedError, match="late check"):
            atomic_write(target, "New content", monitor)

        # Verify write was attempted (temp file created)
        assert write_attempted

    def test_atomic_write_cleans_up_temp_file_on_error(self, tmp_path, monkeypatch):
        """Test that atomic_write cleans up temporary file on error."""
        target = tmp_path / "file.md"

        # Patch Path.write_text to raise an error
        def mock_write_text(self, *args, **kwargs):
            raise OSError("Simulated write error")

        monkeypatch.setattr(Path, 'write_text', mock_write_text)

        # atomic_write should clean up temp file after error
        with pytest.raises(OSError, match="Simulated write error"):
            atomic_write(target, "Content")

        # Check that no temp files remain
        temp_files = list(tmp_path.glob('.*.tmp.*'))
        assert len(temp_files) == 0

    def test_atomic_write_cleans_up_temp_file_on_modification_error(self, tmp_path):
        """Test that atomic_write cleans up temp file on FileModifiedError."""
        target = tmp_path / "file.md"
        target.write_text("Initial content")

        monitor = FileMonitor()
        monitor.record(target)

        # Small sleep to ensure mtime changes on filesystems with low precision (CI environments)
        time.sleep(0.01)

        # Simulate external modification (triggers early check)
        target.write_text("Modified externally")

        # atomic_write should clean up temp file
        with pytest.raises(FileModifiedError):
            atomic_write(target, "New content", monitor)

        # Check that no temp files remain
        temp_files = list(tmp_path.glob('.*.tmp.*'))
        assert len(temp_files) == 0

    def test_atomic_write_preserves_content_on_modification(self, tmp_path):
        """Test that original file is preserved when modification detected."""
        target = tmp_path / "file.md"
        original_content = "Original content"
        target.write_text(original_content)

        monitor = FileMonitor()
        monitor.record(target)

        # Small sleep to ensure mtime changes on filesystems with low precision (CI environments)
        time.sleep(0.01)

        # Simulate external modification
        modified_content = "Modified by external process"
        target.write_text(modified_content)

        # atomic_write should fail and preserve the modified content
        with pytest.raises(FileModifiedError):
            atomic_write(target, "Attempted new content", monitor)

        # Original (modified) content should be preserved
        assert target.read_text() == modified_content

    def test_atomic_write_without_file_monitor(self, tmp_path):
        """Test that atomic_write works without file monitor (no modification checks)."""
        target = tmp_path / "file.md"
        content = "Test content"

        # Should work fine without monitor
        atomic_write(target, content, file_monitor=None)

        assert target.read_text() == content

    def test_atomic_write_with_unicode_content(self, tmp_path):
        """Test that atomic_write handles Unicode content correctly."""
        target = tmp_path / "unicode.md"
        content = "# Test\n- 日本語\n- Emoji: 🤖✅\n- Special: €£¥"

        atomic_write(target, content)

        assert target.read_text(encoding='utf-8') == content

    def test_atomic_write_temp_file_in_same_directory(self, tmp_path):
        """Test that temp file is created in same directory (for atomic rename)."""
        target = tmp_path / "file.md"

        # Track temp file location
        temp_files = []
        original_write_text = Path.write_text

        def track_write_text(self, *args, **kwargs):
            if self.name.startswith('.') and '.tmp.' in self.name:
                temp_files.append(self)
            return original_write_text(self, *args, **kwargs)

        import logsqueak.services.file_operations
        original_func = logsqueak.services.file_operations.Path.write_text
        logsqueak.services.file_operations.Path.write_text = track_write_text

        try:
            atomic_write(target, "Content")

            # Verify temp file was in same directory
            assert len(temp_files) == 1
            assert temp_files[0].parent == target.parent
        finally:
            logsqueak.services.file_operations.Path.write_text = original_func
