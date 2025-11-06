"""Unit tests for FileMonitor."""

import pytest
from pathlib import Path
import time

from logsqueak.services.file_monitor import FileMonitor


class TestFileMonitor:
    """Test FileMonitor class."""

    def test_record_and_check_unmodified(self, tmp_path):
        """Test recording file and checking it hasn't been modified."""
        monitor = FileMonitor()
        test_file = tmp_path / "test.md"
        test_file.write_text("Initial content")

        # Record current state
        monitor.record(test_file)

        # Check immediately - should not be modified
        assert not monitor.is_modified(test_file)

    def test_detect_modification(self, tmp_path):
        """Test detecting file modification."""
        monitor = FileMonitor()
        test_file = tmp_path / "test.md"
        test_file.write_text("Initial content")

        # Record initial state
        monitor.record(test_file)

        # Wait a moment to ensure mtime changes
        time.sleep(0.01)

        # Modify file
        test_file.write_text("Modified content")

        # Should detect modification
        assert monitor.is_modified(test_file)

    def test_refresh_after_reload(self, tmp_path):
        """Test refreshing tracker after reloading modified file."""
        monitor = FileMonitor()
        test_file = tmp_path / "test.md"
        test_file.write_text("Initial content")

        # Record and modify
        monitor.record(test_file)
        time.sleep(0.01)
        test_file.write_text("Modified content")

        # Detect modification
        assert monitor.is_modified(test_file)

        # Refresh after reload
        monitor.refresh(test_file)

        # Should no longer show as modified
        assert not monitor.is_modified(test_file)

    def test_untracked_file_shows_as_modified(self, tmp_path):
        """Test that untracked file is treated as modified."""
        monitor = FileMonitor()
        test_file = tmp_path / "test.md"
        test_file.write_text("Content")

        # File not yet tracked
        assert monitor.is_modified(test_file)

    def test_check_and_reload(self, tmp_path):
        """Test check_and_reload convenience method."""
        monitor = FileMonitor()
        test_file = tmp_path / "test.md"
        test_file.write_text("Initial content")

        monitor.record(test_file)

        # Not modified yet
        assert not monitor.check_and_reload(test_file)

        # Modify file
        time.sleep(0.01)
        test_file.write_text("Modified content")

        # Should return True (needs reload)
        assert monitor.check_and_reload(test_file)

    def test_multiple_files(self, tmp_path):
        """Test tracking multiple files independently."""
        monitor = FileMonitor()

        file1 = tmp_path / "file1.md"
        file2 = tmp_path / "file2.md"

        file1.write_text("File 1")
        file2.write_text("File 2")

        monitor.record(file1)
        monitor.record(file2)

        # Modify only file1
        time.sleep(0.01)
        file1.write_text("File 1 modified")

        # Only file1 should show as modified
        assert monitor.is_modified(file1)
        assert not monitor.is_modified(file2)

    def test_nonexistent_file_raises_error(self, tmp_path):
        """Test that accessing nonexistent file raises FileNotFoundError."""
        monitor = FileMonitor()
        nonexistent = tmp_path / "does-not-exist.md"

        with pytest.raises(FileNotFoundError):
            monitor.record(nonexistent)

        with pytest.raises(FileNotFoundError):
            monitor.is_modified(nonexistent)
