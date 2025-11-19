"""Unit tests for index progress utility."""

from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from logsqueak.utils.index_progress import create_index_progress_callback


class TestCreateIndexProgressCallback:
    """Tests for create_index_progress_callback function."""

    def test_returns_callback_and_cleanup_functions(self):
        """Test that function returns tuple of callback and cleanup."""
        console = Console()

        progress_callback, cleanup = create_index_progress_callback(
            console=console,
            reindex=False,
            has_data=False
        )

        assert callable(progress_callback)
        assert callable(cleanup)

    def test_model_loading_phase(self):
        """Test progress callback during model loading phase (negative current)."""
        console = MagicMock()

        progress_callback, cleanup = create_index_progress_callback(
            console=console,
            reindex=False,
            has_data=False,
            use_echo=True
        )

        # Simulate model loading (negative current)
        progress_callback(-1, 10)

        # Verify initial message was shown
        console.print.assert_called_once()
        assert "Building search index" in str(console.print.call_args)

        # Verify spinner was started
        console.status.assert_called_once()
        assert "Loading embedding model" in str(console.status.call_args)

    def test_encoding_phase(self):
        """Test progress callback during encoding phase."""
        console = MagicMock()
        mock_status = MagicMock()
        console.status.return_value = mock_status

        progress_callback, cleanup = create_index_progress_callback(
            console=console,
            reindex=False,
            has_data=False,
            use_echo=True
        )

        # Simulate model loading first
        progress_callback(-1, 10)

        # Simulate encoding phase (total changes from page count to chunk count)
        progress_callback(50, 100)

        # Verify progress percentage is shown
        mock_status.update.assert_called()
        update_args = str(mock_status.update.call_args)
        assert "50%" in update_args or "Building page index" in update_args

    def test_reindex_shows_rebuilding_message(self):
        """Test that reindex=True shows 'Rebuilding' message."""
        console = MagicMock()

        progress_callback, cleanup = create_index_progress_callback(
            console=console,
            reindex=True,  # Force rebuild
            has_data=False,
            use_echo=True
        )

        # Simulate model loading
        progress_callback(-1, 10)

        # Verify "Rebuilding" message
        console.print.assert_called_once()
        assert "Rebuilding" in str(console.print.call_args)

    def test_has_data_shows_updating_message(self):
        """Test that has_data=True shows 'Updating' message."""
        console = MagicMock()

        progress_callback, cleanup = create_index_progress_callback(
            console=console,
            reindex=False,
            has_data=True,  # Existing data
            use_echo=True
        )

        # Simulate model loading
        progress_callback(-1, 10)

        # Verify "Updating" message
        console.print.assert_called_once()
        assert "Updating" in str(console.print.call_args)

    def test_no_data_shows_building_message(self):
        """Test that has_data=False shows 'Building' message."""
        console = MagicMock()

        progress_callback, cleanup = create_index_progress_callback(
            console=console,
            reindex=False,
            has_data=False,  # No existing data
            use_echo=True
        )

        # Simulate model loading
        progress_callback(-1, 10)

        # Verify "Building" message
        console.print.assert_called_once()
        assert "Building" in str(console.print.call_args)

    def test_cleanup_stops_spinner(self):
        """Test that cleanup function stops any active spinner."""
        console = MagicMock()
        mock_status = MagicMock()
        console.status.return_value = mock_status

        progress_callback, cleanup = create_index_progress_callback(
            console=console,
            reindex=False,
            has_data=False
        )

        # Start spinner
        progress_callback(-1, 10)

        # Call cleanup
        cleanup()

        # Verify spinner was exited
        mock_status.__exit__.assert_called()

    def test_cleanup_safe_when_no_spinner_active(self):
        """Test that cleanup is safe to call when no spinner is active."""
        console = MagicMock()

        progress_callback, cleanup = create_index_progress_callback(
            console=console,
            reindex=False,
            has_data=False
        )

        # Call cleanup without starting spinner
        cleanup()  # Should not raise

    def test_use_echo_false_uses_rprint(self):
        """Test that use_echo=False uses rprint instead of console.print."""
        console = MagicMock()

        progress_callback, cleanup = create_index_progress_callback(
            console=console,
            reindex=False,
            has_data=False,
            use_echo=False  # Use rprint style
        )

        # Simulate model loading
        with patch("rich.print") as mock_rprint:
            progress_callback(-1, 10)

        # Verify rprint was used instead of console.print
        mock_rprint.assert_called()
        console.print.assert_not_called()

    def test_parsing_phase_stops_spinner(self):
        """Test that parsing phase (same total) stops spinner."""
        console = MagicMock()
        mock_status = MagicMock()
        console.status.return_value = mock_status

        progress_callback, cleanup = create_index_progress_callback(
            console=console,
            reindex=False,
            has_data=False
        )

        # Start spinner (model loading)
        progress_callback(-1, 10)

        # Simulate parsing phase (current and total are page counts, not chunk counts)
        progress_callback(5, 10)

        # Verify spinner was stopped
        mock_status.__exit__.assert_called()

    def test_initial_message_shown_only_once(self):
        """Test that initial message is shown only once."""
        console = MagicMock()

        progress_callback, cleanup = create_index_progress_callback(
            console=console,
            reindex=False,
            has_data=False,
            use_echo=True
        )

        # Simulate model loading multiple times
        progress_callback(-1, 10)
        progress_callback(-1, 10)
        progress_callback(-1, 10)

        # Verify message was shown only once
        assert console.print.call_count == 1
