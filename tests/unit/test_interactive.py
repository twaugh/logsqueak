"""Unit tests for interactive user prompts.

Tests user interaction flows for approval, rejection, and editing
of proposed knowledge extractions.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from logsqueak.cli.interactive import (
    confirm_action,
    prompt_for_approval,
    prompt_for_edit,
    show_error,
    show_progress,
    show_success,
    show_warning,
)


class TestPromptForApproval:
    """Test approval prompt functionality."""

    @patch('logsqueak.cli.interactive.click.prompt')
    def test_approval_with_y(self, mock_prompt):
        """Test user approves with 'y'."""
        mock_prompt.return_value = 'y'

        result = prompt_for_approval()

        assert result == 'y'
        mock_prompt.assert_called_once()

    @patch('logsqueak.cli.interactive.click.prompt')
    def test_approval_with_yes(self, mock_prompt):
        """Test user approves with 'yes'."""
        mock_prompt.return_value = 'yes'

        result = prompt_for_approval()

        assert result == 'y'  # Normalized to single letter

    @patch('logsqueak.cli.interactive.click.prompt')
    def test_rejection_with_n(self, mock_prompt):
        """Test user rejects with 'n'."""
        mock_prompt.return_value = 'n'

        result = prompt_for_approval()

        assert result == 'n'

    @patch('logsqueak.cli.interactive.click.prompt')
    def test_rejection_with_no(self, mock_prompt):
        """Test user rejects with 'no'."""
        mock_prompt.return_value = 'no'

        result = prompt_for_approval()

        assert result == 'n'  # Normalized

    @patch('logsqueak.cli.interactive.click.prompt')
    def test_edit_with_e(self, mock_prompt):
        """Test user chooses edit with 'e'."""
        mock_prompt.return_value = 'e'

        result = prompt_for_approval()

        assert result == 'e'

    @patch('logsqueak.cli.interactive.click.prompt')
    def test_edit_with_edit(self, mock_prompt):
        """Test user chooses edit with 'edit'."""
        mock_prompt.return_value = 'edit'

        result = prompt_for_approval()

        assert result == 'e'  # Normalized

    @patch('logsqueak.cli.interactive.click.prompt')
    def test_case_insensitive(self, mock_prompt):
        """Test prompt handles case insensitivity."""
        mock_prompt.return_value = 'Y'

        result = prompt_for_approval()

        assert result == 'y'


class TestConfirmAction:
    """Test confirmation prompt functionality."""

    @patch('logsqueak.cli.interactive.click.confirm')
    def test_confirm_true(self, mock_confirm):
        """Test user confirms action."""
        mock_confirm.return_value = True

        result = confirm_action("Proceed?")

        assert result is True
        mock_confirm.assert_called_once_with("Proceed?", default=False)

    @patch('logsqueak.cli.interactive.click.confirm')
    def test_confirm_false(self, mock_confirm):
        """Test user rejects action."""
        mock_confirm.return_value = False

        result = confirm_action("Proceed?")

        assert result is False

    @patch('logsqueak.cli.interactive.click.confirm')
    def test_confirm_with_default_true(self, mock_confirm):
        """Test confirmation with default True."""
        mock_confirm.return_value = True

        result = confirm_action("Proceed?", default=True)

        assert result is True
        mock_confirm.assert_called_once_with("Proceed?", default=True)


class TestPromptForEdit:
    """Test edit mode functionality."""

    @patch('logsqueak.cli.interactive.click.edit')
    def test_edit_returns_content(self, mock_edit):
        """Test edit returns user's edited content."""
        mock_edit.return_value = "# Comment\nEdited content\nMore content"

        result = prompt_for_edit()

        # Comments should be filtered out
        assert result == "Edited content\nMore content"

    @patch('logsqueak.cli.interactive.click.edit')
    def test_edit_cancelled(self, mock_edit):
        """Test edit returns None when user cancels."""
        mock_edit.return_value = None

        result = prompt_for_edit()

        assert result is None

    @patch('logsqueak.cli.interactive.click.edit')
    def test_edit_filters_comments(self, mock_edit):
        """Test that comment lines are filtered."""
        mock_edit.return_value = "Line 1\n# Comment line\nLine 2\n# Another comment"

        result = prompt_for_edit()

        assert "# Comment" not in result
        assert "Line 1" in result
        assert "Line 2" in result

    @patch('logsqueak.cli.interactive.click.edit')
    @patch('logsqueak.cli.interactive.click.echo')
    def test_edit_error_handling(self, mock_echo, mock_edit):
        """Test edit handles errors gracefully."""
        mock_edit.side_effect = Exception("Editor not found")

        result = prompt_for_edit()

        assert result is None
        mock_echo.assert_called()


class TestProgressMessages:
    """Test progress and status message display."""

    @patch('logsqueak.cli.interactive.click.echo')
    def test_show_progress_normal(self, mock_echo):
        """Test normal progress message."""
        show_progress("Processing...")

        mock_echo.assert_called_once_with("Processing...")

    @patch('logsqueak.cli.interactive.click.echo')
    def test_show_progress_as_step(self, mock_echo):
        """Test progress message as step (with bullet)."""
        show_progress("Extracting knowledge", is_step=True)

        mock_echo.assert_called_once_with("  • Extracting knowledge")

    @patch('logsqueak.cli.interactive.click.secho')
    def test_show_warning(self, mock_secho):
        """Test warning message display."""
        show_warning("Low confidence")

        mock_secho.assert_called_once_with("⚠ Warning: Low confidence", fg='yellow', err=True)

    @patch('logsqueak.cli.interactive.click.secho')
    def test_show_error(self, mock_secho):
        """Test error message display."""
        show_error("File not found")

        mock_secho.assert_called_once_with("✗ Error: File not found", fg='red', err=True)

    @patch('logsqueak.cli.interactive.click.secho')
    def test_show_success(self, mock_secho):
        """Test success message display."""
        show_success("Changes applied")

        mock_secho.assert_called_once_with("✓ Changes applied", fg='green')
