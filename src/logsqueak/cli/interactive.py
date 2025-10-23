"""Interactive user prompts for approval workflows.

This module handles user interaction for approving, rejecting, or editing
proposed knowledge extractions before they are integrated into pages.
"""

import sys
from typing import Optional

import click


def prompt_for_approval() -> str:
    """Prompt user for approval of proposed changes.

    Returns one of:
    - 'y' or 'yes': Approve and apply changes
    - 'n' or 'no': Reject changes (no modifications)
    - 'e' or 'edit': Enter edit mode to modify preview

    Returns:
        User's choice as normalized string ('y', 'n', or 'e')
    """
    while True:
        choice = click.prompt(
            "\nApply these changes?",
            type=click.Choice(['y', 'n', 'e', 'yes', 'no', 'edit'], case_sensitive=False),
            show_choices=True,
            default='n',
        )

        # Normalize to single letter
        normalized = choice.lower()[0]  # 'y', 'n', or 'e'

        if normalized in ['y', 'n', 'e']:
            return normalized

        # Should never reach here due to click.Choice validation
        click.echo("Invalid choice. Please enter y, n, or e.", err=True)


def confirm_action(message: str, default: bool = False) -> bool:
    """Prompt user for yes/no confirmation.

    Args:
        message: Confirmation message to display
        default: Default value if user just presses Enter

    Returns:
        True if user confirms, False otherwise
    """
    return click.confirm(message, default=default)


def prompt_for_edit() -> Optional[str]:
    """Prompt user to edit preview content.

    Opens user's editor (via EDITOR environment variable) to allow
    editing of the proposed changes before applying.

    Returns:
        Edited text if user saved changes, None if user cancelled
    """
    # Get initial text (could be preview content passed as parameter)
    initial_text = "# Edit proposed changes below\n# Lines starting with # will be ignored\n"

    try:
        edited_text = click.edit(initial_text)

        if edited_text is None:
            # User cancelled the edit (empty file or closed without saving)
            return None

        # Filter out comment lines
        lines = [
            line for line in edited_text.split('\n')
            if not line.strip().startswith('#')
        ]

        return '\n'.join(lines).strip()

    except Exception as e:
        click.echo(f"Error opening editor: {e}", err=True)
        return None


def show_progress(message: str, is_step: bool = False) -> None:
    """Show progress message to user.

    Args:
        message: Progress message to display
        is_step: If True, shows as a step (with bullet point)
    """
    if is_step:
        click.echo(f"  • {message}")
    else:
        click.echo(message)


def show_warning(message: str) -> None:
    """Show warning message to user.

    Args:
        message: Warning message to display
    """
    click.secho(f"⚠ Warning: {message}", fg='yellow', err=True)


def show_error(message: str) -> None:
    """Show error message to user.

    Args:
        message: Error message to display
    """
    click.secho(f"✗ Error: {message}", fg='red', err=True)


def show_success(message: str) -> None:
    """Show success message to user.

    Args:
        message: Success message to display
    """
    click.secho(f"✓ {message}", fg='green')
