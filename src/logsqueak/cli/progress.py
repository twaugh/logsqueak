"""Progress feedback display for CLI operations.

Provides clear status messages during extraction workflow.
"""

import click


def show_building_index(page_count: int) -> None:
    """Show index building progress.

    Args:
        page_count: Number of pages being indexed
    """
    click.echo(f"Building page index... (found {page_count} pages)")


def show_processing_journal(journal_date: str) -> None:
    """Show journal processing start.

    Args:
        journal_date: Date being processed (ISO format)
    """
    click.echo(f"\nProcessing journal: {journal_date}")


def show_extracting_knowledge(model: str) -> None:
    """Show extraction phase.

    Args:
        model: LLM model being used
    """
    click.echo(f"\nExtracting knowledge... (using model: {model})")


def show_error(message: str) -> None:
    """Show error message.

    Args:
        message: Error message to display
    """
    click.echo(f"Error: {message}", err=True)


def show_warning(message: str) -> None:
    """Show warning message.

    Args:
        message: Warning message to display
    """
    click.echo(f"Warning: {message}", err=True)
