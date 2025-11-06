"""CLI entry point for Logsqueak."""

import click
from pathlib import Path
from logsqueak.utils.logging import configure_logging, get_logger


logger = get_logger(__name__)


@click.group()
@click.version_option(version="0.1.0", prog_name="logsqueak")
def cli():
    """Logsqueak: Extract lasting knowledge from Logseq journal entries using LLM-powered analysis."""
    # Configure logging on CLI startup
    configure_logging()


@cli.command()
@click.argument("date_or_range", required=False)
def extract(date_or_range: str = None):
    """
    Extract knowledge from Logseq journal entries.

    Examples:
        logsqueak extract                  # Extract from today's journal
        logsqueak extract 2025-01-15       # Extract from specific date
        logsqueak extract 2025-01-10..2025-01-15  # Extract from date range
    """
    logger.info("extract_command_started", date_or_range=date_or_range)

    # Placeholder implementation - will be completed in Phase 6
    click.echo("Logsqueak extract command (placeholder)")

    if date_or_range:
        click.echo(f"Would extract from: {date_or_range}")
    else:
        click.echo("Would extract from today's journal")

    logger.info("extract_command_completed")


def main():
    """Main entry point for setuptools console script."""
    cli()


if __name__ == "__main__":
    main()
