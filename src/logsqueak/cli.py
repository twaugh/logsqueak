"""CLI entry point for Logsqueak."""

import click
from pathlib import Path
from datetime import date, timedelta
from logsqueak.utils.logging import configure_logging, get_logger


logger = get_logger(__name__)


def parse_date_or_range(date_or_range: str | None) -> list[date]:
    """
    Parse a date or date range string into a list of dates.

    Args:
        date_or_range: Either:
            - None or empty string: Returns today's date
            - "YYYY-MM-DD": Returns single date
            - "YYYY-MM-DD..YYYY-MM-DD": Returns list of dates in range (inclusive)

    Returns:
        List of date objects

    Raises:
        ValueError: If date format is invalid or range is reversed
    """
    # Handle None or empty string
    if not date_or_range:
        return [date.today()]

    # Check for range syntax
    if ".." in date_or_range:
        # Validate range format (exactly two dots)
        if date_or_range.count("..") != 1 or "..." in date_or_range:
            raise ValueError("Invalid date range format. Expected: YYYY-MM-DD..YYYY-MM-DD")

        start_str, end_str = date_or_range.split("..")

        # Parse start and end dates
        try:
            start_date = date.fromisoformat(start_str)
        except ValueError as e:
            raise ValueError(f"Invalid date format for start date: {start_str}. Expected: YYYY-MM-DD") from e

        try:
            end_date = date.fromisoformat(end_str)
        except ValueError as e:
            raise ValueError(f"Invalid date format for end date: {end_str}. Expected: YYYY-MM-DD") from e

        # Validate range order
        if start_date > end_date:
            raise ValueError("Start date must be before or equal to end date")

        # Generate list of dates
        dates = []
        current = start_date
        while current <= end_date:
            dates.append(current)
            current += timedelta(days=1)

        return dates
    else:
        # Single date
        try:
            single_date = date.fromisoformat(date_or_range)
            return [single_date]
        except ValueError as e:
            raise ValueError(f"Invalid date format: {date_or_range}. Expected: YYYY-MM-DD") from e


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

    # Parse date or range argument
    try:
        dates = parse_date_or_range(date_or_range)
        logger.info("dates_parsed", count=len(dates), dates=[str(d) for d in dates])
    except ValueError as e:
        logger.error("date_parse_error", error=str(e))
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()

    # Display what we're processing
    if len(dates) == 1:
        click.echo(f"Extracting from journal: {dates[0]}")
    else:
        click.echo(f"Extracting from {len(dates)} journal entries: {dates[0]} to {dates[-1]}")

    # TODO: Remaining implementation in T105-T107
    # - Load journal entries (T105)
    # - Load config (T106)
    # - Initialize services and launch TUI (T107)

    logger.info("extract_command_completed")


def main():
    """Main entry point for setuptools console script."""
    cli()


if __name__ == "__main__":
    main()
