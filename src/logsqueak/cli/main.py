#!/usr/bin/env python3
"""Logsqueak CLI - Extract knowledge from Logseq journals.

This is the main entry point for the Logsqueak command-line tool.
"""

import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import click

from logsqueak.config.loader import load_config


@click.group()
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    help="Path to configuration file (default: ~/.config/logsqueak/config.yaml)",
)
@click.option("--verbose", is_flag=True, help="Enable verbose logging")
@click.option("--version", is_flag=True, help="Show version and exit")
@click.pass_context
def cli(ctx: click.Context, config: Optional[Path], verbose: bool, version: bool):
    """Logsqueak - Extract lasting knowledge from Logseq journals.

    Extract knowledge from your daily journal entries and integrate it into
    your knowledge base using LLM-powered analysis.
    """
    if version:
        click.echo("Logsqueak v0.1.0 (POC)")
        ctx.exit(0)

    # Store global options in context
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config
    ctx.obj["verbose"] = verbose


@cli.command()
@click.argument("date_or_range", required=False, default="today")
@click.option(
    "--dry-run",
    is_flag=True,
    help="Dry-run mode: show diffs instead of writing files",
)
@click.option(
    "--model",
    type=str,
    help="Override LLM model (default: from config)",
)
@click.option(
    "--graph",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Override Logseq graph path (default: from config)",
)
@click.pass_context
def extract(
    ctx: click.Context,
    date_or_range: str,
    dry_run: bool,
    model: Optional[str],
    graph: Optional[Path],
):
    """Extract knowledge from journal entries.

    DATE_OR_RANGE can be:
      - Single date: 2025-01-15 (ISO 8601)
      - Date range: 2025-01-10..2025-01-15 (inclusive)
      - Relative: today, yesterday, last-week

    Examples:
      logsqueak extract                    # Extract from today
      logsqueak extract 2025-01-15        # Extract from specific date
      logsqueak extract --dry-run today   # Dry-run mode (shows diffs, no writes)
    """
    verbose = ctx.obj["verbose"]

    if verbose:
        click.echo(f"Date/range: {date_or_range}")
        click.echo(f"Dry-run: {dry_run}")
        if model:
            click.echo(f"Model override: {model}")
        if graph:
            click.echo(f"Graph override: {graph}")

    # Load configuration
    try:
        config_path = ctx.obj["config_path"]
        config = load_config(config_path)
        click.echo(f"Loading configuration from {config_path or '~/.config/logsqueak/config.yaml'}")
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        ctx.exit(1)
    except Exception as e:
        click.echo(f"Error loading configuration: {e}", err=True)
        ctx.exit(1)

    # Apply overrides
    if model:
        config.llm.model = model
    if graph:
        config.logseq.graph_path = graph

    # Parse date or range
    try:
        dates = parse_date_or_range(date_or_range)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        click.echo(
            'Expected: YYYY-MM-DD, "today", "yesterday", or "YYYY-MM-DD..YYYY-MM-DD"',
            err=True,
        )
        ctx.exit(1)

    if verbose:
        click.echo(f"Processing dates: {[d.isoformat() for d in dates]}")

    # TODO: Implement extraction workflow (T021-T032)
    # For now, just show what we parsed
    click.echo(f"\nConfiguration loaded:")
    click.echo(f"  LLM endpoint: {config.llm.endpoint}")
    click.echo(f"  LLM model: {config.llm.model}")
    click.echo(f"  Graph path: {config.logseq.graph_path}")
    click.echo(f"\nWould process {len(dates)} date(s):")
    for d in dates:
        click.echo(f"  - {d.isoformat()}")
    click.echo(f"\nMode: {'Dry-run (show diffs only)' if dry_run else 'Apply (with confirmation)'}")
    click.echo("\n[Extraction logic not yet implemented - Phase 3 tasks T021-T032]")


def parse_date_or_range(date_str: str) -> list[date]:
    """Parse date or date range argument.

    Args:
        date_str: Date string (ISO, relative, or range)

    Returns:
        List of dates to process

    Raises:
        ValueError: If date format is invalid

    Examples:
        >>> parse_date_or_range("2025-01-15")
        [date(2025, 1, 15)]

        >>> parse_date_or_range("today")
        [date.today()]

        >>> parse_date_or_range("2025-01-10..2025-01-12")
        [date(2025, 1, 10), date(2025, 1, 11), date(2025, 1, 12)]
    """
    # Handle relative dates
    if date_str == "today":
        return [date.today()]
    elif date_str == "yesterday":
        return [date.today() - timedelta(days=1)]
    elif date_str == "last-week":
        # Return last 7 days
        today = date.today()
        return [today - timedelta(days=i) for i in range(7, 0, -1)]

    # Handle date ranges
    if ".." in date_str:
        try:
            start_str, end_str = date_str.split("..")
            start = date.fromisoformat(start_str)
            end = date.fromisoformat(end_str)

            if start > end:
                raise ValueError(
                    f"Invalid date range: start ({start}) is after end ({end})"
                )

            # Generate all dates in range (inclusive)
            dates = []
            current = start
            while current <= end:
                dates.append(current)
                current += timedelta(days=1)

            return dates
        except ValueError as e:
            if "Invalid date range" in str(e):
                raise
            raise ValueError(f'Invalid date format: "{date_str}"') from e

    # Handle single ISO date
    try:
        return [date.fromisoformat(date_str)]
    except ValueError as e:
        raise ValueError(f'Invalid date format: "{date_str}"') from e


def main():
    """Entry point for the CLI."""
    cli(obj={})


if __name__ == "__main__":
    main()
