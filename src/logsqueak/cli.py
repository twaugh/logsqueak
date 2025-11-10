"""CLI entry point for Logsqueak."""

import click
from pathlib import Path
from datetime import date, timedelta
from logsqueak.utils.logging import configure_logging, get_logger
from logsqueak.models.config import Config
from logseq_outline.parser import LogseqOutline
from logseq_outline.graph import GraphPaths


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


def load_journal_entries(graph_path: Path, dates: list[date]) -> dict[str, LogseqOutline]:
    """
    Load journal entries for given dates from Logseq graph.

    Args:
        graph_path: Path to Logseq graph directory
        dates: List of dates to load journals for

    Returns:
        Dictionary mapping date string (YYYY-MM-DD) to LogseqOutline

    Raises:
        ValueError: If graph_path is invalid
        FileNotFoundError: If journal file doesn't exist for a date
    """
    graph = GraphPaths(graph_path)
    journals = {}

    for journal_date in dates:
        # Convert date to Logseq journal format (YYYY_MM_DD)
        date_str = journal_date.strftime("%Y_%m_%d")
        journal_path = graph.get_journal_path(date_str)

        # Check if journal exists
        if not journal_path.exists():
            raise FileNotFoundError(
                f"Journal file not found: {journal_path}\n"
                f"Expected journal for date: {journal_date.strftime('%Y-%m-%d')}"
            )

        # Read and parse journal file
        logger.info("loading_journal", date=str(journal_date), path=str(journal_path))
        journal_text = journal_path.read_text(encoding="utf-8")
        outline = LogseqOutline.parse(journal_text)

        # Store using ISO format (YYYY-MM-DD) for consistency with user input
        journals[journal_date.strftime("%Y-%m-%d")] = outline

    logger.info("journals_loaded", count=len(journals))
    return journals


def load_config() -> Config:
    """
    Load configuration from ~/.config/logsqueak/config.yaml.

    Returns:
        Validated Config instance

    Raises:
        click.ClickException: If config is missing, has invalid permissions, or validation fails
    """
    config_path = Path.home() / ".config" / "logsqueak" / "config.yaml"

    try:
        config = Config.load(config_path)
        logger.info("config_loaded", path=str(config_path))
        return config
    except FileNotFoundError as e:
        logger.error("config_not_found", path=str(config_path))
        raise click.ClickException(str(e))
    except PermissionError as e:
        logger.error("config_permission_error", path=str(config_path))
        raise click.ClickException(str(e))
    except Exception as e:
        logger.error("config_validation_error", error=str(e))
        raise click.ClickException(f"Configuration validation failed:\n{e}")


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

    # Load configuration
    config = load_config()

    # Load journal entries
    try:
        graph_path = Path(config.logseq.graph_path).expanduser()
        journals = load_journal_entries(graph_path, dates)
        logger.info("extract_ready", journal_count=len(journals))
    except (ValueError, FileNotFoundError) as e:
        logger.error("journal_load_error", error=str(e))
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()

    click.echo(f"Loaded {len(journals)} journal(s) successfully")

    # T107: Initialize services
    from logsqueak.services.llm_client import LLMClient
    from logsqueak.services.page_indexer import PageIndexer
    from logsqueak.services.rag_search import RAGSearch
    from logsqueak.services.file_monitor import FileMonitor
    from logsqueak.tui.app import LogsqueakApp

    logger.info("initializing_services")

    # Create LLM client (pass the LLMConfig object)
    llm_client = LLMClient(config=config.llm)
    logger.info("llm_client_initialized", endpoint=str(config.llm.endpoint), model=config.llm.model)

    # Create GraphPaths and ChromaDB path
    graph_paths = GraphPaths(graph_path)
    chroma_db_path = Path.home() / ".cache" / "logsqueak" / "chromadb"
    chroma_db_path.mkdir(parents=True, exist_ok=True)

    # Create page indexer (will build index during Phase 1 background task)
    page_indexer = PageIndexer(graph_paths=graph_paths, db_path=chroma_db_path)
    logger.info("page_indexer_initialized", graph_path=str(graph_path), db_path=str(chroma_db_path))

    # Create RAG search service (uses same db_path as page_indexer)
    rag_search = RAGSearch(db_path=chroma_db_path)
    logger.info("rag_search_initialized", db_path=str(chroma_db_path))

    # Create file monitor
    file_monitor = FileMonitor()
    logger.info("file_monitor_initialized")

    # For single-journal case, use the only journal
    # For multi-journal case, use the first one (or could show picker in future)
    journal_date = list(journals.keys())[0]
    journal_outline = journals[journal_date]

    # Record initial journal file
    journal_path = graph_paths.get_journal_path(journal_date)
    file_monitor.record(journal_path)
    logger.info("journal_recorded_in_file_monitor", date=journal_date, path=str(journal_path))

    # Initialize and launch TUI app
    logger.info("launching_tui", journal_date=journal_date)
    app = LogsqueakApp(
        journal_outline=journal_outline,
        journal_date=journal_date,
        config=config,
        llm_client=llm_client,
        page_indexer=page_indexer,
        rag_search=rag_search,
        file_monitor=file_monitor,
    )

    # Run the TUI app
    app.run()

    logger.info("extract_command_completed")


def main():
    """Main entry point for setuptools console script."""
    cli()


if __name__ == "__main__":
    main()
