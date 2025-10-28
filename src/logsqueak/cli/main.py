#!/usr/bin/env python3
"""Logsqueak CLI - Extract knowledge from Logseq journals.

This is the main entry point for the Logsqueak command-line tool.
"""

import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import click

from logsqueak.cli import progress
from logsqueak.config.loader import load_config
from logsqueak.extraction.extractor import Extractor
from logsqueak.llm.prompt_logger import PromptLogger
from logsqueak.llm.providers.openai_compat import OpenAICompatibleProvider
from logsqueak.logseq.graph import GraphPaths
from logsqueak.models.journal import JournalEntry


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


def build_vector_store(graph_path: Path, ctx: click.Context):
    """Build VectorStore with progress feedback.

    Uses persistent ChromaDB for block-level semantic search.

    Args:
        graph_path: Path to Logseq graph
        ctx: Click context for error handling

    Returns:
        Built VectorStore
    """
    from logsqueak.rag.indexer import IndexBuilder
    from logsqueak.rag.manifest import CacheManifest
    from logsqueak.rag.vector_store import ChromaDBStore

    try:
        progress.show_building_index(0)  # Don't know count yet
        start_time = time.time()

        # Build/load vector store (ChromaDB persists in ~/.cache/logsqueak/chroma)
        persist_dir = Path.home() / ".cache" / "logsqueak" / "chroma"
        vector_store = ChromaDBStore(persist_directory=persist_dir)
        manifest_path = persist_dir.parent / "manifest.json"
        manifest = CacheManifest(manifest_path)
        builder = IndexBuilder(vector_store, manifest)
        stats = builder.build_incremental(graph_path, force=False)

        duration = time.time() - start_time
        total_pages = stats['added'] + stats['updated'] + stats['deleted'] + stats['unchanged']
        newly_indexed = stats['added'] + stats['updated']
        click.echo(
            f"✓ Index built in {duration:.1f}s "
            f"(+{stats['added']} ~{stats['updated']} -{stats['deleted']} ={stats['unchanged']})"
        )

        return vector_store
    except Exception as e:
        progress.show_error(f"Failed to build page index: {e}")
        ctx.exit(1)


def process_journal_date(
    journal_date: date,
    extractor: Extractor,
    vector_store,  # Changed from page_index to vector_store
    graph_path: Path,
    model: str,
    verbose: bool,
    show_diffs: bool = False,
    dry_run: bool = True,
    token_budget: Optional[int] = None,
) -> None:
    """Process a single journal date using the new 5-phase pipeline.

    Args:
        journal_date: Date to process
        extractor: Extractor instance
        vector_store: VectorStore for semantic search
        graph_path: Path to Logseq graph
        model: LLM model name (for progress display)
        verbose: Verbose output flag
        show_diffs: Show diffs in preview (ignored - no preview in new pipeline yet)
        dry_run: If True, show what would happen but don't actually write
        token_budget: Optional token budget (ignored - new pipeline uses top_k)
    """
    if dry_run:
        click.echo("\nWarning: --dry-run mode not yet implemented for new pipeline.", err=True)
        click.echo("The new 5-phase pipeline will execute immediately without approval.", err=True)
        click.echo("Use Ctrl+C to cancel now if you don't want to proceed.\n", err=True)
        import time
        time.sleep(2)

    try:
        # Load journal entry
        progress.show_processing_journal(journal_date.isoformat())

        # Get journal file path from graph
        graph_paths = GraphPaths(graph_path)
        journal_file = graph_paths.get_journal_path(journal_date.strftime("%Y_%m_%d"))

        if not journal_file.exists():
            progress.show_warning(f"No journal entry found for {journal_date.isoformat()}")
            return

        journal = JournalEntry.load(journal_file)

        # Run the new 5-phase pipeline
        progress.show_extracting_knowledge(model)

        num_operations = extractor.extract_and_integrate(
            journal=journal,
            vector_store=vector_store,
            graph_path=graph_path,
            top_k=20,  # Default to top 20 candidates
        )

        if num_operations == 0:
            click.echo(f"No knowledge blocks found in {journal_date.isoformat()}")
        else:
            click.echo(f"\n✓ Successfully integrated {num_operations} knowledge blocks from {journal_date.isoformat()}")

    except Exception as e:
        progress.show_error(f"Failed to process {journal_date.isoformat()}: {e}")
        if verbose:
            import traceback

            traceback.print_exc()


@cli.command()
@click.argument("date_or_range", required=False, default="today")
@click.option(
    "--dry-run",
    is_flag=True,
    help="Dry-run mode: show preview without writing files",
)
@click.option(
    "--show-diffs",
    is_flag=True,
    help="Show diffs of proposed changes in preview",
)
@click.option(
    "--prompt-log-file",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to save prompt logs (default: ~/.cache/logsqueak/prompts/TIMESTAMP.log)",
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
    show_diffs: bool,
    prompt_log_file: Optional[Path],
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

    # Configure logging based on verbose flag
    if verbose:
        # Verbose mode: Show INFO and DEBUG from logsqueak modules
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(levelname)s [%(name)s] %(message)s",
            force=True,  # Override any existing config
        )
    else:
        # Normal mode: Only show WARNING and ERROR
        logging.basicConfig(
            level=logging.WARNING,
            format="%(levelname)s: %(message)s",
            force=True,
        )

    # Set up default prompt log file if not specified
    if prompt_log_file is None:
        from datetime import datetime
        cache_dir = Path.home() / ".cache" / "logsqueak" / "prompts"
        cache_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prompt_log_file = cache_dir / f"{timestamp}.log"

    if verbose:
        click.echo(f"Date/range: {date_or_range}")
        click.echo(f"Dry-run: {dry_run}")
        click.echo(f"Prompt log file: {prompt_log_file}")
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

    # Set up prompt logger (always enabled, logs to file)
    click.echo(f"Prompt logs will be saved to: {prompt_log_file}")
    prompt_logger = PromptLogger(
        log_file=prompt_log_file,
        include_timestamps=True,
        pretty_print=True,
    )

    # Initialize LLM client and extractor
    llm_client = OpenAICompatibleProvider(
        endpoint=config.llm.endpoint,
        api_key=config.llm.api_key,
        model=config.llm.model,
        prompt_logger=prompt_logger,
    )
    extractor = Extractor(llm_client)

    # Build VectorStore (persistent block-level index)
    vector_store = build_vector_store(config.logseq.graph_path, ctx)

    # Process each date
    for journal_date in dates:
        process_journal_date(
            journal_date,
            extractor,
            vector_store,
            config.logseq.graph_path,
            config.llm.model,
            verbose,
            show_diffs,
            dry_run,
            token_budget=config.rag.token_budget,
        )

    # Log summary if prompt inspection was enabled
    if prompt_logger:
        prompt_logger.log_summary()


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


@cli.group()
@click.pass_context
def index(ctx: click.Context):
    """Manage the page index and vector store.

    Commands for building, rebuilding, and inspecting the page index
    used for semantic search.
    """
    pass


@index.command()
@click.option(
    "--graph",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Override Logseq graph path (default: from config)",
)
@click.option(
    "--force",
    is_flag=True,
    help="Force full rebuild (ignore cache)",
)
@click.pass_context
def rebuild(ctx: click.Context, graph: Optional[Path], force: bool):
    """Rebuild the page index.

    Incrementally updates the index by detecting additions, updates, and
    deletions. Use --force to ignore cache and rebuild from scratch.

    Examples:
        logsqueak index rebuild              # Incremental update
        logsqueak index rebuild --force      # Full rebuild
    """
    verbose = ctx.obj["verbose"]

    # Configure logging
    if verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(levelname)s [%(name)s] %(message)s",
            force=True,
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s: %(message)s",
            force=True,
        )

    # Load configuration
    try:
        config_path = ctx.obj["config_path"]
        config = load_config(config_path)
        if verbose:
            click.echo(f"Loading configuration from {config_path or '~/.config/logsqueak/config.yaml'}")
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        ctx.exit(1)
    except Exception as e:
        click.echo(f"Error loading configuration: {e}", err=True)
        ctx.exit(1)

    # Apply graph override
    if graph:
        config.logseq.graph_path = graph

    # Build index using VectorStore
    try:
        from logsqueak.rag.indexer import IndexBuilder
        from logsqueak.rag.manifest import CacheManifest
        from logsqueak.rag.vector_store import ChromaDBStore

        graph_paths = GraphPaths(config.logseq.graph_path)
        page_files = list(graph_paths.pages_dir.glob("*.md"))

        if force:
            click.echo(f"Force rebuild: rebuilding index for {len(page_files)} pages...")
        else:
            click.echo(f"Building index for {len(page_files)} pages...")
        start_time = time.time()

        # Setup vector store and manifest
        vector_store_path = Path.home() / ".cache" / "logsqueak" / "chroma"
        manifest_path = vector_store_path.parent / "manifest.json"

        vector_store = ChromaDBStore(vector_store_path)
        manifest = CacheManifest(manifest_path)

        # Build index incrementally with force flag
        builder = IndexBuilder(vector_store, manifest)
        stats = builder.build_incremental(config.logseq.graph_path, force=force)

        duration = time.time() - start_time
        click.echo(
            f"✓ Index built in {duration:.1f}s "
            f"(+{stats['added']} ~{stats['updated']} -{stats['deleted']} ={stats['unchanged']})"
        )

        # Clean up
        vector_store.close()

    except Exception as e:
        click.echo(f"Error: Failed to build index: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        ctx.exit(1)


@index.command()
@click.option(
    "--graph",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Override Logseq graph path (default: from config)",
)
@click.pass_context
def status(ctx: click.Context, graph: Optional[Path]):
    """Show index status and cache statistics.

    Displays information about the current index state, including
    number of indexed pages and cache location.

    Examples:
        logsqueak index status
    """
    verbose = ctx.obj["verbose"]

    # Load configuration
    try:
        config_path = ctx.obj["config_path"]
        config = load_config(config_path)
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        ctx.exit(1)
    except Exception as e:
        click.echo(f"Error loading configuration: {e}", err=True)
        ctx.exit(1)

    # Apply graph override
    if graph:
        config.logseq.graph_path = graph

    try:
        graph_paths = GraphPaths(config.logseq.graph_path)

        # Check if pages directory exists
        if not graph_paths.pages_dir.exists():
            click.echo(f"Error: Pages directory not found: {graph_paths.pages_dir}", err=True)
            ctx.exit(1)

        # Count pages
        page_files = list(graph_paths.pages_dir.glob("*.md"))

        # Check cache locations
        default_cache = Path.home() / ".cache" / "logsqueak"
        chroma_cache = default_cache / "chroma"
        manifest_path = default_cache / "manifest.json"

        click.echo(f"Graph: {config.logseq.graph_path}")
        click.echo(f"Pages: {len(page_files)}")
        click.echo(f"\nCache locations:")
        click.echo(f"  VectorStore: {chroma_cache} {'(exists)' if chroma_cache.exists() else '(not found)'}")
        click.echo(f"  Manifest: {manifest_path} {'(exists)' if manifest_path.exists() else '(not found)'}")

        # Read manifest if it exists
        if manifest_path.exists():
            from logsqueak.rag.manifest import CacheManifest
            manifest = CacheManifest(manifest_path)
            indexed_pages = manifest.get_all_pages()
            click.echo(f"\nIndexed pages: {len(indexed_pages)}")

            if verbose and indexed_pages:
                click.echo("\nIndexed page names:")
                for page_name in sorted(indexed_pages[:10]):  # Show first 10
                    click.echo(f"  - {page_name}")
                if len(indexed_pages) > 10:
                    click.echo(f"  ... and {len(indexed_pages) - 10} more")
        else:
            click.echo("\nNo index found. Run 'logsqueak index rebuild' to create one.")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        ctx.exit(1)


def main():
    """Entry point for the CLI."""
    cli(obj={})


if __name__ == "__main__":
    main()
