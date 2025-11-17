"""CLI entry point for Logsqueak."""

import click
from pathlib import Path
from datetime import date, timedelta
import sys
from rich.console import Console
from logsqueak.utils.logging import configure_logging, get_logger
from logsqueak.models.config import Config
from logseq_outline.parser import LogseqOutline
from logseq_outline.graph import GraphPaths
from logsqueak.services.llm_wrappers import _augment_outline_with_ids


logger = get_logger(__name__)
console = Console()


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

        # Augment outline with hybrid IDs (content hashes for blocks without explicit id:: properties)
        # This ensures all blocks have stable IDs for LLM classification and tracking
        augmented_outline = _augment_outline_with_ids(outline)

        # Store using ISO format (YYYY-MM-DD) for consistency with user input
        journals[journal_date.strftime("%Y-%m-%d")] = augmented_outline

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

    # Create GraphPaths
    graph_paths = GraphPaths(graph_path)

    # Create page indexer (will build index during Phase 1 background task)
    # PageIndexer creates its own per-graph ChromaDB directory under ~/.cache/logsqueak/chromadb
    page_indexer = PageIndexer(graph_paths=graph_paths)
    logger.info("page_indexer_initialized", graph_path=str(graph_path), db_path=str(page_indexer.db_path))

    # Create RAG search service (uses same per-graph db_path as page_indexer)
    rag_search = RAGSearch(db_path=page_indexer.db_path)
    logger.info("rag_search_initialized", db_path=str(page_indexer.db_path))

    # Create file monitor
    file_monitor = FileMonitor()
    logger.info("file_monitor_initialized")

    # Record all journal files in file monitor
    for journal_date in journals.keys():
        journal_path = graph_paths.get_journal_path(journal_date)
        file_monitor.record(journal_path)
        logger.info("journal_recorded_in_file_monitor", date=journal_date, path=str(journal_path))

    # Initialize and launch TUI app
    logger.info("launching_tui", journal_count=len(journals))
    app = LogsqueakApp(
        journals=journals,
        config=config,
        llm_client=llm_client,
        page_indexer=page_indexer,
        rag_search=rag_search,
        file_monitor=file_monitor,
    )

    # Run the TUI app
    app.run()

    logger.info("extract_command_completed")


@cli.command()
@click.argument("query")
@click.option("--reindex", is_flag=True, help="Force full rebuild of the search index (clears existing data)")
def search(query: str, reindex: bool):
    """
    Search your Logseq knowledge base using semantic search.

    Uses the same RAG search service as the TUI application.
    Automatically updates the index with modified pages (incremental indexing).

    Examples:
        logsqueak search "machine learning best practices"
        logsqueak search "how to debug async code" --reindex
    """
    import asyncio
    from logsqueak.services.page_indexer import PageIndexer
    from logsqueak.services.rag_search import RAGSearch
    from logsqueak.models.edited_content import EditedContent

    logger.info("search_command_started", query=query, reindex=reindex)

    # Load configuration
    config = load_config()
    graph_path = Path(config.logseq.graph_path).expanduser()

    # Initialize services
    logger.info("initializing_search_services")
    graph_paths = GraphPaths(graph_path)

    # Create page indexer
    page_indexer = PageIndexer(graph_paths=graph_paths)
    logger.info("page_indexer_initialized", db_path=str(page_indexer.db_path))

    # Create RAG search service
    rag_search = RAGSearch(db_path=page_indexer.db_path)
    logger.info("rag_search_initialized")

    # Handle --reindex flag (force full rebuild by clearing collection)
    if reindex:
        click.echo("Clearing existing index for full rebuild...")
        # Delete collection and recreate it (ChromaDB doesn't support delete all)
        page_indexer.chroma_client.delete_collection("logsqueak_blocks")
        new_collection = page_indexer.chroma_client.create_collection(
            "logsqueak_blocks",
            metadata={"hnsw:space": "cosine"}
        )
        # Update both page_indexer and rag_search to use new collection
        page_indexer.collection = new_collection
        rag_search.collection = new_collection
        logger.info("index_cleared_for_rebuild")

    # Always update index (incremental indexing is fast - only modified pages)
    has_data = rag_search.has_indexed_data()
    if reindex:
        click.echo("Rebuilding search index...")
    elif not has_data:
        click.echo("Building search index (first run)...")
    else:
        click.echo("Updating search index...")

    # Build/update index with progress indicator
    async def build_index_with_progress():
        total_pages = 0
        current_page = 0
        status_context = None

        def progress_callback(current, total):
            nonlocal total_pages, current_page, status_context
            total_pages = total
            current_page = current

            # Negative current signals model loading phase
            if current < 0:
                # Clear the progress line completely
                sys.stdout.write(f"\r{' ' * 50}\r")
                sys.stdout.flush()
                # Start spinner for model loading
                status_context = console.status("[bold green]Loading embedding model...")
                status_context.__enter__()
            # When current == total, we're in the embedding generation phase
            elif current == total and status_context is not None:
                # Stop model loading spinner
                status_context.__exit__(None, None, None)
                # Start embedding generation spinner
                status_context = console.status("[bold green]Generating embeddings...")
                status_context.__enter__()
            elif current < total:
                # Simple progress indicator for page indexing
                percent = int((current / total) * 100) if total > 0 else 0
                click.echo(f"\rIndexing pages: {current}/{total} ({percent}%)", nl=False)

        try:
            await page_indexer.build_index(progress_callback=progress_callback)
            # Stop spinner if it was started
            if status_context:
                status_context.__exit__(None, None, None)
            click.echo()  # Newline after progress
            if reindex:
                click.echo(f"Index rebuilt successfully ({total_pages} pages)")
            elif not has_data:
                click.echo(f"Index built successfully ({total_pages} pages)")
            else:
                click.echo(f"Index updated successfully ({total_pages} pages checked)")
        except Exception as e:
            # Stop spinner on error
            if status_context:
                status_context.__exit__(None, None, None)
            click.echo()  # Newline after progress
            logger.error("index_build_failed", error=str(e))
            raise click.ClickException(f"Failed to build search index: {e}")

    asyncio.run(build_index_with_progress())

    # Execute search using RAG service
    click.echo(f"\nSearching for: {query}\n")

    async def execute_search():
        try:
            # Create a temporary EditedContent object for the search query
            # (RAG service expects EditedContent for its interface)
            query_content = EditedContent(
                block_id="search_query",
                original_content=query,
                hierarchical_context="",  # Not used for search
                current_content=query
            )

            # Use RAG search service (same as TUI)
            top_k = config.rag.top_k
            chunks_dict = await rag_search.find_candidates(
                edited_content=[query_content],
                original_contexts={"search_query": query},  # Use query as context
                graph_paths=graph_paths,
                top_k=top_k
            )

            # Extract results
            chunks = chunks_dict.get("search_query", [])
            if not chunks:
                click.echo("No results found.")
                return

            # Format results for display
            results = []
            for idx, (page_name, block_id, hierarchical_context) in enumerate(chunks):
                # Calculate a pseudo-confidence based on rank (since RAG returns sorted results)
                # First result = 100%, linearly decreasing
                confidence = max(50, 100 - (idx * (50 // len(chunks))))

                results.append({
                    "page_name": page_name,
                    "block_id": block_id,
                    "confidence": confidence,
                    "snippet": hierarchical_context
                })

            # Display results
            _display_search_results(results, graph_path)

        except Exception as e:
            logger.error("search_execution_failed", error=str(e))
            raise click.ClickException(f"Search failed: {e}")

    asyncio.run(execute_search())
    logger.info("search_command_completed")


def _create_clickable_link(page_name: str, graph_path: Path) -> str:
    """Create a clickable logseq:// hyperlink using OSC 8 escape codes.

    Args:
        page_name: Name of the Logseq page
        graph_path: Path to the Logseq graph directory

    Returns:
        Formatted hyperlink string with OSC 8 escape codes
    """
    from logsqueak.utils.logseq_urls import create_logseq_url

    logseq_url = create_logseq_url(page_name, graph_path)
    # OSC 8 format: \033]8;;URI\033\\TEXT\033]8;;\033\\
    return f"\033]8;;{logseq_url}\033\\{page_name}\033]8;;\033\\"


def _display_search_results(results: list[dict], graph_path: Path):
    """
    Display search results in terminal-friendly format.

    Uses OSC 8 escape codes for clickable links and color coding for readability.
    """
    for idx, result in enumerate(results, 1):
        page_name = result["page_name"]
        confidence = result["confidence"]
        snippet = result["snippet"]

        # Create clickable logseq:// link using OSC 8 escape codes
        clickable_link = _create_clickable_link(page_name, graph_path)

        # Format snippet (exclude frontmatter properties, preserve original indentation)
        snippet_lines = snippet.split('\n')

        # Filter out frontmatter (property lines like "status:: value")
        # Keep only bullet lines, preserving their original indentation
        content_lines = []
        for line in snippet_lines:
            stripped = line.lstrip()
            # Skip empty lines and property lines (contain :: but don't start with -)
            if stripped and stripped.startswith('-'):
                # Convert tabs to 2 spaces and keep original line with indentation intact
                line_with_spaces = line.replace('\t', '  ')
                content_lines.append(line_with_spaces)

        # Show all content lines (full hierarchical context with original formatting)
        if content_lines:
            snippet_display = '\n   '.join(content_lines)  # Indent to align under "Relevance:"
        else:
            # Fallback if no bullet content (shouldn't happen)
            snippet_display = "(no content preview)"

        # Display result
        click.echo(f"{idx}. {clickable_link}")
        click.echo(f"   Relevance: {confidence}%")
        click.echo(f"   {snippet_display}")
        click.echo()  # Blank line between results


def main():
    """Main entry point for setuptools console script."""
    cli()


if __name__ == "__main__":
    main()
