"""Reusable progress feedback for graph indexing operations.

Provides consistent progress reporting across CLI commands and wizard flows.
"""

from typing import Callable, Optional
from rich.console import Console
from rich.status import Status


def create_index_progress_callback(
    console: Console,
    reindex: bool = False,
    has_data: bool = False,
    use_echo: bool = False
) -> tuple[Callable[[int, int], None], Callable[[], None]]:
    """Create progress callback and cleanup function for index building.

    This provides consistent progress feedback across the application:
    - Initial message (Building/Updating/Rebuilding)
    - Spinner during model loading
    - Progress percentage during encoding
    - Silent during parsing phase

    Args:
        console: Rich Console instance for output
        reindex: True if forcing full rebuild (shows "Rebuilding")
        has_data: True if index already exists (shows "Updating" vs "Building")
        use_echo: If True, use console.print instead of rprint (for Click compatibility)

    Returns:
        Tuple of (progress_callback, cleanup_function)
        - progress_callback(current, total): Called by PageIndexer.build_index()
        - cleanup_function(): Call this to stop any active spinner

    Example:
        >>> from rich.console import Console
        >>> console = Console()
        >>> progress_cb, cleanup = create_index_progress_callback(console, has_data=False)
        >>> try:
        ...     await page_indexer.build_index(progress_callback=progress_cb)
        ... finally:
        ...     cleanup()
    """
    # Shared state (closure variables)
    total_pages = 0
    status_context: Optional[Status] = None
    page_count: Optional[int] = None
    showed_initial_message = False

    def progress_callback(current: int, total: int):
        nonlocal total_pages, status_context, page_count, showed_initial_message

        # Negative current signals model loading phase
        if current < 0:
            # Show initial message only when actual work starts
            if not showed_initial_message:
                if reindex:
                    message = "Rebuilding search index..."
                elif not has_data:
                    message = "Building search index..."
                else:
                    message = "Updating search index..."

                if use_echo:
                    console.print(message)
                else:
                    from rich import print as rprint
                    rprint(f"[bold cyan]{message}[/bold cyan]")

                showed_initial_message = True

            # Capture page count for completion message and phase detection
            total_pages = total
            page_count = total
            # Start spinner for model loading
            if status_context:
                status_context.__exit__(None, None, None)
            status_context = console.status("[bold green]Loading embedding model...")
            status_context.__enter__()
            return

        # Detect encoding phase: total has changed from page count to chunk count
        if page_count is not None and total != page_count:
            # Encoding phase - show spinner with progress percentage
            percent = int((current / total) * 100) if total > 0 else 0

            # Start encoding spinner if not already running
            if not status_context:
                status_context = console.status(f"[bold green]Building page index: {percent}%")
                status_context.__enter__()
            else:
                # Update spinner message with current percentage
                status_context.update(f"[bold green]Building page index: {percent}%")
        else:
            # Parsing phase - stop spinner but don't show progress
            if status_context:
                status_context.__exit__(None, None, None)
                status_context = None

    def cleanup():
        """Stop any active spinner (call this in finally block)."""
        nonlocal status_context
        if status_context:
            status_context.__exit__(None, None, None)
            status_context = None

    return progress_callback, cleanup
