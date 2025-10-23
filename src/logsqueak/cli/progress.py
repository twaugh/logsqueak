"""Progress feedback display for CLI operations.

Provides clear status messages during extraction workflow (SC-001).
"""

import click


def show_building_index(page_count: int) -> None:
    """Show index building progress.

    Args:
        page_count: Number of pages being indexed
    """
    click.echo(f"Building page index... (found {page_count} pages)")


def show_index_built(page_count: int, duration: float, cached_count: int, embedded_count: int) -> None:
    """Show index build completion.

    Args:
        page_count: Total pages indexed
        duration: Time taken in seconds
        cached_count: Pages loaded from cache
        embedded_count: Pages newly embedded
    """
    click.echo(f"✓ Indexed {page_count} pages in {duration:.1f}s ({cached_count} cached, {embedded_count} embedded)")


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


def show_extraction_complete(count: int) -> None:
    """Show extraction completion.

    Args:
        count: Number of knowledge blocks found
    """
    click.echo(f"✓ Found {count} knowledge blocks")


def show_matching_progress(
    current: int, total: int, page_name: str, similarity: float, content: str = None
) -> None:
    """Show page matching progress.

    Args:
        current: Current block number (1-indexed)
        total: Total blocks to match
        page_name: Matched page name
        similarity: Similarity score
        content: Optional knowledge content to display
    """
    if content:
        # Show truncated content (first 50 chars)
        truncated = content[:50] + "..." if len(content) > 50 else content
        click.echo(
            f"  {current}/{total}: \"{truncated}\" → \"{page_name}\" (similarity: {similarity:.2f})"
        )
    else:
        click.echo(
            f"  {current}/{total}: Finding candidates... ✓ Matched to \"{page_name}\" (similarity: {similarity:.2f})"
        )


def show_activity_skipped(current: int, total: int) -> None:
    """Show activity log being skipped.

    Args:
        current: Current block number (1-indexed)
        total: Total blocks
    """
    click.echo(f"  {current}/{total}: Activity log detected, skipping")


def show_duplicate_skipped(current: int, total: int, page_name: str, content: str = None) -> None:
    """Show duplicate being skipped.

    Args:
        current: Current block number (1-indexed)
        total: Total blocks
        page_name: Page where duplicate was found
        content: Optional knowledge content to display
    """
    if content:
        truncated = content[:50] + "..." if len(content) > 50 else content
        click.echo(f"  {current}/{total}: \"{truncated}\" → Duplicate on \"{page_name}\", skipping")
    else:
        click.echo(f"  {current}/{total}: Duplicate found on \"{page_name}\", skipping")


def show_preview_header() -> None:
    """Show preview section header."""
    click.echo("\nPreview of proposed changes:")
    click.echo()


def show_apply_prompt() -> str:
    """Show apply confirmation prompt.

    Returns:
        User response (y/n/e)
    """
    return click.prompt("\nApply these changes? [y/N/e]", type=str, default="n").lower()


def show_applying() -> None:
    """Show applying changes message."""
    click.echo("\nApplying changes...")


def show_page_updated(page_name: str, addition_count: int) -> None:
    """Show page update completion.

    Args:
        page_name: Page that was updated
        addition_count: Number of additions made
    """
    click.echo(f"✓ Updated: pages/{page_name}.md ({addition_count} additions)")


def show_complete(total: int, integrated: int, skipped: int) -> None:
    """Show final completion message.

    Args:
        total: Total blocks processed
        integrated: Blocks successfully integrated
        skipped: Blocks skipped
    """
    click.echo(f"\nComplete! Processed {total} blocks, integrated {integrated}, skipped {skipped}.")


def show_dry_run_complete() -> None:
    """Show dry-run completion message."""
    click.echo("\nDry-run complete. No files were modified.")


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
