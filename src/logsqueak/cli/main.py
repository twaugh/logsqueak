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

from logsqueak.cli import interactive, progress
from logsqueak.config.loader import load_config
from logsqueak.extraction.classifier import classify_extractions
from logsqueak.extraction.extractor import Extractor, create_knowledge_block
from logsqueak.integration.integrator import Integrator
from logsqueak.llm.prompt_logger import PromptLogger
from logsqueak.llm.providers.openai_compat import OpenAICompatibleProvider
from logsqueak.logseq.graph import GraphPaths
from logsqueak.models.journal import JournalEntry
from logsqueak.models.knowledge import ActionType
from logsqueak.models.page import PageIndex, TargetPage
from logsqueak.models.preview import ActionStatus, ExtractionPreview, ProposedAction


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


def build_page_index(graph_path: Path, ctx: click.Context) -> PageIndex:
    """Build PageIndex with progress feedback.

    Args:
        graph_path: Path to Logseq graph
        ctx: Click context for error handling

    Returns:
        Built PageIndex
    """
    try:
        graph_paths = GraphPaths(graph_path)
        page_files = list(graph_paths.pages_dir.glob("*.md"))

        progress.show_building_index(len(page_files))
        start_time = time.time()

        page_index = PageIndex.build(graph_path)

        duration = time.time() - start_time
        # Show cache statistics
        cached_count = getattr(page_index, "cached_count", 0)
        computed_count = getattr(page_index, "computed_count", len(page_files))
        progress.show_index_built(len(page_files), duration, cached_count, computed_count)

        return page_index
    except Exception as e:
        progress.show_error(f"Failed to build page index: {e}")
        ctx.exit(1)


def match_knowledge_to_pages(
    knowledge_extractions: list,
    activity_logs: list,
    extractor: Extractor,
    page_index: PageIndex,
    graph_path: Path,
    journal_date: date,
    indent_str: str = "  ",
    token_budget: Optional[int] = None,
) -> tuple[list[ProposedAction], list[str]]:
    """Match extracted knowledge to target pages using RAG.

    Args:
        knowledge_extractions: Extracted knowledge blocks
        activity_logs: Activity logs (for progress display)
        extractor: Extractor instance
        page_index: PageIndex for RAG search
        graph_path: Path to Logseq graph
        journal_date: Journal date for provenance
        indent_str: Indentation string from journal (e.g., "  ", "\t")
        token_budget: Optional token budget for Stage 2 prompts

    Returns:
        Tuple of (proposed_actions, warnings)
    """
    proposed_actions = []
    warnings = []
    total_extractions = len(knowledge_extractions) + len(activity_logs)

    for i, extraction in enumerate(knowledge_extractions, 1):
        # Find target page using RAG (using journal's indentation style)
        selection = extractor.select_target_page(
            extraction.content,
            page_index,
            indent_str=indent_str,
            token_budget=token_budget,
        )

        # Get the top similarity score for progress display
        similar_pages = page_index.find_similar(extraction.content, top_k=1)
        similarity_score = similar_pages[0][1] if similar_pages else 0.0

        # Load target page
        target_page = TargetPage.load(graph_path, selection.target_page)

        if not target_page:
            # Missing page (FR-009, T031)
            warnings.append(f"Target page '{selection.target_page}' does not exist")
            proposed_actions.append(
                ProposedAction(
                    knowledge=create_knowledge_block(
                        extraction,
                        journal_date,
                        selection.target_page,
                        selection.target_section,
                        selection.suggested_action,
                    ),
                    status=ActionStatus.SKIPPED,
                    reason=f"Target page '{selection.target_page}' does not exist",
                    similarity_score=similarity_score,
                )
            )
            progress.show_matching_progress(
                i, len(knowledge_extractions), selection.target_page, similarity_score, extraction.content
            )
            continue

        # Check if target section exists (if specified and action is add_child)
        if (
            selection.target_section
            and selection.suggested_action == ActionType.ADD_CHILD
        ):
            # Try to find the first section in the path
            section = target_page.find_section(selection.target_section[0])
            if not section:
                # Section doesn't exist and we're not creating it
                section_path = " > ".join(selection.target_section)
                warnings.append(
                    f"Section '{section_path}' not found on page '{selection.target_page}'"
                )
                proposed_actions.append(
                    ProposedAction(
                        knowledge=create_knowledge_block(
                            extraction,
                            journal_date,
                            selection.target_page,
                            selection.target_section,
                            selection.suggested_action,
                        ),
                        status=ActionStatus.SKIPPED,
                        reason=f"Section '{section_path}' does not exist (consider using CREATE_SECTION)",
                        similarity_score=similarity_score,
                    )
                )
                progress.show_matching_progress(
                    i, len(knowledge_extractions), selection.target_page, similarity_score, extraction.content
                )
                continue

        # Check for duplicates (FR-017, T024)
        if extractor.is_duplicate(extraction.content, target_page):
            progress.show_duplicate_skipped(i, len(knowledge_extractions), selection.target_page, extraction.content)
            proposed_actions.append(
                ProposedAction(
                    knowledge=create_knowledge_block(
                        extraction,
                        journal_date,
                        selection.target_page,
                        selection.target_section,
                        selection.suggested_action,
                    ),
                    status=ActionStatus.SKIPPED,
                    reason="Duplicate content detected",
                    similarity_score=similarity_score,
                )
            )
            continue

        # Ready to integrate
        progress.show_matching_progress(
            i, len(knowledge_extractions), selection.target_page, similarity_score, extraction.content
        )

        proposed_actions.append(
            ProposedAction(
                knowledge=create_knowledge_block(
                    extraction,
                    journal_date,
                    selection.target_page,
                    selection.target_section,
                    selection.suggested_action,
                ),
                status=ActionStatus.READY,
                similarity_score=similarity_score,
            )
        )

    # Show activity logs that were skipped
    for i, activity in enumerate(activity_logs, 1):
        progress.show_activity_skipped(i + len(knowledge_extractions), total_extractions)

    return proposed_actions, warnings


def process_journal_date(
    journal_date: date,
    extractor: Extractor,
    page_index: PageIndex,
    graph_path: Path,
    model: str,
    verbose: bool,
    show_diffs: bool = False,
    dry_run: bool = True,
    token_budget: Optional[int] = None,
) -> None:
    """Process a single journal date.

    Args:
        journal_date: Date to process
        extractor: Extractor instance
        page_index: PageIndex for RAG search
        graph_path: Path to Logseq graph
        model: LLM model name (for progress display)
        verbose: Verbose output flag
        show_diffs: Show diffs in preview
        dry_run: If True, skip approval and file writes
        token_budget: Optional token budget for Stage 2 prompts
    """
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

        # Stage 1: Extract knowledge blocks
        progress.show_extracting_knowledge(model)
        extractions = extractor.extract_knowledge(journal)

        # Classify into knowledge vs activity logs
        knowledge_extractions, activity_logs = classify_extractions(extractions)

        progress.show_extraction_complete(len(knowledge_extractions))

        if not knowledge_extractions:
            click.echo(f"No knowledge blocks found in {journal_date.isoformat()}")
            return

        # Stage 2: Match each knowledge block to target pages
        proposed_actions, warnings = match_knowledge_to_pages(
            knowledge_extractions,
            activity_logs,
            extractor,
            page_index,
            graph_path,
            journal_date,
            indent_str=journal.outline.indent_str,
            token_budget=token_budget,
        )

        # Display preview (T027)
        preview = ExtractionPreview(
            journal_date=journal_date,
            knowledge_blocks=[action.knowledge for action in proposed_actions],
            proposed_actions=proposed_actions,
            warnings=warnings,
            graph_path=graph_path,
        )

        # Handle approval workflow
        if dry_run:
            # Dry-run mode: show full preview and exit
            progress.show_preview_header()
            click.echo(preview.display(show_diffs=show_diffs))
            progress.show_dry_run_complete()
        else:
            # Normal mode: show summary, then prompt for each action individually
            click.echo(f"\nFound {len(proposed_actions)} knowledge blocks in {journal_file.name}:")

            # Show warnings/skipped if any
            skipped = [a for a in proposed_actions if a.status == ActionStatus.SKIPPED]
            if skipped:
                click.echo(f"\n{len(skipped)} blocks will be skipped:")
                for action in skipped:
                    click.echo(f"  - {action.knowledge.content[:60]}... ({action.reason})")

            ready_actions = [a for a in proposed_actions if a.status == ActionStatus.READY]

            if not ready_actions:
                click.echo("\nNo actions to apply (all skipped or warnings).")
                return

            # Prompt for each action individually with its diff
            approved_actions = []

            for i, action in enumerate(ready_actions, 1):
                kb = action.knowledge

                # Show the block description
                click.echo(f'\n{i}. "{kb.content[:60]}{"..." if len(kb.content) > 60 else ""}"')
                click.echo(f"   Target: {kb.target_page}")
                if kb.target_section:
                    click.echo(f"   Section: {' > '.join(kb.target_section)}")
                click.echo(f"   Action: {kb.suggested_action.value}")
                if action.similarity_score:
                    click.echo(f"   Similarity: {action.similarity_score:.2f}")

                # Show the diff for this specific action
                click.echo("\n   Diff:")
                target_page = TargetPage.load(graph_path, kb.target_page)
                if target_page:
                    diff_text = action.show_diff(target_page)
                    if diff_text:
                        # Indent the diff for readability
                        for line in diff_text.split('\n'):
                            click.echo(f"   {line}")
                else:
                    click.echo(f"   (Target page '{kb.target_page}' does not exist)")

                # Prompt for this specific action
                click.echo()
                choice = interactive.prompt_for_approval()

                if choice == 'y':
                    approved_actions.append(action)
                elif choice == 'n':
                    interactive.show_warning(f"Skipped block {i}")
                elif choice == 'e':
                    interactive.show_warning("Edit mode not yet implemented - skipping this block")

            if not approved_actions:
                interactive.show_warning("No blocks approved - nothing to integrate")
                return

            # Integrate approved actions
            click.echo(f"\nIntegrating {len(approved_actions)} approved blocks...")

            integrator = Integrator(graph_path, page_index=page_index)
            result = integrator.integrate(approved_actions, dry_run=False)

            if result.success:
                interactive.show_success(
                    f"Successfully integrated {result.actions_applied} knowledge blocks into {len(result.modified_pages)} pages"
                )
                if verbose:
                    click.echo(f"Modified pages: {', '.join(result.modified_pages)}")
            else:
                interactive.show_error("Integration failed")
                for error in result.errors:
                    click.echo(f"  - {error}", err=True)

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
    extractor = Extractor(llm_client, model=config.llm.model)

    # Build PageIndex (with cache)
    page_index = build_page_index(config.logseq.graph_path, ctx)

    # Process each date
    for journal_date in dates:
        process_journal_date(
            journal_date,
            extractor,
            page_index,
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


def main():
    """Entry point for the CLI."""
    cli(obj={})


if __name__ == "__main__":
    main()
