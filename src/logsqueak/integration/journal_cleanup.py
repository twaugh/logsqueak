"""Phase 4.5 journal cleanup - add processed:: markers to source journal.

This module handles:
- Finding source blocks in journal that were processed
- Formatting links to integrated knowledge blocks
- Adding processed:: properties to source blocks with links back to targets
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

from logsqueak.logseq.parser import LogseqBlock, LogseqOutline

logger = logging.getLogger(__name__)


def add_processed_markers(
    journal_path: Path,
    processed_blocks_map: Dict[str, List[Tuple[str, str]]],
) -> None:
    """Add processed:: markers to journal blocks that were integrated.

    For each block in the journal that was processed, adds a child block
    with a processed:: property containing links to where the knowledge
    was integrated.

    Args:
        journal_path: Path to journal file
        processed_blocks_map: Dict[original_id -> List[(page_name, new_block_id)]]

    Raises:
        IOError: If journal file cannot be read or written
        ValueError: If source block not found
    """
    logger.info(f"Adding processed markers to journal: {journal_path.name}")

    # Load and parse journal
    journal_content = journal_path.read_text(encoding="utf-8")
    outline = LogseqOutline.parse(journal_content)

    # Process each block that was integrated
    for original_id, page_id_pairs in processed_blocks_map.items():
        logger.debug(f"Adding processed marker for block: {original_id[:8]}...")

        # Find source block in journal
        source_block = outline.find_block_by_id(original_id)
        if not source_block:
            logger.warning(f"Source block not found in journal: {original_id}")
            continue

        # Format links for all integrations of this block
        links = [_format_block_link(page_name, block_id) for page_name, block_id in page_id_pairs]
        processed_value = ", ".join(links)

        # Create processed marker as child block
        _add_processed_marker(source_block, processed_value, outline.indent_str)

        logger.debug(f"Added processed marker with {len(links)} link(s)")

    # Write back to journal
    rendered = outline.render()
    journal_path.write_text(rendered, encoding="utf-8")

    logger.info(f"Added processed markers to {len(processed_blocks_map)} blocks")


def _format_block_link(page_name: str, block_id: str) -> str:
    """Format a link to an integrated knowledge block.

    Creates a markdown link with block reference syntax: [page](((uuid)))

    Args:
        page_name: Target page name (may include .md extension and ___ for namespaces)
        block_id: UUID of the target block

    Returns:
        Formatted link string

    Examples:
        >>> _format_block_link("Project X.md", "abc-123")
        '[Project X](((abc-123)))'

        >>> _format_block_link("Work___Projects.md", "def-456")
        '[Work/Projects](((def-456)))'
    """
    # Remove .md extension if present
    display_name = page_name.replace(".md", "")

    # Replace namespace separator ___ with /
    display_name = display_name.replace("___", "/")

    # Format as markdown link with block ref
    return f"[{display_name}]((({block_id})))"


def _add_processed_marker(
    source_block: LogseqBlock,
    processed_value: str,
    indent_str: str = "  ",
) -> None:
    """Add processed:: property to existing source block.

    Adds a processed:: property containing links to where the knowledge
    was integrated. Does NOT create a new child block or modify id::.

    Args:
        source_block: Block to add property to
        processed_value: Formatted links (e.g., "[Page A](((uuid1))), [Page B](((uuid2)))")
        indent_str: Indentation string from outline (default: "  ")
    """
    # Add processed:: to the block's properties dict
    source_block.properties["processed"] = processed_value

    # Add processed:: as a continuation line (property format)
    # Properties appear as continuation lines indented relative to block content
    property_indent = indent_str * source_block.indent_level + indent_str
    property_line = f"{property_indent}processed:: {processed_value}"

    # Add to continuation_lines (after any existing properties like id::)
    source_block.continuation_lines.append(property_line)
