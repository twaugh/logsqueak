"""Phase 4 execution logic with atomic journal updates.

This module handles:
- Grouping write operations by page to minimize file I/O
- Loading pages as AST, applying operations, and writing back
- Generating UUIDs during write (not before)
- Atomically updating journal blocks after each page write
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Tuple

from logsqueak.integration.journal_cleanup import _add_processed_marker, _format_block_link
from logsqueak.integration.writer import append_to_block, append_to_root, update_block
from logsqueak.logseq.parser import LogseqOutline
from logsqueak.models.knowledge import ActionType

if TYPE_CHECKING:
    from logsqueak.extraction.extractor import WriteOperation

logger = logging.getLogger(__name__)


def execute_write_list(
    write_list: List["WriteOperation"],
    processed_blocks_map: Dict[str, List[Tuple[str, None]]],
    graph_path: Path,
    journal_path: Path,
) -> Dict[str, List[Tuple[str, str]]]:
    """Execute all write operations with atomic journal updates.

    Groups operations by page to minimize file I/O. For each page:
    1. Load and parse page to AST
    2. Apply all operations for that page
    3. Write page back to disk
    4. Immediately update journal blocks with processed:: markers (atomic)

    This ensures consistency: if a page write fails, the journal won't be
    marked as processed.

    Args:
        write_list: List of WriteOperation from Phase 3
        processed_blocks_map: Dict[original_id -> List[(page_name, None)]]
        graph_path: Path to Logseq graph
        journal_path: Path to journal file being processed

    Returns:
        Updated processed_blocks_map with new block IDs:
        Dict[original_id -> List[(page_name, new_block_id)]]

    Raises:
        IOError: If page or journal loading/writing fails
        ValueError: If target block not found
    """
    # Group operations by page_name
    ops_by_page = _group_operations_by_page(write_list)

    logger.info(f"Executing {len(write_list)} write operations across {len(ops_by_page)} pages")

    # Track new block IDs to update processed_blocks_map
    original_to_new_ids: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    # Load journal once at the start (will be updated after each page write)
    journal_outline = LogseqOutline.parse(journal_path.read_text(encoding="utf-8"))

    # Process each page
    for page_name, operations in ops_by_page.items():
        logger.debug(f"Processing page: {page_name} ({len(operations)} operations)")

        # Load page
        page_path = _resolve_page_path(graph_path, page_name)
        outline = LogseqOutline.parse(page_path.read_text(encoding="utf-8"))

        # Apply all operations to this page and track page_level_operations
        page_level_updates: Dict[str, List[str]] = defaultdict(list)
        for op in operations:
            logger.debug(f"Applying {op.action.value} operation to {page_name}")

            new_block_id = _apply_operation(outline, op, page_name)

            # Track new block ID for this original journal block
            original_to_new_ids[op.original_id].append((page_name, new_block_id))
            page_level_updates[op.original_id].append(new_block_id)

        # Write page back to disk
        rendered = outline.render()
        page_path.write_text(rendered, encoding="utf-8")
        logger.info(f"Wrote {len(operations)} operations to {page_name}")

        # Immediately update journal with processed:: markers for this page (atomic)
        for original_id, block_ids in page_level_updates.items():
            source_block = journal_outline.find_block_by_id(original_id)
            if not source_block:
                logger.warning(f"Source block not found in journal: {original_id}")
                continue

            # Format links for this page's integrations
            links = [_format_block_link(page_name, block_id) for block_id in block_ids]
            processed_value = ", ".join(links)

            # Add processed marker to journal block
            _add_processed_marker(source_block, processed_value, journal_outline.indent_str)
            logger.debug(f"Updated journal block {original_id[:8]}... with {len(links)} link(s)")

        # Write updated journal back to disk after each page (atomic)
        journal_rendered = journal_outline.render()
        journal_path.write_text(journal_rendered, encoding="utf-8")
        logger.info(f"Updated journal with processed markers for {page_name}")

    # Update processed_blocks_map with new block IDs
    updated_map = {}
    for original_id, page_id_pairs in processed_blocks_map.items():
        if original_id in original_to_new_ids:
            # Replace None with actual new block IDs
            updated_map[original_id] = original_to_new_ids[original_id]
        else:
            # Keep as-is (shouldn't happen but handle gracefully)
            updated_map[original_id] = page_id_pairs

    logger.info(f"Phase 4 complete: {len(write_list)} operations executed")
    return updated_map


def _group_operations_by_page(
    write_list: List["WriteOperation"],
) -> Dict[str, List["WriteOperation"]]:
    """Group write operations by page_name.

    Args:
        write_list: List of WriteOperation

    Returns:
        Dict mapping page_name to list of operations for that page
    """
    ops_by_page: Dict[str, List["WriteOperation"]] = defaultdict(list)

    for op in write_list:
        ops_by_page[op.page_name].append(op)

    return dict(ops_by_page)


def _resolve_page_path(graph_path: Path, page_name: str) -> Path:
    """Resolve page name to file path.

    Handles namespace pages (/ becomes ___) and .md extension.

    Args:
        graph_path: Path to Logseq graph
        page_name: Page name (e.g., "Project X" or "Work/Projects")

    Returns:
        Path to page markdown file

    Examples:
        >>> _resolve_page_path(Path("/graph"), "Project X")
        Path("/graph/pages/Project X.md")

        >>> _resolve_page_path(Path("/graph"), "Work/Projects")
        Path("/graph/pages/Work___Projects.md")
    """
    # Convert namespace separator / to ___
    filename = page_name.replace("/", "___")

    # Add .md extension
    filename = f"{filename}.md"

    return graph_path / "pages" / filename


def _apply_operation(outline: LogseqOutline, op: "WriteOperation", page_name: str) -> str:
    """Apply a single write operation to the outline AST.

    Generates a new UUID and applies the operation (UPDATE, APPEND_CHILD, APPEND_ROOT).

    Args:
        outline: Parsed LogseqOutline to modify
        op: WriteOperation to apply
        page_name: Name of the page (needed for hybrid ID lookup)

    Returns:
        The new block ID generated for this operation

    Raises:
        ValueError: If target block not found (for UPDATE/APPEND_CHILD)
    """
    if op.action == ActionType.UPDATE:
        # Find target block and update its content
        # target_id is already a clean hybrid_id (no prefix stripping needed)
        target_block = outline.find_block_by_id(op.target_id, page_name=page_name)
        if not target_block:
            raise ValueError(f"Target block not found: {op.target_id}")

        # Update content (preserve existing ID)
        update_block(target_block, op.new_content, preserve_id=True, indent_str=outline.indent_str)

        # Return existing block ID (not a new one)
        return target_block.block_id or target_block.get_hybrid_id(
            parents=[], indent_str=outline.indent_str
        )

    elif op.action == ActionType.APPEND_CHILD:
        # Find target block and append as child
        # target_id is already a clean hybrid_id (no prefix stripping needed)
        target_block = outline.find_block_by_id(op.target_id, page_name=page_name)
        if not target_block:
            raise ValueError(f"Target block not found: {op.target_id}")

        # Append child (generates new UUID)
        new_block_id = append_to_block(target_block, op.new_content, indent_str=outline.indent_str)
        return new_block_id

    elif op.action == ActionType.APPEND_ROOT:
        # Append to page root
        new_block_id = append_to_root(outline, op.new_content, indent_str=outline.indent_str)
        return new_block_id

    else:
        raise ValueError(f"Unsupported action type: {op.action}")
