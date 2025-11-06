"""File operations service for Phase 3 (Integration Review).

Implements atomic two-phase writes with provenance markers following
the file operations contract in specs/002-logsqueak-spec/contracts/file-operations.md.
"""

import uuid
import structlog
from pathlib import Path
from typing import Optional
from logseq_outline.parser import LogseqOutline, LogseqBlock
from logseq_outline.graph import GraphPaths
from logsqueak.models.integration_decision import IntegrationDecision
from logsqueak.services.file_monitor import FileMonitor

logger = structlog.get_logger()


def generate_integration_id(
    knowledge_block_id: str,
    target_page: str,
    action: str,
    target_block_id: Optional[str] = None
) -> str:
    """
    Generate deterministic UUID for integration block.

    Same inputs always produce the same UUID, enabling idempotent operations.

    Args:
        knowledge_block_id: ID of knowledge block being integrated
        target_page: Target page name
        action: Integration action type
        target_block_id: Target block ID (for add_under/replace actions)

    Returns:
        str: Deterministic UUID string
    """
    # Logsqueak namespace UUID (from utils/ids.py)
    namespace = uuid.UUID('32e497fc-abf0-4d71-8cff-e302eb3e2bb0')

    # Combine inputs to create reproducible identifier
    parts = [knowledge_block_id, target_page, action]
    if target_block_id:
        parts.append(target_block_id)

    # Generate UUID v5 (SHA-1 based, deterministic)
    name = ":".join(parts)
    return str(uuid.uuid5(namespace, name))


def write_add_section(
    page_outline: LogseqOutline,
    refined_text: str,
    knowledge_block_id: str,
    target_page: str
) -> str:
    """
    Add knowledge as new root-level block.

    Args:
        page_outline: Target page outline
        refined_text: Content to add
        knowledge_block_id: ID of knowledge block being integrated
        target_page: Target page name

    Returns:
        str: Deterministic UUID of newly created block (for provenance)
    """
    # Generate deterministic UUID
    new_block_id = generate_integration_id(
        knowledge_block_id=knowledge_block_id,
        target_page=target_page,
        action="add_section"
    )

    # Create new block with id:: property
    new_block = LogseqBlock(
        content=[
            refined_text,
            f"id:: {new_block_id}"
        ],
        indent_level=0,
        block_id=None,  # Will be set by parser when it reads the content
        children=[]
    )

    # Add to end of page
    page_outline.blocks.append(new_block)

    return new_block_id


def write_add_under(
    page_outline: LogseqOutline,
    target_block_id: str,
    refined_text: str,
    knowledge_block_id: str,
    target_page: str
) -> str:
    """
    Add knowledge as child of existing block.

    Args:
        page_outline: Target page outline
        target_block_id: ID of parent block
        refined_text: Content to add
        knowledge_block_id: ID of knowledge block being integrated
        target_page: Target page name

    Returns:
        str: Deterministic UUID of newly created block (for provenance)

    Raises:
        ValueError: If target block not found
    """
    # Generate deterministic UUID
    new_block_id = generate_integration_id(
        knowledge_block_id=knowledge_block_id,
        target_page=target_page,
        action="add_under",
        target_block_id=target_block_id
    )

    # Find target block
    target_block = page_outline.find_block_by_id(target_block_id)
    if target_block is None:
        raise ValueError(f"Target block not found: {target_block_id}")

    # Add as child (add_child handles indentation automatically)
    new_child = target_block.add_child(refined_text)

    # Add id:: property to new child
    new_child.set_property("id", new_block_id)

    return new_block_id


def write_replace(
    page_outline: LogseqOutline,
    target_block_id: str,
    refined_text: str,
    knowledge_block_id: str,
    target_page: str
) -> str:
    """
    Replace existing block content while preserving properties and children.

    Args:
        page_outline: Target page outline
        target_block_id: ID of block to replace
        refined_text: New content (may contain multiple lines)
        knowledge_block_id: ID of knowledge block being integrated
        target_page: Target page name

    Returns:
        str: UUID of block (existing or newly added)

    Raises:
        ValueError: If target block not found
    """
    # Find target block
    target_block = page_outline.find_block_by_id(target_block_id)
    if target_block is None:
        raise ValueError(f"Target block not found: {target_block_id}")

    # Separate properties from content lines
    property_lines = []
    for line in target_block.content:
        # Property pattern: key:: value (not starting with bullet)
        if "::" in line and not line.strip().startswith("-"):
            property_lines.append(line)

    # Replace all content lines, preserve all properties
    # refined_text might be multi-line, so split it
    new_content_lines = refined_text.split("\n") if "\n" in refined_text else [refined_text]
    target_block.content = new_content_lines + property_lines

    # Ensure block has id:: property (add deterministic UUID if missing)
    if target_block.block_id is None:
        new_block_id = generate_integration_id(
            knowledge_block_id=knowledge_block_id,
            target_page=target_page,
            action="replace",
            target_block_id=target_block_id
        )
        target_block.set_property("id", new_block_id)
        return new_block_id
    else:
        # If id:: already exists, it's preserved in property_lines above
        return target_block.block_id


def add_provenance(
    journal_block: LogseqBlock,
    target_page: str,
    target_block_id: str
) -> None:
    """
    Add processed:: property to journal block.

    Format: processed:: [Page Name](((uuid))), [Other Page](((uuid2)))

    Args:
        journal_block: Journal block that was processed
        target_page: Name of target page (may contain hierarchy: "Projects/Logsqueak")
        target_block_id: UUID of integrated block
    """
    # Convert hierarchical page names (Projects___Logsqueak -> Projects/Logsqueak)
    display_name = target_page.replace("___", "/")
    provenance_link = f"[{display_name}]((({target_block_id})))"

    # Get existing processed:: value or empty string
    existing = journal_block.get_property("processed") or ""

    # Append new link (comma-separated)
    if existing:
        new_value = f"{existing}, {provenance_link}"
    else:
        new_value = provenance_link

    # Update property (preserves order)
    journal_block.set_property("processed", new_value)


def validate_decision(
    decision: IntegrationDecision,
    page_outline: LogseqOutline
) -> None:
    """
    Validate that decision can still be applied.

    Args:
        decision: Integration decision to validate
        page_outline: Current page outline

    Raises:
        ValueError: If target block doesn't exist or structure invalid
    """
    if decision.action in ["add_under", "replace"]:
        if decision.target_block_id is None:
            raise ValueError(
                f"Action {decision.action} requires target_block_id, but it is None"
            )

        block = page_outline.find_block_by_id(decision.target_block_id)
        if block is None:
            raise ValueError(
                f"Target block not found: {decision.target_block_id}\n"
                f"Page: {decision.target_page}\n"
                f"Possible reasons:\n"
                f"- Block was deleted externally\n"
                f"- Block ID changed (id:: property modified)\n"
                f"- Page structure changed significantly"
            )


def apply_integration(
    decision: IntegrationDecision,
    page_outline: LogseqOutline
) -> str:
    """
    Apply integration decision to page outline.

    Args:
        decision: Integration decision to apply
        page_outline: Page outline to modify (mutated in-place)

    Returns:
        str: Deterministic UUID of newly created/updated block (for provenance)

    Raises:
        ValueError: If action invalid or target block not found
    """
    if decision.action == "add_section":
        return write_add_section(
            page_outline,
            decision.refined_text,
            knowledge_block_id=decision.knowledge_block_id,
            target_page=decision.target_page
        )

    elif decision.action == "add_under":
        return write_add_under(
            page_outline,
            decision.target_block_id,
            decision.refined_text,
            knowledge_block_id=decision.knowledge_block_id,
            target_page=decision.target_page
        )

    elif decision.action == "replace":
        return write_replace(
            page_outline,
            decision.target_block_id,
            decision.refined_text,
            knowledge_block_id=decision.knowledge_block_id,
            target_page=decision.target_page
        )

    else:
        raise ValueError(f"Unknown action: {decision.action}")


async def write_integration_atomic(
    decision: IntegrationDecision,
    journal_date: str,
    graph_paths: GraphPaths,
    file_monitor: FileMonitor
) -> None:
    """
    Atomically write integration to page and update journal provenance.

    This function ensures both operations succeed or none.

    Args:
        decision: Integration decision to execute
        journal_date: Journal date for provenance link (YYYY-MM-DD)
        graph_paths: GraphPaths instance for path resolution
        file_monitor: FileMonitor for concurrent modification detection

    Raises:
        ValueError: If file modified and validation failed, or target block doesn't exist
        FileNotFoundError: If page file doesn't exist
        PermissionError: On file permission errors
        OSError: On file I/O errors
    """
    # Construct file paths using GraphPaths
    page_path = graph_paths.page_path(decision.target_page)
    journal_path = graph_paths.journal_path(journal_date)

    # Check if page file exists
    if not page_path.exists():
        raise FileNotFoundError(
            f"Page not found: {page_path}\n"
            f"Create the page in Logseq before integrating knowledge."
        )

    # Step 1: Check and reload page if modified
    if file_monitor.is_modified(page_path):
        logger.info("file_modified_reload", path=str(page_path))
        page_outline = LogseqOutline.parse(page_path.read_text())
        file_monitor.refresh(page_path)

        # Re-validate decision
        validate_decision(decision, page_outline)
    else:
        page_outline = LogseqOutline.parse(page_path.read_text())

    # Step 2: Check if block already exists (idempotency / retry detection)
    # Generate deterministic UUID to check for existing integration
    expected_block_id = generate_integration_id(
        knowledge_block_id=decision.knowledge_block_id,
        target_page=decision.target_page,
        action=decision.action,
        target_block_id=decision.target_block_id
    )

    existing_block = page_outline.find_block_by_id(expected_block_id)
    if existing_block is not None:
        logger.info(
            "block_already_exists",
            page=decision.target_page,
            block_id=expected_block_id,
            message="Skipping page write (idempotent retry)"
        )
        new_block_id = expected_block_id
        # Skip to journal update (Step 5)
    else:
        # Step 3: Apply integration to page outline
        new_block_id = apply_integration(decision, page_outline)

        # Step 4: Write page (FIRST WRITE)
        try:
            page_path.write_text(page_outline.render())
            file_monitor.refresh(page_path)
            logger.info(
                "page_write_success",
                page=decision.target_page,
                block_id=new_block_id
            )
        except Exception as e:
            logger.error(
                "page_write_failed",
                page=decision.target_page,
                error=str(e)
            )
            raise

    # Step 5: Check and reload journal if modified
    if file_monitor.is_modified(journal_path):
        logger.info("file_modified_reload", path=str(journal_path))
        journal_outline = LogseqOutline.parse(journal_path.read_text())
        file_monitor.refresh(journal_path)
    else:
        journal_outline = LogseqOutline.parse(journal_path.read_text())

    # Step 6: Add provenance to journal
    journal_block = journal_outline.find_block_by_id(decision.knowledge_block_id)
    if journal_block is None:
        raise ValueError(
            f"Knowledge block not found in journal: {decision.knowledge_block_id}"
        )

    add_provenance(journal_block, decision.target_page, new_block_id)

    # Step 7: Write journal (SECOND WRITE - only if page write succeeded)
    try:
        journal_path.write_text(journal_outline.render())
        file_monitor.refresh(journal_path)
        logger.info(
            "journal_provenance_added",
            journal_date=journal_date,
            knowledge_block_id=decision.knowledge_block_id,
            target_page=decision.target_page
        )
    except Exception as e:
        logger.error(
            "journal_write_failed",
            journal_date=journal_date,
            error=str(e)
        )
        raise
