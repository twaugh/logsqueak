"""Writer module for adding knowledge to pages with provenance.

This module handles:
- Updating existing block content
- Appending new blocks as children
- Appending new blocks to page root
- Generating unique IDs for new blocks
"""

import uuid

from logsqueak.logseq.parser import LogseqBlock, LogseqOutline


def _generate_block_id() -> str:
    """Generate a unique UUID for a new block.

    Returns:
        UUID string in standard format (e.g., '65f3a8e0-1234-5678-9abc-def012345678')
    """
    return str(uuid.uuid4())


def update_block(target_block: LogseqBlock, new_content: str, preserve_id: bool = True, indent_str: str = "  ") -> None:
    """Update an existing block's content in place.

    Replaces the block's content while optionally preserving its id:: property.
    This allows updating knowledge blocks without losing their identity, which is
    useful for refreshing outdated information while maintaining links and references.

    Args:
        target_block: Block to update
        new_content: New content to replace with
        preserve_id: If True, preserve existing id:: property (default: True)
        indent_str: Indentation string from outline (e.g., "  ", "\t") (default: "  ")
    """
    # Update content (replace first line only)
    target_block.content[0] = new_content

    # If preserving ID and block has one, keep it
    if preserve_id and target_block.block_id:
        # Ensure id property is set using set_property
        target_block.set_property("id", target_block.block_id)


def append_to_block(target_block: LogseqBlock, new_content: str, indent_str: str = "  ") -> str:
    """Append new content as a child of the target block.

    Creates a new block with a unique UUID and adds it as a child of the specified
    parent block. This is useful for adding related information under an existing
    topic or section without modifying the parent's content.

    Args:
        target_block: Block to append child to
        new_content: Content for the new child block
        indent_str: Indentation string from outline (e.g., "  ", "\t") (default: "  ")

    Returns:
        The generated UUID for the new block
    """
    # Generate UUID for new block
    new_id = _generate_block_id()

    indent_level = target_block.indent_level + 1

    # Create new child block
    new_child = LogseqBlock(
        content=[new_content, f"id:: {new_id}"],
        indent_level=indent_level,
        block_id=new_id,
    )

    # Add as child
    target_block.children.append(new_child)

    return new_id


def append_to_root(outline: LogseqOutline, new_content: str, indent_str: str = "  ") -> str:
    """Append new content to the page root level.

    Creates a new block with a unique UUID and adds it at the root level of the page.
    This is useful when adding standalone information that doesn't belong under any
    existing section, or when the target section doesn't exist.

    Args:
        outline: Page outline
        new_content: Content for the new root-level block
        indent_str: Indentation string from outline (e.g., "  ", "\t") (default: "  ")

    Returns:
        The generated UUID for the new block
    """
    # Generate UUID for new block
    new_id = _generate_block_id()

    # Create new root-level block
    new_block = LogseqBlock(
        content=[new_content, f"id:: {new_id}"],
        indent_level=0,
        block_id=new_id,
    )

    # Add to root blocks
    outline.blocks.append(new_block)

    return new_id
