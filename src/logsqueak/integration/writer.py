"""Writer module for adding knowledge to pages with provenance.

This module handles:
- Finding target sections in page outline (T036)
- Adding provenance links (T037)
- Adding child bullets at appropriate positions (T038)
- Fallback to page end when no section match (T039)
- Generating unique IDs for new blocks
"""

import uuid
from typing import List, Optional

from logsqueak.logseq.parser import LogseqBlock, LogseqOutline
from logsqueak.models.knowledge import ActionType, KnowledgeBlock
from logsqueak.models.page import TargetPage


def add_knowledge_to_page(
    target_page: TargetPage, knowledge: KnowledgeBlock
) -> None:
    """Add knowledge block to target page with provenance.

    This is the main entry point for adding knowledge. It:
    1. Finds the target section (or uses fallback)
    2. Adds provenance link to content
    3. Adds as child bullet at appropriate position

    Args:
        target_page: Page to add knowledge to
        knowledge: Knowledge block to add

    Raises:
        ValueError: If knowledge block is invalid
    """
    # Add provenance link to content
    content_with_provenance = _add_provenance_link(
        knowledge.content, knowledge.source_date
    )

    # Get indentation string from outline
    indent_str = target_page.outline.indent_str

    # Find target section in outline
    target_block = _find_target_section(
        target_page.outline, knowledge.target_section
    )

    if target_block is None:
        # Fallback: Add to page end
        _add_to_page_end(target_page.outline, content_with_provenance, indent_str)
    else:
        # Add as child bullet to target section
        _add_child_bullet(target_block, content_with_provenance, knowledge.suggested_action, indent_str)


def _find_target_section(
    outline: LogseqOutline, section_path: Optional[List[str]]
) -> Optional[LogseqBlock]:
    """Find target section in outline hierarchy (T036).

    Uses TargetPage.find_section() logic to locate the target block
    in the outline hierarchy.

    Args:
        outline: Page outline to search
        section_path: Path to target section (e.g., ["Tech Stack", "Database"])
                     None means page root

    Returns:
        Target LogseqBlock if found, None for page root or not found
    """
    if section_path is None or len(section_path) == 0:
        # No section specified - will add to page root
        return None

    # Search for section by matching content
    # This is a simplified version - full implementation would use
    # TargetPage.find_section() which handles heading matching
    return _find_section_recursive(outline.blocks, section_path, 0)


def _find_section_recursive(
    blocks: List[LogseqBlock], section_path: List[str], depth: int
) -> Optional[LogseqBlock]:
    """Recursively search for section in block tree.

    Args:
        blocks: List of blocks to search
        section_path: Section path to find
        depth: Current depth in section path

    Returns:
        Matching block or None
    """
    if depth >= len(section_path):
        return None

    target_text = section_path[depth]

    for block in blocks:
        # Check if this block matches the current path element
        # Match against block content (first line, strip whitespace)
        block_text = block.content[0].strip()

        # Handle heading syntax (e.g., "## Section Name")
        if block_text.startswith("#"):
            # Extract heading text after # markers
            heading_text = block_text.lstrip("#").strip()
            if heading_text == target_text or block_text == target_text:
                # Found match at this level
                if depth == len(section_path) - 1:
                    # This is the final target
                    return block
                else:
                    # Need to go deeper
                    return _find_section_recursive(
                        block.children, section_path, depth + 1
                    )
        elif target_text in block_text:
            # Simple text match
            if depth == len(section_path) - 1:
                return block
            else:
                return _find_section_recursive(
                    block.children, section_path, depth + 1
                )

    return None


def _generate_block_id() -> str:
    """Generate a unique UUID for a new block.

    Returns:
        UUID string in standard format (e.g., '65f3a8e0-1234-5678-9abc-def012345678')
    """
    return str(uuid.uuid4())


def _add_provenance_link(content: str, source_date) -> str:
    """Add provenance link to knowledge content (T037).

    Appends [[YYYY-MM-DD]] link to content to track source journal.

    Args:
        content: Original knowledge content
        source_date: Date of source journal

    Returns:
        Content with provenance link appended
    """
    # Format date as YYYY-MM-DD
    date_str = source_date.strftime("%Y-%m-%d")
    provenance = f"[[{date_str}]]"

    # Append provenance link
    return f"{content} {provenance}"


def _add_child_bullet(
    parent_block: LogseqBlock, content: str, action: ActionType, indent_str: str
) -> None:
    """Add content as child bullet to parent block (T038).

    Generates a unique UUID for the new block and adds it as an id:: property.

    Args:
        parent_block: Parent block to add child to
        content: Content to add (with provenance)
        action: Suggested action type (ADD_CHILD or CREATE_SECTION)
        indent_str: Indentation string from outline (e.g., "  ", "\t")
    """
    # Generate unique ID for new block
    block_id = _generate_block_id()

    indent_level = parent_block.indent_level + 1

    # Create new child block with id:: property
    new_child = LogseqBlock(
        content=[content, f"id:: {block_id}"],
        indent_level=indent_level,
        block_id=block_id,
    )

    # Add as last child (targeted placement)
    parent_block.children.append(new_child)


def _add_to_page_end(outline: LogseqOutline, content: str, indent_str: str) -> None:
    """Add content to page end as fallback (T039).

    When no clear section match exists, add knowledge at the end
    of the page at root level. Generates a unique UUID for the new block.

    Args:
        outline: Page outline
        content: Content to add (with provenance)
        indent_str: Indentation string from outline (e.g., "  ", "\t")
    """
    # Generate unique ID for new block
    block_id = _generate_block_id()

    # Create new root-level block with id:: property
    new_block = LogseqBlock(
        content=[content, f"id:: {block_id}"],
        indent_level=0,
        block_id=block_id,
    )

    # Add to end of root blocks
    outline.blocks.append(new_block)


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


def write_page_safely(target_page: TargetPage, output_path: Optional = None) -> None:
    """Write modified page to disk safely (T040).

    Uses LogseqOutline.render() to preserve property order and structure.

    Args:
        target_page: Page to write
        output_path: Optional custom output path (defaults to page's file_path)

    Raises:
        IOError: If write fails
    """
    output = output_path or target_page.file_path
    temp_path = output.with_suffix('.tmp')

    try:
        # Render outline preserving structure
        rendered_content = target_page.outline.render()

        # Write atomically (write to temp, then rename)
        temp_path.write_text(rendered_content, encoding='utf-8')
        temp_path.replace(output)
    except Exception as e:
        # Clean up temp file on error
        if temp_path.exists():
            temp_path.unlink()
        raise IOError(f"Failed to write page {output}: {e}") from e
