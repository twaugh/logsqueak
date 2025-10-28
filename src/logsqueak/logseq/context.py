"""Full-context generation and hashing for hybrid block IDs.

This module provides utilities for generating stable block IDs based on
full-context hashing. This is used for the hybrid-ID system where blocks
without explicit id:: properties get IDs based on their content and position
in the document hierarchy.
"""

import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from logsqueak.logseq.parser import LogseqBlock, LogseqOutline


def generate_full_context(block: "LogseqBlock", parents: list["LogseqBlock"], indent_str: str = "  ") -> str:
    """Generate full-context string for a block.

    The full context includes all parent blocks' full content with proper bullet
    formatting, plus the block's own full content. This must be reproducible from
    just the block and its parents (no siblings, no children) since we need to
    recreate the hash when finding blocks by ID.

    NOTE: Blocks with truly identical content and context will have the same hybrid ID.
    This is by design - such blocks are semantically equivalent for RAG purposes.
    Use explicit id:: properties if you need to distinguish identical blocks.

    Args:
        block: The block to generate context for
        parents: List of parent blocks from root to immediate parent
        indent_str: Indentation string (default: "  " for 2 spaces)

    Returns:
        Full context string with bullet markers and indentation

    Examples:
        >>> # Block with no parents
        >>> generate_full_context(block, [])
        "- My content"

        >>> # Block with parents
        >>> generate_full_context(grandchild, [root, child])
        "- Root content\\n  - Child content\\n    - Grandchild content"
    """
    context_parts = []

    # Add all parent blocks with proper indentation
    # Use full content from parents (but filter out property lines)
    # Properties like "id::" are metadata, not semantic content
    for i, parent in enumerate(parents):
        indent = indent_str * i
        # Filter out property lines (lines matching "key:: value" pattern)
        parent_lines = []
        for j, line in enumerate(parent.content):
            # Skip property lines (they start with word characters followed by ::)
            if "::" in line and j > 0:  # Properties are never on first line
                continue
            if j == 0:
                # First line gets the bullet
                parent_lines.append(f"{indent}- {line}")
            else:
                # Continuation lines get extra indentation
                parent_lines.append(f"{indent}  {line}")
        context_parts.extend(parent_lines)

    # Add this block with its indentation
    # Use full content but filter out property lines (metadata, not semantic content)
    indent = indent_str * len(parents)
    for j, line in enumerate(block.content):
        # Skip property lines (they start with word characters followed by ::)
        if "::" in line and j > 0:  # Properties are never on first line
            continue
        if j == 0:
            # First line gets the bullet
            context_parts.append(f"{indent}- {line}")
        else:
            # Continuation lines get extra indentation
            context_parts.append(f"{indent}  {line}")

    # NOTE: We do NOT include children in the hash because:
    # 1. Creates circular dependency (parent hash changes when children added)
    # 2. Cannot be reproduced when finding blocks by ID (we don't have children)
    # Blocks with identical content+context will share IDs, which is acceptable.

    return "\n".join(context_parts)


def generate_content_hash(full_context: str, page_name: str | None = None) -> str:
    """Generate MD5 hash of full context string.

    Args:
        full_context: Full context string to hash
        page_name: Optional page name to prefix content (ensures global uniqueness)

    Returns:
        MD5 hash as hexadecimal string

    Examples:
        >>> generate_content_hash("My content")
        'a1b2c3d4e5f6...'
        >>> generate_content_hash("My content", "Page A")
        'b2c3d4e5f6a1...'  # Different hash due to page name
    """
    # Prefix content with page name for global uniqueness
    content_to_hash = f"{page_name}::{full_context}" if page_name else full_context
    return hashlib.md5(content_to_hash.encode()).hexdigest()


def generate_chunks(outline: "LogseqOutline", page_name: str | None = None) -> list[tuple["LogseqBlock", str, str]]:
    """Generate chunks with full context and hybrid IDs for all blocks.

    This recursively traverses the outline and generates:
    - Full context string (for embedding and uniqueness)
    - Hybrid ID (id:: property if present, otherwise content hash)

    Args:
        outline: Parsed LogseqOutline
        page_name: Optional page name to include in hash (ensures global uniqueness)

    Returns:
        List of (block, full_context, hybrid_id) tuples

    Examples:
        >>> chunks = generate_chunks(outline)
        >>> for block, context, hybrid_id in chunks:
        ...     print(f"{hybrid_id}: {context[:50]}")
        >>> chunks_with_page = generate_chunks(outline, "My Page")
        >>> # Hashes will be different due to page name
    """
    chunks = []
    indent_str = outline.indent_str

    def traverse(block: "LogseqBlock", parents: list["LogseqBlock"]) -> None:
        """Recursively traverse blocks and generate chunks."""
        # Generate full context for this block (with bullets and indentation)
        full_context = generate_full_context(block, parents, indent_str)

        # Determine hybrid ID (id:: property or content hash)
        if block.block_id:
            # Has explicit id:: property
            hybrid_id = block.block_id
        else:
            # Generate hash of full context (with page name for global uniqueness)
            hybrid_id = generate_content_hash(full_context, page_name)

        # Add chunk
        chunks.append((block, full_context, hybrid_id))

        # Traverse children
        new_parents = parents + [block]
        for child in block.children:
            traverse(child, new_parents)

    # Start traversal from root blocks
    for root_block in outline.blocks:
        traverse(root_block, [])

    return chunks
