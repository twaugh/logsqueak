"""Full-context generation and hashing for hybrid block IDs.

This module provides utilities for generating stable block IDs based on
full-context hashing. This is used for the hybrid-ID system where blocks
without explicit id:: properties get IDs based on their content and position
in the document hierarchy.
"""

import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from logseq_outline.parser import LogseqBlock, LogseqOutline


def generate_full_context(
    block: "LogseqBlock",
    parents: list["LogseqBlock"],
    indent_str: str = "  ",
    frontmatter: list[str] | None = None
) -> str:
    """Generate full-context string for a block.

    The full context includes optional frontmatter (page properties), all parent
    blocks' full content with proper bullet formatting, plus the block's own full
    content. This must be reproducible from just the block and its parents (no
    siblings, no children) since we need to recreate the hash when finding blocks by ID.

    NOTE: Blocks with truly identical content and context will have the same hybrid ID.
    This is by design - such blocks are semantically equivalent for RAG purposes.
    Use explicit id:: properties if you need to distinguish identical blocks.

    Args:
        block: The block to generate context for
        parents: List of parent blocks from root to immediate parent
        indent_str: Indentation string (default: "  " for 2 spaces)
        frontmatter: Optional page-level properties/content that appears before first bullet

    Returns:
        Full context string with optional frontmatter, bullet markers and indentation

    Examples:
        >>> # Block with no parents
        >>> generate_full_context(block, [])
        "- My content"

        >>> # Block with parents
        >>> generate_full_context(grandchild, [root, child])
        "- Root content\\n  - Child content\\n    - Grandchild content"

        >>> # Block with frontmatter
        >>> generate_full_context(block, [], frontmatter=["title:: My Page", "tags:: [[Python]]"])
        "title:: My Page\\ntags:: [[Python]]\\n\\n- My content"
    """
    context_parts = []

    # Add frontmatter if provided (page properties)
    # Strip leading/trailing empty lines from frontmatter
    if frontmatter:
        # Find first and last non-empty lines
        non_empty_lines = [line for line in frontmatter if line.strip()]
        if non_empty_lines:
            # Find indices of first and last non-empty lines
            first_idx = next(i for i, line in enumerate(frontmatter) if line.strip())
            last_idx = len(frontmatter) - 1 - next(i for i, line in enumerate(reversed(frontmatter)) if line.strip())
            # Add only the content between first and last non-empty lines
            context_parts.extend(frontmatter[first_idx:last_idx + 1])
            # Add blank line separator between frontmatter and blocks
            context_parts.append("")

    # Add all parent blocks with proper indentation
    # Use full content (all lines) to maximize uniqueness
    # Normalize content to remove outdent markers for stable hashing
    for i, parent in enumerate(parents):
        indent = indent_str * i
        # Join all content lines with proper continuation indentation
        parent_lines = []
        normalized_content = _normalize_content_for_hashing(parent.content)
        for j, line in enumerate(normalized_content):
            if j == 0:
                # First line gets the bullet
                parent_lines.append(f"{indent}- {line}")
            else:
                # Continuation lines get extra indentation
                parent_lines.append(f"{indent}  {line}")
        context_parts.extend(parent_lines)

    # Add this block with its indentation
    # Use full content (all lines) to maximize uniqueness
    # Normalize content to remove outdent markers for stable hashing
    indent = indent_str * len(parents)
    normalized_content = _normalize_content_for_hashing(block.content)
    for j, line in enumerate(normalized_content):
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


def _normalize_content_for_hashing(content_lines: list[str]) -> list[str]:
    """Normalize block content for stable hash generation.

    Removes outdent markers (\x00N\x00) to ensure hash stability regardless
    of parsing mode (strict_indent_preservation=True/False).

    Args:
        content_lines: Block content lines (may contain outdent markers)

    Returns:
        Normalized content lines without outdent markers
    """
    normalized = []
    for line in content_lines:
        # Remove outdent markers if present
        if '\x00' in line:
            parts = line.split('\x00', 2)
            if len(parts) == 3:
                # Extract just the content (skip reduction marker)
                line = parts[2]
        normalized.append(line)
    return normalized


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
    - Full context string (for embedding and uniqueness, includes frontmatter)
    - Hybrid ID (id:: property if present, otherwise content hash)

    Performance optimization: Uses cached contexts from _augment_outline_with_ids() if available,
    avoiding redundant tree traversals and context generation.

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
    frontmatter = outline.frontmatter if outline.frontmatter else None

    def traverse(block: "LogseqBlock", parents: list["LogseqBlock"]) -> None:
        """Recursively traverse blocks and generate chunks."""
        # PERFORMANCE OPTIMIZATION: Use cached context if available
        # Check if we want frontmatter and if appropriate cache exists
        if frontmatter and block._cached_context is not None:
            # Use cached context with frontmatter
            full_context = block._cached_context
        elif not frontmatter and block._cached_context_no_frontmatter is not None:
            # Use cached context without frontmatter
            full_context = block._cached_context_no_frontmatter
        else:
            # Cache miss - generate context normally
            full_context = generate_full_context(block, parents, indent_str, frontmatter)

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
