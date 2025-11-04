"""Shared TUI utility functions.

Common helpers used across multiple TUI screens.
"""

import hashlib
from typing import Optional

from logseq_outline import LogseqBlock


def generate_content_hash(block: LogseqBlock) -> str:
    """
    Generate content hash for block without id:: property.

    Uses MD5 hash of full block content (normalized).

    Args:
        block: LogseqBlock to hash

    Returns:
        MD5 hash string
    """
    content = block.get_full_content(normalize_whitespace=True)
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def find_block_by_id(blocks: list[LogseqBlock], block_id: str) -> Optional[LogseqBlock]:
    """
    Find block in block tree by hybrid ID.

    Args:
        blocks: List of root LogseqBlocks to search
        block_id: Hybrid ID (id:: property or content hash)

    Returns:
        LogseqBlock if found, None otherwise
    """
    def search(blocks_to_search: list[LogseqBlock]) -> Optional[LogseqBlock]:
        for block in blocks_to_search:
            # Get block's hybrid ID (block_id property or content hash)
            current_block_id = block.block_id or generate_content_hash(block)
            if current_block_id == block_id:
                return block
            result = search(block.children)
            if result:
                return result
        return None

    return search(blocks)


def get_block_hierarchy(blocks: list[LogseqBlock], block_id: str) -> list[LogseqBlock]:
    """
    Get hierarchical path from root to target block.

    Args:
        blocks: List of root LogseqBlocks to search
        block_id: Hybrid ID of target block

    Returns:
        List of blocks from root to target (inclusive), or empty list if not found
    """
    def search(blocks_to_search: list[LogseqBlock], path: list[LogseqBlock]) -> Optional[list[LogseqBlock]]:
        for block in blocks_to_search:
            current_path = path + [block]
            # Get block's hybrid ID
            current_block_id = block.block_id or generate_content_hash(block)
            if current_block_id == block_id:
                return current_path
            result = search(block.children, current_path)
            if result:
                return result
        return None

    result = search(blocks, [])
    return result if result else []
