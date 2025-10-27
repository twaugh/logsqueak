"""Page chunking for block-level embeddings.

This module provides chunking functionality for converting Logseq pages into
block-level chunks for vector storage. It reuses the full-context generation
infrastructure from logsqueak.logseq.context.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

from logsqueak.logseq.context import generate_chunks
from logsqueak.logseq.parser import LogseqOutline


@dataclass
class Chunk:
    """Single block chunk for vector storage.

    Attributes:
        full_context_text: Full-context text including all parent blocks
        hybrid_id: Hybrid ID (explicit id:: or content hash)
        page_name: Name of the page this chunk belongs to
        block_content: Just the block's own content (without parents)
        metadata: Additional metadata for filtering/display
    """

    full_context_text: str
    hybrid_id: str
    page_name: str
    block_content: str
    metadata: dict


def chunk_page(outline: LogseqOutline, page_name: str) -> List[Chunk]:
    """Convert page outline into block-level chunks.

    Reuses generate_chunks() from logsqueak.logseq.context to build
    full-context text and hybrid IDs, then wraps results in Chunk dataclass
    with page metadata.

    Note: The hybrid_id is made globally unique by prefixing with page_name
    to avoid collisions when the same block content appears on multiple pages.

    Args:
        outline: Parsed Logseq page outline
        page_name: Name of the page (for metadata)

    Returns:
        List of Chunk objects, one per block in the outline

    Examples:
        >>> outline = LogseqOutline.parse("- Block 1\\n  - Block 2")
        >>> chunks = chunk_page(outline, "Test Page")
        >>> len(chunks)
        2
        >>> chunks[0].page_name
        'Test Page'
        >>> chunks[1].full_context_text
        '- Block 1\\n  - Block 2'
    """
    # Generate chunks using M1.2 infrastructure
    raw_chunks = generate_chunks(outline)

    # Wrap in Chunk dataclass with page metadata
    chunks = []
    # Track IDs to detect duplicates within the same page
    seen_ids = {}

    for block, full_context, hybrid_id in raw_chunks:
        # Make ID globally unique by including page name
        # Format: page_name::hybrid_id::seq (seq only if duplicate)
        global_id = f"{page_name}::{hybrid_id}"

        # Handle duplicates within same page by adding sequence number
        if global_id in seen_ids:
            seen_ids[global_id] += 1
            global_id = f"{global_id}::{seen_ids[global_id]}"
        else:
            seen_ids[global_id] = 0

        chunk = Chunk(
            full_context_text=full_context,
            hybrid_id=global_id,
            page_name=page_name,
            block_content=block.content,
            metadata={
                "page_name": page_name,
                "block_content": block.content,
                "indent_level": block.indent_level,
                "local_hybrid_id": hybrid_id,  # Store original for reference
            },
        )
        chunks.append(chunk)

    return chunks


def chunk_page_file(page_path: Path) -> List[Chunk]:
    """Chunk a Logseq page file.

    Convenience function that reads, parses, and chunks a page file.

    Args:
        page_path: Path to Logseq page markdown file

    Returns:
        List of chunks for the page

    Raises:
        FileNotFoundError: If page file doesn't exist
        ValueError: If page parsing fails
    """
    # Read page content
    page_content = page_path.read_text(encoding="utf-8")

    # Parse outline
    outline = LogseqOutline.parse(page_content)

    # Extract page name from filename (remove .md extension)
    page_name = page_path.stem

    # Chunk the page
    return chunk_page(outline, page_name)
