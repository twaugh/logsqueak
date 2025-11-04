"""Page chunking for block-level embeddings.

This module provides chunking functionality for converting Logseq pages into
block-level chunks for vector storage. It reuses the full-context generation
infrastructure from logseq_outline.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

from logseq_outline import generate_chunks, LogseqOutline


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

    Reuses generate_chunks() from logseq_outline to build
    full-context text and hybrid IDs, then wraps results in Chunk dataclass
    with page metadata.

    Note: The hybrid_id for content-hashed blocks includes the page name
    in the hashed content (not as a prefix), ensuring global uniqueness
    across pages. Within a single page, blocks with truly identical content
    and context will generate the same hybrid ID. In such cases, we only
    index the FIRST occurrence to ensure ChromaDB uniqueness and enable
    reproducible ID lookups.

    Args:
        outline: Parsed Logseq page outline
        page_name: Name of the page (for metadata and hashing)

    Returns:
        List of Chunk objects, one per block (duplicates excluded)

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
    # Generate chunks using M1.2 infrastructure with page_name for hashing
    raw_chunks = generate_chunks(outline, page_name=page_name)

    # Track seen IDs to deduplicate
    seen_ids = set()
    chunks = []

    for block, full_context, hybrid_id in raw_chunks:
        # Skip duplicates - only index first occurrence
        if hybrid_id in seen_ids:
            continue

        # Skip empty or whitespace-only blocks
        # A block is considered empty if all non-property lines are whitespace-only
        # or contain only empty JSON structures
        # Property lines are those containing "::" (e.g., "id:: value")
        # Note: Children of empty blocks are still processed because
        # generate_chunks() recursively traverses all blocks
        def is_property_line(line: str) -> bool:
            """Check if line is a property (key:: value format)."""
            stripped = line.strip()
            return "::" in stripped and len(stripped.split("::", 1)) == 2

        def is_empty_content(line: str) -> bool:
            """Check if line is effectively empty (whitespace or empty JSON structures)."""
            stripped = line.strip()
            return not stripped or stripped in ['{}', '[]', '{ }', '[ ]']

        non_property_lines = [line for line in block.content if not is_property_line(line)]
        if not non_property_lines or all(is_empty_content(line) for line in non_property_lines):
            continue

        seen_ids.add(hybrid_id)

        block_content = block.get_full_content(normalize_whitespace=True)
        chunk = Chunk(
            full_context_text=full_context,
            hybrid_id=hybrid_id,
            page_name=page_name,
            block_content=block_content,
            metadata={
                "page_name": page_name,
                "block_content": block_content,
                "indent_level": block.indent_level,
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
