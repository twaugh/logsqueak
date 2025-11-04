"""Logseq outline parser - Parse and manipulate Logseq markdown files.

This package provides tools for parsing Logseq's outline-based markdown format
into structured AST (Abstract Syntax Tree) and rendering it back to markdown.

Key features:
- Parse Logseq markdown into LogseqBlock tree structure
- Preserve exact property order (insertion order is sacred)
- Support for hybrid ID system (id:: properties and content hashes)
- Full-context generation for semantic search
- Graph path operations for navigating Logseq directories

Example:
    >>> from logseq_outline import LogseqOutline
    >>> outline = LogseqOutline.parse("- My bullet\\n  - Child bullet")
    >>> outline.blocks[0].content[0]
    'My bullet'
    >>> markdown = outline.render()
"""

from logseq_outline.parser import LogseqBlock, LogseqOutline
from logseq_outline.graph import GraphPaths
from logseq_outline.context import (
    generate_full_context,
    generate_content_hash,
    generate_chunks,
)

__version__ = "0.1.0"

__all__ = [
    "LogseqBlock",
    "LogseqOutline",
    "GraphPaths",
    "generate_full_context",
    "generate_content_hash",
    "generate_chunks",
]
