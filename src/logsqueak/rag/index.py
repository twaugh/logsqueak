"""RAG index implementation for semantic page matching.

This module implements the PageIndex build logic with per-page caching.
The main PageIndex class is defined in models/page.py.
"""

from pathlib import Path
from typing import Optional

from logsqueak.models.page import PageIndex as _PageIndex


def build_index(graph_path: Path, cache_dir: Optional[Path] = None) -> _PageIndex:
    """Build RAG index from Logseq graph.

    Convenience function that delegates to PageIndex.build().

    Per-page caching details:
    - Cache location: ~/.cache/logsqueak/embeddings/
    - Cache validation: mtime-based
    - Each page has own .pkl file
    - Contains: embedding + mtime + text

    Performance (tested scale: 566 pages, 2.3MB):
    - First run: ~20 seconds (embed all pages)
    - Subsequent runs: <1 second (load from cache)
    - After modifying 5 pages: ~1.5 seconds (561 cached + 5 new)

    Args:
        graph_path: Path to Logseq graph directory
        cache_dir: Optional cache directory (default: ~/.cache/logsqueak/embeddings/)

    Returns:
        Built PageIndex ready for semantic search
    """
    return _PageIndex.build(graph_path, cache_dir)


# Re-export PageIndex for convenience
PageIndex = _PageIndex
