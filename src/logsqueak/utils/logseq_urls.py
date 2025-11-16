"""Logseq URL utilities.

Provides helpers for creating logseq:// protocol URLs.
"""

from pathlib import Path
from urllib.parse import quote


def create_logseq_url(page_name: str, graph_path: Path, block_id: str = None) -> str:
    """Create a logseq:// URL for a page or block.

    Args:
        page_name: Name of the Logseq page
        graph_path: Path to the Logseq graph directory
        block_id: Optional block UUID to link directly to a block

    Returns:
        Formatted logseq:// URL

    Example:
        >>> url = create_logseq_url("Python Programming", Path("/home/user/graph"))
        >>> print(url)
        logseq://graph/graph?page=Python%20Programming
        >>> url = create_logseq_url("Python Programming", Path("/home/user/graph"),
        ...                         block_id="67ed05bf-4e74-4087-a9de-d9e25166d1b9")
        >>> print(url)
        logseq://graph/graph?block-id=67ed05bf-4e74-4087-a9de-d9e25166d1b9
    """
    graph_name = graph_path.name
    encoded_graph = quote(graph_name, safe='')

    if block_id:
        # Block reference: logseq://graph/{graph_name}?block-id={block_uuid}
        encoded_block_id = quote(block_id, safe='')
        return f"logseq://graph/{encoded_graph}?block-id={encoded_block_id}"
    else:
        # Page reference: logseq://graph/{graph_name}?page={page_name}
        encoded_page = quote(page_name, safe='')
        return f"logseq://graph/{encoded_graph}?page={encoded_page}"
