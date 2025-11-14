"""Logseq URL utilities.

Provides helpers for creating logseq:// protocol URLs.
"""

from pathlib import Path
from urllib.parse import quote


def create_logseq_url(page_name: str, graph_path: Path) -> str:
    """Create a logseq:// URL for a page.

    Args:
        page_name: Name of the Logseq page
        graph_path: Path to the Logseq graph directory

    Returns:
        Formatted logseq:// URL

    Example:
        >>> url = create_logseq_url("Python Programming", Path("/home/user/graph"))
        >>> print(url)
        logseq://graph/graph?page=Python%20Programming
    """
    graph_name = graph_path.name

    # URL format: logseq://graph/(encoded-graph-name)?page=(encoded-page-name)
    encoded_graph = quote(graph_name, safe='')
    encoded_page = quote(page_name, safe='')
    return f"logseq://graph/{encoded_graph}?page={encoded_page}"
