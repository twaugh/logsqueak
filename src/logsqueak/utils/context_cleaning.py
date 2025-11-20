"""Utilities for cleaning block context for LLM consumption."""

import re


def strip_id_property(context: str) -> str:
    """
    Strip id:: properties from block context for LLM prompts.

    When using short IDs in XML attributes or when id:: properties would
    conflict with the ID mapping system, we need to remove them from the
    block content.

    Args:
        context: Hierarchical block context with id:: properties

    Returns:
        Context with id:: properties removed

    Example:
        >>> context = "- Example block\\n  id:: 65f3a8e0-1234-5678\\n  - Child"
        >>> strip_id_property(context)
        "- Example block\\n  - Child"
    """
    lines = context.split('\n')
    filtered_lines = []

    for line in lines:
        # Skip lines that are id:: properties (with any indentation)
        if re.match(r'^\s*id::\s+\S', line):
            continue
        filtered_lines.append(line)

    return '\n'.join(filtered_lines)
