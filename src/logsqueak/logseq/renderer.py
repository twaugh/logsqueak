"""Logseq markdown renderer with minimal changes guarantee.

This module implements rendering of LogseqOutline back to markdown,
ensuring property order preservation and minimal modifications.
"""

from logsqueak.logseq.parser import LogseqBlock, LogseqOutline


def render(outline: LogseqOutline) -> str:
    """Render LogseqOutline back to markdown.

    This is a convenience function that calls outline.render().

    IMPORTANT: Minimal changes guarantee (FR-008):
    - Preserves exact order of all blocks
    - Maintains original indentation (2 spaces per level)
    - NEVER reorders properties (insertion order sacred)
    - Children modifications are targeted and minimal
    - Uses _original_line when available for perfect round-trip

    Args:
        outline: Parsed outline to render

    Returns:
        Rendered Logseq markdown
    """
    return outline.render()


def render_block(block: LogseqBlock, indent_level: int = 0) -> str:
    """Render a single block and its children.

    Args:
        block: Block to render
        indent_level: Override indent level (default: use block.indent_level)

    Returns:
        Rendered block as markdown lines
    """
    lines = []

    # Use original line if available and block hasn't been modified
    if block._original_line and not block.children:
        lines.append(block._original_line)
    else:
        # Render block with correct indentation
        indent = "  " * (indent_level if indent_level > 0 else block.indent_level)
        lines.append(f"{indent}- {block.content}")

        # Render all children
        for child in block.children:
            child_lines = render_block(
                child, indent_level=block.indent_level + 1 if indent_level > 0 else 0
            )
            lines.append(child_lines)

    return "\n".join(lines)
