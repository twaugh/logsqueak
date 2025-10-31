"""Markdown rendering utilities for TUI display.

This module provides functions to render LogseqBlock objects as markdown
suitable for display in the TUI, with optional property filtering.
"""

from logsqueak.logseq.parser import LogseqBlock


def render_block_for_display(
    block: LogseqBlock,
    indent_str: str = "  ",
    exclude_properties: set[str] | None = None,
    show_children: bool = True,
) -> str:
    """Render a LogseqBlock as markdown for TUI display.

    This renderer differs from LogseqOutline.render() in that it:
    - Can exclude specific properties (e.g., 'id' for cleaner display)
    - Is designed for human readability, not round-trip fidelity
    - Can optionally render without children (for flat lists)

    Args:
        block: Block to render
        indent_str: Indentation string (default: "  ")
        exclude_properties: Set of property names to exclude (e.g., {"id"})
        show_children: Whether to render child blocks (default: True)

    Returns:
        Rendered markdown string

    Example:
        >>> block = LogseqBlock(...)
        >>> render_block_for_display(block, exclude_properties={"id"})
        "- Meeting notes
          participants:: [[Alice]], [[Bob]]
          - Discussed project timeline"
    """
    if exclude_properties is None:
        exclude_properties = set()

    lines = []
    indent = indent_str * block.indent_level

    # Render first line with bullet
    if not block.content:
        lines.append(f"{indent}-")
    elif block.content[0] == "":
        lines.append(f"{indent}-")
    else:
        lines.append(f"{indent}- {block.content[0]}")

    # Render continuation lines, filtering out excluded properties
    for line in block.content[1:]:
        # Skip lines that are properties we want to exclude
        if _is_excluded_property(line, exclude_properties):
            continue

        # Handle indent reduction markers (from parser)
        if line.startswith('\x00'):
            parts = line.split('\x00', 2)
            reduction = int(parts[1])
            content = parts[2]
            # Calculate actual indent
            base_continuation = indent + "  "
            actual_indent = base_continuation[:-reduction] if reduction > 0 else base_continuation
            lines.append(f"{actual_indent}{content}")
        else:
            lines.append(f"{indent}  {line}")

    # Render children if requested
    if show_children:
        for child in block.children:
            child_lines = render_block_for_display(
                child,
                indent_str=indent_str,
                exclude_properties=exclude_properties,
                show_children=True,
            )
            lines.append(child_lines)

    return "\n".join(lines)


def _is_excluded_property(line: str, exclude_properties: set[str]) -> bool:
    """Check if a line is a property that should be excluded.

    Args:
        line: Content line to check
        exclude_properties: Set of property names to exclude

    Returns:
        True if line is an excluded property, False otherwise

    Example:
        >>> _is_excluded_property("id:: abc123", {"id"})
        True
        >>> _is_excluded_property("tags:: [[project]]", {"id"})
        False
    """
    if not exclude_properties:
        return False

    # Strip leading whitespace to check property name
    stripped = line.lstrip()

    # Check if line matches "property:: value" pattern
    if "::" not in stripped:
        return False

    # Extract property name (everything before "::")
    prop_name = stripped.split("::", 1)[0].strip()

    return prop_name in exclude_properties


def render_block_flat(
    block: LogseqBlock,
    parent_content: str = "",
    indent_str: str = "  ",
    exclude_properties: set[str] | None = None,
) -> list[tuple[LogseqBlock, str, int]]:
    """Render a block hierarchy as a flat list with visual indentation.

    This is useful for selection UIs where you want to show all blocks
    in a scrollable list but maintain visual hierarchy.

    Args:
        block: Root block to render
        parent_content: Content of parent blocks (for context)
        indent_str: Indentation string
        exclude_properties: Properties to exclude

    Returns:
        List of (block, rendered_content, depth) tuples

    Example:
        >>> blocks = render_block_flat(root_block, exclude_properties={"id"})
        >>> for block, content, depth in blocks:
        ...     print(f"{indent_str * depth}{content}")
    """
    if exclude_properties is None:
        exclude_properties = set()

    results = []

    # Render this block (without children)
    content = render_block_for_display(
        block,
        indent_str=indent_str,
        exclude_properties=exclude_properties,
        show_children=False,
    )

    # Add to results
    results.append((block, content, block.indent_level))

    # Recursively process children
    for child in block.children:
        child_results = render_block_flat(
            child,
            parent_content=content,
            indent_str=indent_str,
            exclude_properties=exclude_properties,
        )
        results.extend(child_results)

    return results
