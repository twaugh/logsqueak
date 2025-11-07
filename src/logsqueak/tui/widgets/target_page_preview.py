"""TargetPagePreview widget for showing integration preview with green bar indicator.

This widget displays the target page content with the new knowledge block
visually integrated at the insertion point, marked with a green bar.

New design: Custom Rich Text rendering with block-level highlighting.
"""

import re
from typing import Optional
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, ScrollableContainer
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static

from logseq_outline.parser import LogseqOutline, LogseqBlock
from logseq_outline.context import generate_full_context, generate_content_hash


def render_inline_syntax(content: str, base_style: str = "") -> Text:
    """Render inline Logseq syntax within a string.

    Handles (in order of specificity):
    1. [[Page Links]]
    2. #[[Link Tags]]
    3. #tags
    4. **bold**
    5. *italic*
    6. URLs

    Args:
        content: String content to render
        base_style: Base style to apply to unstyled text (e.g., "dim italic")

    Returns:
        Rich Text object with styled segments
    """
    text = Text()

    if not content:
        return text

    # Process text sequentially, applying patterns in order of specificity
    # We need to handle overlapping patterns carefully
    pos = 0

    while pos < len(content):
        remaining = content[pos:]
        matched = False

        # Try each pattern in order (most specific first)

        # 1. Link tag: #[[Page Name]]
        link_tag_match = re.match(r'^#\[\[([^\]]+)\]\]', remaining)
        if link_tag_match:
            text.append("#", style="green")
            text.append(link_tag_match.group(1), style="bold blue")
            pos += link_tag_match.end()
            matched = True
            continue

        # 2. Page link: [[Page Name]]
        page_link_match = re.match(r'^\[\[([^\]]+)\]\]', remaining)
        if page_link_match:
            text.append(page_link_match.group(1), style="bold blue")
            pos += page_link_match.end()
            matched = True
            continue

        # 3. Bold: **text**
        bold_match = re.match(r'^\*\*([^*]+)\*\*', remaining)
        if bold_match:
            text.append(bold_match.group(1), style="bold")
            pos += bold_match.end()
            matched = True
            continue

        # 4. Italic: *text* (but not part of **)
        italic_match = re.match(r'^\*([^*]+)\*(?!\*)', remaining)
        if italic_match:
            text.append(italic_match.group(1), style="italic")
            pos += italic_match.end()
            matched = True
            continue

        # 5. URL: http:// or https://
        url_match = re.match(r'^(https?://[^\s]+)', remaining)
        if url_match:
            text.append(url_match.group(1), style="blue underline")
            pos += url_match.end()
            matched = True
            continue

        # 6. Tag: #word (at start or after whitespace)
        # Check if we're at start or after whitespace
        if pos == 0 or content[pos - 1].isspace():
            tag_match = re.match(r'^#([\w-]+)', remaining)
            if tag_match:
                text.append(f"#{tag_match.group(1)}", style="green")
                pos += tag_match.end()
                matched = True
                continue

        # No pattern matched, add single character with base style
        if not matched:
            text.append(content[pos], style=base_style)
            pos += 1

    return text


def render_property_line(line: str) -> Text:
    """Render a property line with nested syntax.

    Handles Logseq properties like:
    - key:: value
    - key:: #tag, #value, other value
    - key:: [[Page Link]], more text

    Args:
        line: Property line to render (may include indentation)

    Returns:
        Rich Text object with styled segments

    Example:
        "  key:: #tag, other value"
        -> "  " + "key:" (bold yellow) + " " + "#tag" (green) + ", other value" (dim italic)
    """
    # Match property pattern: optional indent + key + :: + value
    match = re.match(r'^(\s*)([\w-]+)::\s*(.*)$', line)

    if not match:
        # Not a property line, return as plain text
        return Text(line)

    indent, key, value = match.groups()

    text = Text()

    # Add indentation (preserve it exactly)
    text.append(indent)

    # Add key (bold yellow)
    text.append(f"{key}:", style="bold yellow")

    # Add space separator
    text.append(" ")

    # Parse value for inline syntax (tags, links, etc.)
    # Use dim italic as base style for property values
    value_text = render_inline_syntax(value, base_style="dim italic")
    text.append_text(value_text)

    return text


def wrap_text_to_width(text: Text, max_width: int) -> list[Text]:
    """Wrap a Rich Text object to a maximum width.

    Args:
        text: Rich Text to wrap
        max_width: Maximum width per line

    Returns:
        List of Text objects (one per wrapped line)
    """
    if max_width <= 0:
        return [text]

    plain = text.plain
    if len(plain) <= max_width:
        return [text]

    # Manual word wrapping using Rich's slicing (preserves styles correctly)
    lines = []
    current_pos = 0

    while current_pos < len(plain):
        # Determine wrap point
        if current_pos + max_width >= len(plain):
            # Last segment - take the rest
            segment_end = len(plain)
        else:
            # Find last space within max_width
            search_end = current_pos + max_width
            wrap_point = plain.rfind(' ', current_pos, search_end + 1)

            if wrap_point <= current_pos:
                # No space found, hard wrap at max_width
                segment_end = search_end
            else:
                # Wrap at the space
                segment_end = wrap_point

        # Use Rich Text slicing to preserve styles
        # This correctly handles all styles within the slice
        segment_text = text[current_pos:segment_end]

        lines.append(segment_text)

        # Move to next position (skip space if we wrapped at one)
        current_pos = segment_end
        if current_pos < len(plain) and plain[current_pos] == ' ':
            current_pos += 1

    return lines if lines else [Text("")]


def render_block_content(
    block: LogseqBlock,
    max_width: int,
    indent_str: str = "  ",
) -> list[Text]:
    """Render a block's content into styled Text lines with word wrapping.

    Args:
        block: LogseqBlock to render
        max_width: Maximum width before wrapping
        indent_str: Indentation string per level (default: 2 spaces)

    Returns:
        List of Rich Text objects (one per rendered line)
    """
    lines = []
    base_indent = indent_str * block.indent_level
    bullet = "• "
    hanging_indent_width = len(bullet)  # Width of "• "

    for i, content_line in enumerate(block.content):
        is_first_line = (i == 0)

        # Determine if this is a property line
        is_property = "::" in content_line and not content_line.strip().startswith("-")

        # Skip id:: properties completely (don't render them at all)
        if is_property:
            # Check if this is an id:: property
            property_match = re.match(r'^\s*(id)::\s*', content_line)
            if property_match:
                # Skip this line entirely - don't render id:: properties
                continue

        # Render the content with syntax highlighting
        if is_property:
            rendered = render_property_line(content_line)
        else:
            # For first line (block content), just render inline syntax
            # For continuation lines, they're already part of block content
            rendered = render_inline_syntax(content_line.strip() if is_first_line else content_line.lstrip())

        # Build the full line with indentation
        if is_first_line:
            # First line: base indent + bullet + content
            prefix = base_indent + bullet
        else:
            # Continuation/property lines: base indent + hanging indent (same width as bullet)
            prefix = base_indent + (" " * hanging_indent_width)

        # Calculate available width for content
        available_width = max_width - len(prefix)

        # Wrap the rendered content
        wrapped_segments = wrap_text_to_width(rendered, available_width)

        # Add each wrapped segment with appropriate prefix
        for j, segment in enumerate(wrapped_segments):
            if j == 0:
                # First wrapped segment: use the calculated prefix
                line = Text(prefix)
                line.append_text(segment)
            else:
                # Subsequent wrapped segments: hanging indent WITHOUT the content prefix
                # For the first line, the prefix already has the base indent + bullet/hanging
                # For wrapped lines, we want the same indentation as the continuation
                wrapped_prefix = base_indent + (" " * hanging_indent_width)
                line = Text(wrapped_prefix)
                line.append_text(segment)

            lines.append(line)

    return lines


def render_outline_with_tracking(
    outline: LogseqOutline,
    max_width: int,
    indent_str: str = "  ",
    skip_hashes: bool = False,
) -> tuple[list[Text], dict[str, list[int]]]:
    """Render full outline and track which lines belong to which blocks.

    Args:
        outline: LogseqOutline to render
        max_width: Maximum width before wrapping
        indent_str: Indentation string per level
        skip_hashes: If True, only track blocks with explicit IDs (optimization)

    Returns:
        Tuple of:
        - List of rendered Text lines
        - Dict mapping block IDs to list of line numbers
    """
    all_lines: list[Text] = []
    block_map: dict[str, list[int]] = {}

    def render_block_recursive(block: LogseqBlock, parent_blocks: list[LogseqBlock]) -> None:
        """Recursively render block and its children."""
        # Generate block ID (explicit or content hash)
        if block.block_id:
            # Block has explicit ID - always track it
            block_id = block.block_id
        elif skip_hashes:
            # Skip hash generation optimization - don't track this block
            block_id = None
        else:
            # Generate content hash for this block
            full_context = generate_full_context(block, parent_blocks)
            block_id = generate_content_hash(full_context)

        # Render this block's content
        start_line = len(all_lines)
        block_lines = render_block_content(block, max_width, indent_str)
        all_lines.extend(block_lines)
        end_line = len(all_lines)

        # Track which lines belong to this block (if we have an ID)
        if block_id:
            block_map[block_id] = list(range(start_line, end_line))

        # Render children
        for child in block.children:
            render_block_recursive(child, parent_blocks + [block])

    # Render all root blocks
    for root_block in outline.blocks:
        render_block_recursive(root_block, [])

    return all_lines, block_map


class _LineGutter(Widget):
    """Gutter widget showing line indicators."""

    marked_lines: reactive[set[int]] = reactive(set())
    total_lines: reactive[int] = reactive(0)

    DEFAULT_CSS = """
    _LineGutter {
        width: 2;
        background: $surface;
    }
    """

    def render(self) -> str:
        """Render the gutter with line markers."""
        lines = []
        for i in range(self.total_lines):
            if i in self.marked_lines:
                lines.append("[green]▌[/green]")
            else:
                lines.append(" ")
        return "\n".join(lines)


class TargetPagePreview(Widget):
    """Scrollable preview of target page with block-level highlighting.

    New design using custom Rich Text rendering:
    - Parses Logseq markdown into block tree
    - Renders each block with syntax highlighting
    - Tracks which lines belong to highlighted block
    - Shows green bar in gutter for highlighted block
    """

    # Custom CSS for widget
    DEFAULT_CSS = """
    TargetPagePreview {
        border: solid white;
        background: $surface;
        height: auto;
    }

    TargetPagePreview:focus-within {
        border: heavy blue;
    }

    TargetPagePreview Horizontal {
        height: auto;
        width: 100%;
    }

    TargetPagePreview Static {
        background: $surface;
        color: $text;
        width: 100%;
        height: auto;
    }

    TargetPagePreview ScrollableContainer {
        width: 100%;
        height: 100%;
    }
    """

    def __init__(self, *args, **kwargs):
        """Initialize TargetPagePreview with gutter and content."""
        super().__init__(*args, id="target-page-preview", **kwargs)
        self.can_focus = True
        self._content = ""
        self._highlight_block_id: Optional[str] = None
        self._gutter: Optional[_LineGutter] = None
        self._content_widget: Optional[Static] = None

    def compose(self) -> ComposeResult:
        """Compose the gutter and content widget."""
        with ScrollableContainer():
            with Horizontal():
                self._gutter = _LineGutter()
                yield self._gutter
                self._content_widget = Static("*No preview available*", expand=True, shrink=False)
                yield self._content_widget

    async def on_mount(self) -> None:
        """Widget mounted - load any pending content."""
        # If content was set before widgets were available, load it now
        if self._content and self._content_widget and self._gutter:
            await self._render_content()

    async def on_resize(self) -> None:
        """Widget resized - re-render if we have content."""
        # If we have pending content and now have a width, render it
        if self._content and self._content_widget and self._gutter:
            if self.size.width > 0:
                await self._render_content()

    async def load_preview(
        self, content: str, highlight_block_id: Optional[str] = None
    ) -> None:
        """Load preview content and highlight specified block.

        Args:
            content: Full Logseq page content (markdown)
            highlight_block_id: Block ID to highlight (explicit id:: or content hash)
        """
        self._content = content
        self._highlight_block_id = highlight_block_id

        # Only update if widgets are available
        if not self._content_widget or not self._gutter:
            # Widgets not ready yet - on_mount will handle it
            return

        await self._render_content()

    async def _render_content(self) -> None:
        """Internal method to render the current content."""
        content = self._content
        highlight_block_id = self._highlight_block_id

        # Widget needs to have a width before we can render (height can be auto)
        if self.size.width == 0:
            # Will be called again after layout via on_resize
            return

        if not content:
            self._content_widget.update("*No preview available*")
            self._gutter.total_lines = 0
            self._gutter.marked_lines = set()
            return

        # Parse the content into blocks
        outline = LogseqOutline.parse(content)

        # Optimization: Check if highlight_block_id exists as an explicit ID in the content
        # If so, we can skip hash generation entirely
        has_explicit_highlight_id = False
        if highlight_block_id:
            # Quick scan for "id:: <highlight_block_id>" in the content
            id_pattern = f"id:: {highlight_block_id}"
            has_explicit_highlight_id = id_pattern in content

        # Render with tracking
        # Use available width for wrapping (widget width minus gutter)
        widget_width = self.size.width
        gutter_width = 2  # _LineGutter width
        max_width = max(widget_width - gutter_width, 40) if widget_width > 0 else 80

        rendered_lines, block_map = render_outline_with_tracking(
            outline, max_width=max_width, skip_hashes=has_explicit_highlight_id
        )

        # Join lines into single rich text
        combined_text = Text()
        for i, line in enumerate(rendered_lines):
            if i > 0:
                combined_text.append("\n")
            combined_text.append_text(line)

        # Update content widget
        self._content_widget.update(combined_text)

        # Update gutter
        self._gutter.total_lines = len(rendered_lines)

        # Mark lines for highlighted block
        if highlight_block_id and highlight_block_id in block_map:
            self._gutter.marked_lines = set(block_map[highlight_block_id])
        else:
            self._gutter.marked_lines = set()

        # Force a refresh to ensure the update is displayed
        self.refresh()

    def clear(self) -> None:
        """Clear preview content."""
        self._content = ""
        self._highlight_block_id = None
        if self._content_widget:
            self._content_widget.update("*No preview available*")
        if self._gutter:
            self._gutter.total_lines = 0
            self._gutter.marked_lines = set()
