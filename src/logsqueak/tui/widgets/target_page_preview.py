"""TargetPagePreview widget for showing integration preview with green bar indicator.

This widget displays the target page content with the new knowledge block
visually integrated at the insertion point, marked with a green bar.
"""

from typing import Callable
from textual.widgets import Markdown, Static, RichLog
from textual.containers import Horizontal, Vertical, ScrollableContainer
from textual.reactive import reactive
from textual.widget import Widget
from textual.app import ComposeResult
from markdown_it import MarkdownIt
from markdown_it.rules_inline import StateInline
from rich.text import Text
from rich.markdown import Markdown as RichMarkdown
import re


def logseq_page_link_plugin(md: MarkdownIt) -> None:
    """Plugin to parse Logseq [[page links]] syntax.

    Converts [[Page Name]] to [Page Name](page:Page Name) for proper rendering.
    """
    def parse_page_link(state: StateInline, silent: bool) -> bool:
        # Check if we're at [[
        if state.src[state.pos:state.pos+2] != "[[":
            return False

        # Find closing ]]
        start = state.pos + 2
        end = state.src.find("]]", start)

        if end == -1:
            return False

        if not silent:
            # Create link token
            token = state.push("link_open", "a", 1)
            token.attrs = {"href": f"page:{state.src[start:end]}", "class": "logseq-page-link"}

            token = state.push("text", "", 0)
            token.content = state.src[start:end]

            state.push("link_close", "a", -1)

        state.pos = end + 2
        return True

    md.inline.ruler.before("link", "logseq_page_link", parse_page_link)


def logseq_tag_plugin(md: MarkdownIt) -> None:
    """Plugin to parse Logseq #tag syntax.

    Converts #tag to <span class="logseq-tag">#tag</span>.
    """
    def parse_tag(state: StateInline, silent: bool) -> bool:
        # Check if we're at #
        if state.src[state.pos] != "#":
            return False

        # Check if preceded by whitespace or start of line
        if state.pos > 0 and not state.src[state.pos-1].isspace():
            return False

        # Find end of tag (alphanumeric, underscore, hyphen)
        start = state.pos + 1
        end = start
        while end < len(state.src) and (state.src[end].isalnum() or state.src[end] in "_-"):
            end += 1

        if end == start:
            return False

        if not silent:
            token = state.push("html_inline", "", 0)
            tag_name = state.src[state.pos:end]
            token.content = f'<span class="logseq-tag">{tag_name}</span>'

        state.pos = end
        return True

    md.inline.ruler.before("text", "logseq_tag", parse_tag)


def logseq_property_plugin(md: MarkdownIt) -> None:
    """Plugin to parse Logseq property syntax.

    Converts key:: value to styled property display: key: value
    """
    def parse_property(state: StateInline, silent: bool) -> bool:
        # Look for property pattern at start of line (after whitespace)
        # Properties are: word-chars followed by :: followed by value

        # Check if we're at the start of a potential property key
        pos = state.pos
        src = state.src

        # Find :: delimiter
        delimiter_pos = src.find("::", pos)
        if delimiter_pos == -1 or delimiter_pos - pos > 50:  # Reasonable key length limit
            return False

        # Check if this looks like a property (key before ::)
        key_part = src[pos:delimiter_pos]

        # Property keys should be at start of line (after whitespace) or after list marker
        # For inline parsing, we need to check if we're at the logical start
        line_start = src.rfind("\n", 0, pos) + 1
        prefix = src[line_start:pos]

        # Check if prefix is only whitespace or list marker + whitespace
        if not (prefix.strip() == "" or prefix.strip() == "-"):
            return False

        # Key should not contain spaces or special chars (except - and _)
        if not key_part or not all(c.isalnum() or c in "-_" for c in key_part):
            return False

        # Find end of value (end of line or certain delimiters)
        value_start = delimiter_pos + 2
        value_end = src.find("\n", value_start)
        if value_end == -1:
            value_end = len(src)

        value_part = src[value_start:value_end].strip()

        if not silent:
            # Render as: <span class="logseq-property"><span class="property-key">key:</span> <span class="property-value">value</span></span>
            token = state.push("html_inline", "", 0)
            token.content = (
                f'<span class="logseq-property">'
                f'<span class="property-key">{key_part}:</span> '
                f'<span class="property-value">{value_part}</span>'
                f'</span>'
            )

        state.pos = value_end
        return True

    md.inline.ruler.before("text", "logseq_property", parse_property)


def create_logseq_parser() -> MarkdownIt:
    """Create a MarkdownIt parser with Logseq extensions."""
    parser = MarkdownIt()
    parser.use(logseq_page_link_plugin)
    parser.use(logseq_tag_plugin)
    parser.use(logseq_property_plugin)
    return parser


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
    """Scrollable preview of target page with integration point marked.

    Composite widget with:
    - Left gutter showing insertion indicator
    - Markdown content with Logseq syntax support
    """

    # Reactive attribute for insertion line
    insertion_line = reactive(-1)

    # Custom CSS for Logseq-specific elements
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
    }

    TargetPagePreview Markdown {
        background: $surface;
        color: $text;
    }

    TargetPagePreview .logseq-page-link {
        color: $accent;
        text-style: bold;
    }

    TargetPagePreview .logseq-tag {
        color: $success;
    }

    TargetPagePreview .logseq-property {
        color: $text-muted;
        text-style: italic;
    }

    TargetPagePreview .property-key {
        color: $warning;
        text-style: bold;
    }

    TargetPagePreview .property-value {
        color: $text-muted;
    }
    """

    def __init__(self, *args, **kwargs):
        """Initialize TargetPagePreview with gutter and markdown."""
        super().__init__(*args, id="target-page-preview", **kwargs)
        self.can_focus = True
        self._preview_content = ""
        self._markdown: Markdown | None = None
        self._gutter: _LineGutter | None = None

    def compose(self) -> ComposeResult:
        """Compose the gutter and markdown widget."""
        with Horizontal():
            self._gutter = _LineGutter()
            yield self._gutter
            self._markdown = Markdown(
                "*No preview available*",
                parser_factory=create_logseq_parser,
            )
            yield self._markdown

    async def load_preview(self, content: str, insertion_line: int = -1) -> None:
        """Load preview content and mark insertion point.

        Args:
            content: Full page content with integrated block
            insertion_line: Line number where new content is inserted (-1 for none)
        """
        self._preview_content = content
        self.insertion_line = insertion_line

        # Only update if widgets are available
        if not self._markdown or not self._gutter:
            return

        # Prepare content with property line breaks
        final_content = self._prepare_content()

        # Update markdown
        await self._markdown.update(final_content)

        # Update gutter
        total_lines = len(final_content.split("\n"))
        self._gutter.total_lines = total_lines
        if insertion_line >= 0:
            self._gutter.marked_lines = {insertion_line}
        else:
            self._gutter.marked_lines = set()

    def _prepare_content(self) -> str:
        """Prepare content for rendering (add line breaks for continuation lines).

        Also transforms Logseq-specific syntax for better display:
        - Properties: 'key:: value' → '**key:** *value*'
        - Tags: '#tag' → '`#tag`' (code formatting)
        - Link tags: '#[[Page Name]]' → '`#`**Page Name**'
        """
        content = self._preview_content

        if not content:
            return "*No preview available*"

        lines = content.split("\n")

        # Process lines for proper rendering
        processed_lines = []
        for i, line in enumerate(lines):
            # Check if next line is a continuation line (not a bullet, not empty)
            next_is_continuation = False
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                # Continuation line: has indentation but no bullet marker
                stripped = next_line.lstrip()
                if stripped and not stripped.startswith("-") and len(next_line) > len(stripped):
                    next_is_continuation = True

            # Detect and transform Logseq properties (key:: value pattern)
            stripped_current = line.lstrip()
            is_property = "::" in line and not stripped_current.startswith("-")

            if is_property:
                # Transform "key:: value" to styled "key: value"
                # Match property pattern
                match = re.match(r'^(\s*)([\w-]+)::\s*(.*)$', line)
                if match:
                    indent, key, value = match.groups()

                    # Transform tags in the property value before wrapping in italic
                    # Handle #[[link-tag]] first (more specific pattern)
                    value = re.sub(r'#\[\[([^\]]+)\]\]', r'`#`**\1**', value)
                    # Then handle regular #tags
                    value = re.sub(r'#([\w-]+)', r'`#\1`', value)

                    # Use markdown bold for key, italic for value
                    # Note: nested markdown (italic containing code) should work
                    line = f"{indent}**{key}:** *{value}*"
            else:
                # Transform tags in non-property lines
                # Handle #[[link-tag]] first (more specific pattern)
                line = re.sub(r'#\[\[([^\]]+)\]\]', r'`#`**\1**', line)
                # Then handle regular #tags
                line = re.sub(r'(?<=\s)#([\w-]+)', r'`#\1`', line)
                # Also catch tags at start of line/value
                line = re.sub(r'^(\s*)#([\w-]+)', r'\1`#\2`', line)

            # Check if current line is a continuation line
            is_continuation = (
                stripped_current
                and not stripped_current.startswith("-")
                and len(line) > len(stripped_current)
                and not is_property  # Don't double-process properties
            )

            # Add double-space at end to force markdown line break
            # This is needed for continuation lines and properties
            if is_property or next_is_continuation or is_continuation:
                line = line.rstrip() + "  "

            processed_lines.append(line)

        return "\n".join(processed_lines)

    async def clear(self) -> None:
        """Clear preview content."""
        self._preview_content = ""
        self.insertion_line = -1
        if self._markdown:
            await self._markdown.update("*No preview available*")
        if self._gutter:
            self._gutter.total_lines = 0
            self._gutter.marked_lines = set()
