"""Custom TreeView widget with multi-line wrapping support for Logseq-like display."""

from dataclasses import dataclass, field
from typing import Optional
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static


@dataclass
class TreeNode:
    """A node in the tree with multi-line content support."""

    content: str
    children: list["TreeNode"] = field(default_factory=list)
    expanded: bool = True
    id: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.id is None:
            import uuid
            self.id = str(uuid.uuid4())


class TreeLine(Static):
    """A single line in the tree (may be part of a multi-line node)."""

    DEFAULT_CSS = """
    TreeLine {
        height: auto;
        padding: 0 1;
    }

    TreeLine.selected {
        background: $accent;
        color: $text;
    }

    TreeLine.continuation {
        /* Continuation lines of wrapped text */
    }
    """

    def __init__(self,
                 text: Text,
                 node_id: str,
                 line_index: int = 0,
                 is_continuation: bool = False):
        super().__init__(text)
        self.node_id = node_id
        self.line_index = line_index
        self.is_continuation = is_continuation
        if is_continuation:
            self.add_class("continuation")


class TreeView(VerticalScroll):
    """Custom tree widget with multi-line wrapping support."""

    DEFAULT_CSS = """
    TreeView {
        height: 1fr;
        border: solid $accent;
    }
    """

    class NodeSelected(Message):
        """Posted when a node is selected."""

        def __init__(self, node_id: str, node: TreeNode):
            super().__init__()
            self.node_id = node_id
            self.node = node

    selected_node_id: reactive[Optional[str]] = reactive(None)

    def __init__(self, root_nodes: list[TreeNode], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.root_nodes = root_nodes
        self._node_map: dict[str, TreeNode] = {}
        self._build_node_map()

    def _build_node_map(self):
        """Build a map of node IDs to nodes for quick lookup."""
        self._node_map.clear()

        def visit(node: TreeNode):
            self._node_map[node.id] = node
            for child in node.children:
                visit(child)

        for root in self.root_nodes:
            visit(root)

    def compose(self) -> ComposeResult:
        """Render all tree nodes."""
        yield from self._render_all_nodes()

    def _render_all_nodes(self) -> list[TreeLine]:
        """Render all nodes in the tree."""
        lines = []
        for node in self.root_nodes:
            lines.extend(self._render_node(node, depth=0))
        return lines

    def _render_node(self, node: TreeNode, depth: int) -> list[TreeLine]:
        """Render a single node and its children."""
        lines = []

        # Calculate indentation
        indent = "  " * depth

        # Choose icon based on state
        if node.children:
            icon = "▼" if node.expanded else "▶"
        else:
            icon = "•"

        # Wrap content to available width
        # Reserve space for indentation and icon
        available_width = max(40, self.size.width - len(indent) - 3)
        wrapped_lines = self._wrap_text(node.content, available_width)

        # Render first line with icon
        if wrapped_lines:
            first_line = wrapped_lines[0]
            text = Text()
            text.append(f"{indent}{icon} ", style="bold cyan")
            text.append_text(Text.from_markup(first_line))

            line_widget = TreeLine(text, node.id, line_index=0, is_continuation=False)
            lines.append(line_widget)

            # Render continuation lines
            continuation_indent = indent + "  "
            for i, wrapped_line in enumerate(wrapped_lines[1:], start=1):
                text = Text()
                text.append(continuation_indent, style="dim")
                text.append_text(Text.from_markup(wrapped_line))

                line_widget = TreeLine(text, node.id, line_index=i, is_continuation=True)
                lines.append(line_widget)

        # Render children if expanded
        if node.expanded and node.children:
            for child in node.children:
                lines.extend(self._render_node(child, depth + 1))

        return lines

    def _wrap_text(self, text: str, width: int) -> list[str]:
        """Wrap text to specified width, preserving words."""
        if not text:
            return [""]

        # Simple word-wrapping algorithm
        words = text.split()
        lines = []
        current_line = []
        current_length = 0

        for word in words:
            # Strip markup for length calculation (rough approximation)
            word_length = len(word.replace('[', '').replace(']', '').replace('*', ''))

            if current_length + word_length + len(current_line) <= width:
                current_line.append(word)
                current_length += word_length
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = word_length

        if current_line:
            lines.append(' '.join(current_line))

        return lines or [""]

    def refresh_tree(self):
        """Refresh the entire tree display."""
        # Remove all existing children
        for child in list(self.children):
            child.remove()

        # Re-render
        for line in self._render_all_nodes():
            self.mount(line)

    def toggle_node(self, node_id: str):
        """Toggle expansion state of a node."""
        node = self._node_map.get(node_id)
        if node and node.children:
            node.expanded = not node.expanded
            self.refresh_tree()
            self._restore_selection()

    def _restore_selection(self):
        """Restore selection highlighting after refresh."""
        if self.selected_node_id:
            self._update_selection_display()

    def _update_selection_display(self):
        """Update visual selection state."""
        # Remove selection from all lines
        for child in self.children:
            if isinstance(child, TreeLine):
                child.remove_class("selected")

        # Add selection to lines belonging to selected node
        if self.selected_node_id:
            for child in self.children:
                if isinstance(child, TreeLine) and child.node_id == self.selected_node_id:
                    child.add_class("selected")

    def select_node(self, node_id: str):
        """Select a node by ID."""
        if node_id in self._node_map:
            self.selected_node_id = node_id
            self._update_selection_display()
            self._scroll_to_selected()

            # Post message
            node = self._node_map[node_id]
            self.post_message(self.NodeSelected(node_id, node))

    def _scroll_to_selected(self):
        """Scroll to make the selected node visible."""
        if not self.selected_node_id:
            return

        # Find the first TreeLine of the selected node
        for child in self.children:
            if isinstance(child, TreeLine) and child.node_id == self.selected_node_id:
                if child.line_index == 0:
                    # Scroll to make this widget visible
                    self.scroll_to_widget(child, animate=False)
                    break

    def on_click(self, event):
        """Handle click events on tree lines."""
        # Find the TreeLine that was clicked
        widget = event.widget
        while widget and not isinstance(widget, TreeLine):
            widget = widget.parent

        if isinstance(widget, TreeLine):
            node_id = widget.node_id

            # If clicking the first line and node has children, toggle expansion
            if widget.line_index == 0 and self._node_map[node_id].children:
                # Check if click was on the icon area (rough approximation)
                self.toggle_node(node_id)

            # Always select the node
            self.select_node(node_id)

    def action_cursor_down(self):
        """Move selection down."""
        if not self.selected_node_id:
            # Select first node
            if self.root_nodes:
                self.select_node(self.root_nodes[0].id)
            return

        # Find next node in display order
        lines = list(self.children)
        current_index = None

        for i, line in enumerate(lines):
            if isinstance(line, TreeLine) and line.node_id == self.selected_node_id:
                if line.line_index == 0:  # Only match first line of node
                    current_index = i
                    break

        if current_index is not None:
            # Find next node (skip continuation lines)
            for i in range(current_index + 1, len(lines)):
                if isinstance(lines[i], TreeLine) and lines[i].line_index == 0:
                    self.select_node(lines[i].node_id)
                    break

    def action_cursor_up(self):
        """Move selection up."""
        if not self.selected_node_id:
            return

        # Find previous node in display order
        lines = list(self.children)
        current_index = None

        for i, line in enumerate(lines):
            if isinstance(line, TreeLine) and line.node_id == self.selected_node_id:
                if line.line_index == 0:
                    current_index = i
                    break

        if current_index is not None:
            # Find previous node (skip continuation lines)
            for i in range(current_index - 1, -1, -1):
                if isinstance(lines[i], TreeLine) and lines[i].line_index == 0:
                    self.select_node(lines[i].node_id)
                    break

    def action_toggle_expand(self):
        """Toggle expansion of selected node."""
        if self.selected_node_id:
            self.toggle_node(self.selected_node_id)
