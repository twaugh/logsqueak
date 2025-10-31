#!/usr/bin/env python3
"""Proof-of-concept demo for the new selection-focused UI.

This demonstrates what Phase 1 would look like with:
- Flat list display (instead of tree)
- Rendered Logseq markdown (excluding 'id' property)
- Highlight-style selection (knowledge blocks visually distinct)
- Reason display at bottom

Run with: python demo_selection_ui.py
"""

from textual.app import App, ComposeResult
from textual.containers import Vertical, ScrollableContainer
from textual.widgets import Header, Footer, Static, ListView, ListItem
from textual.binding import Binding
from rich.text import Text
from rich.markdown import Markdown


# Sample data simulating journal blocks with various Logseq features
SAMPLE_BLOCKS = [
    {
        "content": "TODO Review [[Pull Request #423]]",
        "classification": "activity",
        "confidence": 0.97,
        "reason": None,
        "indent": 0,
    },
    {
        "content": "DONE Meeting with [[Alice]] about [[Project Phoenix]]",
        "classification": "activity",
        "confidence": 0.95,
        "reason": None,
        "indent": 0,
    },
    {
        "content": "**Decision**: We'll use *microservices architecture* based on DDD principles",
        "classification": "knowledge",
        "confidence": 0.92,
        "reason": "documents architectural decision for future reference",
        "indent": 1,
    },
    {
        "content": "Each service gets its own database - see [[Database Per Service Pattern]]",
        "classification": "knowledge",
        "confidence": 0.88,
        "reason": "key design principle for service boundaries",
        "indent": 2,
    },
    {
        "content": "Avoid `shared databases` to prevent coupling between services",
        "classification": "knowledge",
        "confidence": 0.85,
        "reason": "rationale behind database-per-service pattern",
        "indent": 2,
    },
    {
        "content": "TODO Set up [[CI/CD Pipeline]] for new microservices",
        "classification": "activity",
        "confidence": 0.91,
        "reason": None,
        "indent": 1,
    },
    {
        "content": "Had lunch with the team",
        "classification": "activity",
        "confidence": 0.98,
        "reason": None,
        "indent": 0,
    },
    {
        "content": "Read article: **Domain-Driven Design Distilled** by [[Vaughn Vernon]]",
        "classification": "activity",
        "confidence": 0.72,
        "reason": None,
        "indent": 0,
    },
    {
        "content": "Bounded contexts should define service boundaries, NOT technical layers",
        "classification": "knowledge",
        "confidence": 0.94,
        "reason": "insight about service decomposition strategy",
        "indent": 1,
    },
    {
        "content": "Example: `UserContext`, `OrderContext`, `PaymentContext` - all separate services",
        "classification": "knowledge",
        "confidence": 0.87,
        "reason": "concrete examples of bounded context application",
        "indent": 2,
    },
    {
        "content": "DONE Worked on documentation for [[Onboarding Guide]]",
        "classification": "activity",
        "confidence": 0.96,
        "reason": None,
        "indent": 0,
    },
    {
        "content": "CANCELLED Follow up on pending code review (team was disbanded)",
        "classification": "activity",
        "confidence": 0.89,
        "reason": None,
        "indent": 0,
    },
    {
        "content": "TODO Write unit tests for [[Authentication Service]]",
        "classification": "activity",
        "confidence": 0.93,
        "reason": None,
        "indent": 0,
    },
]


class BlockListItem(ListItem):
    """A single block in the selection list."""

    def __init__(self, block_data: dict, index: int):
        super().__init__()
        self.block_data = block_data
        self.block_index = index

    def compose(self) -> ComposeResult:
        """Render the block with selection indicator and content."""
        block = self.block_data
        indent = "  " * block["indent"]

        # Icon based on classification
        if block["classification"] == "knowledge":
            icon = "✓"
            icon_style = "bold green"
        elif block["classification"] == "activity":
            icon = "✗"
            icon_style = "dim"
        else:
            icon = "⊙"
            icon_style = "yellow"

        # Confidence
        conf_str = ""
        if block["confidence"]:
            conf_str = f" ({int(block['confidence'] * 100)}%)"

        # Process content with markdown-like rendering
        content = block['content']

        # Build the text with manual markdown rendering
        text = Text()
        text.append(f"{icon} ", style=icon_style)
        text.append(indent)

        # Simple markdown parsing for common Logseq patterns
        self._render_markdown_to_text(content, text)

        text.append(conf_str, style="dim")

        yield Static(text)

    def _render_markdown_to_text(self, content: str, text: Text) -> None:
        """Render markdown-like content to Rich Text with styles.

        Supports:
        - **bold**
        - *italic*
        - `code`
        - [[links]]
        - TODO/DONE/CANCELLED markers (with checkboxes)
        - ~~strikethrough~~
        """
        import re

        # Handle TODO/DONE/CANCELLED markers with checkbox indicators
        if content.startswith("TODO "):
            text.append("☐ ", style="bold yellow")  # Empty checkbox
            text.append("TODO ", style="yellow")
            content = content[5:]
        elif content.startswith("DONE "):
            text.append("☑ ", style="bold green")  # Checked checkbox
            text.append("DONE ", style="green")
            content = content[5:]
        elif content.startswith("CANCELLED ") or content.startswith("CANCELED "):
            prefix_len = 10 if content.startswith("CANCELLED ") else 9
            text.append("☒ ", style="bold dim")  # Crossed checkbox
            text.append(content[:prefix_len], style="dim strike")
            content = content[prefix_len:]

        # Process the rest with regex
        i = 0
        while i < len(content):
            # Check for [[links]]
            if content[i:i+2] == "[[":
                end = content.find("]]", i+2)
                if end != -1:
                    link_text = content[i+2:end]
                    text.append(link_text, style="bold blue underline")
                    i = end + 2
                    continue

            # Check for **bold**
            if content[i:i+2] == "**":
                end = content.find("**", i+2)
                if end != -1:
                    bold_text = content[i+2:end]
                    text.append(bold_text, style="bold")
                    i = end + 2
                    continue

            # Check for *italic*
            if content[i] == "*" and (i == 0 or content[i-1] != "*"):
                end = content.find("*", i+1)
                if end != -1 and (end+1 >= len(content) or content[end+1] != "*"):
                    italic_text = content[i+1:end]
                    text.append(italic_text, style="italic")
                    i = end + 1
                    continue

            # Check for `code`
            if content[i] == "`":
                end = content.find("`", i+1)
                if end != -1:
                    code_text = content[i+1:end]
                    text.append(code_text, style="bold cyan on #222222")
                    i = end + 1
                    continue

            # Check for ~~strikethrough~~
            if content[i:i+2] == "~~":
                end = content.find("~~", i+2)
                if end != -1:
                    strike_text = content[i+2:end]
                    text.append(strike_text, style="strike dim")
                    i = end + 2
                    continue

            # Check for ~strikethrough~
            if content[i] == "~" and (i == 0 or content[i-1] != "~"):
                end = content.find("~", i+1)
                if end != -1 and (end+1 >= len(content) or content[end+1] != "~"):
                    strike_text = content[i+1:end]
                    text.append(strike_text, style="strike dim")
                    i = end + 1
                    continue

            # Regular character
            text.append(content[i])
            i += 1


class SelectionUIDemo(App):
    """Demo app showing the new selection-focused UI."""

    CSS = """
    Screen {
        background: $surface;
    }

    #header-bar {
        dock: top;
        height: 3;
        background: $panel;
        padding: 1;
        border-bottom: solid $primary;
    }

    #blocks-container {
        height: 1fr;
        border: solid $primary;
    }

    ListView {
        height: 100%;
        background: $surface;
    }

    ListItem {
        height: auto;
        padding: 0 1;
    }

    ListItem:hover {
        background: $boost;
    }

    ListItem.-active {
        background: $accent;
    }

    #reason-bar {
        dock: bottom;
        height: auto;
        min-height: 3;
        background: $boost;
        border-top: solid $primary;
        padding: 1;
    }

    #reason-text {
        color: $text;
    }

    #instructions {
        color: $text-muted;
        text-style: italic;
    }
    """

    BINDINGS = [
        ("j", "cursor_down", "Down"),
        ("k", "cursor_up", "Up"),
        Binding("K", "mark_knowledge", "Mark Knowledge", show=True),
        Binding("A", "mark_activity", "Mark Activity", show=True),
        Binding("n", "continue", "Next →", show=True),
        ("q", "quit", "Quit"),
    ]

    def __init__(self):
        super().__init__()
        self.blocks = SAMPLE_BLOCKS.copy()
        self.current_index = 0

    def compose(self) -> ComposeResult:
        """Layout the demo UI."""
        yield Header()

        # Header bar
        with Vertical(id="header-bar"):
            yield Static("📝 Pick blocks to integrate into pages - 2025-10-27")
            yield Static(
                "Use j/k to navigate, K to mark as knowledge, A to mark as activity",
                id="instructions"
            )

        # Blocks list
        with ScrollableContainer(id="blocks-container"):
            yield ListView(id="blocks-list")

        # Reason bar
        with Vertical(id="reason-bar"):
            yield Static("Select a knowledge block to see why it was classified", id="reason-text")

        yield Footer()

    def on_mount(self) -> None:
        """Populate list after mounting."""
        list_view = self.query_one("#blocks-list", ListView)
        for i, block in enumerate(self.blocks):
            list_view.append(BlockListItem(block, i))

    def action_cursor_down(self) -> None:
        """Move cursor down."""
        list_view = self.query_one("#blocks-list", ListView)
        list_view.action_cursor_down()
        self._update_reason_bar()

    def action_cursor_up(self) -> None:
        """Move cursor up."""
        list_view = self.query_one("#blocks-list", ListView)
        list_view.action_cursor_up()
        self._update_reason_bar()

    def action_mark_knowledge(self) -> None:
        """Mark current block as knowledge."""
        list_view = self.query_one("#blocks-list", ListView)
        if list_view.index is not None and 0 <= list_view.index < len(self.blocks):
            self.blocks[list_view.index]["classification"] = "knowledge"
            self.blocks[list_view.index]["confidence"] = None  # User override
            self._refresh_list()
            self._update_reason_bar()

    def action_mark_activity(self) -> None:
        """Mark current block as activity."""
        list_view = self.query_one("#blocks-list", ListView)
        if list_view.index is not None and 0 <= list_view.index < len(self.blocks):
            self.blocks[list_view.index]["classification"] = "activity"
            self.blocks[list_view.index]["confidence"] = None
            self.blocks[list_view.index]["reason"] = None
            self._refresh_list()
            self._update_reason_bar()

    def action_continue(self) -> None:
        """Continue to next phase."""
        knowledge_count = sum(1 for b in self.blocks if b["classification"] == "knowledge")
        self.notify(f"Would proceed with {knowledge_count} knowledge blocks")

    def _update_reason_bar(self) -> None:
        """Update reason bar based on current selection."""
        list_view = self.query_one("#blocks-list", ListView)
        reason_text = self.query_one("#reason-text", Static)

        if list_view.index is None or list_view.index < 0 or list_view.index >= len(self.blocks):
            reason_text.update("Select a knowledge block to see why it was classified")
            return

        block = self.blocks[list_view.index]

        if block["classification"] == "knowledge" and block["reason"]:
            reason_text.update(f"💡 Why knowledge: {block['reason']}")
        else:
            reason_text.update("")

    def _refresh_list(self) -> None:
        """Refresh the list view after state changes."""
        list_view = self.query_one("#blocks-list", ListView)
        current_index = list_view.index

        # Clear and rebuild
        list_view.clear()
        for i, block in enumerate(self.blocks):
            list_view.append(BlockListItem(block, i))

        # Restore cursor position
        if current_index is not None:
            list_view.index = current_index

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        """Handle list view highlight changes."""
        self._update_reason_bar()


if __name__ == "__main__":
    app = SelectionUIDemo()
    app.run()
