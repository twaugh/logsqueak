#!/usr/bin/env python3
"""Demo app for TargetPagePreview widget.

This demonstrates the widget's capabilities:
- Markdown rendering (bullets, bold, links)
- Logseq [[page links]]
- Logseq #tags
- Green bar insertion indicator
- Nested structure with proper indentation
"""

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Header, Footer, Button, Static
from logsqueak.tui.widgets.target_page_preview import TargetPagePreview
from logseq_outline.parser import LogseqOutline
from logseq_outline.context import generate_full_context, generate_content_hash


# Sample Logseq page content demonstrating various features
SAMPLE_PAGES = {
    "basic": """- Python Concepts
  A collection of fundamental Python programming concepts.
  - Functions
    Reusable blocks of code that perform specific tasks.
  - Decorators
  - Generators""",

    "with_formatting": """- [[Python]] Best Practices
  Following established conventions helps maintain code quality and readability across teams.
  - Use **type hints** for clarity
    Introduced in Python 3.5, type hints help catch bugs early and improve IDE support.
  - Follow #PEP8 style guide
  - See [official docs](https://docs.python.org)
  - Related: [[Code Quality]]""",

    "nested_structure": """- Web Development
  Modern web development encompasses both client-side and server-side technologies.
  - Frontend
    The client-side of web applications, running in the user's browser.
    - React
      - Hooks
        Introduced in React 16.8, hooks allow functional components to use state and lifecycle features.
      - Context API
    - CSS
      - Flexbox
      - Grid
  - Backend
    - REST APIs
    - GraphQL
    - Databases
      - PostgreSQL
      - MongoDB""",

    "mixed_content": """- Project: Logsqueak Knowledge Extraction
  A TUI application for extracting lasting knowledge from Logseq journal entries using LLM-powered analysis.
  tags:: #project #logseq #tui
  status:: in-progress
  - **Phase 1**: Block Selection ✅
    Users can navigate journal blocks in a hierarchical tree view with LLM classification.
  - **Phase 2**: Content Editing ✅
  - **Phase 3**: Integration Review 🚧
    - Implement [[TargetPagePreview]] widget
      Shows preview of new content in target page context with insertion indicator.
    - Add green bar insertion indicator
    - Test with real [[Logseq]] data
  - See [[Project Management]] for timeline""",
}


class DemoApp(App):
    """Demo application for TargetPagePreview widget."""

    CSS = """
    #controls {
        height: 8;
        padding: 1;
        background: $boost;
    }

    TargetPagePreview {
        border: solid white;
        height: 1fr;
    }

    Button {
        margin: 0 1;
    }

    #info {
        padding: 1;
        background: $panel;
        color: $text;
    }
    """

    BINDINGS = [
        ("1", "load_basic", "Basic"),
        ("2", "load_formatted", "Formatted"),
        ("3", "load_nested", "Nested"),
        ("4", "load_mixed", "Mixed"),
        ("i", "toggle_insertion", "Toggle Insertion"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self):
        super().__init__()
        self.current_page = "basic"
        self.highlight_enabled = False

    def compose(self) -> ComposeResult:
        """Compose the demo UI."""
        yield Header()

        with Container(id="controls"):
            yield Static(
                "Press 1-4 to load different samples | Press 'i' to toggle block highlight",
                id="info"
            )
            with Horizontal():
                yield Button("Basic (1)", id="btn-basic", variant="primary")
                yield Button("Formatted (2)", id="btn-formatted")
                yield Button("Nested (3)", id="btn-nested")
                yield Button("Mixed (4)", id="btn-mixed")
                yield Button("Toggle Insert (i)", id="btn-insert", variant="success")

        yield TargetPagePreview()

        yield Footer()

    def on_mount(self) -> None:
        """Load initial sample on mount."""
        self.call_later(self.load_sample, "basic")

    async def load_sample(self, sample_name: str) -> None:
        """Load a sample page into the preview."""
        self.current_page = sample_name
        content = SAMPLE_PAGES[sample_name]

        # Parse to find a block to highlight
        highlight_block_id = None
        if self.highlight_enabled:
            outline = LogseqOutline.parse(content)
            if outline.blocks:
                # Find first child block (more interesting than root)
                if outline.blocks[0].children:
                    target_block = outline.blocks[0].children[0]
                    # If it has explicit ID, use it
                    if target_block.block_id:
                        highlight_block_id = target_block.block_id
                    else:
                        # Generate content hash
                        full_context = generate_full_context(target_block, [outline.blocks[0]])
                        highlight_block_id = generate_content_hash(full_context)

        preview = self.query_one(TargetPagePreview)
        await preview.load_preview(content, highlight_block_id=highlight_block_id)

        # Update button states
        for btn_id, btn_sample in [
            ("btn-basic", "basic"),
            ("btn-formatted", "with_formatting"),
            ("btn-nested", "nested_structure"),
            ("btn-mixed", "mixed_content"),
        ]:
            button = self.query_one(f"#{btn_id}", Button)
            button.variant = "primary" if btn_sample == sample_name or (
                btn_id == "btn-formatted" and sample_name == "with_formatting"
            ) or (
                btn_id == "btn-nested" and sample_name == "nested_structure"
            ) or (
                btn_id == "btn-mixed" and sample_name == "mixed_content"
            ) else "default"

    async def action_load_basic(self) -> None:
        """Load basic sample."""
        await self.load_sample("basic")

    async def action_load_formatted(self) -> None:
        """Load formatted sample."""
        await self.load_sample("with_formatting")

    async def action_load_nested(self) -> None:
        """Load nested sample."""
        await self.load_sample("nested_structure")

    async def action_load_mixed(self) -> None:
        """Load mixed content sample."""
        await self.load_sample("mixed_content")

    async def action_toggle_insertion(self) -> None:
        """Toggle block highlight."""
        self.highlight_enabled = not self.highlight_enabled

        # Update button
        button = self.query_one("#btn-insert", Button)
        button.variant = "success" if self.highlight_enabled else "default"

        # Reload current sample with new highlight state
        await self.load_sample(self.current_page)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_actions = {
            "btn-basic": self.action_load_basic,
            "btn-formatted": self.action_load_formatted,
            "btn-nested": self.action_load_nested,
            "btn-mixed": self.action_load_mixed,
            "btn-insert": self.action_toggle_insertion,
        }

        action = button_actions.get(event.button.id)
        if action:
            action()


if __name__ == "__main__":
    app = DemoApp()
    app.run()
