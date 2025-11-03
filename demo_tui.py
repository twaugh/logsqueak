#!/usr/bin/env python3
"""
Interactive TUI Demo with Dummy Data

Demonstrates the redesigned 4-phase workflow:
1. Phase 1: Select knowledge blocks
2. Phase 2: Choose content version (original vs reworded)
3. Phase 3: Accept/reject integration decisions
4. Phase 4: Execute writes (simulated)

Run with: python demo_tui.py
"""

import asyncio
from dataclasses import dataclass
from typing import Literal

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical, ScrollableContainer
from textual.screen import Screen
from textual.widgets import Footer, Header, Label, Static, Tree
from textual.widgets.tree import TreeNode
from rich.text import Text

# Import markdown rendering from the actual codebase
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from logsqueak.tui.markdown import render_markdown_to_markup


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_hierarchical_context(block: dict, journal_date: str = "2025-01-15") -> str:
    """
    Format hierarchical context for a knowledge block (for Phase 3 decisions).

    Args:
        block: Knowledge block with 'parent' and 'content' fields
        journal_date: Journal entry date

    Returns:
        Formatted context string with tree structure
    """
    lines = []
    lines.append(f"[dim]Journal {journal_date}[/]")

    if block['parent']:
        # Render parent with markdown
        parent_rendered = render_markdown_to_markup(block['parent'])
        lines.append(f"[dim]  {parent_rendered}[/]")

        # Render current block with markdown and indent
        content_rendered = render_markdown_to_markup(block['content'])
        lines.append(f"[dim]    {content_rendered}[/]")
    else:
        # No parent - just show the block under journal
        content_rendered = render_markdown_to_markup(block['content'])
        lines.append(f"[dim]  {content_rendered}[/]")

    return "\n".join(lines)


# ============================================================================
# DUMMY DATA
# ============================================================================

DUMMY_JOURNAL = """
- Had great discussion with [[Sarah]] about **microservices** today
  - She mentioned that `bounded contexts` should drive service boundaries
  - Not technical layers (controller/service/repository pattern)
  - Reference: [[Domain-Driven Design]] book
- Learned about `ChromaDB` for vector storage
  - Much easier to set up than [[Pinecone]]
  - Good for prototyping [[RAG]] systems
  - See code: `pip install chromadb`
- DONE Buy groceries tomorrow
  - TODO Milk
  - TODO Eggs
  - TODO Bread
- Read article about **testing strategies**
  - Integration tests are more valuable than unit tests for system behavior
  - But unit tests are faster for [[TDD]] workflow
  - Link: https://martinfowler.com/articles/practical-test-pyramid.html
"""

DUMMY_KNOWLEDGE_BLOCKS = [
    {
        "id": "block_1",
        "content": "She mentioned that `bounded contexts` should drive service boundaries\nNot technical layers (controller/service/repository pattern)\nThis keeps business logic cohesive",
        "parent": "Had great discussion with [[Sarah]] about **microservices** today",
        "llm_reason": "Documents architectural principle about microservice design",
        "confidence": 0.92,
        "reworded": "**Bounded contexts** from [[Domain-Driven Design]] should be the primary driver for defining microservice boundaries, rather than technical layer separation (e.g., `controller`/`service`/`repository`).\n\nThis approach keeps business logic cohesive and aligned with domain concepts.",
    },
    {
        "id": "block_2",
        "content": "Learned about `ChromaDB` for vector storage\nMuch easier to set up than [[Pinecone]]\nGood for prototyping [[RAG]] systems\nInstall: `pip install chromadb`",
        "parent": None,
        "llm_reason": "Technical knowledge about vector database selection",
        "confidence": 0.88,
        "reworded": "`ChromaDB` is a **lightweight vector database** well-suited for prototyping [[RAG]] systems.\n\nKey advantages:\n- Simpler setup compared to hosted solutions like [[Pinecone]]\n- Easy installation: `pip install chromadb`\n- Great for local development",
    },
    {
        "id": "block_3",
        "content": "Integration tests are more valuable than unit tests for **system behavior**\nBut unit tests are faster for [[TDD]] workflow\nNeed both - see test pyramid",
        "parent": "Read article about **testing strategies**",
        "llm_reason": "Testing philosophy and trade-offs",
        "confidence": 0.76,
        "reworded": "**Integration tests** provide better validation of overall system behavior, while **unit tests** remain valuable for rapid feedback during [[TDD]] workflows.\n\nSee the [test pyramid](https://martinfowler.com/articles/practical-test-pyramid.html) for guidance on balancing both approaches.",
    },
]

DUMMY_INTEGRATION_DECISIONS = [
    {
        "block_id": "block_1",
        "page": "Software Architecture",
        "action": "add_section",
        "confidence": 0.89,
        "reasoning": "Relevant architectural principle not yet documented on this page",
        "target_block": None,
    },
    {
        "block_id": "block_2",
        "page": "RAG Implementation",
        "action": "add_under",
        "confidence": 0.91,
        "reasoning": "Fits under existing 'Vector Databases' section",
        "target_block": "Vector Databases",
    },
    {
        "block_id": "block_3",
        "page": "Testing Philosophy",
        "action": "add_section",
        "confidence": 0.72,
        "reasoning": "New perspective to add to testing discussion",
        "target_block": None,
    },
]


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class BlockState:
    block_id: str
    classification: Literal["pending", "knowledge"]
    confidence: float | None
    source: Literal["user", "llm"]
    llm_classification: Literal["knowledge"] | None
    llm_confidence: float | None
    reason: str | None


@dataclass
class ContentVersion:
    block_id: str
    original_content: str
    reworded_content: str | None
    selected_version: Literal["original", "reworded"]
    rewording_complete: bool


@dataclass
class AppState:
    current_phase: int
    block_states: dict[str, BlockState]
    content_versions: dict[str, ContentVersion]
    integration_decisions: list[dict]
    current_decision_index: int


# ============================================================================
# PHASE 1: KNOWLEDGE SELECTION
# ============================================================================

class Phase1Screen(Screen):
    """Phase 1: Select knowledge blocks from journal."""

    BINDINGS = [
        ("j", "cursor_down", "Down"),
        ("k", "cursor_up", "Up"),
        ("down", "cursor_down", "Down"),
        ("up", "cursor_up", "Up"),
        Binding("space", "toggle_selection", "Toggle", priority=True),
        ("a", "accept_all", "Accept AI"),
        ("c", "clear_all", "Clear All"),
        ("n", "continue", "Next →"),
    ]

    CSS = """
    Phase1Screen {
        layout: vertical;
    }

    #status-bar {
        dock: top;
        height: 3;
        background: $panel;
        padding: 1;
    }

    #tree-container {
        height: 1fr;
    }

    #reason-bar {
        dock: bottom;
        height: 3;
        background: $boost;
        border-top: solid $primary;
        padding: 1;
    }
    """

    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self.block_tree = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="status-bar"):
            yield Label("Phase 1: Pick blocks to integrate into pages")
            yield Static("LLM suggestions loaded (3 knowledge blocks identified)")
        with Container(id="tree-container"):
            yield Label("Loading...")
        with Container(id="reason-bar"):
            yield Static("Navigate and press Space to toggle knowledge selection", id="reason-text")
        yield Footer()

    async def on_mount(self) -> None:
        # Create tree with dummy data
        self.block_tree = Tree("Journal 2025-01-15")
        self.block_tree.root.expand()

        # Add blocks with LLM suggestions shown
        for block in DUMMY_KNOWLEDGE_BLOCKS:
            parent_label = block['parent'] if block['parent'] else ""
            if parent_label:
                # Render parent with markdown
                parent_rendered = render_markdown_to_markup(parent_label)
                parent_node = self.block_tree.root.add(parent_rendered, expand=True)
                # Render child with markdown
                content_rendered = render_markdown_to_markup(block['content'])
                child_node = parent_node.add(
                    f"[on dark_blue]{content_rendered}[/] ({int(block['confidence']*100)}%)",
                    data=block['id']
                )
            else:
                # Render content with markdown
                content_rendered = render_markdown_to_markup(block['content'])
                self.block_tree.root.add(
                    f"[on dark_blue]{content_rendered}[/] ({int(block['confidence']*100)}%)",
                    data=block['id']
                )

        # Replace loading with tree
        container = self.query_one("#tree-container", Container)
        await container.query("Label").remove()
        await container.mount(self.block_tree)

    def action_toggle_selection(self) -> None:
        if not self.block_tree or not self.block_tree.cursor_node:
            return

        node = self.block_tree.cursor_node
        block_id = node.data

        if not block_id:
            return

        # Toggle selection
        block_state = self.state.block_states.get(block_id)
        if block_state:
            if block_state.classification == "knowledge":
                block_state.classification = "pending"
                block_state.source = "llm"
                # Show LLM suggestion again
                block = next(b for b in DUMMY_KNOWLEDGE_BLOCKS if b['id'] == block_id)
                content_rendered = render_markdown_to_markup(block['content'])
                node.set_label(f"[on dark_blue]{content_rendered}[/] ({int(block['confidence']*100)}%)")
            else:
                block_state.classification = "knowledge"
                block_state.source = "user"
                block = next(b for b in DUMMY_KNOWLEDGE_BLOCKS if b['id'] == block_id)
                content_rendered = render_markdown_to_markup(block['content'])
                node.set_label(f"[on dark_green]{content_rendered}[/]")

    def action_accept_all(self) -> None:
        """Accept all LLM knowledge suggestions."""
        for block_id, block_state in self.state.block_states.items():
            if block_state.llm_classification == "knowledge":
                block_state.classification = "knowledge"
                block_state.confidence = block_state.llm_confidence
                # Keep source as "llm" to show very dark green background

        # Update tree visuals
        for node in self.block_tree.root.children:
            if node.data and self.state.block_states[node.data].classification == "knowledge":
                block = next(b for b in DUMMY_KNOWLEDGE_BLOCKS if b['id'] == node.data)
                content_rendered = render_markdown_to_markup(block['content'])
                # Use very dark green (#004400) for LLM-accepted blocks
                node.set_label(f"[on #004400]{content_rendered}[/]")
            # Also check children
            for child in node.children:
                if child.data and self.state.block_states[child.data].classification == "knowledge":
                    block = next(b for b in DUMMY_KNOWLEDGE_BLOCKS if b['id'] == child.data)
                    content_rendered = render_markdown_to_markup(block['content'])
                    # Use very dark green (#004400) for LLM-accepted blocks
                    child.set_label(f"[on #004400]{content_rendered}[/]")

        status = self.query_one("#status-bar Static", Static)
        status.update("Accepted 3 knowledge blocks")

    def action_clear_all(self) -> None:
        """Clear all selections."""
        for block_state in self.state.block_states.values():
            block_state.classification = "pending"

        # Update tree to show suggestions again
        for node in self.block_tree.root.children:
            if node.data:
                block = next(b for b in DUMMY_KNOWLEDGE_BLOCKS if b['id'] == node.data)
                content_rendered = render_markdown_to_markup(block['content'])
                node.set_label(f"[on dark_blue]{content_rendered}[/] ({int(block['confidence']*100)}%)")
            for child in node.children:
                if child.data:
                    block = next(b for b in DUMMY_KNOWLEDGE_BLOCKS if b['id'] == child.data)
                    content_rendered = render_markdown_to_markup(block['content'])
                    child.set_label(f"[on dark_blue]{content_rendered}[/] ({int(block['confidence']*100)}%)")

    def action_cursor_down(self) -> None:
        if self.block_tree:
            self.block_tree.action_cursor_down()
            self._update_reason_bar()

    def action_cursor_up(self) -> None:
        if self.block_tree:
            self.block_tree.action_cursor_up()
            self._update_reason_bar()

    def _update_reason_bar(self) -> None:
        if not self.block_tree or not self.block_tree.cursor_node:
            return

        block_id = self.block_tree.cursor_node.data
        if not block_id:
            return

        block = next((b for b in DUMMY_KNOWLEDGE_BLOCKS if b['id'] == block_id), None)
        if block:
            reason_bar = self.query_one("#reason-text", Static)
            reason_bar.update(f"🤖 LLM suggests knowledge ({int(block['confidence']*100)}%): {block['llm_reason']}")

    def action_continue(self) -> None:
        # Check if at least one knowledge block selected
        knowledge_count = sum(
            1 for bs in self.state.block_states.values()
            if bs.classification == "knowledge"
        )

        if knowledge_count == 0:
            status = self.query_one("#status-bar Static", Static)
            status.update("⚠ Please select at least one knowledge block")
            return

        self.state.current_phase = 2
        self.app.push_screen(Phase2Screen(self.state))


# ============================================================================
# PHASE 2: CONTENT REWORDING
# ============================================================================

class Phase2Screen(Screen):
    """Phase 2: Review and accept reworded content."""

    BINDINGS = [
        Binding("j", "cursor_down", "Down", priority=True),
        Binding("k", "cursor_up", "Up", priority=True),
        Binding("down", "cursor_down", "Down", priority=True),
        Binding("up", "cursor_up", "Up", priority=True),
        Binding("space", "toggle_version", "Toggle", priority=True),
        ("a", "accept_all_rewording", "Accept Rewording"),
        ("o", "use_all_original", "Use Original"),
        ("n", "continue", "Next →"),
        ("q", "back", "← Back"),
    ]

    CSS = """
    Phase2Screen {
        layout: vertical;
    }

    #status-bar {
        dock: top;
        height: 3;
        background: $panel;
        padding: 1;
    }

    #tree-container {
        height: 1fr;
    }

    #detail-panel {
        dock: bottom;
        height: 7;
        background: $panel;
        border-top: solid $primary;
        padding: 1;
    }
    """

    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self.rewording_tree = None
        self.knowledge_blocks = [
            bid for bid, bs in state.block_states.items()
            if bs.classification == "knowledge"
        ]

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="status-bar"):
            yield Label("Phase 2: Review reworded content")
            yield Static(f"Showing {len(self.knowledge_blocks)} knowledge blocks in context")
        with Container(id="tree-container"):
            yield Label("Loading...")
        with Container(id="detail-panel"):
            yield Static("Enter: expand/collapse | Space: toggle version | 'a': accept all | 'o': use all original | 'n': next →", id="detail-text")
        yield Footer()

    async def on_mount(self) -> None:
        # Create tree with full journal hierarchy, showing only knowledge blocks and their parents
        self.rewording_tree = Tree("Journal 2025-01-15")
        self.rewording_tree.root.expand()

        # Build tree structure from dummy data
        # We need to reconstruct the hierarchy showing only knowledge blocks
        for block in DUMMY_KNOWLEDGE_BLOCKS:
            block_id = block['id']
            if block_id not in self.knowledge_blocks:
                continue

            cv = self.state.content_versions[block_id]

            if block['parent']:
                # Find or create parent node (non-selectable)
                parent_rendered = render_markdown_to_markup(block['parent'])
                parent_node = None

                # Check if parent already exists
                for child in self.rewording_tree.root.children:
                    if str(child.label) == parent_rendered:
                        parent_node = child
                        break

                if not parent_node:
                    # Add parent as dimmed, non-selectable node (no data)
                    parent_label = f"[dim]{parent_rendered}[/]"
                    parent_node = self.rewording_tree.root.add(parent_label, expand=True)
                    # Disable the parent node so it can't be selected
                    parent_node.allow_expand = True

                # Add knowledge block as expandable node with both versions as children
                self._add_knowledge_node(parent_node, block, block_id, cv)
            else:
                # Root-level knowledge block
                self._add_knowledge_node(self.rewording_tree.root, block, block_id, cv)

        # Replace loading with tree
        container = self.query_one("#tree-container", Container)
        await container.query("Label").remove()
        await container.mount(self.rewording_tree)

        # Set up tree event handler and select first knowledge block
        self.rewording_tree.focus()

        # Move cursor to first knowledge block (skip root and any parent nodes)
        if self.rewording_tree.cursor_node and self.rewording_tree.cursor_node.data is None:
            # Start on root or parent, move to first knowledge block
            self.action_cursor_down()
        else:
            self._update_detail_panel()

    def _add_knowledge_node(self, parent_node: TreeNode, block: dict, block_id: str, cv: ContentVersion) -> None:
        """
        Add a knowledge block node with expandable original/reworded children.

        Args:
            parent_node: Parent tree node to add to
            block: Block data dict
            block_id: Block ID
            cv: ContentVersion for this block
        """
        # Main knowledge node shows selected version with indicator
        version_indicator = "✓" if cv.selected_version == "reworded" else "○"

        # Use the selected version for the main label
        if cv.selected_version == "reworded" and cv.reworded_content:
            display_content = cv.reworded_content
        else:
            display_content = block['content']

        # Render full content with markdown (no truncation)
        content_rendered = render_markdown_to_markup(display_content)
        main_label = f"[b]{version_indicator}[/b] {content_rendered}"

        # Create main node with block_id as data (collapsed by default)
        main_node = parent_node.add(main_label, data=block_id, expand=False)

        # Add original version child (non-selectable, for display only)
        # Show full original content with markdown
        original_rendered = render_markdown_to_markup(block['content'])
        original_indicator = "✓" if cv.selected_version == "original" else "○"
        original_label = f"{original_indicator} Original: {original_rendered}"
        main_node.add(original_label, data=None)  # No data = non-selectable

        # Add reworded version child (non-selectable, for display only)
        # Show full reworded content with markdown
        reworded_rendered = render_markdown_to_markup(block['reworded'])
        reworded_indicator = "✓" if cv.selected_version == "reworded" else "○"
        reworded_label = f"{reworded_indicator} Reworded: {reworded_rendered}"
        main_node.add(reworded_label, data=None)  # No data = non-selectable

    def action_cursor_down(self) -> None:
        """Move cursor down, skipping non-knowledge nodes."""
        if not self.rewording_tree:
            return

        # Move down until we find a knowledge block node
        original_cursor = self.rewording_tree.cursor_node
        self.rewording_tree.action_cursor_down()

        # If we landed on a non-knowledge node (no data), keep moving
        max_attempts = 20  # Prevent infinite loop
        attempts = 0
        while (self.rewording_tree.cursor_node
               and self.rewording_tree.cursor_node.data is None
               and attempts < max_attempts
               and self.rewording_tree.cursor_node != original_cursor):
            self.rewording_tree.action_cursor_down()
            attempts += 1

        self._update_detail_panel()

    def action_cursor_up(self) -> None:
        """Move cursor up, skipping non-knowledge nodes."""
        if not self.rewording_tree:
            return

        # Move up until we find a knowledge block node
        original_cursor = self.rewording_tree.cursor_node
        self.rewording_tree.action_cursor_up()

        # If we landed on a non-knowledge node (no data), keep moving
        max_attempts = 20  # Prevent infinite loop
        attempts = 0
        while (self.rewording_tree.cursor_node
               and self.rewording_tree.cursor_node.data is None
               and attempts < max_attempts
               and self.rewording_tree.cursor_node != original_cursor):
            self.rewording_tree.action_cursor_up()
            attempts += 1

        self._update_detail_panel()

    def action_toggle_version(self) -> None:
        if not self.rewording_tree or not self.rewording_tree.cursor_node:
            return

        block_id = self.rewording_tree.cursor_node.data
        if not block_id or block_id not in self.knowledge_blocks:
            return

        cv = self.state.content_versions[block_id]

        # Toggle version
        cv.selected_version = "reworded" if cv.selected_version == "original" else "original"

        # Update tree node label
        self._refresh_tree_node(self.rewording_tree.cursor_node, block_id)

        # Update detail panel
        self._update_detail_panel()

    def action_accept_all_rewording(self) -> None:
        for block_id in self.knowledge_blocks:
            self.state.content_versions[block_id].selected_version = "reworded"

        # Refresh all tree nodes
        self._refresh_all_nodes()

        # Update detail panel
        self._update_detail_panel()

        status = self.query_one("#status-bar Static", Static)
        status.update(f"Accepted all {len(self.knowledge_blocks)} reworded versions")

    def action_use_all_original(self) -> None:
        for block_id in self.knowledge_blocks:
            self.state.content_versions[block_id].selected_version = "original"

        # Refresh all tree nodes
        self._refresh_all_nodes()

        # Update detail panel
        self._update_detail_panel()

        status = self.query_one("#status-bar Static", Static)
        status.update(f"Using original content for all {len(self.knowledge_blocks)} blocks")

    def _refresh_tree_node(self, node: TreeNode, block_id: str) -> None:
        """Refresh a single tree node's label and children to show current version selection."""
        block = next(b for b in DUMMY_KNOWLEDGE_BLOCKS if b['id'] == block_id)
        cv = self.state.content_versions[block_id]

        # Update main node label with selected version
        version_indicator = "✓" if cv.selected_version == "reworded" else "○"

        # Use the selected version for the main label
        if cv.selected_version == "reworded" and cv.reworded_content:
            display_content = cv.reworded_content
        else:
            display_content = block['content']

        # Render full content with markdown (no truncation)
        content_rendered = render_markdown_to_markup(display_content)
        main_label = f"[b]{version_indicator}[/b] {content_rendered}"
        node.set_label(main_label)

        # Update child nodes (original and reworded indicators)
        if len(node.children) >= 2:
            # Update original child - show full content
            original_rendered = render_markdown_to_markup(block['content'])
            original_indicator = "✓" if cv.selected_version == "original" else "○"
            original_label = f"{original_indicator} Original: {original_rendered}"
            node.children[0].set_label(original_label)

            # Update reworded child - show full content
            reworded_rendered = render_markdown_to_markup(block['reworded'])
            reworded_indicator = "✓" if cv.selected_version == "reworded" else "○"
            reworded_label = f"{reworded_indicator} Reworded: {reworded_rendered}"
            node.children[1].set_label(reworded_label)

    def _refresh_all_nodes(self) -> None:
        """Refresh all tree node labels."""
        if not self.rewording_tree:
            return

        def refresh_node(node: TreeNode) -> None:
            if node.data and node.data in self.knowledge_blocks:
                self._refresh_tree_node(node, node.data)
            for child in node.children:
                refresh_node(child)

        refresh_node(self.rewording_tree.root)

    def _update_detail_panel(self) -> None:
        """Update the detail panel to show help text."""
        if not self.rewording_tree or not self.rewording_tree.cursor_node:
            return

        block_id = self.rewording_tree.cursor_node.data
        if not block_id or block_id not in self.knowledge_blocks:
            # Cursor on a parent node, not a knowledge block
            detail = self.query_one("#detail-text", Static)
            detail.update("[dim]Navigate to a knowledge block to see options[/]")
            return

        cv = self.state.content_versions[block_id]

        # Show help text about expanding and toggling
        lines = []
        if cv.selected_version == "reworded":
            lines.append("[green]✓ Using reworded version[/]")
        else:
            lines.append("[dim]○ Using original version[/]")
        lines.append("")
        lines.append("[dim]Enter: expand/collapse | Space: toggle version[/]")
        lines.append("[dim]'a': accept all rewording | 'o': use all original[/]")
        lines.append("[dim]'n': continue to next phase →[/]")

        detail = self.query_one("#detail-text", Static)
        detail.update("\n".join(lines))

    @on(Tree.NodeHighlighted)
    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted) -> None:
        """Update detail panel when tree selection changes."""
        self._update_detail_panel()

    def action_continue(self) -> None:
        self.state.current_phase = 3
        self.app.push_screen(Phase3Screen(self.state))

    def action_back(self) -> None:
        self.app.pop_screen()


# ============================================================================
# PHASE 3: INTEGRATION DECISIONS
# ============================================================================

class Phase3Screen(Screen):
    """Phase 3: Accept/reject integration decisions."""

    BINDINGS = [
        ("n", "next_decision", "Next →"),
        ("p", "prev_decision", "← Prev"),
        Binding("space", "toggle_decision", "Toggle", priority=True),
        ("a", "accept_all", "Accept All"),
        ("s", "skip_all", "Skip All"),
        Binding("enter", "continue", "Continue →", priority=True),
        ("q", "back", "← Back"),
    ]

    CSS = """
    Phase3Screen {
        layout: vertical;
    }

    #status-bar {
        dock: top;
        height: 3;
        background: $panel;
        padding: 1;
    }

    #decision-container {
        height: 1fr;
        overflow-y: auto;
        padding: 1;
    }

    .section {
        border: solid $primary;
        padding: 1;
        margin-bottom: 1;
    }

    .section-title {
        color: $accent;
        text-style: bold;
    }

    .accepted {
        background: $success-darken-2;
    }

    .skipped {
        background: $error-darken-3;
    }
    """

    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self.decisions = DUMMY_INTEGRATION_DECISIONS
        self.decision_states = {i: "pending" for i in range(len(self.decisions))}
        self.current_index = 0

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="status-bar"):
            yield Label("Phase 3: Accept or reject integrations")
            yield Static(f"Decision 1 of {len(self.decisions)}", id="decision-counter")
        with ScrollableContainer(id="decision-container"):
            yield Label("Loading...")
        yield Footer()

    async def on_mount(self) -> None:
        await self._show_decision(self.current_index)

    async def _show_decision(self, index: int) -> None:
        if index >= len(self.decisions):
            return

        decision = self.decisions[index]
        block = next(b for b in DUMMY_KNOWLEDGE_BLOCKS if b['id'] == decision['block_id'])
        cv = self.state.content_versions[decision['block_id']]

        # Update counter
        counter = self.query_one("#decision-counter", Static)
        counter.update(f"Decision {index + 1} of {len(self.decisions)}")

        # Build decision display
        container = self.query_one("#decision-container", ScrollableContainer)
        await container.remove_children()

        # Status indicator
        status = self.decision_states[index]
        if status == "accepted":
            await container.mount(Label("[green]✓ ACCEPTED[/]"))
        elif status == "skipped":
            await container.mount(Label("[red]✗ SKIPPED[/]"))
        else:
            await container.mount(Label("[yellow]⊙ PENDING[/]"))

        # Knowledge block context
        section1 = Vertical(classes="section")
        await container.mount(section1)
        await section1.mount(Label("Knowledge Block:", classes="section-title"))

        # Show hierarchical context
        context = format_hierarchical_context(block)
        await section1.mount(Static(context, markup=True))
        await section1.mount(Label(""))  # Spacing

        content = cv.reworded_content if cv.selected_version == "reworded" else block['content']
        content_rendered = render_markdown_to_markup(content)
        version_label = "(reworded)" if cv.selected_version == "reworded" else "(original)"
        await section1.mount(Static(f"{content_rendered} [dim]{version_label}[/]", markup=True))

        # Integration decision
        section2 = Vertical(classes="section")
        await container.mount(section2)
        await section2.mount(Label("Integration Decision:", classes="section-title"))
        await section2.mount(Static(f"Target page: [b]{decision['page']}[/]"))

        action_map = {
            "add_section": "Add as new section",
            "add_under": f"Add under '{decision['target_block']}'",
            "replace": f"Replace '{decision['target_block']}'",
        }
        await section2.mount(Static(f"Action: {action_map[decision['action']]}"))
        await section2.mount(Static(f"Confidence: {int(decision['confidence']*100)}%"))
        await section2.mount(Static(f"Reasoning: {decision['reasoning']}"))

    def action_toggle_decision(self) -> None:
        current_state = self.decision_states[self.current_index]

        if current_state == "pending":
            self.decision_states[self.current_index] = "accepted"
        elif current_state == "accepted":
            self.decision_states[self.current_index] = "skipped"
        else:
            self.decision_states[self.current_index] = "pending"

        asyncio.create_task(self._show_decision(self.current_index))

    def action_accept_all(self) -> None:
        for i in range(len(self.decisions)):
            self.decision_states[i] = "accepted"
        asyncio.create_task(self._show_decision(self.current_index))

        status = self.query_one("#status-bar Static", Static)
        status.update(f"Accepted all {len(self.decisions)} integration decisions")

    def action_skip_all(self) -> None:
        for i in range(len(self.decisions)):
            self.decision_states[i] = "skipped"
        asyncio.create_task(self._show_decision(self.current_index))

        status = self.query_one("#status-bar Static", Static)
        status.update(f"Skipped all {len(self.decisions)} integration decisions")

    def action_next_decision(self) -> None:
        if self.current_index < len(self.decisions) - 1:
            self.current_index += 1
            asyncio.create_task(self._show_decision(self.current_index))

    def action_prev_decision(self) -> None:
        if self.current_index > 0:
            self.current_index -= 1
            asyncio.create_task(self._show_decision(self.current_index))

    def action_continue(self) -> None:
        accepted_count = sum(1 for s in self.decision_states.values() if s == "accepted")

        if accepted_count == 0:
            status = self.query_one("#status-bar Static", Static)
            status.update("⚠ No integrations accepted - nothing to write")
            return

        self.state.current_phase = 4
        self.app.push_screen(Phase4Screen(self.state, self.decision_states, accepted_count))

    def action_back(self) -> None:
        self.app.pop_screen()


# ============================================================================
# PHASE 4: WRITE OPERATIONS
# ============================================================================

class Phase4Screen(Screen):
    """Phase 4: Execute write operations (simulated)."""

    BINDINGS = [
        ("enter", "exit", "Exit"),
    ]

    CSS = """
    Phase4Screen {
        layout: vertical;
    }

    #status-bar {
        dock: top;
        height: 3;
        background: $panel;
        padding: 1;
    }

    #results-container {
        height: 1fr;
        padding: 1;
    }
    """

    def __init__(self, state: AppState, decision_states: dict, accepted_count: int):
        super().__init__()
        self.state = state
        self.decision_states = decision_states
        self.accepted_count = accepted_count

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="status-bar"):
            yield Label("Phase 4: Write operations complete")
            yield Static(f"Integrated {self.accepted_count} blocks into pages")
        with ScrollableContainer(id="results-container"):
            yield Label("[green]✓ All write operations successful[/]")
            yield Label("")
            yield Label("Integrated blocks:")

            for i, (idx, state) in enumerate(self.decision_states.items()):
                if state == "accepted":
                    decision = DUMMY_INTEGRATION_DECISIONS[idx]
                    block = next(b for b in DUMMY_KNOWLEDGE_BLOCKS if b['id'] == decision['block_id'])
                    # Render markdown - let Textual handle truncation
                    content_rendered = render_markdown_to_markup(block['content'])
                    yield Label(f"  • {content_rendered} → {decision['page']}")

            yield Label("")
            yield Label("Journal entry updated with provenance markers")
            yield Label("Press Enter to exit")
        yield Footer()

    def action_exit(self) -> None:
        self.app.exit()


# ============================================================================
# MAIN APP
# ============================================================================

class DemoApp(App):
    """Interactive TUI Demo Application."""

    CSS = """
    Screen {
        background: $surface;
    }
    """

    def on_mount(self) -> None:
        # Initialize state with dummy data
        block_states = {}
        content_versions = {}

        for block in DUMMY_KNOWLEDGE_BLOCKS:
            # Initialize block states with LLM suggestions
            block_states[block['id']] = BlockState(
                block_id=block['id'],
                classification="pending",
                confidence=None,
                source="llm",
                llm_classification="knowledge",
                llm_confidence=block['confidence'],
                reason=block['llm_reason'],
            )

            # Initialize content versions
            content_versions[block['id']] = ContentVersion(
                block_id=block['id'],
                original_content=block['content'],
                reworded_content=block['reworded'],
                selected_version="original",
                rewording_complete=True,
            )

        state = AppState(
            current_phase=1,
            block_states=block_states,
            content_versions=content_versions,
            integration_decisions=[],
            current_decision_index=0,
        )

        # Start with Phase 1
        self.push_screen(Phase1Screen(state))


if __name__ == "__main__":
    app = DemoApp()
    app.run()
