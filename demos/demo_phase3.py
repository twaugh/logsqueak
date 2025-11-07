#!/usr/bin/env python3
"""Demo for Phase 3: Integration Review Screen.

Demonstrates the integration review workflow with:
- Knowledge block display with context
- Decision list showing target pages
- Target page preview with insertion indicator
- Navigation and approval actions
"""

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Header, Footer, Static, ListView, ListItem, Label
from textual.binding import Binding
from logsqueak.tui.widgets.target_page_preview import TargetPagePreview
from logseq_outline.parser import LogseqOutline, LogseqBlock
from logseq_outline.context import generate_full_context, generate_content_hash
import logging

# Set up file logging
logging.basicConfig(
    filename='/tmp/demo_phase3.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True  # Force reconfiguration
)
logger = logging.getLogger(__name__)


# Sample data from test-graph
KNOWLEDGE_BLOCKS = [
    {
        "id": "kb1",
        "number": 1,
        "total": 3,
        "context": "Journal: 2025-01-15\n  - Morning session\n    - Python learning",
        "content": "**List comprehensions** are a concise way to create lists in Python. They combine map() and filter() operations into a single expression: `[x*2 for x in range(10) if x % 2 == 0]`",
        "decisions": [
            {
                "target_page": "Python",
                "action": "add_under",
                "target_block_title": "List Operations",
                "confidence": 0.92,
                "reasoning": "This is a fundamental Python feature that belongs under list operations",
            },
            {
                "target_page": "Functional Programming",
                "action": "add_section",
                "target_block_title": None,
                "confidence": 0.78,
                "reasoning": "List comprehensions are a functional programming concept",
            },
        ],
    },
    {
        "id": "kb2",
        "number": 2,
        "total": 3,
        "context": "Journal: 2025-01-15\n  - Afternoon\n    - Docker setup",
        "content": "Docker **multi-stage builds** reduce image size by using multiple FROM statements. Build artifacts from earlier stages can be copied to later stages with `COPY --from=builder`, eliminating build dependencies from the final image.",
        "decisions": [
            {
                "target_page": "Docker",
                "action": "add_under",
                "target_block_title": "Best Practices",
                "confidence": 0.95,
                "reasoning": "Multi-stage builds are a Docker best practice for production images",
            },
        ],
    },
    {
        "id": "kb3",
        "number": 3,
        "total": 3,
        "context": "Journal: 2025-01-16\n  - Reading notes\n    - CI/CD article",
        "content": "**Continuous deployment** differs from continuous delivery in that code changes automatically deploy to production after passing tests, without human approval. This requires high confidence in automated testing and monitoring.",
        "decisions": [
            {
                "target_page": "CI/CD",
                "action": "add_section",
                "target_block_title": None,
                "confidence": 0.88,
                "reasoning": "This clarifies an important distinction in CI/CD terminology",
            },
            {
                "target_page": "DevOps",
                "action": "add_under",
                "target_block_title": "Deployment Strategies",
                "confidence": 0.82,
                "reasoning": "Continuous deployment is a key DevOps practice",
            },
        ],
    },
]

# Sample target page content (what would be shown in preview)
TARGET_PAGES = {
    "Python": """- Python Programming
  - Core Concepts
    - Variables and Types
    - Control Flow
  - List Operations
    - Creating Lists
    - Indexing and Slicing
    - List Methods
  - Functions
    - Defining Functions
    - Lambda Functions""",
    "Functional Programming": """- Functional Programming Paradigm
  - Core Principles
    - Immutability
    - Pure Functions
    - First-Class Functions
  - Techniques
    - Map and Filter
    - Reduce Operations""",
    "Docker": """- Docker Containerization
  - Images
    - Building Images
    - Image Layers
  - Best Practices
    - Security
    - Performance Optimization
  - Networking
    - Bridge Networks
    - Host Networking""",
    "CI/CD": """- Continuous Integration and Deployment
  - CI Fundamentals
    - Automated Testing
    - Build Automation
  - CD Fundamentals
    - Deployment Pipelines
    - Environment Management""",
    "DevOps": """- DevOps Practices
  - Deployment Strategies
    - Blue-Green Deployment
    - Canary Releases
  - Monitoring
    - Metrics Collection
    - Alerting""",
}


class Phase3Demo(App):
    """Demo app for Phase 3 integration review."""

    CSS = """
    #main-container {
        height: 100%;
    }

    #content-row {
        height: 1fr;
    }

    #left-panel {
        width: 1fr;
        border: solid $primary;
        padding: 1;
    }

    #knowledge-block {
        height: auto;
        border: solid green;
        padding: 1;
        margin-bottom: 1;
    }

    #context {
        color: $text-muted;
        margin-bottom: 1;
    }

    #decisions-container {
        height: 1fr;
        border: solid yellow;
    }

    #decisions-list {
        height: 100%;
    }

    #right-panel {
        width: 2fr;
    }

    #preview-header {
        height: 3;
        background: $boost;
        padding: 1;
    }

    TargetPagePreview {
        height: 1fr;
    }

    #status-bar {
        height: 3;
        background: $panel;
        padding: 1;
    }

    ListItem {
        padding: 1;
    }

    ListItem > Label {
        width: 100%;
    }
    """

    BINDINGS = [
        Binding("j", "next_decision", "Next Decision", show=True),
        Binding("k", "prev_decision", "Prev Decision", show=True),
        Binding("y", "accept", "Accept (✓)", show=True),
        Binding("n", "next_block", "Next Block", show=True),
        Binding("a", "accept_all", "Accept All", show=True),
        Binding("q", "quit", "Quit", show=True),
    ]

    def __init__(self):
        super().__init__()
        self.current_block_idx = 0
        self.current_decision_idx = 0
        self.completed_decisions = set()  # Set of (block_id, decision_idx)

    def compose(self) -> ComposeResult:
        """Compose the Phase 3 UI."""
        yield Header()

        with Vertical(id="main-container"):
            with Horizontal(id="content-row"):
                # Left panel: Knowledge block + decision list
                with Container(id="left-panel"):
                    yield Static(id="knowledge-block")
                    with Container(id="decisions-container"):
                        yield ListView(id="decisions-list")

                # Right panel: Target page preview
                with Vertical(id="right-panel"):
                    yield Static(id="preview-header")
                    yield TargetPagePreview()

            yield Static(id="status-bar")

        yield Footer()

    async def on_mount(self) -> None:
        """Load initial content."""
        self.call_later(self.update_display)

    async def update_display(self) -> None:
        """Update all display elements for current state."""
        block = KNOWLEDGE_BLOCKS[self.current_block_idx]

        # Update knowledge block display
        context_lines = block["context"].split("\n")
        context_text = "\n".join(f"[dim]{line}[/dim]" for line in context_lines)

        kb_widget = self.query_one("#knowledge-block", Static)
        kb_widget.update(
            f"[bold]Knowledge Block {block['number']} of {block['total']}[/bold]\n\n"
            f"{context_text}\n\n"
            f"{block['content']}"
        )

        # Update decisions list
        decisions_list = self.query_one("#decisions-list", ListView)
        await decisions_list.clear()

        for idx, decision in enumerate(block["decisions"]):
            # Check if this decision is completed
            is_completed = (block["id"], idx) in self.completed_decisions
            is_selected = idx == self.current_decision_idx

            status = "✓" if is_completed else "⊙"
            style = "green" if is_completed else "yellow" if is_selected else "white"

            label_text = (
                f"[{style}]{status} {decision['target_page']}[/{style}]\n"
                f"  {decision['action']} "
            )
            if decision["target_block_title"]:
                label_text += f"'{decision['target_block_title']}'"
            label_text += f"\n  Confidence: {decision['confidence']:.0%}"

            item = ListItem(Label(label_text))
            await decisions_list.append(item)

        # Highlight current decision
        if 0 <= self.current_decision_idx < len(block["decisions"]):
            decisions_list.index = self.current_decision_idx

        # Update preview header
        decision = block["decisions"][self.current_decision_idx]
        header = self.query_one("#preview-header", Static)
        header.update(
            f"[bold]Target Page: {decision['target_page']}[/bold]\n"
            f"Reasoning: {decision['reasoning']}"
        )

        # Update target page preview
        preview = self.query_one(TargetPagePreview)
        page_content = TARGET_PAGES.get(decision["target_page"], "*Page not found*")

        # Generate preview with integrated new block
        preview_content = page_content
        highlight_block_id = None

        if page_content != "*Page not found*":
            # Parse the target page
            outline = LogseqOutline.parse(page_content)

            # Create a new block for the knowledge content
            new_block_content = block["content"]
            new_block_id = f"new-kb-{block['id']}"  # Temporary ID for highlighting

            # Find where to insert based on action
            action = decision["action"]
            target_title = decision.get("target_block_title")

            if action == "add_under" and target_title:
                # Find the target block and add as child
                def add_under_block(blocks, parents=[]):
                    for parent_block in blocks:
                        if parent_block.content and target_title.lower() in parent_block.content[0].lower():
                            # Found the target - add new block as child
                            # Note: add_child only takes the first line, but we want the full content
                            # So we'll manually create the child block
                            new_child = LogseqBlock(
                                content=[new_block_content],  # Wrapped in list - this is the full content
                                indent_level=parent_block.indent_level + 1,
                                block_id=new_block_id
                            )
                            parent_block.children.append(new_child)
                            return True
                        # Recursively search children
                        if add_under_block(parent_block.children, parents + [parent_block]):
                            return True
                    return False

                add_under_block(outline.blocks)
            elif action == "add_section":
                # Add as new root-level block
                new_root = LogseqBlock(
                    content=[new_block_content],
                    indent_level=0,
                    block_id=new_block_id
                )
                outline.blocks.append(new_root)

            # Render the modified outline
            preview_content = outline.render()

            # Re-parse to get the actual hash IDs that were generated
            reparsed = LogseqOutline.parse(preview_content)

            # Find the block with our new content
            def find_new_block(blocks, parents=[]):
                for block in blocks:
                    # Check if this block contains our new content
                    if block.content and new_block_content in block.content[0]:
                        # Generate its hash
                        full_context = generate_full_context(block, parents)
                        return generate_content_hash(full_context)
                    # Search children
                    result = find_new_block(block.children, parents + [block])
                    if result:
                        return result
                return None

            highlight_block_id = find_new_block(reparsed.blocks)
        else:
            highlight_block_id = None

        await preview.load_preview(preview_content, highlight_block_id=highlight_block_id)

        # Update status bar
        completed_count = sum(
            1 for bid, _ in self.completed_decisions if bid == block["id"]
        )
        pending_count = len(block["decisions"]) - completed_count

        status = self.query_one("#status-bar", Static)
        status.update(
            f"Block {block['number']} of {block['total']} | "
            f"{len(block['decisions'])} decisions: "
            f"✓ {completed_count} completed, ⊙ {pending_count} pending"
        )

    def action_next_decision(self) -> None:
        """Navigate to next decision."""
        block = KNOWLEDGE_BLOCKS[self.current_block_idx]
        if self.current_decision_idx < len(block["decisions"]) - 1:
            self.current_decision_idx += 1
            self.call_later(self.update_display)

    def action_prev_decision(self) -> None:
        """Navigate to previous decision."""
        if self.current_decision_idx > 0:
            self.current_decision_idx -= 1
            self.call_later(self.update_display)

    def action_accept(self) -> None:
        """Accept current decision."""
        block = KNOWLEDGE_BLOCKS[self.current_block_idx]
        self.completed_decisions.add((block["id"], self.current_decision_idx))
        self.call_later(self.update_display)

    async def action_accept_all(self) -> None:
        """Accept all pending decisions for current block."""
        block = KNOWLEDGE_BLOCKS[self.current_block_idx]
        for idx in range(len(block["decisions"])):
            self.completed_decisions.add((block["id"], idx))
        await self.action_next_block()

    async def action_next_block(self) -> None:
        """Move to next knowledge block."""
        if self.current_block_idx < len(KNOWLEDGE_BLOCKS) - 1:
            self.current_block_idx += 1
            self.current_decision_idx = 0
            await self.update_display()
        else:
            # Show completion
            status = self.query_one("#status-bar", Static)
            total_completed = len(self.completed_decisions)
            status.update(
                f"[green]✓ All blocks reviewed! {total_completed} decisions accepted.[/green]"
            )


if __name__ == "__main__":
    app = Phase3Demo()
    app.run()
