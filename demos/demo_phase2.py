#!/usr/bin/env python3
"""Demo for Phase 2: Content Editing Screen.

Demonstrates the content editing workflow with:
- Original hierarchical context (read-only)
- LLM reworded version (when available)
- Editable current content panel
- Navigation and editing actions

Features demonstrated:
- Three-panel vertical layout (original, LLM, editable)
- Keyboard navigation (j/k for blocks, Tab to focus editor)
- Accept LLM version ('a' key)
- Revert to original ('r' key)
- Auto-save on navigation
- LLM rewording status

Run: python demos/demo_phase2.py
"""

from pathlib import Path
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer
from logsqueak.tui.screens.content_editing import Phase2Screen
from logsqueak.models.edited_content import EditedContent
from logseq_outline.parser import LogseqBlock, LogseqOutline
from logseq_outline.graph import GraphPaths


def create_sample_blocks() -> list[LogseqBlock]:
    """Create sample knowledge blocks for demonstration."""
    return [
        # Block 1: Python async patterns
        LogseqBlock(
            content=["Learned about [[Python]] asyncio patterns while debugging slow API calls today #async #concurrency"],
            indent_level=2,
            block_id="block-async-1",
            children=[
                LogseqBlock(
                    content=["**asyncio.create_task()** enables concurrent operations without blocking"],
                    indent_level=3,
                    block_id="block-async-2",
                    children=[]
                ),
                LogseqBlock(
                    content=["This is different from *await* which blocks until the coroutine completes"],
                    indent_level=3,
                    block_id="block-async-3",
                    children=[]
                ),
            ]
        ),
        # Block 2: Docker networking
        LogseqBlock(
            content=["[[Docker]] containers on same bridge network can communicate by service name #docker #networking"],
            indent_level=1,
            block_id="block-docker-1",
            children=[
                LogseqBlock(
                    content=["No need to use IP addresses - **Docker DNS** handles name resolution automatically"],
                    indent_level=2,
                    block_id="block-docker-2",
                    children=[]
                ),
            ]
        ),
        # Block 3: Git workflow (multi-line)
        LogseqBlock(
            content=[
                "TIL: `git rebase -i HEAD~N` lets you interactively squash commits #git #workflow",
                "This is really useful when cleaning up commits before creating a PR",
                "You can reorder, edit, squash, or drop commits in the interactive editor"
            ],
            indent_level=1,
            block_id="block-git-1",
            children=[
                LogseqBlock(
                    content=["Use **fixup** instead of **squash** to discard commit messages"],
                    indent_level=2,
                    block_id="block-git-2",
                    children=[]
                ),
            ]
        ),
    ]


def create_sample_edited_content() -> list[EditedContent]:
    """Create sample EditedContent objects with LLM rewording.

    Each EditedContent represents a knowledge block with:
    - original_content: Single block content (without parent context)
    - hierarchical_context: Full hierarchical context (with all parent blocks)
    - reworded_content: LLM version without temporal context (single block)
    - current_content: What user edits (starts as original_content)
    """
    return [
        # Block 1: Async - LLM rewording complete (single line block)
        EditedContent(
            block_id="block-async-1",
            original_content="Learned about [[Python]] asyncio patterns while debugging slow API calls today #async #concurrency",
            hierarchical_context=(
                "2025-01-15 - Tuesday\n"
                "  Morning session\n"
                "    Learned about [[Python]] asyncio patterns while debugging slow API calls today #async #concurrency\n"
                "      **asyncio.create_task()** enables concurrent operations without blocking\n"
                "      This is different from *await* which blocks until the coroutine completes"
            ),
            reworded_content="[[Python]] asyncio patterns for concurrent operations #async #concurrency",
            current_content="Learned about [[Python]] asyncio patterns while debugging slow API calls today #async #concurrency",
            rewording_complete=True
        ),
        # Block 2: Docker - LLM rewording complete (single line block)
        EditedContent(
            block_id="block-docker-1",
            original_content="[[Docker]] containers on same bridge network can communicate by service name #docker #networking",
            hierarchical_context=(
                "2025-01-15 - Tuesday\n"
                "  Afternoon debugging\n"
                "    [[Docker]] containers on same bridge network can communicate by service name #docker #networking\n"
                "      No need to use IP addresses - **Docker DNS** handles name resolution automatically"
            ),
            reworded_content="[[Docker]] service name resolution on bridge networks - containers communicate by service name via Docker DNS #docker #networking",
            current_content="[[Docker]] containers on same bridge network can communicate by service name #docker #networking",
            rewording_complete=True
        ),
        # Block 3: Git - LLM rewording in progress (multi-line block - shows continuation lines)
        EditedContent(
            block_id="block-git-1",
            original_content=(
                "TIL: `git rebase -i HEAD~N` lets you interactively squash commits #git #workflow\n"
                "This is really useful when cleaning up commits before creating a PR\n"
                "You can reorder, edit, squash, or drop commits in the interactive editor"
            ),
            hierarchical_context=(
                "2025-01-16 - Wednesday\n"
                "  Evening reading\n"
                "    TIL: `git rebase -i HEAD~N` lets you interactively squash commits #git #workflow\n"
                "    This is really useful when cleaning up commits before creating a PR\n"
                "    You can reorder, edit, squash, or drop commits in the interactive editor\n"
                "      Use **fixup** instead of **squash** to discard commit messages"
            ),
            reworded_content=None,  # LLM hasn't finished yet
            current_content=(
                "TIL: `git rebase -i HEAD~N` lets you interactively squash commits #git #workflow\n"
                "This is really useful when cleaning up commits before creating a PR\n"
                "You can reorder, edit, squash, or drop commits in the interactive editor"
            ),
            rewording_complete=False
        ),
    ]


class Phase2DemoApp(App):
    """Demo application for Phase 2 Content Editing screen."""

    TITLE = "Logsqueak - Phase 2: Content Editing Demo"
    SUB_TITLE = "Interactive Content Editing and LLM Rewording"

    def __init__(self):
        super().__init__()
        self.blocks = create_sample_blocks()
        self.edited_content = create_sample_edited_content()

    def on_mount(self) -> None:
        """Push Phase2Screen on mount."""
        # Create journal outline from blocks
        # Note: source_text is required but not actually used in demo
        journal_outline = LogseqOutline(
            blocks=self.blocks,
            source_text="# Demo journal content",
            indent_str="  ",
            frontmatter=[]
        )

        # Create a dummy graph_paths (demo doesn't need real paths)
        # Create temporary directory for demo
        demo_graph_path = Path("/tmp/demo-graph")
        demo_graph_path.mkdir(parents=True, exist_ok=True)
        (demo_graph_path / "pages").mkdir(exist_ok=True)
        (demo_graph_path / "journals").mkdir(exist_ok=True)

        graph_paths = GraphPaths(demo_graph_path)

        screen = Phase2Screen(
            blocks=self.blocks,
            edited_content=self.edited_content,
            journal_outline=journal_outline,
            graph_paths=graph_paths,
            llm_client=None,  # No real LLM client in demo
            rag_search=None,  # No real RAG search in demo
            auto_start_workers=False  # Disable workers for demo
        )
        self.push_screen(screen)


if __name__ == "__main__":
    app = Phase2DemoApp()
    app.run()
