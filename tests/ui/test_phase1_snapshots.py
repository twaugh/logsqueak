"""Snapshot tests for Phase 1 initial render.

Uses pytest-textual-snapshot to capture and compare rendered UI.
"""

import pytest
from textual.app import App
from logsqueak.tui.screens.block_selection import Phase1Screen
from logsqueak.models.block_state import BlockState
from logseq_outline.parser import LogseqBlock, LogseqOutline




class Phase1TestApp(App):
    """Test app wrapper for Phase1Screen."""

    def __init__(self, screen: Phase1Screen):
        super().__init__()
        self.test_screen = screen

    def on_mount(self) -> None:
        """Push the test screen on mount."""
        self.push_screen(self.test_screen)

@pytest.fixture
def sample_blocks():
    """Create sample blocks for testing."""
    blocks = [
        LogseqBlock(
            content=["2025-01-15"],
            indent_level=0,
            block_id="root",
            children=[
                LogseqBlock(
                    content=["TODO Review PR #123"],
                    indent_level=1,
                    block_id="block-1",
                    children=[]
                ),
                LogseqBlock(
                    content=["Learned that asyncio.create_task() enables concurrent operations"],
                    indent_level=1,
                    block_id="block-2",
                    children=[
                        LogseqBlock(
                            content=["This is different from await which blocks"],
                            indent_level=2,
                            block_id="block-2-1",
                            children=[]
                        )
                    ]
                ),
                LogseqBlock(
                    content=["Had lunch with the team"],
                    indent_level=1,
                    block_id="block-3",
                    children=[]
                ),
            ]
        ),
    ]
    return blocks


@pytest.fixture
def journal_outline(sample_blocks):
    """Create a LogseqOutline from sample blocks."""
    return LogseqOutline(blocks=sample_blocks, source_text="", frontmatter=[])


@pytest.fixture
def journals(journal_outline):
    """Create journals dict for Phase1Screen."""
    return {"2025-01-15": journal_outline}


def test_phase1_initial_render(snap_compare, journals):
    """Test Phase 1 screen initial appearance matches snapshot."""
    screen = Phase1Screen(journals=journals)
    app = Phase1TestApp(screen)

    assert snap_compare(
        app,
        terminal_size=(120, 40)
    )


def test_phase1_with_llm_suggestions(snap_compare, journals):
    """Test Phase 1 screen with LLM suggestions displayed."""
    # Set up block states with LLM suggestions
    block_states = {
        "block-1": BlockState(
            block_id="block-1",
            classification="pending",
            source="user",
        ),
        "block-2": BlockState(
            block_id="block-2",
            classification="knowledge",
            confidence=0.92,
            source="llm",
            llm_classification="knowledge",
            llm_confidence=0.92,
            reason="Explains asyncio concurrency patterns"
        ),
        "block-2-1": BlockState(
            block_id="block-2-1",
            classification="knowledge",
            confidence=0.88,
            source="llm",
            llm_classification="knowledge",
            llm_confidence=0.88,
            reason="Clarifies difference between async patterns"
        ),
        "block-3": BlockState(
            block_id="block-3",
            classification="pending",
            source="user",
        ),
    }

    screen = Phase1Screen(
        journals=journals,
        initial_block_states=block_states
    )
    app = Phase1TestApp(screen)

    assert snap_compare(
        app,
        terminal_size=(120, 40)
    )


def test_phase1_with_user_selections(snap_compare, journals):
    """Test Phase 1 screen with user selections (green highlights)."""
    # Set up block states with user selections
    block_states = {
        "block-1": BlockState(
            block_id="block-1",
            classification="knowledge",
            confidence=1.0,
            source="user",
        ),
        "block-2": BlockState(
            block_id="block-2",
            classification="knowledge",
            confidence=0.92,
            source="llm",
            llm_classification="knowledge",
            llm_confidence=0.92,
            reason="Explains asyncio concurrency patterns"
        ),
        "block-2-1": BlockState(
            block_id="block-2-1",
            classification="pending",
            source="user",
        ),
        "block-3": BlockState(
            block_id="block-3",
            classification="knowledge",
            confidence=1.0,
            source="user",
        ),
    }

    screen = Phase1Screen(
        journals=journals,
        initial_block_states=block_states
    )
    app = Phase1TestApp(screen)

    assert snap_compare(
        app,
        terminal_size=(120, 40)
    )


def test_phase1_with_background_tasks(snap_compare, journals):
    """Test Phase 1 screen with background tasks in progress."""
    from logsqueak.models.background_task import BackgroundTask

    test_screen = Phase1Screen(journals=journals)

    # Set up background tasks
    test_screen.background_tasks = {
        "llm_classification": BackgroundTask(
            task_type="llm_classification",
            status="running",
            progress_current=3,
            progress_total=5,
        ),
        "page_indexing": BackgroundTask(
            task_type="page_indexing",
            status="running",
            progress_percentage=45.5,
        ),
    }

    app = Phase1TestApp(test_screen)

    assert snap_compare(
        app,
        terminal_size=(120, 40)
    )


def test_phase1_with_processed_blocks(snap_compare, sample_blocks, journals):
    """Test Phase 1 screen with previously processed blocks."""
    # Create blocks with extracted-to:: property
    blocks = [
        LogseqBlock(
            content=["2025-01-15"],
            indent_level=0,
            block_id="root",
            children=[
                LogseqBlock(
                    content=[
                        "Learned about Python async patterns",
                        "extracted-to:: [Async Patterns](((uuid-123)))"
                    ],
                    indent_level=1,
                    block_id="block-1",
                    children=[]
                ),
                LogseqBlock(
                    content=[
                        "Found a bug in the authentication flow",
                        "extracted-to:: [Bug Fixes](((uuid-456)))"
                    ],
                    indent_level=1,
                    block_id="block-2",
                    children=[]
                ),
                LogseqBlock(
                    content=["New idea: use Redis for caching"],
                    indent_level=1,
                    block_id="block-3",
                    children=[]
                ),
            ]
        ),
    ]

    # Set up block states - LLM suggests block-2 (processed) and block-3 (not processed)
    block_states = {
        "block-1": BlockState(
            block_id="block-1",
            classification="pending",
            source="user",
        ),
        "block-2": BlockState(
            block_id="block-2",
            classification="pending",
            source="user",
            llm_classification="knowledge",
            llm_confidence=0.85,
            reason="Documents bug discovery"
        ),
        "block-3": BlockState(
            block_id="block-3",
            classification="pending",
            source="user",
            llm_classification="knowledge",
            llm_confidence=0.90,
            reason="New technical idea"
        ),
    }

    # Create journal outline from blocks
    test_outline = LogseqOutline(blocks=blocks, source_text="", frontmatter=[])
    test_journals = {"2025-01-15": test_outline}

    screen = Phase1Screen(
        journals=test_journals,
        initial_block_states=block_states
    )
    app = Phase1TestApp(screen)

    assert snap_compare(
        app,
        terminal_size=(120, 40)
    )
