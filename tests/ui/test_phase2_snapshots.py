"""Snapshot tests for Phase 2 initial render.

Uses pytest-textual-snapshot to verify the visual layout and
structure of the Phase2Screen on initial load.
"""

import pytest
from textual.app import App
from logsqueak.tui.screens.content_editing import Phase2Screen
from logsqueak.models.edited_content import EditedContent
from logsqueak.models.background_task import BackgroundTaskState
from logseq_outline.parser import LogseqBlock


class Phase2TestApp(App):
    """Test app wrapper for Phase2Screen."""

    def __init__(self, screen: Phase2Screen):
        super().__init__()
        self.test_screen = screen

    def on_mount(self) -> None:
        """Push the test screen on mount."""
        self.push_screen(self.test_screen)


@pytest.fixture
def sample_blocks_for_snapshot():
    """Create sample blocks with hierarchical context for snapshot testing."""
    # Parent block (for context display)
    parent = LogseqBlock(
        content=["Daily Notes - 2025-01-15"],
        indent_level=0,
        block_id="parent-1",
        children=[]
    )

    # Knowledge block with parent
    blocks = [
        LogseqBlock(
            content=["Today I learned about Python decorators"],
            indent_level=1,
            block_id="block-1",
            children=[]
        ),
        LogseqBlock(
            content=["Fixed the authentication bug in the login module"],
            indent_level=0,
            block_id="block-2",
            children=[]
        ),
    ]

    return blocks


@pytest.fixture
def sample_edited_content_for_snapshot():
    """Create EditedContent with various states for snapshot testing."""
    return [
        EditedContent(
            block_id="block-1",
            original_content="Today I learned about Python decorators",
            hierarchical_context="Today I learned about Python decorators",
            current_content="Today I learned about Python decorators",
            reworded_content="Python decorators are functions that modify other functions",
            rewording_complete=True
        ),
        EditedContent(
            block_id="block-2",
            original_content="Fixed the authentication bug in the login module",
            hierarchical_context="Fixed the authentication bug in the login module",
            current_content="Fixed the authentication bug in the login module",
            reworded_content=None,  # Still pending
            rewording_complete=False
        ),
    ]


def test_phase2_initial_render_snapshot(
    snap_compare,
    sample_blocks_for_snapshot,
    sample_edited_content_for_snapshot
):
    """Test Phase 2 screen initial render matches snapshot."""
    screen = Phase2Screen(
        blocks=sample_blocks_for_snapshot,
        edited_content=sample_edited_content_for_snapshot,
        auto_start_workers=False
    )
    app = Phase2TestApp(screen)

    # Take snapshot of initial render
    assert snap_compare(app, terminal_size=(120, 40))


def test_phase2_with_llm_rewording_complete(
    snap_compare,
    sample_blocks_for_snapshot,
    sample_edited_content_for_snapshot
):
    """Test Phase 2 screen with LLM rewording complete."""
    screen = Phase2Screen(
        blocks=sample_blocks_for_snapshot,
        edited_content=sample_edited_content_for_snapshot,
        auto_start_workers=False
    )
    app = Phase2TestApp(screen)

    # Take snapshot showing LLM reworded content
    assert snap_compare(app, terminal_size=(120, 40))


def test_phase2_second_block_snapshot(
    snap_compare,
    sample_blocks_for_snapshot,
    sample_edited_content_for_snapshot
):
    """Test Phase 2 screen on second block matches snapshot."""
    screen = Phase2Screen(
        blocks=sample_blocks_for_snapshot,
        edited_content=sample_edited_content_for_snapshot,
        auto_start_workers=False
    )

    # Set to second block
    screen.current_block_index = 1

    app = Phase2TestApp(screen)

    # Take snapshot showing second block (with pending rewording)
    assert snap_compare(app, terminal_size=(120, 40))


def test_phase2_three_panel_layout_snapshot(
    snap_compare,
    sample_blocks_for_snapshot,
    sample_edited_content_for_snapshot
):
    """Test that three-panel layout is visible in snapshot."""
    screen = Phase2Screen(
        blocks=sample_blocks_for_snapshot,
        edited_content=sample_edited_content_for_snapshot,
        auto_start_workers=False
    )
    app = Phase2TestApp(screen)

    # Snapshot should show:
    # 1. Original hierarchical context panel (read-only)
    # 2. LLM reworded version panel
    # 3. Current editable version panel
    assert snap_compare(app, terminal_size=(140, 50))


def test_phase2_status_panel_snapshot(
    snap_compare,
    sample_blocks_for_snapshot,
    sample_edited_content_for_snapshot
):
    """Test that status panel shows background task progress."""
    screen = Phase2Screen(
        blocks=sample_blocks_for_snapshot,
        edited_content=sample_edited_content_for_snapshot,
        auto_start_workers=False
    )

    # Set some background task states for snapshot
    screen.page_indexing_state = BackgroundTaskState.IN_PROGRESS
    screen.rag_search_state = BackgroundTaskState.PENDING

    app = Phase2TestApp(screen)

    # Snapshot should show status panel with task progress
    assert snap_compare(app, terminal_size=(120, 40))


def test_phase2_footer_snapshot(
    snap_compare,
    sample_blocks_for_snapshot,
    sample_edited_content_for_snapshot
):
    """Test that footer shows keyboard shortcuts."""
    screen = Phase2Screen(
        blocks=sample_blocks_for_snapshot,
        edited_content=sample_edited_content_for_snapshot,
        auto_start_workers=False
    )
    app = Phase2TestApp(screen)

    # Snapshot should show footer with shortcuts:
    # j/k: Navigate, Tab: Focus, a: Accept LLM, r: Revert, n: Next, q: Back
    assert snap_compare(app, terminal_size=(120, 40))
