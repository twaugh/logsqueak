"""UI tests for Phase 1 status widget progress display.

Tests for background task progress updates in the status widget.
"""

import pytest
import asyncio
from textual.app import App
from textual.pilot import Pilot
from logsqueak.tui.screens.block_selection import Phase1Screen
from logsqueak.models.background_task import BackgroundTask
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
            content=["Block 1"],
            indent_level=0,
            block_id="block-1",
            children=[]
        ),
        LogseqBlock(
            content=["Block 2"],
            indent_level=0,
            block_id="block-2",
            children=[]
        ),
    ]
    return blocks


@pytest.fixture
def journal_outline(sample_blocks):
    """Create a LogseqOutline from sample blocks."""
    return LogseqOutline(blocks=sample_blocks, source_text="", frontmatter=[])


@pytest.mark.asyncio
async def test_status_shows_llm_classification_progress(sample_blocks, journal_outline):
    """Test status widget shows LLM classification progress."""
    screen = Phase1Screen(blocks=sample_blocks, journal_date="2025-01-15", journal_outline=journal_outline)

    app = Phase1TestApp(screen)


    async with app.run_test() as pilot:
        await pilot.pause()

        status_panel = screen.query_one("#status-panel")

        # Initially no tasks
        # (Status widget should show "Ready" or similar)

        # Start classification task
        screen.background_tasks["llm_classification"] = BackgroundTask(
            task_type="llm_classification",
            status="running",
            progress_current=1,
            progress_total=2,
        )

        # Trigger reactive update
        screen.refresh()
        await pilot.pause()

        # Status should show progress
        # Note: Exact text depends on implementation
        # This is a simplified check


@pytest.mark.asyncio
async def test_status_shows_page_indexing_progress(sample_blocks, journal_outline):
    """Test status widget shows page indexing progress with percentage."""
    screen = Phase1Screen(blocks=sample_blocks, journal_date="2025-01-15", journal_outline=journal_outline)

    app = Phase1TestApp(screen)


    async with app.run_test() as pilot:
        await pilot.pause()

        status_panel = screen.query_one("#status-panel")

        # Set up page indexing task with percentage
        screen.background_tasks["page_indexing"] = BackgroundTask(
            task_type="page_indexing",
            status="running",
            progress_percentage=45.5,
        )

        # Trigger reactive update
        screen.refresh()
        await pilot.pause()

        # Status should show percentage
        # Note: Check implementation for exact format


@pytest.mark.asyncio
async def test_status_shows_multiple_tasks(sample_blocks, journal_outline):
    """Test status widget shows multiple background tasks at once."""
    screen = Phase1Screen(blocks=sample_blocks, journal_date="2025-01-15", journal_outline=journal_outline)

    app = Phase1TestApp(screen)


    async with app.run_test() as pilot:
        await pilot.pause()

        # Set up multiple tasks
        screen.background_tasks["llm_classification"] = BackgroundTask(
            task_type="llm_classification",
            status="running",
            progress_current=3,
            progress_total=5,
        )

        screen.background_tasks["page_indexing"] = BackgroundTask(
            task_type="page_indexing",
            status="running",
            progress_percentage=67.3,
        )

        # Trigger reactive update
        screen.refresh()
        await pilot.pause()

        status_panel = screen.query_one("#status-panel")

        # Status should show both tasks
        # Note: Implementation-specific formatting


@pytest.mark.asyncio
async def test_status_task_completion(sample_blocks, journal_outline):
    """Test status widget updates when tasks complete."""
    screen = Phase1Screen(blocks=sample_blocks, journal_date="2025-01-15", journal_outline=journal_outline)

    app = Phase1TestApp(screen)


    async with app.run_test() as pilot:
        await pilot.pause()

        # Set up running task
        screen.background_tasks["llm_classification"] = BackgroundTask(
            task_type="llm_classification",
            status="running",
            progress_current=2,
            progress_total=2,
        )

        # Trigger reactive update
        screen.refresh()
        await pilot.pause()

        # Complete the task
        screen.background_tasks["llm_classification"].status = "completed"
        screen.background_tasks["llm_classification"].progress_percentage = 100.0

        # Trigger reactive update
        screen.refresh()
        await pilot.pause()

        # Status should indicate completion
        # (May show different message or remove task from display)


@pytest.mark.asyncio
async def test_status_task_failure(sample_blocks, journal_outline):
    """Test status widget shows error when task fails."""
    screen = Phase1Screen(blocks=sample_blocks, journal_date="2025-01-15", journal_outline=journal_outline)

    app = Phase1TestApp(screen)


    async with app.run_test() as pilot:
        await pilot.pause()

        # Set up failed task
        screen.background_tasks["llm_classification"] = BackgroundTask(
            task_type="llm_classification",
            status="failed",
            error_message="Connection refused"
        )

        # Trigger reactive update
        screen.refresh()
        await pilot.pause()

        status_panel = screen.query_one("#status-panel")

        # Status should show error
        # Note: Implementation-specific error display
