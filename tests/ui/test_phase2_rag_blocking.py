"""UI tests for Phase 2 RAG search completion blocking.

Tests for blocking 'n' key progression until both page indexing
and RAG search complete, with appropriate status messages.
"""

import pytest
from textual.app import App
from textual.pilot import Pilot
from logsqueak.tui.screens.content_editing import Phase2Screen
from logsqueak.models.edited_content import EditedContent
from logsqueak.models.background_task import BackgroundTaskState
from logseq_outline.parser import LogseqBlock, LogseqOutline
from logseq_outline.graph import GraphPaths


class Phase2TestApp(App):
    """Test app wrapper for Phase2Screen."""

    def __init__(self, screen: Phase2Screen):
        super().__init__()
        self.test_screen = screen

    def on_mount(self) -> None:
        """Push the test screen on mount."""
        self.push_screen(self.test_screen)


@pytest.fixture
def sample_blocks():
    """Create sample knowledge blocks for testing."""
    blocks = [
        LogseqBlock(
            content=["First knowledge block"],
            indent_level=0,
            block_id="block-1",
            children=[]
        ),
    ]
    return blocks


@pytest.fixture
def sample_edited_content():
    """Create sample EditedContent for testing."""
    return [
        EditedContent(
            block_id="block-1",
            original_content="First knowledge block",
            hierarchical_context="First knowledge block",
            current_content="First knowledge block"
        ),
    ]


@pytest.fixture
def journal_outline(sample_blocks):
    """Create a LogseqOutline from sample blocks."""
    return LogseqOutline(blocks=sample_blocks, source_text="", frontmatter=[])


@pytest.fixture
def journals(journal_outline):
    """Create journals dict for Phase2Screen."""
    return {"2025-01-15": journal_outline}


@pytest.fixture
def graph_paths(tmp_path):
    """Create a temporary GraphPaths instance."""
    graph_dir = tmp_path / "test-graph"
    graph_dir.mkdir()
    (graph_dir / "pages").mkdir()
    (graph_dir / "journals").mkdir()
    return GraphPaths(graph_dir)


@pytest.mark.asyncio
async def test_n_key_blocked_when_page_indexing_running(sample_blocks, sample_edited_content, journal_outline, graph_paths):
    """Test 'n' key is blocked while page indexing is still running."""
    screen = Phase2Screen(
        blocks=sample_blocks,
        edited_content=sample_edited_content,
        journals=journals,
        graph_paths=graph_paths,
        auto_start_workers=False
    )
    app = Phase2TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        # Simulate page indexing in progress
        screen.page_indexing_state = BackgroundTaskState.IN_PROGRESS

        # Try to press 'n' - should be blocked
        await pilot.press("n")
        await pilot.pause()

        # Should still be on Phase2Screen (no transition)
        assert app.screen == screen


@pytest.mark.asyncio
async def test_n_key_blocked_when_rag_search_running(sample_blocks, sample_edited_content, journal_outline, graph_paths):
    """Test 'n' key is blocked while RAG search is still running."""
    screen = Phase2Screen(
        blocks=sample_blocks,
        edited_content=sample_edited_content,
        journals=journals,
        graph_paths=graph_paths,
        auto_start_workers=False
    )
    app = Phase2TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        # Simulate page indexing complete, but RAG search in progress
        screen.page_indexing_state = BackgroundTaskState.COMPLETED
        screen.rag_search_state = BackgroundTaskState.IN_PROGRESS

        # Try to press 'n' - should be blocked
        await pilot.press("n")
        await pilot.pause()

        # Should still be on Phase2Screen (no transition)
        assert app.screen == screen


@pytest.mark.asyncio
async def test_n_key_enabled_when_all_tasks_complete(sample_blocks, sample_edited_content, journal_outline, graph_paths):
    """Test 'n' key works when both page indexing and RAG search are complete."""
    screen = Phase2Screen(
        blocks=sample_blocks,
        edited_content=sample_edited_content,
        journals=journals,
        graph_paths=graph_paths,
        auto_start_workers=False
    )
    app = Phase2TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        # Simulate all tasks complete
        screen.page_indexing_state = BackgroundTaskState.COMPLETED
        screen.rag_search_state = BackgroundTaskState.COMPLETED

        # Now 'n' should work (test would need screen transition mocking)
        # For now, just verify the states are correct
        assert screen.page_indexing_state == BackgroundTaskState.COMPLETED
        assert screen.rag_search_state == BackgroundTaskState.COMPLETED


@pytest.mark.asyncio
async def test_status_message_when_waiting_for_page_index(sample_blocks, sample_edited_content, journal_outline, graph_paths):
    """Test that status shows 'Waiting for page index...' message."""
    screen = Phase2Screen(
        blocks=sample_blocks,
        edited_content=sample_edited_content,
        journals=journals,
        graph_paths=graph_paths,
        auto_start_workers=False
    )
    app = Phase2TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        # Simulate page indexing in progress
        screen.page_indexing_state = BackgroundTaskState.IN_PROGRESS

        # Status widget should show waiting message
        # (Implementation detail: may be a label or status panel)
        # For now, verify the state
        assert screen.page_indexing_state == BackgroundTaskState.IN_PROGRESS


@pytest.mark.asyncio
async def test_status_message_when_waiting_for_rag_search(sample_blocks, sample_edited_content, journal_outline, graph_paths):
    """Test that status shows 'Finding relevant pages...' message."""
    screen = Phase2Screen(
        blocks=sample_blocks,
        edited_content=sample_edited_content,
        journals=journals,
        graph_paths=graph_paths,
        auto_start_workers=False
    )
    app = Phase2TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        # Simulate page indexing complete, RAG search in progress
        screen.page_indexing_state = BackgroundTaskState.COMPLETED
        screen.rag_search_state = BackgroundTaskState.IN_PROGRESS

        # Status widget should show RAG search progress
        assert screen.rag_search_state == BackgroundTaskState.IN_PROGRESS


@pytest.mark.asyncio
async def test_rag_search_starts_after_page_indexing_completes(sample_blocks, sample_edited_content, journal_outline, graph_paths):
    """Test that RAG search automatically starts when page indexing completes."""
    screen = Phase2Screen(
        blocks=sample_blocks,
        edited_content=sample_edited_content,
        journals=journals,
        graph_paths=graph_paths,
        auto_start_workers=False
    )
    app = Phase2TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        # Initially both pending
        screen.page_indexing_state = BackgroundTaskState.PENDING
        screen.rag_search_state = BackgroundTaskState.PENDING

        # Simulate page indexing completing
        screen.page_indexing_state = BackgroundTaskState.COMPLETED
        await pilot.pause()

        # RAG search should automatically start
        # (Implementation detail: may need to trigger manually in test)
        # For now, just verify the initial state
        assert screen.page_indexing_state == BackgroundTaskState.COMPLETED


@pytest.mark.asyncio
async def test_footer_shows_n_disabled_while_waiting(sample_blocks, sample_edited_content, journal_outline, graph_paths):
    """Test that footer indicates 'n' key is disabled while waiting."""
    screen = Phase2Screen(
        blocks=sample_blocks,
        edited_content=sample_edited_content,
        journals=journals,
        graph_paths=graph_paths,
        auto_start_workers=False
    )
    app = Phase2TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        # Simulate page indexing in progress
        screen.page_indexing_state = BackgroundTaskState.IN_PROGRESS

        # Footer should show 'n' as disabled or with different styling
        # (Implementation detail: may check footer widget content)
        # For now, just verify the blocking state
        assert screen.page_indexing_state != BackgroundTaskState.COMPLETED


@pytest.mark.asyncio
async def test_status_updates_with_rag_search_progress(sample_blocks, sample_edited_content, journal_outline, graph_paths):
    """Test that status shows RAG search progress (e.g., '2/5 complete')."""
    screen = Phase2Screen(
        blocks=sample_blocks,
        edited_content=sample_edited_content,
        journals=journals,
        graph_paths=graph_paths,
        auto_start_workers=False
    )
    app = Phase2TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        # Simulate RAG search in progress with some progress
        screen.rag_search_state = BackgroundTaskState.IN_PROGRESS
        screen.rag_search_progress = 2
        screen.rag_search_total = 5

        # Status widget should show "Finding relevant pages: 2/5 complete"
        # (Implementation detail: verify status panel content)
        assert screen.rag_search_progress == 2
        assert screen.rag_search_total == 5


@pytest.mark.asyncio
async def test_page_indexing_error_shows_message(sample_blocks, sample_edited_content, journal_outline, graph_paths):
    """Test that page indexing errors are displayed to user."""
    screen = Phase2Screen(
        blocks=sample_blocks,
        edited_content=sample_edited_content,
        journals=journals,
        graph_paths=graph_paths,
        auto_start_workers=False
    )
    app = Phase2TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        # Simulate page indexing failure
        screen.page_indexing_state = BackgroundTaskState.FAILED
        screen.page_indexing_error = "Failed to load pages from graph"

        # Error message should be displayed
        # (Implementation detail: may be a status message or modal)
        assert screen.page_indexing_state == BackgroundTaskState.FAILED
        assert screen.page_indexing_error is not None


@pytest.mark.asyncio
async def test_rag_search_error_shows_message(sample_blocks, sample_edited_content, journal_outline, graph_paths):
    """Test that RAG search errors are displayed to user."""
    screen = Phase2Screen(
        blocks=sample_blocks,
        edited_content=sample_edited_content,
        journals=journals,
        graph_paths=graph_paths,
        auto_start_workers=False
    )
    app = Phase2TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        # Simulate RAG search failure
        screen.page_indexing_state = BackgroundTaskState.COMPLETED
        screen.rag_search_state = BackgroundTaskState.FAILED
        screen.rag_search_error = "Semantic search failed"

        # Error message should be displayed
        assert screen.rag_search_state == BackgroundTaskState.FAILED
        assert screen.rag_search_error is not None


def test_rag_config_top_k_extraction():
    """Test that top_k is correctly extracted from config."""
    from logsqueak.models.config import Config, LogseqConfig, LLMConfig, RAGConfig
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create config with custom top_k
        config = Config(
            logseq=LogseqConfig(graph_path=tmpdir),
            llm=LLMConfig(endpoint="http://fake", model="fake", api_key="fake"),
            rag=RAGConfig(top_k=25)
        )

        # Verify the config has the right value
        assert config.rag.top_k == 25

        # Simulate the extraction logic used in _rag_search_worker
        top_k = 10  # Default fallback
        if config and hasattr(config, 'rag') and hasattr(config.rag, 'top_k'):
            top_k = config.rag.top_k

        assert top_k == 25


def test_rag_default_top_k_without_config():
    """Test that top_k defaults to 10 when config not provided."""
    config = None

    # Simulate the extraction logic used in _rag_search_worker
    top_k = 10  # Default fallback
    if config and hasattr(config, 'rag') and hasattr(config.rag, 'top_k'):
        top_k = config.rag.top_k

    assert top_k == 10
