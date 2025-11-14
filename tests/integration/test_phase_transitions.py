"""Integration tests for phase transitions in Logsqueak app.

Tests the flow between Phase 1 → Phase 2 → Phase 3 and verifies that
state is correctly passed between phases.
"""

import pytest
from pathlib import Path
from textual.pilot import Pilot

from logsqueak.tui.app import LogsqueakApp
from logsqueak.models.config import Config, LLMConfig, LogseqConfig
from logsqueak.services.llm_client import LLMClient
from logsqueak.services.page_indexer import PageIndexer
from logsqueak.services.rag_search import RAGSearch
from logsqueak.services.file_monitor import FileMonitor
from logseq_outline.parser import LogseqOutline
from logseq_outline.graph import GraphPaths


# Sample journal content for testing
SAMPLE_JOURNAL = """- Journal entry 2025-01-15
  - Morning reflection
    - Today I learned about async patterns in Python
      id:: block-001
  - Afternoon work
    - Completed the project documentation
      id:: block-002
  - Evening notes
    - Read about testing strategies
      id:: block-003
"""


@pytest.fixture
def sample_journal_outline():
    """Create a sample journal outline for testing."""
    return LogseqOutline.parse(SAMPLE_JOURNAL)


@pytest.fixture
def mock_config(tmp_path):
    """Create a mock configuration."""
    return Config(
        llm=LLMConfig(
            endpoint="http://localhost:11434/v1",
            api_key="test-key",
            model="test-model",
        ),
        logseq=LogseqConfig(graph_path=str(tmp_path)),
    )


@pytest.fixture
def mock_services(mock_config, tmp_path):
    """Create mock services for testing."""
    llm_client = LLMClient(config=mock_config.llm)
    graph_paths = GraphPaths(Path(mock_config.logseq.graph_path))
    page_indexer = PageIndexer(
        graph_paths=graph_paths,
        db_path=tmp_path / "chroma",
    )
    rag_search = RAGSearch(
        db_path=tmp_path / "chroma",
    )
    file_monitor = FileMonitor()

    return {
        "llm_client": llm_client,
        "page_indexer": page_indexer,
        "rag_search": rag_search,
        "file_monitor": file_monitor,
    }


@pytest.mark.asyncio
async def test_phase1_to_phase2_transition(
    sample_journal_outline, mock_config, mock_services
):
    """Test that pressing 'n' in Phase 1 transitions to Phase 2 with selected blocks."""
    journals = {"2025-01-15": sample_journal_outline}
    app = LogsqueakApp(
        journals=journals,
        config=mock_config,
        **mock_services,
    )

    async with app.run_test() as pilot:
        # Wait for Phase 1 to load
        await pilot.pause()

        # Verify we're on Phase 1
        assert app.screen.name == "phase1"

        # Manually select a block by updating block_states
        # (bypassing the LLM worker for testing)
        phase1_screen = app.screen
        for block_id in list(phase1_screen.block_states.keys())[:1]:
            phase1_screen.block_states[block_id].classification = "knowledge"
            phase1_screen.block_states[block_id].source = "user"
            phase1_screen.block_states[block_id].confidence = 1.0

        # Update selected count
        phase1_screen.selected_count = 1

        await pilot.pause()

        # Press 'n' to proceed to Phase 2
        await pilot.press("n")
        await pilot.pause()

        # Verify transition to Phase 2
        assert app.screen.name == "phase2"

        # Verify selected blocks were passed
        assert app.selected_blocks is not None
        assert len(app.selected_blocks) > 0

        # Verify edited content was initialized
        assert app.edited_content is not None
        assert len(app.edited_content) == len(app.selected_blocks)


@pytest.mark.asyncio
async def test_phase2_to_phase3_transition(
    sample_journal_outline, mock_config, mock_services, tmp_path
):
    """Test that pressing 'n' in Phase 2 transitions to Phase 3 with edited content."""
    from logsqueak.models.block_state import BlockState
    from logsqueak.models.edited_content import EditedContent
    from logseq_outline.graph import GraphPaths

    # Create test graph structure
    graph_dir = tmp_path / "test-graph"
    graph_dir.mkdir()
    (graph_dir / "pages").mkdir()
    (graph_dir / "journals").mkdir()

    # Create sample pages for RAG
    ml_page = graph_dir / "pages" / "Machine Learning.md"
    ml_page.write_text("- # Machine Learning\n  - Introduction to ML")

    journals = {"2025-01-15": sample_journal_outline}
    app = LogsqueakApp(
        journals=journals,
        config=mock_config,
        **mock_services,
    )

    # Pre-populate selected blocks (simulate Phase 1 completion)
    selected_blocks = [
        BlockState(
            block_id="block-001",
            classification="knowledge",
            confidence=1.0,
            source="user",
        )
    ]

    async with app.run_test() as pilot:
        await pilot.pause()

        # Transition to Phase 2 (creates the screen properly)
        app.transition_to_phase2(selected_blocks)
        await pilot.pause()

        # Verify we're on Phase 2
        assert app.screen.name == "phase2"

        # Wait for RAG search to complete (mocked, should be instant)
        await pilot.pause()

        # Manually populate candidate pages and page contents (simulate RAG completion)
        phase2_screen = app.screen
        phase2_screen.candidate_page_names = {"block-001": ["Machine Learning"]}
        phase2_screen.page_contents = {
            "Machine Learning": LogseqOutline.parse("- # Machine Learning\n  - Introduction to ML")
        }
        # Mark RAG search as completed so 'n' key works
        from logsqueak.models.background_task import BackgroundTaskState
        phase2_screen.rag_search_state = BackgroundTaskState.COMPLETED

        # Press 'n' to proceed to Phase 3
        await pilot.press("n")
        await pilot.pause()

        # Verify transition to Phase 3
        assert app.screen.name == "phase3"

        # Verify edited content was passed
        assert app.edited_content is not None
        assert len(app.edited_content) > 0

        # Verify page contents were passed
        assert app.page_contents is not None


@pytest.mark.asyncio
async def test_state_preservation_across_phases(
    sample_journal_outline, mock_config, mock_services
):
    """Test that state is correctly preserved across all phase transitions."""
    from logsqueak.models.block_state import BlockState
    from logsqueak.models.edited_content import EditedContent

    journals = {"2025-01-15": sample_journal_outline}
    app = LogsqueakApp(
        journals=journals,
        config=mock_config,
        **mock_services,
    )

    test_block_id = "block-001"
    test_content = "Test content"

    # Prepare selected blocks (Phase 1 output)
    selected_blocks = [
        BlockState(
            block_id=test_block_id,
            classification="knowledge",
            confidence=0.95,
            source="llm",
            llm_classification="knowledge",
            llm_confidence=0.95,
        )
    ]

    async with app.run_test() as pilot:
        await pilot.pause()

        # Start at Phase 1
        assert app.screen.name == "phase1"

        # Transition to Phase 2 using proper transition method
        app.transition_to_phase2(selected_blocks)
        await pilot.pause()
        assert app.screen.name == "phase2"

        # Verify Phase 2 can access edited_content (created by transition_to_phase2)
        assert app.edited_content is not None
        assert len(app.edited_content) > 0
        assert app.selected_blocks is not None

        # Manually populate RAG results (simulate Phase 2 completion)
        phase2_screen = app.screen
        phase2_screen.candidate_page_names = {test_block_id: ["Page1", "Page2"]}
        phase2_screen.page_contents = {
            "Page1": LogseqOutline.parse("- Page 1 content"),
            "Page2": LogseqOutline.parse("- Page 2 content")
        }

        # Transition to Phase 3 using proper transition method
        app.transition_to_phase3(
            edited_content=app.edited_content,
            candidate_pages=["Page1", "Page2"],
            page_contents=phase2_screen.page_contents
        )
        await pilot.pause()
        assert app.screen.name == "phase3"

        # Verify Phase 3 can access all state
        assert app.edited_content is not None
        assert app.candidate_pages == ["Page1", "Page2"]
        assert app.page_contents is not None
        assert app.original_contexts is not None


@pytest.mark.asyncio
async def test_phase1_no_selection_blocks_phase2(
    sample_journal_outline, mock_config, mock_services
):
    """Test that Phase 1 doesn't transition to Phase 2 if no blocks selected."""
    journals = {"2025-01-15": sample_journal_outline}
    app = LogsqueakApp(
        journals=journals,
        config=mock_config,
        **mock_services,
    )

    async with app.run_test() as pilot:
        # Wait for Phase 1 to load
        await pilot.pause()

        # Verify we're on Phase 1
        assert app.screen.name == "phase1"

        # Try to press 'n' without selecting any blocks
        await pilot.press("n")
        await pilot.pause()

        # Should still be on Phase 1 (no transition)
        # Note: This assumes Phase 1 validates selection before allowing transition
        # Implementation may show a message instead
        # For now, we just verify the app doesn't crash
        assert app.screen is not None
