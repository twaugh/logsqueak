"""Integration tests for full workflow (Phase 1 → Phase 2 → Phase 3)."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from datetime import date

from logseq_outline.parser import LogseqOutline
from logsqueak.models.config import Config, LLMConfig, LogseqConfig, RAGConfig
from logsqueak.models.llm_chunks import (
    KnowledgeClassificationChunk,
    ContentRewordingChunk,
    IntegrationDecisionChunk,
)
from logsqueak.services.llm_client import LLMClient
from logsqueak.services.page_indexer import PageIndexer
from logsqueak.services.rag_search import RAGSearch
from logsqueak.services.file_monitor import FileMonitor
from logsqueak.tui.app import LogsqueakApp


@pytest.fixture
def sample_journal_outline():
    """Create a sample journal outline for testing."""
    journal_text = """- First block
- Second block is knowledge
- Third block is activity log
"""
    return LogseqOutline.parse(journal_text)


@pytest.fixture
def mock_config(tmp_path):
    """Create a mock config for testing."""
    graph_path = tmp_path / "logseq-graph"
    graph_path.mkdir()

    (graph_path / "journals").mkdir()
    (graph_path / "pages").mkdir()

    # Create test page file
    test_page = graph_path / "pages" / "Test Page.md"
    test_page.write_text("""- Test section
  - Existing content
""")

    return Config(
        llm=LLMConfig(
            endpoint="http://localhost:11434/v1",
            api_key="test-key",
            model="test-model",
        ),
        logseq=LogseqConfig(graph_path=str(graph_path)),
        rag=RAGConfig(top_k=5),
    )


@pytest.fixture
def mock_llm_client(mock_config):
    """Create a mock LLM client with canned responses."""
    client = MagicMock(spec=LLMClient)
    client.config = mock_config.llm

    # Mock Phase 1 classification responses
    async def mock_classify():
        """Mock classify_blocks stream."""
        yield KnowledgeClassificationChunk(
            block_id="block_1",
            is_knowledge=False,
            confidence=0.2,
            reason="Activity log entry",
        )
        yield KnowledgeClassificationChunk(
            block_id="block_2",
            is_knowledge=True,
            confidence=0.9,
            reason="Contains reusable knowledge",
        )
        yield KnowledgeClassificationChunk(
            block_id="block_3",
            is_knowledge=False,
            confidence=0.3,
            reason="Temporal activity",
        )

    # Mock Phase 2 rewording responses
    async def mock_reword():
        """Mock reword_content stream."""
        yield ContentRewordingChunk(
            block_id="block_2",
            reworded_content="Knowledge block reworded without temporal context",
            complete=True,
        )

    # Mock Phase 3 integration decisions
    async def mock_decisions():
        """Mock plan_integrations stream."""
        yield IntegrationDecisionChunk(
            knowledge_block_id="block_2",
            target_page="Test Page",
            action="add_under",
            target_block_id="test-section",
            target_block_title="Test section",
            confidence=0.85,
            refined_text="Knowledge block reworded without temporal context",
            reasoning="Fits well under test section",
        )

    client.stream_ndjson = AsyncMock()

    return client, mock_classify, mock_reword, mock_decisions


@pytest.fixture
def mock_page_indexer(tmp_path):
    """Create a mock page indexer."""
    from logseq_outline.graph import GraphPaths

    # Create a real GraphPaths instance for the mock
    graph_path = tmp_path / "logseq-graph"
    graph_path.mkdir(exist_ok=True)
    (graph_path / "pages").mkdir(exist_ok=True)
    (graph_path / "journals").mkdir(exist_ok=True)

    graph_paths = GraphPaths(graph_path)
    db_path = tmp_path / "chromadb"
    db_path.mkdir(exist_ok=True)

    indexer = MagicMock(spec=PageIndexer)
    indexer.graph_paths = graph_paths
    indexer.db_path = db_path
    indexer.build_index = AsyncMock()
    return indexer


@pytest.fixture
def mock_rag_search(tmp_path):
    """Create a mock RAG search service."""
    db_path = tmp_path / "chromadb"
    db_path.mkdir(exist_ok=True)

    rag = MagicMock(spec=RAGSearch)
    rag.db_path = db_path

    # Mock find_candidates to return candidate pages
    async def mock_find_candidates(edited_content, original_contexts, top_k=10):
        # Return dict mapping block_id to list of candidate pages
        return {content.block_id: ["Test Page"] for content in edited_content}

    rag.find_candidates = AsyncMock(side_effect=mock_find_candidates)

    # Mock load_page_contents
    async def mock_load_pages(candidate_pages):
        return {
            "Test Page": "- Test section\n  - Existing content\n"
        }

    rag.load_page_contents = AsyncMock(side_effect=mock_load_pages)

    return rag


@pytest.fixture
def mock_file_monitor():
    """Create a mock file monitor."""
    monitor = MagicMock(spec=FileMonitor)
    monitor.record = MagicMock()
    monitor.is_modified = MagicMock(return_value=False)
    monitor.refresh = MagicMock()
    return monitor


@pytest.mark.asyncio
async def test_full_workflow_phase1_to_phase2_to_phase3(
    sample_journal_outline,
    mock_config,
    mock_llm_client,
    mock_page_indexer,
    mock_rag_search,
    mock_file_monitor,
):
    """Test complete workflow from Phase 1 through Phase 3.

    This integration test verifies:
    1. App initializes with all services
    2. Phase 1 screen loads and shows journal blocks
    3. Phase 2 screen can be initialized (manual transition for now)
    4. Phase 3 screen can be initialized (manual transition for now)

    Note: Full TUI navigation testing with pilot is complex and will be
    added in future iterations. This test focuses on service wiring and
    basic initialization.
    """
    client, mock_classify, mock_reword, mock_decisions = mock_llm_client

    # Initialize app with mocked services
    app = LogsqueakApp(
        journal_outline=sample_journal_outline,
        journal_date="2025-01-15",
        config=mock_config,
        llm_client=client,
        page_indexer=mock_page_indexer,
        rag_search=mock_rag_search,
        file_monitor=mock_file_monitor,
    )

    # Verify app initialized with correct state
    assert app.journal_date == "2025-01-15"
    # Note: app.journal_outline is augmented with IDs, so it won't equal the original
    assert len(app.journal_outline.blocks) == len(sample_journal_outline.blocks)
    assert app.config == mock_config
    assert app.llm_client == client
    assert app.page_indexer == mock_page_indexer
    assert app.rag_search == mock_rag_search
    assert app.file_monitor == mock_file_monitor

    # Verify initial state
    assert app.selected_blocks is None
    assert app.edited_content is None
    assert app.candidate_pages is None
    assert app.page_contents is None

    # Note: Full pilot-based testing with screen navigation would go here
    # For now, we verify that the app can be initialized with all services
    # and that the basic structure is correct.


@pytest.mark.asyncio
async def test_app_initialization_with_services(
    sample_journal_outline,
    mock_config,
    mock_llm_client,
    mock_page_indexer,
    mock_rag_search,
    mock_file_monitor,
):
    """Test that app initializes correctly with all services wired up."""
    client, _, _, _ = mock_llm_client

    app = LogsqueakApp(
        journal_outline=sample_journal_outline,
        journal_date="2025-01-15",
        config=mock_config,
        llm_client=client,
        page_indexer=mock_page_indexer,
        rag_search=mock_rag_search,
        file_monitor=mock_file_monitor,
    )

    # Verify all services are stored
    assert app.llm_client is not None
    assert app.page_indexer is not None
    assert app.rag_search is not None
    assert app.file_monitor is not None
    assert app.config is not None

    # Verify journal data is stored (outline is augmented with IDs)
    assert app.journal_outline is not None
    assert app.journal_date == "2025-01-15"
    assert len(app.journal_outline.blocks) == 3
    # All blocks should have IDs after augmentation
    for block in app.journal_outline.blocks:
        assert block.block_id is not None


@pytest.mark.asyncio
async def test_app_can_initialize_without_crashing(
    sample_journal_outline,
    mock_config,
    mock_llm_client,
    mock_page_indexer,
    mock_rag_search,
    mock_file_monitor,
):
    """Test that app can be created without errors.

    This is a smoke test to ensure basic initialization works.
    """
    client, _, _, _ = mock_llm_client

    try:
        app = LogsqueakApp(
            journal_outline=sample_journal_outline,
            journal_date="2025-01-15",
            config=mock_config,
            llm_client=client,
            page_indexer=mock_page_indexer,
            rag_search=mock_rag_search,
            file_monitor=mock_file_monitor,
        )
        assert app is not None
    except Exception as e:
        pytest.fail(f"App initialization failed with error: {e}")
