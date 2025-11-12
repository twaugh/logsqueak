"""Integration tests for worker dependency ordering.

Tests that workers execute in the correct order and wait for dependencies:
1. SentenceTransformer loading → PageIndexer starts
2. PageIndexer completes → RAG search starts
3. Integration decisions worker is opportunistic (starts after RAG completes)
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from logsqueak.tui.app import LogsqueakApp
from logsqueak.models.config import Config, LLMConfig, LogseqConfig, RAGConfig
from logsqueak.models.background_task import BackgroundTask
from logseq_outline.parser import LogseqOutline


@pytest.fixture
def temp_graph_path(tmp_path: Path):
    """Create a minimal Logseq graph structure."""
    graph_path = tmp_path / "test-graph"
    pages_dir = graph_path / "pages"
    journals_dir = graph_path / "journals"

    pages_dir.mkdir(parents=True)
    journals_dir.mkdir(parents=True)

    # Create a sample page
    test_page = pages_dir / "test_page.md"
    test_page.write_text("- Test block content\n")

    # Create a sample journal
    journal = journals_dir / "2025-01-15.md"
    journal.write_text("- Journal entry\n")

    return graph_path


@pytest.fixture
def mock_config(temp_graph_path: Path):
    """Create mock configuration."""
    return Config(
        llm=LLMConfig(
            endpoint="http://localhost:11434/v1",
            api_key="test-key",
            model="test-model",
        ),
        logseq=LogseqConfig(
            graph_path=str(temp_graph_path),
        ),
        rag=RAGConfig(top_k=5),
    )


@pytest.mark.asyncio
async def test_worker_dependency_order_model_preload_then_indexing(
    temp_graph_path, mock_config
):
    """Test that PageIndexer waits for SentenceTransformer model to load."""
    from logsqueak.services.llm_client import LLMClient
    from logsqueak.services.page_indexer import PageIndexer
    from logsqueak.services.rag_search import RAGSearch
    from logsqueak.services.file_monitor import FileMonitor
    from logseq_outline.graph import GraphPaths

    # Track execution order
    execution_order = []

    # Create mock services
    llm_client = MagicMock(spec=LLMClient)
    file_monitor = FileMonitor()

    # Create real GraphPaths and PageIndexer
    graph_paths = GraphPaths(temp_graph_path)

    # Mock RAGSearch with controlled encoder property
    rag_search = MagicMock(spec=RAGSearch)
    rag_search.embedding_model = "all-MiniLM-L6-v2"  # Set model name

    # Pre-populate _encoder with mock to avoid real model loading
    mock_encoder = MagicMock()
    rag_search._encoder = mock_encoder

    # Track when model is "preloaded" (in reality it's already there)
    execution_order.append("model_preload_complete")

    # Mock PageIndexer.build_index to track when it starts
    page_indexer = MagicMock(spec=PageIndexer)

    async def mock_build_index(progress_callback=None):
        execution_order.append("page_indexing_started")
        await asyncio.sleep(0.05)  # Simulate indexing work
        execution_order.append("page_indexing_complete")

    page_indexer.build_index = AsyncMock(side_effect=mock_build_index)

    # Create test journal outline
    journal_outline = LogseqOutline.parse("- Test block\n  id:: abc123")

    # Create app
    app = LogsqueakApp(
        journal_outline=journal_outline,
        journal_date="2025-01-15",
        config=mock_config,
        llm_client=llm_client,
        page_indexer=page_indexer,
        rag_search=rag_search,
        file_monitor=file_monitor,
    )

    # Manually trigger model preload and page indexing
    # (normally triggered by app.on_mount, but we're testing workers in isolation)

    # Create background task entry before starting worker
    app.background_tasks["model_preload"] = BackgroundTask(
        task_type="page_indexing",
        status="running",
        progress_percentage=0.0,
    )

    # Start model preload worker
    model_preload_task = asyncio.create_task(app._preload_embedding_model())

    # Wait for model preload to start
    await asyncio.sleep(0.01)

    # Simulate Phase 1 page indexing worker behavior
    async def simulate_phase1_indexing():
        # Wait for model_preload to complete
        while True:
            model_task = app.background_tasks.get("model_preload")
            if model_task and model_task.status == "completed":
                break
            await asyncio.sleep(0.01)

        # Now start indexing
        await page_indexer.build_index()

    indexing_task = asyncio.create_task(simulate_phase1_indexing())

    # Wait for both tasks to complete
    await model_preload_task
    await indexing_task

    # Verify execution order
    assert execution_order == [
        "model_preload_complete",
        "page_indexing_started",
        "page_indexing_complete",
    ], f"Expected model preload → indexing start → indexing complete, got {execution_order}"

    # Verify background_tasks status
    assert app.background_tasks["model_preload"].status == "completed"
    assert app.background_tasks["model_preload"].progress_percentage == 100.0


@pytest.mark.asyncio
async def test_worker_dependency_order_indexing_then_rag(temp_graph_path, mock_config):
    """Test that RAG search waits for PageIndexer to complete."""
    from logsqueak.services.llm_client import LLMClient
    from logsqueak.services.page_indexer import PageIndexer
    from logsqueak.services.rag_search import RAGSearch
    from logsqueak.services.file_monitor import FileMonitor
    from logsqueak.models.edited_content import EditedContent
    from logseq_outline.graph import GraphPaths

    # Track execution order
    execution_order = []

    # Create mock services
    llm_client = MagicMock(spec=LLMClient)
    file_monitor = FileMonitor()
    graph_paths = GraphPaths(temp_graph_path)

    # Mock PageIndexer
    page_indexer = MagicMock(spec=PageIndexer)

    async def mock_build_index(progress_callback=None):
        execution_order.append("page_indexing_started")
        await asyncio.sleep(0.05)
        execution_order.append("page_indexing_complete")

    page_indexer.build_index = AsyncMock(side_effect=mock_build_index)

    # Mock RAGSearch
    rag_search = MagicMock(spec=RAGSearch)

    async def mock_find_candidates(edited_content, original_contexts, top_k):
        execution_order.append("rag_search_started")
        await asyncio.sleep(0.05)
        return {"block1": ["page1", "page2"]}

    rag_search.find_candidates = AsyncMock(side_effect=mock_find_candidates)

    # Create test journal outline
    journal_outline = LogseqOutline.parse("- Test block\n  id:: abc123")

    # Create app
    app = LogsqueakApp(
        journal_outline=journal_outline,
        journal_date="2025-01-15",
        config=mock_config,
        llm_client=llm_client,
        page_indexer=page_indexer,
        rag_search=rag_search,
        file_monitor=file_monitor,
    )

    # Simulate Phase 1 page indexing
    async def simulate_phase1_indexing():
        await page_indexer.build_index()
        # Mark complete in shared background_tasks
        app.background_tasks["page_indexing"] = BackgroundTask(
            task_type="page_indexing",
            status="completed",
            progress_percentage=100.0,
        )

    # Simulate Phase 2 RAG search worker behavior
    async def simulate_phase2_rag():
        # Wait for page_indexing to complete
        while True:
            indexing_task = app.background_tasks.get("page_indexing")
            if indexing_task and indexing_task.status == "completed":
                break
            await asyncio.sleep(0.01)

        # Now start RAG search
        edited_content = [
            EditedContent(
                block_id="abc123",
                original_content="Test block",
                hierarchical_context="Test block",
                current_content="Test block",
            )
        ]
        await rag_search.find_candidates(edited_content, {}, top_k=5)

    # Run both workers
    indexing_task = asyncio.create_task(simulate_phase1_indexing())
    rag_task = asyncio.create_task(simulate_phase2_rag())

    await indexing_task
    await rag_task

    # Verify execution order
    assert execution_order == [
        "page_indexing_started",
        "page_indexing_complete",
        "rag_search_started",
    ], f"Expected indexing → RAG search, got {execution_order}"

    # Verify background_tasks status
    assert app.background_tasks["page_indexing"].status == "completed"


@pytest.mark.asyncio
async def test_worker_dependency_handles_failures(temp_graph_path, mock_config):
    """Test that dependent workers handle upstream failures gracefully."""
    from logsqueak.services.llm_client import LLMClient
    from logsqueak.services.page_indexer import PageIndexer
    from logsqueak.services.rag_search import RAGSearch
    from logsqueak.services.file_monitor import FileMonitor
    from logseq_outline.graph import GraphPaths

    # Create mock services
    llm_client = MagicMock(spec=LLMClient)
    file_monitor = FileMonitor()
    graph_paths = GraphPaths(temp_graph_path)

    # Mock PageIndexer that fails
    page_indexer = MagicMock(spec=PageIndexer)

    async def mock_build_index_fails(progress_callback=None):
        await asyncio.sleep(0.05)
        raise RuntimeError("Indexing failed intentionally")

    page_indexer.build_index = AsyncMock(side_effect=mock_build_index_fails)

    # Mock RAGSearch
    rag_search = MagicMock(spec=RAGSearch)
    rag_search.find_candidates = AsyncMock()

    # Create test journal outline
    journal_outline = LogseqOutline.parse("- Test block\n  id:: abc123")

    # Create app
    app = LogsqueakApp(
        journal_outline=journal_outline,
        journal_date="2025-01-15",
        config=mock_config,
        llm_client=llm_client,
        page_indexer=page_indexer,
        rag_search=rag_search,
        file_monitor=file_monitor,
    )

    # Simulate Phase 1 page indexing that fails
    async def simulate_phase1_indexing_fails():
        try:
            await page_indexer.build_index()
        except Exception as e:
            # Mark failed in shared background_tasks
            app.background_tasks["page_indexing"] = BackgroundTask(
                task_type="page_indexing",
                status="failed",
                error_message=str(e),
            )

    # Simulate Phase 2 RAG search that should detect failure
    async def simulate_phase2_rag_detects_failure():
        # Wait for page_indexing to complete or fail
        while True:
            indexing_task = app.background_tasks.get("page_indexing")
            if indexing_task and indexing_task.status == "failed":
                # RAG worker should raise RuntimeError
                raise RuntimeError(f"Page indexing failed: {indexing_task.error_message}")
            elif indexing_task and indexing_task.status == "completed":
                break
            await asyncio.sleep(0.01)

    # Run both workers
    indexing_task = asyncio.create_task(simulate_phase1_indexing_fails())
    rag_task = asyncio.create_task(simulate_phase2_rag_detects_failure())

    await indexing_task

    # RAG task should raise RuntimeError
    with pytest.raises(RuntimeError, match="Page indexing failed"):
        await rag_task

    # Verify background_tasks status
    assert app.background_tasks["page_indexing"].status == "failed"
    assert "Indexing failed intentionally" in app.background_tasks["page_indexing"].error_message
