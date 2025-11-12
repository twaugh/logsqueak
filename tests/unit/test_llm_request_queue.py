"""Unit tests for LLM request queue serialization."""

import pytest
import pytest_asyncio
import asyncio
from unittest.mock import AsyncMock, Mock, patch

from logsqueak.tui.app import LogsqueakApp, LLMRequestPriority
from logsqueak.models.config import Config, LLMConfig, LogseqConfig, RAGConfig
from logsqueak.services.llm_client import LLMClient
from logsqueak.services.page_indexer import PageIndexer
from logsqueak.services.rag_search import RAGSearch
from logsqueak.services.file_monitor import FileMonitor
from logseq_outline.parser import LogseqOutline
from logseq_outline.graph import GraphPaths


class TestLLMRequestQueue:
    """Test LLM request queue serialization."""

    @pytest.fixture
    def mock_config(self, tmp_path):
        """Create mock configuration."""
        # Create graph directory
        graph_path = tmp_path / "test-graph"
        graph_path.mkdir(parents=True, exist_ok=True)

        return Config(
            llm=LLMConfig(
                endpoint="http://localhost:11434/v1",
                api_key="test-key",
                model="test-model"
            ),
            logseq=LogseqConfig(
                graph_path=str(graph_path)
            ),
            rag=RAGConfig()
        )

    @pytest.fixture
    def mock_journal_outline(self):
        """Create mock journal outline."""
        markdown = """- Test block
  id:: abc123
- Another block
  id:: def456"""
        return LogseqOutline.parse(markdown)

    @pytest.fixture
    def mock_llm_client(self, mock_config):
        """Create mock LLM client."""
        return LLMClient(mock_config.llm)

    @pytest.fixture
    def mock_page_indexer(self, tmp_path):
        """Create mock page indexer."""
        graph_paths = GraphPaths(tmp_path / "test-graph")
        return PageIndexer(
            graph_paths=graph_paths,
            db_path=tmp_path / "chroma",
            embedding_model="all-mpnet-base-v2"
        )

    @pytest.fixture
    def mock_rag_search(self, tmp_path):
        """Create mock RAG search."""
        return RAGSearch(
            db_path=tmp_path / "chroma",
            embedding_model="all-mpnet-base-v2"
        )

    @pytest.fixture
    def mock_file_monitor(self):
        """Create mock file monitor."""
        return FileMonitor()

    @pytest_asyncio.fixture
    async def app(
        self,
        mock_config,
        mock_journal_outline,
        mock_llm_client,
        mock_page_indexer,
        mock_rag_search,
        mock_file_monitor
    ):
        """Create test app instance."""
        app = LogsqueakApp(
            journal_outline=mock_journal_outline,
            journal_date="2025-01-15",
            config=mock_config,
            llm_client=mock_llm_client,
            page_indexer=mock_page_indexer,
            rag_search=mock_rag_search,
            file_monitor=mock_file_monitor
        )
        # Don't actually mount the app (which would start UI)
        # Just initialize the queue consumer manually
        app._llm_queue_consumer_started = False
        return app

    @pytest.mark.asyncio
    async def test_concurrent_requests_execute_sequentially(self, app):
        """Test that concurrent calls to acquire_llm_slot execute sequentially."""
        # Start the queue consumer
        app._start_llm_queue_consumer()

        # Track execution order
        execution_order = []

        async def mock_request(request_id: str, priority: LLMRequestPriority, duration: float = 0.1):
            """Simulate an LLM request."""
            await app.acquire_llm_slot(request_id, priority)
            try:
                execution_order.append(f"{request_id}_start")
                await asyncio.sleep(duration)
                execution_order.append(f"{request_id}_end")
            finally:
                app.release_llm_slot(request_id)

        # Launch two concurrent requests
        await asyncio.gather(
            mock_request("request1", LLMRequestPriority.CLASSIFICATION, 0.1),
            mock_request("request2", LLMRequestPriority.REWORDING, 0.1)
        )

        # Verify they executed sequentially (start -> end -> start -> end)
        # Not concurrently (start -> start -> end -> end)
        assert execution_order == [
            "request1_start",
            "request1_end",
            "request2_start",
            "request2_end"
        ]

    @pytest.mark.asyncio
    async def test_second_request_waits_for_first_to_complete(self, app):
        """Test that second request waits for first to complete."""
        app._start_llm_queue_consumer()

        first_started = asyncio.Event()
        first_can_complete = asyncio.Event()
        second_started = asyncio.Event()

        async def first_request():
            """First request that blocks until signaled."""
            await app.acquire_llm_slot("first", LLMRequestPriority.CLASSIFICATION)
            try:
                first_started.set()
                await first_can_complete.wait()
            finally:
                app.release_llm_slot("first")

        async def second_request():
            """Second request that should wait."""
            await app.acquire_llm_slot("second", LLMRequestPriority.REWORDING)
            try:
                second_started.set()
            finally:
                app.release_llm_slot("second")

        # Start both requests
        task1 = asyncio.create_task(first_request())
        task2 = asyncio.create_task(second_request())

        # Wait for first to start
        await first_started.wait()

        # Give second a chance to start (but it shouldn't yet)
        await asyncio.sleep(0.2)
        assert not second_started.is_set(), "Second request started before first completed"

        # Allow first to complete
        first_can_complete.set()

        # Wait for second to start
        await asyncio.wait_for(second_started.wait(), timeout=1.0)

        # Wait for both to complete
        await asyncio.gather(task1, task2)

    @pytest.mark.asyncio
    async def test_errors_in_first_request_dont_block_second(self, app):
        """Test that errors in first request don't block second request."""
        app._start_llm_queue_consumer()

        execution_order = []

        async def failing_request():
            """Request that fails after acquiring slot."""
            await app.acquire_llm_slot("failing", LLMRequestPriority.CLASSIFICATION)
            try:
                execution_order.append("failing_start")
                raise RuntimeError("Test error")
            finally:
                app.release_llm_slot("failing")
                execution_order.append("failing_release")

        async def successful_request():
            """Request that should succeed after first fails."""
            await app.acquire_llm_slot("successful", LLMRequestPriority.REWORDING)
            try:
                execution_order.append("successful_start")
            finally:
                app.release_llm_slot("successful")
                execution_order.append("successful_release")

        # Run failing request first, then successful
        try:
            await failing_request()
        except RuntimeError:
            pass  # Expected

        await successful_request()

        # Verify both executed and second wasn't blocked
        assert execution_order == [
            "failing_start",
            "failing_release",
            "successful_start",
            "successful_release"
        ]

    @pytest.mark.asyncio
    async def test_multiple_workers_can_queue_without_deadlock(self, app):
        """Test that multiple workers can queue requests without deadlock."""
        app._start_llm_queue_consumer()

        execution_count = {"count": 0}

        async def worker_request(worker_id: int, priority: LLMRequestPriority):
            """Simulate a worker making an LLM request."""
            request_id = f"worker_{worker_id}"
            await app.acquire_llm_slot(request_id, priority)
            try:
                execution_count["count"] += 1
                await asyncio.sleep(0.01)  # Simulate work
            finally:
                app.release_llm_slot(request_id)

        # Launch 5 workers with different priorities
        workers = [
            worker_request(1, LLMRequestPriority.CLASSIFICATION),
            worker_request(2, LLMRequestPriority.REWORDING),
            worker_request(3, LLMRequestPriority.INTEGRATION),
            worker_request(4, LLMRequestPriority.CLASSIFICATION),
            worker_request(5, LLMRequestPriority.REWORDING),
        ]

        # Wait for all workers to complete (with timeout to detect deadlock)
        await asyncio.wait_for(
            asyncio.gather(*workers),
            timeout=5.0
        )

        # Verify all workers executed
        assert execution_count["count"] == 5

    @pytest.mark.asyncio
    async def test_priority_ordering(self, app):
        """Test that requests are processed in priority order."""
        app._start_llm_queue_consumer()

        # Block the queue with a long-running request
        first_can_complete = asyncio.Event()

        async def blocking_request():
            await app.acquire_llm_slot("blocking", LLMRequestPriority.INTEGRATION)
            try:
                await first_can_complete.wait()
            finally:
                app.release_llm_slot("blocking")

        # Start blocking request
        blocking_task = asyncio.create_task(blocking_request())
        await asyncio.sleep(0.1)  # Give it time to acquire slot

        # Now queue requests with different priorities
        execution_order = []

        async def tracked_request(request_id: str, priority: LLMRequestPriority):
            await app.acquire_llm_slot(request_id, priority)
            try:
                execution_order.append(request_id)
            finally:
                app.release_llm_slot(request_id)

        # Queue in reverse priority order (integration, rewording, classification)
        # But they should execute in priority order (classification, rewording, integration)
        tasks = [
            asyncio.create_task(tracked_request("integration", LLMRequestPriority.INTEGRATION)),
            asyncio.create_task(tracked_request("rewording", LLMRequestPriority.REWORDING)),
            asyncio.create_task(tracked_request("classification", LLMRequestPriority.CLASSIFICATION)),
        ]

        await asyncio.sleep(0.1)  # Let all queue up

        # Release the blocking request
        first_can_complete.set()

        # Wait for all to complete
        await asyncio.gather(blocking_task, *tasks)

        # Verify priority order: classification (1) < rewording (2) < integration (3)
        assert execution_order == ["classification", "rewording", "integration"]
