"""Main Logsqueak TUI Application.

This module defines the main Textual App class that manages screen transitions
between Phase 1 (Block Selection), Phase 2 (Content Editing), and Phase 3
(Integration Review).

## Worker Dependency Coordination

The app coordinates background workers across phases to ensure correct execution order:

**Dependency Chain:**
```
Phase 1:
  model_preload (app-level) ──► SentenceTransformer loads in thread pool
    └─ [BLOCKS] → page_indexing (Phase1Screen) ──► PageIndexer.build_index()

Phase 2:
  rag_search (Phase2Screen)
    └─ [WAITS FOR] page_indexing ──► RAGSearch.find_candidates()
      └─ [TRIGGERS] llm_decisions (opportunistic)

Phase 3:
  llm_decisions (Phase3Screen or Phase2Screen)
    └─ [WAITS FOR] rag_search completion
```

**Coordination Mechanism:**
- `app.background_tasks: Dict[str, BackgroundTask]` is a shared dict accessible from all screens
- Workers register their status in this dict when they start/complete/fail
- Dependent workers poll this dict to check if their dependencies are ready
- Polling uses `asyncio.sleep(0.1)` to avoid busy-waiting

**Key Dependencies:**
1. **model_preload → page_indexing**: PageIndexer uses the SentenceTransformer
   model to generate embeddings, so the model must be loaded before indexing can start.
2. **page_indexing → rag_search**: RAG search cannot query ChromaDB until the
   vector index is built.
3. **rag_search → llm_decisions**: Integration decisions need candidate pages
   from RAG search results.

See tests/integration/test_worker_dependencies.py for dependency ordering tests.
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
import asyncio
from dataclasses import dataclass
from enum import IntEnum

from textual.app import App
from textual.binding import Binding
from textual.widgets import Footer, Header
from textual.worker import Worker
import structlog

from logseq_outline.parser import LogseqOutline, LogseqBlock
from logseq_outline.graph import GraphPaths
from logsqueak.models.config import Config
from logsqueak.models.block_state import BlockState
from logsqueak.models.edited_content import EditedContent
from logsqueak.services.llm_client import LLMClient
from logsqueak.services.page_indexer import PageIndexer
from logsqueak.services.rag_search import RAGSearch
from logsqueak.services.file_monitor import FileMonitor
from logsqueak.services.llm_wrappers import _augment_outline_with_ids
from logsqueak.tui.screens import Phase1Screen, Phase2Screen, Phase3Screen

logger = structlog.get_logger()


class LLMRequestPriority(IntEnum):
    """Priority levels for LLM requests (lower number = higher priority)."""
    CLASSIFICATION = 1  # Phase 1: Block classification
    REWORDING = 2  # Phase 2: Content rewording
    INTEGRATION = 3  # Phase 3: Integration decisions


@dataclass
class LLMRequest:
    """Represents a queued LLM request."""
    priority: LLMRequestPriority
    request_id: str
    ready_event: asyncio.Event  # Set when request can proceed


class LogsqueakApp(App):
    """Main Logsqueak TUI Application.

    Manages screen transitions and shared state across all three phases.
    """

    CSS = """
    Screen {
        background: $surface;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=True, priority=True),
        Binding("q", "quit", "Quit", show=False),
    ]

    def __init__(
        self,
        journals: Dict[str, LogseqOutline],
        config: Config,
        llm_client: LLMClient,
        page_indexer: PageIndexer,
        rag_search: RAGSearch,
        file_monitor: FileMonitor,
    ):
        """Initialize the Logsqueak app.

        Args:
            journals: Dictionary mapping date string (YYYY-MM-DD) to LogseqOutline
            config: Application configuration
            llm_client: LLM client for streaming responses
            page_indexer: Page indexing service for RAG
            rag_search: Semantic search service
            file_monitor: File modification tracking service
        """
        super().__init__()

        # Augment all journal outlines with temporary IDs for blocks without explicit id:: properties
        # This ensures all blocks have stable IDs for LLM classification and tracking
        self.journals = {
            date: _augment_outline_with_ids(outline)
            for date, outline in journals.items()
        }

        # Store services
        self.config = config
        self.llm_client = llm_client
        self.page_indexer = page_indexer
        self.rag_search = rag_search
        self.file_monitor = file_monitor

        # Phase state tracking (shared across screens)
        self.selected_blocks: Optional[List[BlockState]] = None
        self.edited_content: Optional[List[EditedContent]] = None
        self.candidate_pages: Optional[List[str]] = None
        self.page_contents: Optional[Dict[str, str]] = None
        self.original_contexts: Optional[Dict[str, str]] = None
        self.integration_decisions: List = []  # Shared list populated by Phase2, read by Phase3

        # Shared background task status tracking (for worker dependency coordination)
        # This dict is shared across all screens so workers can check dependencies
        from logsqueak.models.background_task import BackgroundTask
        self.background_tasks: Dict[str, BackgroundTask] = {}

        # LLM request queue (serializes concurrent LLM requests)
        # Requests are processed sequentially with priority order:
        # 1. Classification (Phase 1) - highest priority
        # 2. Rewording (Phase 2) - medium priority
        # 3. Integration (Phase 3) - lowest priority
        # Within same priority, FIFO order is maintained.
        self.llm_request_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._llm_queue_consumer_started = False
        self._current_llm_request: Optional[str] = None  # Current request ID being processed
        self._active_llm_workers: Dict[str, Worker] = {}  # Track active Textual Worker instances for cancellation

        # Calculate total blocks across all journals
        total_blocks = sum(len(outline.blocks) for outline in self.journals.values())

        logger.info(
            "app_initialized",
            num_journals=len(self.journals),
            journal_dates=list(self.journals.keys()),
            total_blocks=total_blocks,
        )

    def on_mount(self) -> None:
        """Called when app is mounted. Start with Phase 1."""
        logger.info("app_on_mount_started")

        # Start LLM request queue consumer worker
        self._start_llm_queue_consumer()

        # Create and push Phase 1 screen FIRST (so it shows immediately)
        logger.info("creating_phase1_screen")
        phase1_screen = Phase1Screen(
            journals=self.journals,
            llm_client=self.llm_client,
            auto_start_workers=True,  # Auto-start LLM classification
            name="phase1",
        )
        logger.info("pushing_phase1_screen")
        # Create model_preload task BEFORE pushing screen so workers can see it exists
        # This prevents race condition where page_indexing worker starts before preload task exists
        from logsqueak.models.background_task import BackgroundTask
        self.background_tasks["model_preload"] = BackgroundTask(
            task_type="model_preload",
            status="running",
        )

        self.push_screen(phase1_screen)
        logger.info("phase1_screen_pushed")

        # Defer actual model preloading with a delay so UI renders first
        # The task exists now, so workers can poll it
        self.set_timer(0.1, self._start_model_preload)
        logger.info("app_on_mount_finished")

    def _start_llm_queue_consumer(self) -> None:
        """Start the LLM request queue consumer worker."""
        if not self._llm_queue_consumer_started:
            self._llm_queue_consumer_started = True
            self.run_worker(self._consume_llm_queue(), name="llm_queue_consumer", exclusive=True)
            logger.info("llm_queue_consumer_started")

    async def _consume_llm_queue(self) -> None:
        """Consumer worker that processes LLM requests sequentially.

        This worker runs for the lifetime of the app and processes requests from
        the priority queue. Only one LLM request executes at a time.

        Requests are processed in priority order:
        1. Classification (Phase 1)
        2. Rewording (Phase 2)
        3. Integration (Phase 3)

        Within the same priority level, FIFO order is maintained.
        """
        logger.info("llm_queue_consumer_running")

        try:
            while True:
                # Get next request from queue (blocks if queue is empty)
                priority, request_id, ready_event = await self.llm_request_queue.get()

                logger.info(
                    "llm_queue_processing_request",
                    request_id=request_id,
                    priority=priority,
                )

                # Track current request
                self._current_llm_request = request_id

                # Signal that this request can proceed
                ready_event.set()

                # Wait for the request to clear (worker will call release_llm_slot)
                # We poll instead of using another event to keep the implementation simple
                while self._current_llm_request == request_id:
                    await asyncio.sleep(0.1)

                logger.info(
                    "llm_queue_request_completed",
                    request_id=request_id,
                )

        except asyncio.CancelledError:
            logger.info("llm_queue_consumer_cancelled")
            raise
        except Exception as e:
            logger.error("llm_queue_consumer_error", error=str(e))
            raise

    async def acquire_llm_slot(
        self,
        request_id: str,
        priority: LLMRequestPriority,
    ) -> None:
        """Acquire a slot in the LLM request queue.

        This method blocks until the request can proceed. Workers should call
        this before making LLM API calls, and call release_llm_slot() when done.

        Args:
            request_id: Unique identifier for this request (for logging)
            priority: Priority level for this request

        Example:
            await app.acquire_llm_slot("classification_worker", LLMRequestPriority.CLASSIFICATION)
            try:
                async for chunk in llm_client.stream_ndjson(...):
                    # Process chunks
            finally:
                app.release_llm_slot("classification_worker")
        """
        ready_event = asyncio.Event()
        request = LLMRequest(
            priority=priority,
            request_id=request_id,
            ready_event=ready_event,
        )

        logger.info(
            "llm_queue_request_submitted",
            request_id=request_id,
            priority=priority.name,
        )

        # Submit to queue (priority queue uses tuple: (priority, secondary_key, payload))
        # We use request_id as secondary key to maintain FIFO within same priority
        await self.llm_request_queue.put((priority, request_id, ready_event))

        # Wait for our turn
        logger.debug(
            "llm_queue_request_waiting",
            request_id=request_id,
        )
        await ready_event.wait()

        logger.info(
            "llm_queue_request_ready",
            request_id=request_id,
        )

    def release_llm_slot(self, request_id: str) -> None:
        """Release the LLM request slot.

        Workers must call this when their LLM request is complete (or failed).

        Args:
            request_id: Same identifier used in acquire_llm_slot()
        """
        if self._current_llm_request == request_id:
            logger.info(
                "llm_queue_slot_released",
                request_id=request_id,
            )
            self._current_llm_request = None
        else:
            logger.warning(
                "llm_queue_slot_release_mismatch",
                request_id=request_id,
                current_request=self._current_llm_request,
            )

    def register_llm_worker(self, worker_name: str, worker: Worker) -> None:
        """Register an LLM worker for potential cancellation.

        Args:
            worker_name: Name of the worker (e.g., "llm_classification", "llm_rewording")
            worker: The Textual Worker instance running the worker coroutine
        """
        self._active_llm_workers[worker_name] = worker
        logger.debug(
            "llm_worker_registered",
            worker_name=worker_name,
            worker_state=worker.state.name if hasattr(worker, 'state') else None
        )

    def cancel_llm_worker(self, worker_name: str) -> None:
        """Cancel an active LLM worker.

        This is called during screen transitions to stop workers that are no longer needed.

        Args:
            worker_name: Name of the worker to cancel (e.g., "llm_classification")
        """
        worker = self._active_llm_workers.get(worker_name)
        if worker:
            # Check if worker is still running (Textual Worker has is_running property)
            if hasattr(worker, 'is_running') and worker.is_running:
                logger.info(
                    "llm_worker_cancelling",
                    worker_name=worker_name
                )
                worker.cancel()
            # Remove from active workers
            del self._active_llm_workers[worker_name]
        else:
            logger.debug(
                "llm_worker_cancel_skipped",
                worker_name=worker_name,
                reason="not found"
            )

    def cancel_all_llm_workers(self) -> None:
        """Cancel all active LLM workers.

        This is useful for cleanup or during app shutdown.
        """
        for worker_name in list(self._active_llm_workers.keys()):
            self.cancel_llm_worker(worker_name)

    def _start_model_preload(self) -> None:
        """Start the model preload worker (called after screen is visible)."""
        self.run_worker(self._preload_embedding_model(), name="model_preload")

    async def _preload_embedding_model(self) -> None:
        """Preload SentenceTransformer model in background during Phase 1.

        This triggers lazy loading of the embedding model so it's ready for PageIndexer
        to use. The model is loaded in a thread pool to avoid blocking the async event loop
        (model initialization is CPU-intensive).

        Worker Dependency: PageIndexer cannot start until this completes because it uses
        the SentenceTransformer model to generate embeddings for blocks.

        Note: The background task is created in on_mount() before this worker starts,
        so dependent workers can see it exists and poll for completion.
        """
        import asyncio

        try:
            logger.info("preloading_embedding_model", phase="phase1")

            # Check if encoder is already set (e.g., mocked in tests)
            if self.rag_search._encoder is not None:
                logger.info("embedding_model_already_loaded", phase="phase1")
            else:
                # Run the model loading in a thread pool to avoid blocking
                # the async event loop (SentenceTransformer import and init is CPU-heavy)
                def _load_model():
                    """Load SentenceTransformer model (runs in executor thread)."""
                    from sentence_transformers import SentenceTransformer
                    return SentenceTransformer(self.rag_search.embedding_model)

                loop = asyncio.get_event_loop()
                encoder = await loop.run_in_executor(
                    None,  # Use default executor
                    _load_model
                )

                # Store the loaded encoder in both rag_search and page_indexer
                # Both services need the same encoder instance to avoid re-loading
                self.rag_search._encoder = encoder
                self.page_indexer._encoder = encoder

            # Mark complete (other workers poll for this status)
            # Don't set progress_percentage - model loading is not granularly measurable
            self.background_tasks["model_preload"].status = "completed"

            logger.info("embedding_model_preloaded", phase="phase1")
        except Exception as e:
            # Mark failed (if task still exists)
            if "model_preload" in self.background_tasks:
                self.background_tasks["model_preload"].status = "failed"
                self.background_tasks["model_preload"].error_message = str(e)

            # Non-fatal - model will load on-demand in Phase 2 if preload fails
            logger.warning(
                "embedding_model_preload_failed",
                error=str(e),
                fallback="will load on-demand in Phase 2"
            )

    def transition_to_phase2(self, selected_blocks: List[BlockState]) -> None:
        """Transition from Phase 1 to Phase 2.

        Args:
            selected_blocks: List of BlockState objects selected in Phase 1
        """
        logger.info(
            "transitioning_to_phase2",
            num_selected=len(selected_blocks),
        )

        # Cancel Phase 1 workers (classification)
        self.cancel_llm_worker("llm_classification")

        # Store selected blocks
        self.selected_blocks = selected_blocks

        # Initialize edited content from selected blocks
        self.edited_content = []
        for block_state in selected_blocks:
            # Find the actual block and its context using generate_chunks
            from logseq_outline.context import generate_chunks, generate_full_context

            block = None
            hierarchical_context = ""
            journal_outline = None

            # First pass: Find the block across all journals using hybrid IDs
            for date, outline in self.journals.items():
                for found_block, context, hybrid_id in generate_chunks(outline):
                    if hybrid_id == block_state.block_id:
                        block = found_block
                        journal_outline = outline
                        # Don't use context yet - it includes frontmatter
                        break
                if block:
                    break

            # Second pass: Generate hierarchical context WITHOUT frontmatter
            # We need to rebuild the parent chain to generate clean context
            if block and journal_outline:
                # Find parent chain by walking the tree
                def find_block_with_parents(target_block, blocks, parents=[]):
                    for b in blocks:
                        if b is target_block:
                            return parents
                        result = find_block_with_parents(target_block, b.children, parents + [b])
                        if result is not None:
                            return result
                    return None

                parents = find_block_with_parents(block, journal_outline.blocks) or []

                # Generate context WITHOUT frontmatter
                hierarchical_context = generate_full_context(
                    block,
                    parents,
                    indent_str=journal_outline.indent_str,
                    frontmatter=None  # Exclude frontmatter
                )

            if block:
                # Get user-facing content (excludes id:: property)
                user_content = block.get_user_content()

                # Create EditedContent
                edited = EditedContent(
                    block_id=block_state.block_id,
                    original_content=user_content,
                    hierarchical_context=hierarchical_context,
                    current_content=user_content,
                )
                self.edited_content.append(edited)

        # Extract the selected blocks (LogseqBlock objects)
        from logseq_outline.graph import GraphPaths
        from pathlib import Path

        selected_logseq_blocks = []
        for block_state in selected_blocks:
            # Search across all journals
            block = None
            for outline in self.journals.values():
                block = outline.find_block_by_id(block_state.block_id)
                if block:
                    selected_logseq_blocks.append(block)
                    break

        # Create GraphPaths from config
        graph_paths = GraphPaths(Path(self.config.logseq.graph_path))

        # For Phase 2, we pass journals dict instead of single outline
        # Phase2Screen will need to handle multiple journals
        phase2_screen = Phase2Screen(
            blocks=selected_logseq_blocks,
            edited_content=self.edited_content,
            journals=self.journals,
            graph_paths=graph_paths,
            llm_client=self.llm_client,
            rag_search=self.rag_search,
            auto_start_workers=True,
            name="phase2",
        )
        self.push_screen(phase2_screen)

    def transition_to_phase3(
        self,
        edited_content: List[EditedContent],
        candidate_pages: List[str],
        page_contents: Dict[str, LogseqOutline],
    ) -> None:
        """Transition from Phase 2 to Phase 3.

        Args:
            edited_content: List of EditedContent from Phase 2
            candidate_pages: List of candidate page names from RAG search
            page_contents: Dict mapping page names to LogseqOutline objects

        Note:
            Integration decisions are accessed via self.integration_decisions,
            which is a shared list populated by Phase 2's background worker.
        """
        logger.info(
            "transitioning_to_phase3",
            num_blocks=len(edited_content),
            num_candidates=len(candidate_pages),
        )

        # Cancel Phase 2 workers (rewording)
        self.cancel_llm_worker("llm_rewording")
        # Note: llm_decisions worker may still be running and that's okay
        # It populates the shared decisions list that Phase 3 uses

        # Store Phase 2 outputs
        self.edited_content = edited_content
        self.candidate_pages = candidate_pages
        self.page_contents = page_contents

        # Generate original contexts for each block
        self.original_contexts = {}
        for content in edited_content:
            self.original_contexts[content.block_id] = content.hierarchical_context

        # Filter journal blocks to only include edited ones (preserving order)
        # Collect blocks from all journals
        edited_block_ids = {ec.block_id for ec in edited_content}
        all_journal_blocks = []
        for outline in self.journals.values():
            all_journal_blocks.extend(outline.blocks)

        filtered_journal_blocks = self._filter_blocks_by_ids(
            all_journal_blocks,
            edited_block_ids
        )

        # Combine all journal content for preview (concatenate all journals)
        journal_contents = []
        for date in sorted(self.journals.keys()):
            journal_contents.append(f"# {date}\n\n")
            journal_contents.append(self.journals[date].render())
            journal_contents.append("\n\n")
        combined_journal_content = "".join(journal_contents)

        # Create and push Phase 3 screen with shared decisions list
        # Phase 3 will use self.integration_decisions which is being populated
        # by Phase 2's background worker
        phase3_screen = Phase3Screen(
            journal_blocks=filtered_journal_blocks,
            edited_content=edited_content,
            page_contents=page_contents,
            journals=self.journals,
            journal_content=combined_journal_content,  # Combined journal content for preview
            llm_client=None,  # Decisions are generated in Phase 2
            graph_paths=GraphPaths(Path(self.config.logseq.graph_path)),
            file_monitor=self.file_monitor,
            decisions=self.integration_decisions,  # Pass shared list (may still be populating)
            auto_start_workers=False,  # Phase 2 worker is already running
            name="phase3",
        )
        self.push_screen(phase3_screen)

    def _filter_blocks_by_ids(
        self,
        blocks: List[LogseqBlock],
        block_ids: set[str]
    ) -> List[LogseqBlock]:
        """Recursively filter blocks to only include those with IDs in the set.

        Args:
            blocks: List of LogseqBlock to filter
            block_ids: Set of block IDs to include

        Returns:
            Filtered list of LogseqBlock
        """
        filtered = []
        for block in blocks:
            # Check if this block should be included
            if block.block_id in block_ids:
                filtered.append(block)
            # Recursively filter children
            elif block.children:
                child_matches = self._filter_blocks_by_ids(block.children, block_ids)
                if child_matches:
                    # If any children match, include them
                    filtered.extend(child_matches)
        return filtered

    def action_quit(self) -> None:
        """Handle quit action.

        In Phase 3, shows a warning about partial journal state.
        In Phase 1-2, quits immediately.
        """
        logger.info("app_quit_requested", current_screen=self.screen.name)

        # Check if we're in Phase 3
        if self.screen.name == "phase3":
            # Show warning dialog about partial journal state
            # For now, just exit (dialog implementation deferred to polish phase)
            logger.warning(
                "phase3_quit_warning",
                message="Quitting from Phase 3 may leave journal in partial state",
            )

        self.exit()
