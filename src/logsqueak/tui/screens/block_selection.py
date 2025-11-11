"""Phase 1 Screen: Block Selection.

This screen displays journal blocks in a tree view, streams LLM classification
results, and allows users to select knowledge blocks for extraction.
"""

from typing import Dict, Optional, Callable, Any
from textual.app import ComposeResult
from textual.screen import Screen
from textual.containers import Container, Vertical, Horizontal
from textual.widgets import Footer, Header, Static
from textual.reactive import reactive
from textual.worker import Worker
from textual.binding import Binding
from textual._context import NoActiveAppError
import structlog

from logseq_outline.parser import LogseqBlock, LogseqOutline
from logsqueak.models.block_state import BlockState
from logsqueak.models.background_task import BackgroundTask
from logsqueak.tui.widgets import BlockTree, StatusPanel, BlockDetailPanel
from logsqueak.services.llm_client import LLMClient
from logsqueak.services.llm_wrappers import classify_blocks

logger = structlog.get_logger()


class Phase1Screen(Screen):
    """Phase 1: Block Selection screen.

    Displays journal blocks, streams LLM suggestions, allows user selection.
    """

    CSS = """
    BlockTree {
        height: 1fr;
    }

    BlockDetailPanel {
        height: auto;
        min-height: 16;
        max-height: 30;
    }
    """

    BINDINGS = [
        ("j", "cursor_down", "Down"),
        ("k", "cursor_up", "Up"),
        ("down", "cursor_down", "Down"),
        ("up", "cursor_up", "Up"),
        ("J", "jump_next_knowledge", "Next Knowledge"),
        ("K", "jump_prev_knowledge", "Prev Knowledge"),
        Binding("space", "toggle_selection", "Toggle Selection", priority=True),
        ("a", "accept_all_suggestions", "Accept All"),
        ("c", "clear_all_selections", "Clear All"),
        ("r", "reset_to_llm", "Reset to LLM"),
        ("n", "next_phase", "Next Phase"),
        ("q", "quit", "Quit"),
    ]

    # Reactive attributes for state management
    selected_count: reactive[int] = reactive(0)
    current_block_id: reactive[Optional[str]] = reactive(None)

    def __init__(
        self,
        blocks: list[LogseqBlock],
        journal_date: str,
        journal_outline: LogseqOutline,
        llm_client: Optional[LLMClient] = None,
        initial_block_states: Optional[Dict[str, BlockState]] = None,
        llm_stream_fn: Optional[Callable] = None,
        auto_start_workers: bool = True,
        *args,
        **kwargs
    ):
        """Initialize Phase1Screen.

        Args:
            blocks: List of LogseqBlock objects from journal
            journal_date: Date string for journal (e.g., "2025-01-15")
            journal_outline: Full journal outline (for LLM classification)
            llm_client: LLM client instance (None for testing with mock)
            initial_block_states: Optional pre-populated block states
            llm_stream_fn: Optional mock LLM streaming function (for testing)
            auto_start_workers: Whether to auto-start background workers on mount
        """
        super().__init__(*args, **kwargs)
        self.blocks = blocks
        self.journal_date = journal_date
        self.journal_outline = journal_outline
        self.llm_client = llm_client
        self.llm_stream_fn = llm_stream_fn
        self.auto_start_workers = auto_start_workers

        # Initialize block states
        if initial_block_states:
            self.block_states = initial_block_states
        else:
            self.block_states = self._initialize_block_states(blocks)

        # Background tasks tracking - use app.background_tasks in production, local dict in tests
        self._test_background_tasks: Dict[str, BackgroundTask] = {}

        # Workers
        self._llm_worker: Optional[Worker] = None
        self._indexing_worker: Optional[Worker] = None

    def _initialize_block_states(
        self,
        blocks: list[LogseqBlock]
    ) -> Dict[str, BlockState]:
        """Recursively initialize BlockState for all blocks.

        Args:
            blocks: List of LogseqBlock objects

        Returns:
            Dictionary mapping block_id to BlockState
        """
        states = {}

        for block in blocks:
            if block.block_id:
                states[block.block_id] = BlockState(
                    block_id=block.block_id,
                    classification="pending",
                    source="user",
                )

            # Recursively process children
            child_states = self._initialize_block_states(block.children)
            states.update(child_states)

        return states

    def compose(self) -> ComposeResult:
        """Compose the screen layout.

        Layout:
        - Header
        - Main container (vertical split):
          - BlockTree (top 60%)
          - BlockDetailPanel (bottom 40%)
        - StatusPanel
        - Footer
        """
        yield Header()

        with Container(id="main-container"):
            with Vertical():
                # Block tree (top panel)
                yield BlockTree(
                    f"Journal Blocks ({self.journal_date})",
                    blocks=self.blocks,
                    block_states=self.block_states,
                )

                # Selected block details (bottom panel)
                yield BlockDetailPanel()

        # Status panel for background tasks (uses app-level shared dict or test dict)
        from logsqueak.tui.app import LogsqueakApp
        if isinstance(self.app, LogsqueakApp):
            yield StatusPanel(background_tasks=self.app.background_tasks)
        else:
            # Test environment - use test dict
            yield StatusPanel(background_tasks=self._test_background_tasks)
        yield Footer()

    @property
    def _background_tasks(self) -> Dict[str, BackgroundTask]:
        """Get app-level background tasks dict (or test dict in test mode)."""
        from logsqueak.tui.app import LogsqueakApp
        if isinstance(self.app, LogsqueakApp):
            return self.app.background_tasks
        return self._test_background_tasks

    @property
    def background_tasks(self) -> Dict[str, BackgroundTask]:
        """Backwards compatibility property for tests."""
        return self._background_tasks

    @background_tasks.setter
    def background_tasks(self, value: Dict[str, BackgroundTask]) -> None:
        """Allow tests to set background tasks dict."""
        # In test mode (or before app is created), replace the test dict
        from logsqueak.tui.app import LogsqueakApp
        try:
            if not isinstance(self.app, LogsqueakApp):
                self._test_background_tasks = value
        except NoActiveAppError:
            # App not created yet (common in tests)
            self._test_background_tasks = value

    def on_mount(self) -> None:
        """Called when screen is mounted."""
        logger.info("phase1_on_mount_started")

        # Start background tasks automatically (unless disabled for testing)
        if self.auto_start_workers:
            self.start_llm_classification()
            self.start_page_indexing()

        # Update bottom panel with initial block
        self._update_current_block()

        logger.info("phase1_on_mount_finished")

    def on_unmount(self) -> None:
        """Called when screen is unmounted."""
        # Cancel any running workers
        if self._llm_worker:
            self._llm_worker.cancel()
        if self._indexing_worker:
            self._indexing_worker.cancel()

    def on_tree_node_highlighted(self, event: BlockTree.NodeHighlighted) -> None:
        """Called when tree cursor moves (handles arrow keys and j/k)."""
        self._update_current_block()

    # Keyboard actions

    def action_cursor_down(self) -> None:
        """Move cursor down in tree."""
        tree = self.query_one(BlockTree)
        tree.action_cursor_down()
        # Note: _update_current_block() called by on_tree_node_highlighted event

    def action_cursor_up(self) -> None:
        """Move cursor up in tree."""
        tree = self.query_one(BlockTree)
        tree.action_cursor_up()
        # Note: _update_current_block() called by on_tree_node_highlighted event

    def action_jump_next_knowledge(self) -> None:
        """Jump to next LLM-suggested knowledge block."""
        tree = self.query_one(BlockTree)
        current_line = tree.cursor_line

        next_line = tree.find_next_knowledge_block(current_line)
        if next_line is not None:
            tree.cursor_line = next_line
            tree.scroll_to_line(next_line)
            # Note: _update_current_block() called by on_tree_node_highlighted event

    def action_jump_prev_knowledge(self) -> None:
        """Jump to previous LLM-suggested knowledge block."""
        tree = self.query_one(BlockTree)
        current_line = tree.cursor_line

        prev_line = tree.find_previous_knowledge_block(current_line)
        if prev_line is not None:
            tree.cursor_line = prev_line
            tree.scroll_to_line(prev_line)
            # Note: _update_current_block() called by on_tree_node_highlighted event

    def action_toggle_selection(self) -> None:
        """Toggle selection on current block."""
        tree = self.query_one(BlockTree)
        block_id = tree.get_current_block_id()

        if block_id and block_id in self.block_states:
            state = self.block_states[block_id]

            if state.classification == "knowledge":
                # Already selected (green checkmark) → Deselect
                state.classification = "pending"
                state.source = "user"
                state.confidence = None
            else:
                # Not selected → Select as user choice (green checkmark)
                state.classification = "knowledge"
                state.source = "user"
                state.confidence = 1.0

            # Update visual
            tree.update_block_label(block_id)
            self._update_selected_count()
            self._update_current_block()

    def action_accept_all_suggestions(self) -> None:
        """Accept all LLM suggestions (keeps existing user selections)."""
        tree = self.query_one(BlockTree)

        for block_id, state in self.block_states.items():
            if state.llm_classification == "knowledge":
                # Accept LLM suggestion (only if not already selected)
                if state.classification != "knowledge":
                    state.classification = "knowledge"
                    state.source = "llm"
                    state.confidence = state.llm_confidence or 0.0

                    # Update visual
                    tree.update_block_label(block_id)

        self._update_selected_count()
        self._update_current_block()

    def action_clear_all_selections(self) -> None:
        """Clear all selections."""
        tree = self.query_one(BlockTree)

        for block_id, state in self.block_states.items():
            state.classification = "pending"
            state.source = "user"

            # Update visual
            tree.update_block_label(block_id)

        self._update_selected_count()
        self._update_current_block()

    def action_reset_to_llm(self) -> None:
        """Reset current block to LLM suggestion."""
        tree = self.query_one(BlockTree)
        block_id = tree.get_current_block_id()

        if block_id and block_id in self.block_states:
            state = self.block_states[block_id]

            if state.llm_classification == "knowledge":
                # Reset to LLM suggestion
                state.classification = "knowledge"
                state.source = "llm"
                state.confidence = state.llm_confidence or 0.0

                # Update visual
                tree.update_block_label(block_id)
                self._update_selected_count()
                self._update_current_block()

    def action_next_phase(self) -> None:
        """Proceed to Phase 2 (only if blocks selected)."""
        if self.selected_count > 0:
            # Collect all selected blocks
            selected_blocks = [
                state for state in self.block_states.values()
                if state.classification == "knowledge"
            ]

            logger.info(
                "phase1_next_requested",
                num_selected=len(selected_blocks),
            )

            # Call app transition method
            from logsqueak.tui.app import LogsqueakApp
            if isinstance(self.app, LogsqueakApp):
                self.app.transition_to_phase2(selected_blocks)

    def action_quit(self) -> None:
        """Quit application."""
        self.app.exit()

    # Helper methods

    def _update_current_block(self) -> None:
        """Update bottom panel with current block details."""
        try:
            tree = self.query_one(BlockTree)
            block_id = tree.get_current_block_id()

            if block_id and block_id in self.block_states:
                self.current_block_id = block_id

                # Find the block
                block = self._find_block_by_id(self.blocks, block_id)
                if block:
                    state = self.block_states[block_id]

                    # Update block detail panel (async call)
                    panel = self.query_one(BlockDetailPanel)
                    self.call_later(panel.show_block, block, state)
            elif tree.cursor_line >= 0:
                # Cursor is on a line but block_id might be None (root node)
                # Try to get the first actual block
                try:
                    node = tree.get_node_at_line(tree.cursor_line)
                    if node and node.data:
                        block_id = node.data
                        block = self._find_block_by_id(self.blocks, block_id)
                        if block and block_id in self.block_states:
                            state = self.block_states[block_id]
                            panel = self.query_one(BlockDetailPanel)
                            self.call_later(panel.show_block, block, state)
                except:
                    pass
        except:
            # Widget not ready yet
            pass

    def _find_block_by_id(
        self,
        blocks: list[LogseqBlock],
        block_id: str
    ) -> Optional[LogseqBlock]:
        """Recursively find LogseqBlock by id."""
        for block in blocks:
            if block.block_id == block_id:
                return block

            result = self._find_block_by_id(block.children, block_id)
            if result:
                return result

        return None

    def _update_selected_count(self) -> None:
        """Update count of selected blocks."""
        count = sum(
            1 for state in self.block_states.values()
            if state.classification == "knowledge"
        )
        self.selected_count = count

    # Background tasks

    def start_llm_classification(self) -> None:
        """Start LLM classification worker."""
        if self.llm_stream_fn:
            # Use mock function (for testing)
            self._llm_worker = self.run_worker(
                self._llm_classification_worker_mock(),
                exclusive=False,
                thread=False
            )
        elif self.llm_client:
            # Use real LLM client
            self._llm_worker = self.run_worker(
                self._llm_classification_worker(),
                exclusive=False,
                thread=False
            )
        else:
            logger.warning("No LLM client or mock function provided, skipping classification")

    async def _llm_classification_worker_mock(self) -> None:
        """Mock LLM classification worker (for testing)."""
        # Create background task
        self._background_tasks["llm_classification"] = BackgroundTask(
            task_type="llm_classification",
            status="running",
            progress_current=0,
            progress_total=len(self.block_states),
        )

        status_panel = self.query_one(StatusPanel)
        status_panel.update_status()

        # Stream results (LLM only returns knowledge blocks, omits activity blocks)
        count = 0
        async for result in self.llm_stream_fn():
            block_id = result["block_id"]
            if block_id in self.block_states:
                state = self.block_states[block_id]

                # All results are knowledge blocks (LLM filters out activity blocks)
                state.llm_classification = "knowledge"
                state.llm_confidence = result["confidence"]
                state.reason = result["reason"]

                # Update visual (shows robot emoji but block not selected yet)
                tree = self.query_one(BlockTree)
                tree.update_block_label(block_id)

                # Update bottom panel if this is the currently selected block
                if block_id == self.current_block_id:
                    self._update_current_block()

            count += 1

            # Update progress
            self._background_tasks["llm_classification"].progress_current = count
            status_panel.update_status()

        # Mark complete and remove from background tasks
        del self._background_tasks["llm_classification"]
        status_panel.update_status()

        self._update_selected_count()

    async def _llm_classification_worker(self) -> None:
        """Real LLM classification worker using classify_blocks wrapper."""
        # Create background task
        self._background_tasks["llm_classification"] = BackgroundTask(
            task_type="llm_classification",
            status="running",
            progress_current=0,
            progress_total=len(self.block_states),
        )

        status_panel = self.query_one(StatusPanel)
        status_panel.update_status()

        try:
            # Stream results from LLM (only returns knowledge blocks)
            count = 0
            async for chunk in classify_blocks(self.llm_client, self.journal_outline):
                block_id = chunk.block_id
                if block_id in self.block_states:
                    state = self.block_states[block_id]

                    # All chunks are knowledge blocks (LLM filters out activity blocks)
                    state.llm_classification = "knowledge"
                    state.llm_confidence = chunk.confidence
                    state.reason = chunk.reason

                    # Update visual (shows robot emoji but block not selected yet)
                    tree = self.query_one(BlockTree)
                    tree.update_block_label(block_id)

                    # Update bottom panel if this is the currently selected block
                    if block_id == self.current_block_id:
                        self._update_current_block()

                    logger.info(
                        "llm_classification_chunk",
                        block_id=block_id,
                        confidence=chunk.confidence,
                        reason=chunk.reason[:100] if chunk.reason else None
                    )
                else:
                    logger.warning(
                        "llm_classification_unknown_block",
                        block_id=block_id
                    )

                count += 1

                # Update progress
                self._background_tasks["llm_classification"].progress_current = count
                status_panel.update_status()

            # Mark complete and remove from background tasks
            del self._background_tasks["llm_classification"]
            status_panel.update_status()

            self._update_selected_count()

            logger.info(
                "llm_classification_complete",
                total_knowledge_blocks=count
            )

        except Exception as e:
            # Mark failed
            self._background_tasks["llm_classification"].status = "failed"
            self._background_tasks["llm_classification"].error_message = str(e)
            status_panel.update_status()

            logger.error(
                "llm_classification_error",
                error=str(e),
                exc_info=True
            )

    def start_page_indexing(self) -> None:
        """Start page indexing worker."""
        # Create background task
        self._background_tasks["page_indexing"] = BackgroundTask(
            task_type="page_indexing",
            status="running",
            progress_percentage=0.0,
        )

        status_panel = self.query_one(StatusPanel)
        status_panel.update_status()

        # Launch worker
        self._indexing_worker = self.run_worker(
            self._page_indexing_worker(),
            exclusive=False,
            thread=False
        )

    async def _page_indexing_worker(self) -> None:
        """Page indexing worker - builds ChromaDB vector index for RAG search.

        Worker Dependency: This worker MUST wait for model_preload to complete before starting.
        The SentenceTransformer model must be loaded before PageIndexer can generate embeddings.

        Coordination: Polls app.background_tasks["model_preload"] until status="completed".
        """
        import asyncio
        from logsqueak.tui.app import LogsqueakApp

        # Wait for SentenceTransformer model to load (T108b)
        logger.info("page_indexing_waiting_for_model", phase="phase1")

        # Poll for model preload completion
        while True:
            if isinstance(self.app, LogsqueakApp):
                model_task = self.app.background_tasks.get("model_preload")
                if model_task is None:
                    # Task deleted - it completed successfully
                    logger.info("page_indexing_model_ready", phase="phase1")
                    break
                elif model_task.status == "failed":
                    # Model failed to load - proceed anyway (will load on-demand)
                    logger.warning(
                        "page_indexing_model_failed",
                        phase="phase1",
                        fallback="will load on-demand"
                    )
                    break

            # Wait before polling again
            await asyncio.sleep(0.1)

        # Now implement real PageIndexer (T108c)
        try:
            logger.info("page_indexing_starting", phase="phase1")

            # Get PageIndexer from app
            if not isinstance(self.app, LogsqueakApp):
                logger.error("page_indexing_no_app")
                return

            page_indexer = self.app.page_indexer

            # Define progress callback to update status panel
            def on_progress(current: int, total: int):
                """Update progress percentage as pages are indexed."""
                percentage = (current / total) * 100.0
                self._background_tasks["page_indexing"].progress_percentage = percentage
                self._background_tasks["page_indexing"].progress_current = current
                self._background_tasks["page_indexing"].progress_total = total

                # Update status panel
                try:
                    status_panel = self.query_one(StatusPanel)
                    status_panel.update_status()
                except Exception:
                    pass  # Widget not mounted yet

            # Run PageIndexer in thread pool to avoid blocking UI
            # (file I/O and SentenceTransformer.encode() are CPU-intensive)
            # Create a wrapper to run the async function in a new event loop in the thread
            def _run_indexing():
                """Run indexing in a thread with its own event loop."""
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(page_indexer.build_index(progress_callback=on_progress))
                finally:
                    loop.close()

            # Run in thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _run_indexing)

            # Mark complete and remove from background tasks
            del self._background_tasks["page_indexing"]

            # Update status panel
            try:
                status_panel = self.query_one(StatusPanel)
                status_panel.update_status()
            except Exception:
                pass  # Widget not mounted yet

            logger.info("page_indexing_complete", phase="phase1")

        except Exception as e:
            # Mark failed
            self._background_tasks["page_indexing"].status = "failed"
            self._background_tasks["page_indexing"].error_message = str(e)

            # Update status panel
            try:
                status_panel = self.query_one(StatusPanel)
                status_panel.update_status()
            except Exception:
                pass  # Widget not mounted yet

            logger.error(
                "page_indexing_error",
                phase="phase1",
                error=str(e),
                exc_info=True
            )
