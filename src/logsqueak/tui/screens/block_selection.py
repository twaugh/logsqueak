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

        # Background tasks
        self.background_tasks: Dict[str, BackgroundTask] = {}

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

        yield StatusPanel(background_tasks=self.background_tasks)
        yield Footer()

    def on_mount(self) -> None:
        """Called when screen is mounted."""
        # Start background tasks automatically (unless disabled for testing)
        if self.auto_start_workers:
            self.start_llm_classification()
            self.start_page_indexing()

        # Update bottom panel with initial block
        self._update_current_block()

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
            # TODO: Transition to Phase2Screen
            # For now, just a placeholder
            pass

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
        self.background_tasks["llm_classification"] = BackgroundTask(
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
            self.background_tasks["llm_classification"].progress_current = count
            status_panel.update_status()

        # Mark complete
        self.background_tasks["llm_classification"].status = "completed"
        self.background_tasks["llm_classification"].progress_percentage = 100.0
        status_panel.update_status()

        self._update_selected_count()

    async def _llm_classification_worker(self) -> None:
        """Real LLM classification worker using classify_blocks wrapper."""
        # Create background task
        self.background_tasks["llm_classification"] = BackgroundTask(
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
                self.background_tasks["llm_classification"].progress_current = count
                status_panel.update_status()

            # Mark complete
            self.background_tasks["llm_classification"].status = "completed"
            self.background_tasks["llm_classification"].progress_percentage = 100.0
            status_panel.update_status()

            self._update_selected_count()

            logger.info(
                "llm_classification_complete",
                total_knowledge_blocks=count
            )

        except Exception as e:
            # Mark failed
            self.background_tasks["llm_classification"].status = "failed"
            self.background_tasks["llm_classification"].error_message = str(e)
            status_panel.update_status()

            logger.error(
                "llm_classification_error",
                error=str(e),
                exc_info=True
            )

    def start_page_indexing(self) -> None:
        """Start page indexing worker."""
        # Create background task
        self.background_tasks["page_indexing"] = BackgroundTask(
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
        """Page indexing worker (simulated for now)."""
        import asyncio

        # Simulate indexing progress
        for progress in range(0, 101, 10):
            await asyncio.sleep(0.2)

            self.background_tasks["page_indexing"].progress_percentage = float(progress)

            # Update status panel if it's mounted
            try:
                status_panel = self.query_one(StatusPanel)
                status_panel.update_status()
            except Exception:
                pass  # Widget not mounted yet

        # Mark complete
        self.background_tasks["page_indexing"].status = "completed"

        # Update status panel if it's mounted
        try:
            status_panel = self.query_one(StatusPanel)
            status_panel.update_status()
        except Exception:
            pass  # Widget not mounted yet
