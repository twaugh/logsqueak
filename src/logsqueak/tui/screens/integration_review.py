"""Phase 3 Screen: Integration Review.

This screen allows users to review integration decisions for each knowledge block,
see target page previews, and accept/skip decisions for writing to pages.
"""

from typing import Optional, List, Dict, AsyncIterator
from textual.app import ComposeResult
from textual.screen import Screen
from textual.containers import Container, Vertical, VerticalScroll
from textual.widgets import Static, Footer, Label
from textual.reactive import reactive
from logseq_outline.parser import LogseqBlock, LogseqOutline
from logseq_outline.graph import GraphPaths
from logseq_outline.context import generate_full_context, generate_content_hash
from logsqueak.models.integration_decision import IntegrationDecision
from logsqueak.models.edited_content import EditedContent
from logsqueak.models.background_task import BackgroundTask, BackgroundTaskState
from logsqueak.models.llm_chunks import IntegrationDecisionChunk
from logsqueak.tui.widgets.target_page_preview import TargetPagePreview
from logsqueak.tui.widgets.decision_list import DecisionList
from logsqueak.tui.widgets.status_panel import StatusPanel
from logsqueak.services.file_operations import write_integration_atomic
from logsqueak.services.file_monitor import FileMonitor
from logsqueak.services.llm_client import LLMClient
from logsqueak.services.llm_wrappers import plan_integrations
from logsqueak.services.llm_helpers import filter_skip_exists_blocks
import structlog

logger = structlog.get_logger()


class Phase3Screen(Screen):
    """Phase 3: Integration Review screen.

    Displays integration decisions for each knowledge block with:
    1. Journal context panel (top, full width)
    2. Decision list (bottom-left)
    3. Target page preview (bottom-right)
    """

    DEFAULT_CSS = """
    Phase3Screen {
        layout: vertical;
    }

    #phase3-container {
        height: 100%;
        layout: vertical;
    }

    #block-counter {
        height: auto;
        padding: 0 1;
    }

    #content-panels {
        height: 1fr;
        layout: vertical;
    }

    #bottom-row {
        height: 2fr;
        layout: horizontal;
    }

    #journal-preview {
        height: 1fr;
        border: solid white;
    }

    #decision-list {
        height: 1fr;
        width: 1fr;
        min-width: 30;
        border: solid white;
    }

    #target-page-preview {
        height: 1fr;
        width: 2fr;
    }
    """

    BINDINGS = [
        ("j", "navigate_next_decision", "Next decision"),
        ("k", "navigate_previous_decision", "Previous decision"),
        ("down", "navigate_next_decision", "Next decision"),
        ("up", "navigate_previous_decision", "Previous decision"),
        ("y", "accept_decision", "Accept decision"),
        ("n", "next_block", "Next block"),
        ("a", "accept_all", "Accept all pending"),
        ("tab", "focus_preview", "Focus preview"),
        ("q", "back", "Back"),
    ]

    # Reactive state
    current_block_index = reactive(0)
    current_decision_index = reactive(0)
    llm_decisions_state = reactive(BackgroundTaskState.PENDING)
    llm_decisions_error: Optional[str] = None

    def __init__(
        self,
        journal_blocks: list[LogseqBlock],
        edited_content: list[EditedContent],
        page_contents: Dict[str, LogseqOutline],
        journals: Dict[str, LogseqOutline],
        journal_content: str,
        llm_client: Optional[LLMClient] = None,
        graph_paths: Optional[GraphPaths] = None,
        file_monitor: Optional[FileMonitor] = None,
        decisions: Optional[list[IntegrationDecision]] = None,
        auto_start_workers: bool = True,
        **kwargs
    ):
        """Initialize Phase 3 screen.

        Args:
            journal_blocks: Original journal blocks (for context)
            edited_content: Refined content from Phase 2
            page_contents: Dict mapping page names to LogseqOutline (from Phase 2 RAG)
            journals: Dictionary mapping date string (YYYY-MM-DD) to LogseqOutline
            journal_content: Full journal content (markdown text for preview)
            llm_client: LLM client instance (None for testing with pre-generated decisions)
            graph_paths: GraphPaths instance for file operations
            file_monitor: FileMonitor for concurrent modification detection
            decisions: Pre-generated decisions (for testing, None to generate with LLM)
            auto_start_workers: Whether to auto-start background workers (default True)
        """
        super().__init__(**kwargs)
        self.journal_blocks = journal_blocks
        self.edited_content = edited_content
        self.page_contents = page_contents
        self.journals = journals
        self.journal_content = journal_content
        self.llm_client = llm_client
        self.graph_paths = graph_paths
        self.file_monitor = file_monitor or FileMonitor()
        self.auto_start_workers = auto_start_workers

        # Build mapping from block_id to journal date
        from logseq_outline.context import generate_chunks
        self.block_to_date = {}
        for date, outline in journals.items():
            for block, _, hybrid_id in generate_chunks(outline):
                self.block_to_date[hybrid_id] = date

        # Map block_id to EditedContent for quick lookup
        self.edited_content_map = {ec.block_id: ec for ec in edited_content}

        # Decisions will be populated by LLM worker or passed in for testing
        # IMPORTANT: Use 'is None' check instead of 'or []' to preserve empty list references
        self.decisions = decisions if decisions is not None else []
        self.decisions_by_block = self._group_decisions_by_block(self.decisions)

        # Background tasks tracking - use app.background_tasks in production, local dict in tests
        self._test_background_tasks: Dict[str, BackgroundTask] = {}

        # Track which blocks have decisions ready (for navigation blocking)
        # If decisions are pre-generated, mark all blocks as ready
        self.decisions_ready: Dict[str, bool] = {}
        if decisions:
            for block_id in self.decisions_by_block.keys():
                self.decisions_ready[block_id] = True

        # Track skipped blocks (for status display)
        self.skipped_block_count = 0

        # Track decision count for polling updates (when using shared list from Phase 2)
        self._last_known_decision_count = len(self.decisions)

        # Store polling timer so we can stop it when complete
        self._polling_timer = None

    def _group_decisions_by_block(self, decisions: List[IntegrationDecision]) -> dict:
        """Group decisions by knowledge_block_id.

        Args:
            decisions: All integration decisions

        Returns:
            Dict mapping block_id to list of decisions
        """
        grouped = {}
        for decision in decisions:
            block_id = decision.knowledge_block_id
            if block_id not in grouped:
                grouped[block_id] = []
            grouped[block_id].append(decision)
        return grouped

    def compose(self) -> ComposeResult:
        """Compose the Phase 3 screen layout."""
        with Container(id="phase3-container"):
            # Header showing block progress
            yield Label("", id="block-counter")

            # Three-panel layout: journal context on top, decisions and preview below
            with Vertical(id="content-panels"):
                # Top: Journal context (full width)
                journal_preview = TargetPagePreview(id="journal-preview")
                journal_preview.border_title = "Journal Context"
                yield journal_preview

                # Bottom row: Decision list and target page preview
                with Container(id="bottom-row"):
                    # Bottom-left: Decision list (with graph_path for clickable links)
                    graph_path = self.graph_paths.graph_path if self.graph_paths else None
                    decision_list = DecisionList(graph_path=graph_path)
                    decision_list.border_title = "Integration Points (Ctrl+Click)"
                    yield decision_list

                    # Bottom-right: Target page preview
                    target_preview = TargetPagePreview()
                    target_preview.border_title = "Target Page Preview"
                    yield target_preview

            # Status panel for background tasks (uses app-level shared dict or test dict)
            from logsqueak.tui.app import LogsqueakApp
            if isinstance(self.app, LogsqueakApp):
                yield StatusPanel(background_tasks=self.app.background_tasks)
            else:
                # Test environment - use test dict
                yield StatusPanel(background_tasks=self._test_background_tasks)

        # Footer with keyboard shortcuts
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
        from logsqueak.tui.app import LogsqueakApp
        from textual._context import NoActiveAppError
        try:
            if not isinstance(self.app, LogsqueakApp):
                self._test_background_tasks = value
        except NoActiveAppError:
            self._test_background_tasks = value

    def on_mount(self) -> None:
        """Initialize screen when mounted."""
        # Display first block
        self.call_later(self._display_current_block)

        # Start background workers if enabled
        if self.auto_start_workers and self.llm_client:
            # Start LLM decision generation worker
            self.run_worker(self._llm_decisions_worker(), name="llm_decisions")
            logger.info("llm_decisions_worker_started")
        else:
            # If decisions are pre-generated from Phase 2, check task status
            # The Phase 2 worker may still be adding decisions to the shared list

            # Check if all decisions are already complete
            blocks_ready = len(self.decisions_ready)
            total_blocks = len(self.edited_content)
            all_complete = blocks_ready >= total_blocks

            # Check task status to determine what to do
            from logsqueak.tui.app import LogsqueakApp
            task = None
            if isinstance(self.app, LogsqueakApp):
                task = self.app.background_tasks.get("llm_decisions")

            if task is None and self.llm_client:
                # Worker never started - start it now
                logger.info("phase3_starting_worker", reason="task not found")
                self.run_worker(self._llm_decisions_worker(), name="llm_decisions")
            elif task and task.status == "running":
                # Still running - poll for updates if not all complete
                if not all_complete:
                    self._polling_timer = self.set_interval(0.5, self._check_for_new_decisions)
                    logger.info("phase3_started_polling", blocks_ready=blocks_ready, total_blocks=total_blocks)
                else:
                    logger.info("phase3_all_decisions_ready", blocks_ready=blocks_ready)
            elif task and task.status == "completed":
                # Already done
                logger.info(
                    "llm_decisions_already_complete_phase3",
                    total_blocks=blocks_ready,
                    total_decisions=len(self.decisions)
                )
            elif task and task.status == "failed":
                # Worker failed in Phase 2
                logger.warning("phase3_worker_failed", error=task.error_message)

            # Update status panel
            if isinstance(self.app, LogsqueakApp):
                status_panel = self.query_one(StatusPanel)
                status_panel.update_status()

    def watch_current_decision_index(self, old_index: int, new_index: int) -> None:
        """React to changes in current_decision_index.

        This is called automatically by Textual's reactive system whenever
        current_decision_index changes, regardless of how it changed.
        """
        if old_index != new_index:
            self.call_later(self._update_decision_display)

    async def _display_current_block(self) -> None:
        """Display current knowledge block and its decisions."""
        if not self.journal_blocks:
            return

        current_block = self.journal_blocks[self.current_block_index]
        block_id = current_block.block_id

        # Update block counter
        counter = self.query_one("#block-counter", Label)
        counter.update(
            f"Block {self.current_block_index + 1} of {len(self.journal_blocks)}"
        )

        # Display journal context with highlighted block
        journal_preview = self.query_one("#journal-preview", TargetPagePreview)
        # Get the date for this block
        block_date = self.block_to_date.get(block_id)
        if block_date:
            journal_preview.border_title = f"Journal Context: {block_date}"
        else:
            # Fallback for multi-journal case
            dates = sorted(self.journals.keys())
            if len(dates) == 1:
                journal_preview.border_title = f"Journal Context: {dates[0]}"
            else:
                journal_preview.border_title = f"Journal Context: {dates[0]} to {dates[-1]}"
        await journal_preview.load_preview(self.journal_content, block_id)

        # Get decisions for this block
        block_decisions = self.decisions_by_block.get(block_id, [])

        # Display decisions in list
        decision_list = self.query_one(DecisionList)
        decision_list.load_decisions(block_decisions, self.current_decision_index)

        # Display preview for current decision
        await self._update_preview()

    async def _update_preview(self) -> None:
        """Update target page preview with current decision."""
        block_id = self.journal_blocks[self.current_block_index].block_id
        block_decisions = self.decisions_by_block.get(block_id, [])

        preview = self.query_one("#target-page-preview", TargetPagePreview)

        if not block_decisions or self.current_decision_index >= len(block_decisions):
            preview.border_title = "Target Page Preview"
            if not block_decisions:
                # Show helpful message when no relevant pages found
                message = "No relevant pages found\n\n"
                message += "The AI did not find any suitable pages to integrate this knowledge block.\n\n"
                message += "This could mean:\n"
                message += "- The knowledge is new and doesn't fit existing pages\n"
                message += "- You may want to create a new page for this topic\n"
                message += "- The content might be too specific to this journal entry"
                await preview.load_preview(message, None)
            else:
                preview.clear()
            return

        decision = block_decisions[self.current_decision_index]

        # Update border title with target page name
        preview.border_title = f"Target Page Preview: {decision.target_page}"

        # Generate preview content with integrated block
        preview_text, highlight_block_id = self._generate_preview_with_integration(decision)

        await preview.load_preview(preview_text, highlight_block_id)

    def _generate_preview_with_integration(
        self, decision: IntegrationDecision
    ) -> tuple[str, str | None]:
        """Generate preview with integrated content.

        Args:
            decision: Integration decision to preview

        Returns:
            Tuple of (preview_text, highlight_block_id)
        """
        if not self.graph_paths:
            # No graph paths - return placeholder
            return self._generate_placeholder_preview(decision), None

        # Try to load the target page
        try:
            page_path = self.graph_paths.get_page_path(decision.target_page)
            if page_path.exists():
                page_content = page_path.read_text()
            else:
                # Page doesn't exist yet - create minimal structure
                page_content = f"- {decision.target_page}"
        except Exception as e:
            logger.error("failed_to_load_page", error=str(e), page=decision.target_page)
            return self._generate_placeholder_preview(decision), None

        # Parse the target page
        outline = LogseqOutline.parse(page_content)

        # Create a new block for the knowledge content
        new_block_content = decision.refined_text
        new_block_id = f"preview-{decision.knowledge_block_id}"

        # Find where to insert based on action
        action = decision.action
        target_block_id = decision.target_block_id

        if action == "add_under" and target_block_id:
            # Find the target block and add as child (pass page name for hash matching)
            target_found = self._add_block_under(
                outline, target_block_id, new_block_content, new_block_id, decision.target_page
            )
            if not target_found:
                logger.warning(
                    "target_block_not_found_fallback_to_section",
                    target_id=target_block_id,
                    target_page=decision.target_page,
                    knowledge_block_id=decision.knowledge_block_id,
                    reason="Block may have been modified/deleted since RAG indexing, adding as section instead"
                )
                # Fallback: add as new root-level section if target not found
                new_root = LogseqBlock(
                    content=[new_block_content],
                    indent_level=0,
                    block_id=new_block_id
                )
                outline.blocks.append(new_root)
        elif action == "add_section":
            # Add as new root-level block
            new_root = LogseqBlock(
                content=[new_block_content],
                indent_level=0,
                block_id=new_block_id
            )
            outline.blocks.append(new_root)

        # Render the modified outline
        preview_content = outline.render()

        # Re-parse to get the actual hash IDs
        reparsed = LogseqOutline.parse(preview_content)

        # Find the block with our new content to get its hash
        highlight_block_id = self._find_block_by_content(reparsed.blocks, new_block_content)

        return preview_content, highlight_block_id

    def _add_block_under(
        self, outline: LogseqOutline, target_id: str, content: str, block_id: str, page_name: str
    ) -> bool:
        """Add a new block under the target block.

        Args:
            outline: Outline to modify
            target_id: ID of target block to add under
            content: Content for new block
            block_id: ID for new block
            page_name: Page name (for content hash matching)

        Returns:
            True if target was found and block added
        """
        # Get frontmatter and indent_str from outline (needed for hash matching)
        frontmatter = outline.frontmatter if outline.frontmatter else None
        indent_str = outline.indent_str

        def search_and_add(blocks: list[LogseqBlock], parents: list[LogseqBlock]) -> bool:
            for parent_block in blocks:
                # Check if this is the target (by explicit ID or content hash)
                block_matches = False
                if parent_block.block_id == target_id:
                    block_matches = True
                else:
                    # Try content hash (MUST include page_name, frontmatter, and indent_str to match RAG indexing)
                    full_context = generate_full_context(parent_block, parents, indent_str, frontmatter)
                    content_hash = generate_content_hash(full_context, page_name)
                    if content_hash == target_id:
                        block_matches = True

                if block_matches:
                    # Found the target - add new block as child
                    new_child = LogseqBlock(
                        content=[content],
                        indent_level=parent_block.indent_level + 1,
                        block_id=block_id
                    )
                    parent_block.children.append(new_child)
                    return True

                # Recursively search children
                if search_and_add(parent_block.children, parents + [parent_block]):
                    return True
            return False

        return search_and_add(outline.blocks, [])

    def _find_block_by_content(
        self, blocks: list[LogseqBlock], content: str, parents: list[LogseqBlock] = None
    ) -> str | None:
        """Find a block by its content and return its hash.

        Args:
            blocks: Blocks to search
            content: Content to find
            parents: Parent blocks for context

        Returns:
            Content hash of found block, or None
        """
        if parents is None:
            parents = []

        for block in blocks:
            # Check if this block contains our content
            if block.content and content in block.content[0]:
                # Generate its hash
                full_context = generate_full_context(block, parents)
                return generate_content_hash(full_context)

            # Search children
            result = self._find_block_by_content(block.children, content, parents + [block])
            if result:
                return result

        return None

    def _generate_placeholder_preview(self, decision: IntegrationDecision) -> str:
        """Generate placeholder preview when page can't be loaded.

        Args:
            decision: Integration decision

        Returns:
            Placeholder text
        """
        return f"""Target Page: {decision.target_page}
Action: {decision.action}
Confidence: {decision.confidence:.0%}

{decision.refined_text}

[Unable to load target page for preview]
"""

    def action_navigate_next_decision(self) -> None:
        """Navigate to next decision for current block."""
        block_id = self.journal_blocks[self.current_block_index].block_id
        block_decisions = self.decisions_by_block.get(block_id, [])

        if self.current_decision_index < len(block_decisions) - 1:
            self.current_decision_index += 1
            # The watch_current_decision_index watcher will handle the update

    def action_navigate_previous_decision(self) -> None:
        """Navigate to previous decision for current block."""
        if self.current_decision_index > 0:
            self.current_decision_index -= 1
            # The watch_current_decision_index watcher will handle the update

    async def _update_decision_display(self) -> None:
        """Update decision list and preview after navigation."""
        block_id = self.journal_blocks[self.current_block_index].block_id
        block_decisions = self.decisions_by_block.get(block_id, [])

        # Update decision list highlight
        decision_list = self.query_one(DecisionList)
        decision_list.set_current_index(self.current_decision_index)

        # Update preview
        await self._update_preview()

    async def action_accept_decision(self) -> None:
        """Accept current decision and trigger write operation."""
        block_id = self.journal_blocks[self.current_block_index].block_id
        block_decisions = self.decisions_by_block.get(block_id, [])

        if not block_decisions or self.current_decision_index >= len(block_decisions):
            return

        decision = block_decisions[self.current_decision_index]

        # Skip if already completed
        if decision.write_status == "completed":
            logger.info("decision_already_completed", decision_id=id(decision))
            return

        # Perform write operation
        try:
            await self.write_integration(decision)
            decision.write_status = "completed"
            logger.info("decision_accepted", decision_id=id(decision))
        except Exception as e:
            decision.write_status = "failed"
            decision.error_message = str(e)
            logger.error("decision_write_failed", error=str(e))

        # Refresh decision list to show updated status
        decision_list = self.query_one(DecisionList)
        decision_list.load_decisions(block_decisions, self.current_decision_index)

    async def write_integration(self, decision: IntegrationDecision) -> None:
        """Write integration to disk (mock for now).

        Args:
            decision: Decision to write

        Raises:
            ValueError: If write fails
        """
        # Full implementation would call write_integration_atomic
        # For now, this is a placeholder that tests can mock
        if self.graph_paths:
            # Get the journal date for this block
            journal_date = self.block_to_date.get(decision.knowledge_block_id)
            if not journal_date:
                # Fallback: use first date if we can't determine which journal
                journal_date = sorted(self.journals.keys())[0]

            await write_integration_atomic(
                decision=decision,
                journal_date=journal_date,
                graph_paths=self.graph_paths,
                file_monitor=self.file_monitor
            )

    def action_next_block(self) -> None:
        """Advance to next knowledge block.

        Blocks navigation if next block's decisions aren't ready yet.
        Shows completion summary if at last block.
        """
        if self.current_block_index >= len(self.journal_blocks) - 1:
            # At last block - show completion summary
            self.run_worker(self._show_completion_summary(), exclusive=True)
            return

        # Check if next block's decisions are ready
        next_block_id = self.journal_blocks[self.current_block_index + 1].block_id
        if not self.decisions_ready.get(next_block_id, False):
            # Block not ready yet - show message
            logger.info(
                "navigation_blocked_decisions_pending",
                next_block_id=next_block_id
            )
            # TODO: Show user feedback about waiting for decisions
            return

        # Navigate to next block
        self.current_block_index += 1
        self.current_decision_index = 0
        self.call_later(self._display_current_block)

    async def action_accept_all(self) -> None:
        """Accept all pending decisions for current block."""
        block_id = self.journal_blocks[self.current_block_index].block_id
        block_decisions = self.decisions_by_block.get(block_id, [])

        for decision in block_decisions:
            if decision.write_status == "pending":
                try:
                    await self.write_integration(decision)
                    decision.write_status = "completed"
                except Exception as e:
                    decision.write_status = "failed"
                    decision.error_message = str(e)

        # Refresh decision list
        decision_list = self.query_one(DecisionList)
        decision_list.load_decisions(block_decisions, self.current_decision_index)

        # Advance to next block after accepting all
        self.action_next_block()

    def action_focus_preview(self) -> None:
        """Focus the target page preview widget."""
        preview = self.query_one("#target-page-preview", TargetPagePreview)
        preview.focus()

    def action_back(self) -> None:
        """Return to previous screen."""
        self.dismiss()

    # Background worker methods

    def _check_for_new_decisions(self) -> None:
        """Poll for new decisions added to the shared list by Phase 2 worker.

        This is called periodically when Phase3 receives pre-generated decisions
        from Phase 2, as the Phase 2 worker may still be streaming in new decisions.
        """
        current_count = len(self.decisions)

        logger.debug(
            "polling_for_decisions",
            current=current_count,
            last_known=self._last_known_decision_count
        )

        # Check if new decisions have been added
        if current_count != self._last_known_decision_count:
            # Re-group decisions by block
            self.decisions_by_block = self._group_decisions_by_block(self.decisions)

            # Mark newly ready blocks
            newly_ready_count = 0
            for block_id in self.decisions_by_block.keys():
                if block_id not in self.decisions_ready:
                    self.decisions_ready[block_id] = True
                    newly_ready_count += 1
                    logger.info(
                        "new_decisions_detected_phase3",
                        block_id=block_id,
                        num_decisions=len(self.decisions_by_block[block_id])
                    )

            # Update progress in background task if it exists
            from logsqueak.tui.app import LogsqueakApp
            if isinstance(self.app, LogsqueakApp) and "llm_decisions" in self.app.background_tasks:
                task = self.app.background_tasks["llm_decisions"]

                # Update progress based on blocks with decisions ready
                blocks_ready = len(self.decisions_ready)
                total_blocks = len(self.edited_content)
                task.progress_current = blocks_ready

                # Check if task finished (completed or failed) - stop polling
                if task.status in ("completed", "failed"):
                    # Task finished - stop polling
                    if self._polling_timer is not None:
                        self._polling_timer.stop()
                        self._polling_timer = None
                        logger.info("phase3_stopped_polling", reason=f"task {task.status}")

                    if task.status == "completed":
                        logger.info(
                            "llm_decisions_complete_phase3_polling",
                            total_blocks=blocks_ready,
                            total_decisions=current_count
                        )

                # Update status panel
                status_panel = self.query_one(StatusPanel)
                status_panel.update_status()

            # Update display if we're viewing a block that just got decisions
            current_block_id = self.journal_blocks[self.current_block_index].block_id
            if current_block_id in self.decisions_by_block:
                self.call_later(self._display_current_block)

            # Update the known count
            self._last_known_decision_count = current_count

    async def _convert_chunks_to_decisions(
        self,
        chunk_stream: AsyncIterator[IntegrationDecisionChunk]
    ) -> AsyncIterator[IntegrationDecision]:
        """Convert IntegrationDecisionChunk stream to IntegrationDecision stream.

        Adds refined_text from edited_content to each chunk.

        Args:
            chunk_stream: Stream of IntegrationDecisionChunk from LLM

        Yields:
            IntegrationDecision with refined_text populated
        """
        async for chunk in chunk_stream:
            # Look up the refined_text from edited_content
            edited_content = self.edited_content_map.get(chunk.knowledge_block_id)
            if not edited_content:
                logger.warning(
                    "chunk_missing_edited_content",
                    block_id=chunk.knowledge_block_id
                )
                continue

            # Convert chunk to full decision
            decision = IntegrationDecision(
                knowledge_block_id=chunk.knowledge_block_id,
                target_page=chunk.target_page,
                action=chunk.action,
                target_block_id=chunk.target_block_id,
                target_block_title=chunk.target_block_title,
                confidence=chunk.confidence,
                refined_text=edited_content.current_content,
                reasoning=chunk.reasoning,
                write_status="pending",
            )
            yield decision

    async def _llm_decisions_worker(self) -> None:
        """Worker: Generate integration decisions using LLM with batching and filtering."""
        from logsqueak.tui.app import LogsqueakApp

        # Create background task
        if isinstance(self.app, LogsqueakApp):
            self.app.background_tasks["llm_decisions"] = BackgroundTask(
                task_type="llm_decisions",
                status="running",
                progress_current=0,
                progress_total=len(self.edited_content),
            )

            status_panel = self.query_one(StatusPanel)
            status_panel.update_status()

        try:
            # Acquire LLM slot (blocks until request can proceed)
            from logsqueak.tui.app import LLMRequestPriority
            request_id = "llm_decisions_phase3"
            await self.app.acquire_llm_slot(request_id, LLMRequestPriority.INTEGRATION)

            try:
                # Step 1: Stream raw decisions from LLM
                logger.info("llm_decisions_starting", num_blocks=len(self.edited_content))

                # Note: This code path is no longer used in production.
                # Decisions are now generated in Phase 2 and passed via shared list.
                # This is only for testing with mocked LLM client.
                raise NotImplementedError(
                    "Phase 3 LLM worker is deprecated - decisions should come from Phase 2"
                )

                # Step 2: Convert IntegrationDecisionChunk to IntegrationDecision
                # (add refined_text from edited_content)
                converted_stream = self._convert_chunks_to_decisions(raw_decision_stream)

                # Step 3: Filter out skip_exists blocks and track count
                filtered_stream = filter_skip_exists_blocks(converted_stream)

                # Step 4: Process filtered decisions (already grouped by block via per-block LLM calls)
                block_count = 0
                decisions_in_current_block = []
                current_block_id = None

                async for decision in filtered_stream:
                    # Track when we move to a new block
                    if current_block_id is not None and decision.knowledge_block_id != current_block_id:
                        # Finished a block - store its decisions
                        self.decisions.extend(decisions_in_current_block)
                        self.decisions_by_block[current_block_id] = decisions_in_current_block

                        # Mark block as ready for navigation
                        self.decisions_ready[current_block_id] = True

                        # Update display if this is the current block
                        journal_block_id = self.journal_blocks[self.current_block_index].block_id
                        if current_block_id == journal_block_id:
                            self.call_later(self._display_current_block)

                        block_count += 1

                        # Update progress
                        from logsqueak.tui.app import LogsqueakApp
                        if isinstance(self.app, LogsqueakApp):
                            self.app.background_tasks["llm_decisions"].progress_current = block_count
                            status_panel = self.query_one(StatusPanel)
                            status_panel.update_status()

                        logger.info(
                            "llm_decisions_batch_complete",
                            block_id=current_block_id,
                            num_decisions=len(decisions_in_current_block)
                        )

                        decisions_in_current_block = []

                    # Add decision to current block
                    decisions_in_current_block.append(decision)
                    current_block_id = decision.knowledge_block_id

                # Handle final block if there are any remaining decisions
                if decisions_in_current_block:
                    self.decisions.extend(decisions_in_current_block)
                    self.decisions_by_block[current_block_id] = decisions_in_current_block
                    self.decisions_ready[current_block_id] = True

                    # Update display if this is the current block
                    journal_block_id = self.journal_blocks[self.current_block_index].block_id
                    if current_block_id == journal_block_id:
                        self.call_later(self._display_current_block)

                    block_count += 1

                    logger.info(
                        "llm_decisions_batch_complete",
                        block_id=current_block_id,
                        num_decisions=len(decisions_in_current_block)
                    )

                # Get skipped count from filter
                self.skipped_block_count = filtered_stream.skipped_count

                # Mark complete
                self.llm_decisions_state = BackgroundTaskState.COMPLETED
                from logsqueak.tui.app import LogsqueakApp
                if isinstance(self.app, LogsqueakApp):
                    self.app.background_tasks["llm_decisions"].status = "completed"
                    status_panel = self.query_one(StatusPanel)
                    status_panel.update_status()

                # Update block counter to show skipped count
                self._update_block_counter_with_skip_count()

                logger.info(
                    "llm_decisions_complete",
                    total_blocks=block_count,
                    skipped_blocks=self.skipped_block_count
                )

            finally:
                # Release LLM slot
                self.app.release_llm_slot(request_id)

        except Exception as e:
            # Mark failed
            self.llm_decisions_state = BackgroundTaskState.FAILED
            self.llm_decisions_error = str(e)
            from logsqueak.tui.app import LogsqueakApp
            if isinstance(self.app, LogsqueakApp):
                self.app.background_tasks["llm_decisions"].status = "failed"
                self.app.background_tasks["llm_decisions"].error_message = str(e)
                status_panel = self.query_one(StatusPanel)
                status_panel.update_status()

            logger.error(
                "llm_decisions_error",
                error=str(e),
                exc_info=True
            )

    def _update_block_counter_with_skip_count(self) -> None:
        """Update block counter to show new integrations vs already recorded."""
        try:
            counter = self.query_one("#block-counter", Label)
            total_blocks = len(self.edited_content)
            new_integrations = total_blocks - self.skipped_block_count
            counter.update(
                f"Block {self.current_block_index + 1} of {len(self.journal_blocks)} | "
                f"{new_integrations} new integrations, {self.skipped_block_count} already recorded"
            )
        except Exception:
            pass  # Widget not mounted yet

    async def _show_completion_summary(self) -> None:
        """Show completion summary screen with statistics and journal link."""
        # Calculate statistics
        total_decisions = len(self.decisions)
        completed_count = sum(1 for d in self.decisions if d.write_status == "completed")
        failed_count = sum(1 for d in self.decisions if d.write_status == "failed")
        pending_count = sum(1 for d in self.decisions if d.write_status == "pending")

        total_blocks = len(self.edited_content)
        new_integrations = total_blocks - self.skipped_block_count

        # Build summary text in Logseq bullet format
        summary_lines = [
            "- # Knowledge Extraction Complete",
            "  - ## Summary Statistics",
            f"    - **Total knowledge blocks processed:** {total_blocks}",
            f"    - **New integrations:** {new_integrations}",
            f"    - **Already recorded:** {self.skipped_block_count}",
            "  - ## Write Operations",
            f"    - **Total integration points:** {total_decisions}",
            f"    - **Completed:** {completed_count} ✓",
            f"    - **Pending:** {pending_count} ⊙",
        ]

        if failed_count > 0:
            summary_lines.append(f"    - **Failed:** {failed_count} ⚠")

        # Add journal link(s)
        if self.graph_paths:
            dates = sorted(self.journals.keys())
            if len(dates) == 1:
                journal_path = self.graph_paths.get_journal_path(dates[0])
                summary_lines.extend([
                    "  - ## Journal Entry",
                    "    - Journal entry updated with provenance markers:",
                    f"      `{journal_path}`",
                    "    - You can view the processed blocks in your Logseq graph with the `extracted-to::` property.",
                ])
            else:
                summary_lines.extend([
                    "  - ## Journal Entries",
                    "    - Journal entries updated with provenance markers:",
                ])
                for date in dates:
                    journal_path = self.graph_paths.get_journal_path(date)
                    summary_lines.append(f"      - `{journal_path}`")
                summary_lines.append("    - You can view the processed blocks in your Logseq graph with the `extracted-to::` property.")

        summary_lines.extend([
            "  - ---",
            "  - Press 'q' to return to content editing or Ctrl+Q to exit the application.",
        ])

        summary_text = "\n".join(summary_lines)

        # Update counter first (synchronous)
        counter = self.query_one("#block-counter", Label)
        counter.update("✓ Processing Complete")

        # Clear decision list and target preview (synchronous)
        decision_list = self.query_one(DecisionList)
        decision_list.clear()

        target_preview = self.query_one("#target-page-preview", TargetPagePreview)
        target_preview.border_title = ""
        target_preview.clear()

        # Load summary into journal preview (async)
        journal_preview = self.query_one("#journal-preview", TargetPagePreview)
        journal_preview.border_title = "Extraction Complete"
        await journal_preview.load_preview(summary_text, None)

        logger.info(
            "completion_summary_displayed",
            total_blocks=total_blocks,
            new_integrations=new_integrations,
            skipped=self.skipped_block_count,
            completed=completed_count,
            failed=failed_count,
            pending=pending_count
        )
