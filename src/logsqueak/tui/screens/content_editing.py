"""Phase 2 Screen: Content Editing.

This screen allows users to review and refine knowledge blocks before integration.
Users can see LLM-suggested rewordings, manually edit content, and review RAG search results.
"""

from typing import Optional, Dict
import asyncio
from textual.app import ComposeResult
from textual.screen import Screen
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Footer, Label
from textual.reactive import reactive
from logseq_outline.parser import LogseqBlock, LogseqOutline
from logsqueak.models.edited_content import EditedContent
from logsqueak.models.background_task import BackgroundTask, BackgroundTaskState
from logsqueak.tui.widgets.content_editor import ContentEditor
from logsqueak.tui.widgets.status_panel import StatusPanel
from logsqueak.tui.widgets.target_page_preview import TargetPagePreview
from logsqueak.services.llm_client import LLMClient
from logsqueak.services.llm_wrappers import reword_content, plan_integrations
from logsqueak.services.llm_helpers import filter_skip_exists_blocks
from logsqueak.services.rag_search import RAGSearch
from logsqueak.models.integration_decision import IntegrationDecision
from logsqueak.models.llm_chunks import IntegrationDecisionChunk
from logseq_outline.graph import GraphPaths
import structlog

logger = structlog.get_logger()


class Phase2Screen(Screen):
    """Phase 2: Content Editing screen.

    Displays selected knowledge blocks with three vertical panels (top to bottom):
    1. Top: Original hierarchical context (read-only)
    2. Middle: LLM reworded version (when available)
    3. Bottom: Current editable content (TextArea)
    """

    # CSS for balanced vertical layout
    DEFAULT_CSS = """
    Phase2Screen {
        layout: vertical;
    }

    #phase2-container {
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

    #original-panel {
        height: 1fr;
        min-height: 3;
        layout: vertical;
    }

    #llm-panel {
        height: 1fr;
        min-height: 3;
        layout: vertical;
    }

    #editor-panel {
        height: 1fr;
        min-height: 4;
        border: solid $accent;
        border-title-align: center;
        layout: vertical;
    }

    #original-context {
        height: 1fr;
    }

    #llm-reworded {
        height: 1fr;
    }

    ContentEditor {
        height: 1fr;
    }
    """

    BINDINGS = [
        ("j", "navigate_next", "Next block"),
        ("k", "navigate_previous", "Previous block"),
        ("down", "navigate_next", "Next block"),
        ("up", "navigate_previous", "Previous block"),
        ("tab", "toggle_focus", "Focus editor"),
        ("a", "accept_llm", "Accept LLM version"),
        ("r", "revert_original", "Revert to original"),
        ("n", "next_phase", "Continue"),
        ("q", "back", "Back"),
    ]

    # Reactive state
    current_block_index = reactive(0)
    page_indexing_state = reactive(BackgroundTaskState.PENDING)
    rag_search_state = reactive(BackgroundTaskState.PENDING)
    rag_search_progress = reactive(0)
    rag_search_total = reactive(0)
    page_indexing_error: Optional[str] = None
    rag_search_error: Optional[str] = None

    def __init__(
        self,
        blocks: list[LogseqBlock],
        edited_content: list[EditedContent],
        journals: Dict[str, LogseqOutline],
        graph_paths: GraphPaths,
        llm_client: Optional[LLMClient] = None,
        rag_search: Optional[RAGSearch] = None,
        auto_start_workers: bool = True,
        **kwargs
    ):
        """Initialize Phase 2 screen.

        Args:
            blocks: List of selected knowledge blocks
            edited_content: List of EditedContent for each block
            journals: Dictionary mapping date string (YYYY-MM-DD) to LogseqOutline
            graph_paths: GraphPaths instance for loading page contents
            llm_client: LLM client instance (None for testing)
            rag_search: RAG search service instance (None for testing)
            auto_start_workers: Whether to auto-start background workers (default True)
        """
        super().__init__(**kwargs)
        self.blocks = blocks
        self.edited_content = edited_content
        self.journals = journals
        self.graph_paths = graph_paths
        self.llm_client = llm_client
        self.rag_search = rag_search
        self.auto_start_workers = auto_start_workers

        # Map block_id to EditedContent for quick lookup
        self.edited_content_map = {ec.block_id: ec for ec in edited_content}

        # Background tasks tracking - use app.background_tasks in production, local dict in tests
        self._test_background_tasks: Dict[str, BackgroundTask] = {}

        # Storage for RAG search results (for Phase 3)
        self.candidate_page_names: Dict[str, list[str]] = {}  # block_id -> page names
        self.page_contents: Dict[str, LogseqOutline] = {}  # page_name -> outline

    def compose(self) -> ComposeResult:
        """Compose the Phase 2 screen layout."""
        with Container(id="phase2-container"):
            # Header showing block progress
            yield Label("", id="block-counter")

            # Three-panel layout (vertical - top to bottom)
            with Vertical(id="content-panels"):
                # Top panel: Original context (using TargetPagePreview for syntax highlighting)
                with Container(id="original-panel"):
                    yield TargetPagePreview(id="original-context")

                # Middle panel: LLM reworded version (using TargetPagePreview for syntax highlighting)
                with Container(id="llm-panel"):
                    yield TargetPagePreview(id="llm-reworded")

                # Bottom panel: Editable current content
                with Container(id="editor-panel"):
                    yield ContentEditor()

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

    async def on_mount(self) -> None:
        """Handle screen mount event."""
        # Set border titles for panels
        self.query_one("#original-panel").border_title = "Original Context"
        self.query_one("#llm-panel").border_title = "LLM Reworded"
        self.query_one("#editor-panel").border_title = "Current Content (Editable)"

        # Update initial display
        await self._update_display()

        # Start background workers if enabled
        if self.auto_start_workers:
            self._start_background_workers()

    def _start_background_workers(self) -> None:
        """Start background workers for LLM rewording and RAG search."""
        if self.llm_client:
            # Start LLM rewording worker
            llm_worker = self.run_worker(self._llm_rewording_worker(), name="llm_rewording")
            logger.info("llm_rewording_worker_started")

            # Register worker for cancellation
            from logsqueak.tui.app import LogsqueakApp
            if isinstance(self.app, LogsqueakApp):
                self.app.register_llm_worker("llm_rewording", llm_worker)

        if self.rag_search:
            # Start RAG search worker (will be implemented in T097a)
            self.run_worker(self._rag_search_worker(), name="rag_search")
            logger.info("rag_search_worker_started")

    def _clean_context_for_display(self, text: str) -> str:
        """Clean context text for display by removing id:: properties.

        Removes:
        - id:: property lines (internal metadata)

        Args:
            text: Context text

        Returns:
            Clean text for display
        """
        if not text:
            return ""

        lines = text.split('\n')
        clean_lines = []

        for line in lines:
            # Skip id:: property lines
            stripped = line.strip()
            if stripped.startswith('id::'):
                continue

            clean_lines.append(line)

        return '\n'.join(clean_lines)

    def _convert_to_logseq_bullets(self, text: str) -> str:
        """Convert plain indented text to Logseq bullet format.

        Takes text with indentation (spaces) and converts to bullets.
        Example:
            "Parent\n  Child\n    Grandchild"
        Becomes:
            "- Parent\n  - Child\n    - Grandchild"

        Args:
            text: Indented plain text

        Returns:
            Logseq markdown with bullets
        """
        if not text:
            return ""

        lines = text.split('\n')
        result = []

        for line in lines:
            if not line.strip():
                result.append(line)
                continue

            # Count leading whitespace (spaces or tabs)
            stripped = line.lstrip()
            indent_count = len(line) - len(stripped)
            # Preserve the original indentation characters (spaces or tabs)
            indent = line[:indent_count]

            # Add bullet if line doesn't already have one
            if not stripped.startswith('- '):
                result.append(f"{indent}- {stripped}")
            else:
                result.append(line)

        return '\n'.join(result)

    async def _update_display(self) -> None:
        """Update all display panels for current block."""
        if not self.edited_content:
            return

        # Get current edited content
        current_ec = self.edited_content[self.current_block_index]
        current_block = self.blocks[self.current_block_index]

        # Update block counter
        counter = self.query_one("#block-counter", Label)
        counter.update(
            f"Block {self.current_block_index + 1} of {len(self.edited_content)}"
        )

        # Update original context panel using TargetPagePreview
        original_context = self.query_one("#original-context", TargetPagePreview)
        # Clean context (remove id:: properties), then convert to bullets
        clean_context = self._clean_context_for_display(current_ec.hierarchical_context)
        context_with_bullets = self._convert_to_logseq_bullets(clean_context)
        await original_context.load_preview(context_with_bullets)

        # Update LLM reworded panel using TargetPagePreview
        llm_reworded = self.query_one("#llm-reworded", TargetPagePreview)
        if current_ec.rewording_complete and current_ec.reworded_content:
            # Format as single block for display
            content_as_block = f"- {current_ec.reworded_content}"
            await llm_reworded.load_preview(content_as_block)
        else:
            # Clear preview and show waiting message
            llm_reworded.clear()

        # Update editor with current content
        editor = self.query_one(ContentEditor)
        editor.load_content(current_ec.current_content)

    async def action_navigate_next(self) -> None:
        """Navigate to next block (j or down arrow)."""
        # Only navigate if editor is not focused
        editor = self.query_one(ContentEditor)
        if editor.has_focus:
            return

        # Save current content before navigating
        self._save_current_content()

        # Move to next block
        if self.current_block_index < len(self.edited_content) - 1:
            old_index = self.current_block_index
            self.current_block_index += 1
            logger.info("user_action_navigate_next", from_index=old_index, to_index=self.current_block_index)
            await self._update_display()

    async def action_navigate_previous(self) -> None:
        """Navigate to previous block (k or up arrow)."""
        # Only navigate if editor is not focused
        editor = self.query_one(ContentEditor)
        if editor.has_focus:
            return

        # Save current content before navigating
        self._save_current_content()

        # Move to previous block
        if self.current_block_index > 0:
            old_index = self.current_block_index
            self.current_block_index -= 1
            logger.info("user_action_navigate_previous", from_index=old_index, to_index=self.current_block_index)
            await self._update_display()

    def action_toggle_focus(self) -> None:
        """Toggle focus on/off the editor (Tab key)."""
        editor = self.query_one(ContentEditor)

        if editor.has_focus:
            # Unfocus the editor
            self.set_focus(None)
            logger.info("user_action_toggle_focus", focused=False)
        else:
            # Focus the editor
            editor.focus()
            logger.info("user_action_toggle_focus", focused=True)

    def action_accept_llm(self) -> None:
        """Accept LLM reworded version ('a' key)."""
        # Only work when editor is not focused
        editor = self.query_one(ContentEditor)
        if editor.has_focus:
            return

        # Get current edited content
        current_ec = self.edited_content[self.current_block_index]

        # Check if LLM rewording is available
        if not current_ec.rewording_complete or not current_ec.reworded_content:
            return

        # Update current content with reworded version
        current_ec.current_content = current_ec.reworded_content

        # Update editor display
        editor.load_content(current_ec.current_content)

        logger.info(
            "llm_version_accepted",
            block_id=current_ec.block_id
        )

    def action_revert_original(self) -> None:
        """Revert to original content ('r' key)."""
        # Only work when editor is not focused
        editor = self.query_one(ContentEditor)
        if editor.has_focus:
            return

        # Get current edited content
        current_ec = self.edited_content[self.current_block_index]

        # Revert to original
        current_ec.current_content = current_ec.original_content

        # Update editor display
        editor.load_content(current_ec.current_content)

        logger.info(
            "content_reverted",
            block_id=current_ec.block_id
        )

    def action_next_phase(self) -> None:
        """Proceed to next phase ('n' key)."""
        # Check if RAG search is complete
        if self.rag_search_state != BackgroundTaskState.COMPLETED:
            # Block progression
            return

        # Save current content
        self._save_current_content()

        # Start LLM decisions worker if not already started
        # (normally starts after rewording completes, but user might press 'n' before that)
        from logsqueak.tui.app import LogsqueakApp
        task_not_started = True
        if isinstance(self.app, LogsqueakApp):
            task_not_started = "llm_decisions" not in self.app.background_tasks

        if self.llm_client and task_not_started:
            # Save all content before starting worker (in case there are unsaved edits)
            self._save_all_content()
            logger.info("saved_current_edits_before_integration_on_transition")

            logger.info("llm_decisions_worker_starting_on_transition")
            self.run_worker(self._llm_decisions_worker(), name="llm_decisions")

        logger.info("phase2_complete", blocks_edited=len(self.edited_content))

        # Update refined_text in all existing decisions with latest edited content
        # This ensures decisions use the user's final edits, even if they were
        # created by the worker while user was still editing
        from logsqueak.tui.app import LogsqueakApp
        if isinstance(self.app, LogsqueakApp):
            for decision in self.app.integration_decisions:
                edited_content = self.edited_content_map.get(decision.knowledge_block_id)
                if edited_content:
                    decision.refined_text = edited_content.current_content
                    logger.debug(
                        "updated_decision_refined_text",
                        block_id=decision.knowledge_block_id,
                        refined_text_length=len(decision.refined_text)
                    )

        # Call app transition method
        from logsqueak.tui.app import LogsqueakApp
        if isinstance(self.app, LogsqueakApp):
            # Convert candidate_page_names dict to list of unique page names
            unique_pages = sorted(set(
                page for pages in self.candidate_page_names.values() for page in pages
            ))

            # No need to pass decisions - Phase3 will access app.integration_decisions directly
            self.app.transition_to_phase3(
                edited_content=self.edited_content,
                candidate_pages=unique_pages,
                page_contents=self.page_contents,
            )

    def action_back(self) -> None:
        """Go back to Phase 1 ('q' key)."""
        # Save current content
        self._save_current_content()

        logger.info("user_action_back_to_phase1")
        # TODO: Transition back to Phase 1
        self.app.pop_screen()

    def _save_current_content(self) -> None:
        """Save editor content to current EditedContent."""
        if not self.edited_content:
            return

        editor = self.query_one(ContentEditor)
        current_ec = self.edited_content[self.current_block_index]

        # Update current content from editor
        current_ec.current_content = editor.get_content()

    def _save_all_content(self) -> None:
        """Save current editor content before starting integration worker.

        This ensures that any unsaved edits in the currently displayed block
        are captured before the Integration Decision worker reads the content.
        """
        if not self.edited_content:
            return

        try:
            editor = self.query_one(ContentEditor)
            current_ec = self.edited_content[self.current_block_index]
            current_ec.current_content = editor.get_content()
        except Exception:
            # Widget might not be mounted in test scenarios
            pass

    # Background worker methods

    async def _llm_rewording_worker(self) -> None:
        """Worker: Generate LLM reworded versions using reword_content wrapper."""
        # Create background task
        self._background_tasks["llm_rewording"] = BackgroundTask(
            task_type="llm_rewording",
            status="running",
            progress_current=0,
            progress_total=len(self.edited_content),
        )

        status_panel = self.query_one(StatusPanel)
        status_panel.update_status()

        try:
            # Acquire LLM slot (blocks until request can proceed)
            from logsqueak.tui.app import LLMRequestPriority
            request_id = "llm_rewording"
            await self.app.acquire_llm_slot(request_id, LLMRequestPriority.REWORDING)

            try:
                # Stream reworded content from LLM
                count = 0
                # Get first journal for indent detection (all should have same indent)
                first_journal = next(iter(self.journals.values()))
                async for chunk in reword_content(
                    self.llm_client,
                    self.edited_content,
                    first_journal
                ):
                    block_id = chunk.block_id
                    if block_id in self.edited_content_map:
                        ec = self.edited_content_map[block_id]

                        # Update reworded content
                        ec.reworded_content = chunk.reworded_content
                        ec.rewording_complete = True

                        # Update display if this is the current block
                        current_ec = self.edited_content[self.current_block_index]
                        if block_id == current_ec.block_id:
                            # Update LLM panel
                            llm_reworded = self.query_one("#llm-reworded", TargetPagePreview)
                            content_as_block = f"- {chunk.reworded_content}"
                            await llm_reworded.load_preview(content_as_block)

                        logger.info(
                            "llm_rewording_chunk",
                            block_id=block_id,
                            reworded_length=len(chunk.reworded_content)
                        )
                    else:
                        logger.warning(
                            "llm_rewording_unknown_block",
                            block_id=block_id
                        )

                    count += 1

                    # Update progress
                    self._background_tasks["llm_rewording"].progress_current = count
                    status_panel.update_status()

                # Mark complete and remove from background tasks
                del self._background_tasks["llm_rewording"]
                status_panel.update_status()

                logger.info(
                    "llm_rewording_complete",
                    total_blocks=count
                )

                # NOTE: LLM decisions worker is NOT started here.
                # It will start after RAG search completes (in _rag_search_worker),
                # because it requires page_contents to be populated first.

            finally:
                # Release LLM slot
                self.app.release_llm_slot(request_id)

        except asyncio.CancelledError:
            # Worker was cancelled (e.g., during screen transition)
            logger.info("llm_rewording_cancelled")
            # Clean up background task
            if "llm_rewording" in self._background_tasks:
                del self._background_tasks["llm_rewording"]
                try:
                    status_panel = self.query_one(StatusPanel)
                    status_panel.update_status()
                except Exception:
                    pass  # Widget not mounted (screen already transitioned)
            raise  # Re-raise to properly cancel the task

        except Exception as e:
            # Mark failed
            self._background_tasks["llm_rewording"].status = "failed"
            self._background_tasks["llm_rewording"].error_message = str(e)
            status_panel.update_status()

            logger.error(
                "llm_rewording_error",
                error=str(e),
                exc_info=True
            )

    async def _llm_decisions_worker(self) -> None:
        """Worker: Generate integration decisions using LLM.

        This runs in Phase 2 after rewording completes, so decisions
        are ready when user transitions to Phase 3.
        """
        from logsqueak.tui.app import LogsqueakApp

        # Create background task in shared app dict
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
            request_id = "llm_decisions"
            await self.app.acquire_llm_slot(request_id, LLMRequestPriority.INTEGRATION)

            try:
                # Stream raw decisions from LLM
                logger.info("llm_decisions_starting_phase2", num_blocks=len(self.edited_content))

                raw_decision_stream = plan_integrations(
                    self.llm_client,
                    self.edited_content,
                    self.page_contents,
                    self.candidate_chunks  # Pass hierarchical chunks
                )

                # Convert chunks to full decisions (add refined_text)
                async def convert_chunks():
                    async for chunk in raw_decision_stream:
                        edited_content = self.edited_content_map.get(chunk.knowledge_block_id)
                        if not edited_content:
                            logger.warning(
                                "chunk_missing_edited_content_phase2",
                                block_id=chunk.knowledge_block_id
                            )
                            continue

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

                converted_stream = convert_chunks()

                # Filter out skip_exists blocks
                filtered_stream = filter_skip_exists_blocks(converted_stream)

                # Collect all decisions (already grouped by block via per-block LLM calls)
                block_count = 0
                decisions_in_current_block = []
                current_block_id = None

                async for decision in filtered_stream:
                    # Track when we move to a new block
                    if current_block_id is not None and decision.knowledge_block_id != current_block_id:
                        # Finished a block - store its decisions
                        from logsqueak.tui.app import LogsqueakApp
                        if isinstance(self.app, LogsqueakApp):
                            before_count = len(self.app.integration_decisions)
                            self.app.integration_decisions.extend(decisions_in_current_block)
                            after_count = len(self.app.integration_decisions)
                            logger.info(
                                "decisions_added_to_shared_list",
                                before=before_count,
                                after=after_count,
                                batch_size=len(decisions_in_current_block)
                            )

                        block_count += 1
                        decisions_in_current_block = []

                    # Add decision to current block
                    decisions_in_current_block.append(decision)
                    current_block_id = decision.knowledge_block_id

                # Handle final block if there are any remaining decisions
                if decisions_in_current_block:
                    from logsqueak.tui.app import LogsqueakApp
                    if isinstance(self.app, LogsqueakApp):
                        before_count = len(self.app.integration_decisions)
                        self.app.integration_decisions.extend(decisions_in_current_block)
                        after_count = len(self.app.integration_decisions)
                        logger.info(
                            "decisions_added_to_shared_list",
                            before=before_count,
                            after=after_count,
                            batch_size=len(decisions_in_current_block)
                        )

                    block_count += 1
                    logger.info(
                        "llm_decisions_batch_complete_phase2",
                        block_id=current_block_id,
                        num_decisions=len(decisions_in_current_block)
                    )

                # Mark complete
                from logsqueak.tui.app import LogsqueakApp
                if isinstance(self.app, LogsqueakApp):
                    logger.info("phase2_marking_llm_decisions_complete")
                    self.app.background_tasks["llm_decisions"].status = "completed"
                    logger.info("phase2_task_completed", remaining_tasks=list(self.app.background_tasks.keys()))
                    status_panel = self.query_one(StatusPanel)
                    status_panel.update_status()
                    logger.info("phase2_status_panel_updated")

                # Get total count from app's shared list
                total_decisions = 0
                from logsqueak.tui.app import LogsqueakApp
                if isinstance(self.app, LogsqueakApp):
                    total_decisions = len(self.app.integration_decisions)

                logger.info(
                    "llm_decisions_complete_phase2",
                    total_blocks=block_count,
                    total_decisions=total_decisions
                )

            finally:
                # Release LLM slot
                self.app.release_llm_slot(request_id)

        except Exception as e:
            # Mark failed (if task still exists)
            from logsqueak.tui.app import LogsqueakApp
            if isinstance(self.app, LogsqueakApp):
                if "llm_decisions" in self.app.background_tasks:
                    self.app.background_tasks["llm_decisions"].status = "failed"
                    self.app.background_tasks["llm_decisions"].error_message = str(e)
                    status_panel = self.query_one(StatusPanel)
                    status_panel.update_status()

            logger.error(
                "llm_decisions_error_phase2",
                error=str(e),
                exc_info=True
            )

    async def _page_indexing_worker(self) -> None:
        """Worker: Build page index with progress."""
        # TODO: Implement page indexing
        pass

    async def _rag_search_worker(self) -> None:
        """Worker: Perform RAG search for candidate pages and load page contents.

        Worker Dependency: This worker MUST wait for page_indexing to complete before starting.
        The ChromaDB vector index must be built before RAG search can query it.

        Coordination: Polls app.background_tasks["page_indexing"] until status="completed".
        """
        import asyncio
        from logsqueak.tui.app import LogsqueakApp

        # Create background task
        self._background_tasks["rag_search"] = BackgroundTask(
            task_type="rag_search",
            status="running",
            progress_current=0,
            progress_total=2,  # Two steps: search + load
        )

        status_panel = self.query_one(StatusPanel)
        status_panel.update_status()

        try:
            # Wait for PageIndexer to complete (T108d)
            logger.info("rag_search_waiting_for_indexer", phase="phase2")

            # Poll for page indexing completion
            while True:
                if isinstance(self.app, LogsqueakApp):
                    indexing_task = self.app.background_tasks.get("page_indexing")
                    if indexing_task is None:
                        # Task deleted - it completed successfully
                        logger.info("rag_search_indexer_ready", phase="phase2")
                        break
                    elif indexing_task.status == "failed":
                        # Indexing failed - cannot proceed with RAG search
                        error_msg = indexing_task.error_message or "Unknown error"
                        raise RuntimeError(f"Page indexing failed: {error_msg}")

                # Wait before polling again
                await asyncio.sleep(0.1)

            # Step 1: Find candidate pages for each block
            logger.info("rag_search_starting", num_blocks=len(self.edited_content))

            # Build original_contexts dict for RAG search
            original_contexts = {
                ec.block_id: ec.hierarchical_context
                for ec in self.edited_content
            }

            # Find candidate chunks (now returns hierarchical chunks, not just page names)
            self.candidate_chunks = await self.rag_search.find_candidates(
                edited_content=self.edited_content,
                original_contexts=original_contexts,
                graph_paths=self.graph_paths,
                top_k=10  # TODO: Get from config
            )

            # Update progress
            self._background_tasks["rag_search"].progress_current = 1
            status_panel.update_status()

            # Extract unique pages and load page outlines (for frontmatter/properties)
            unique_pages = set()
            for chunks in self.candidate_chunks.values():
                for page_name, _, _ in chunks:
                    unique_pages.add(page_name)

            logger.info(
                "rag_search_chunks_found",
                unique_pages=len(unique_pages),
                total_chunks=sum(len(chunks) for chunks in self.candidate_chunks.values())
            )

            # Load page outlines for frontmatter/properties
            logger.info("rag_page_loading_starting")
            self.page_contents = {}
            for page_name in unique_pages:
                page_path = self.graph_paths.get_page_path(page_name)
                if page_path.exists():
                    try:
                        page_text = page_path.read_text()
                        from logseq_outline.parser import LogseqOutline
                        self.page_contents[page_name] = LogseqOutline.parse(page_text)
                    except Exception as e:
                        logger.error("page_load_failed", page_name=page_name, error=str(e))

            # Mark complete and remove from background tasks
            self.rag_search_state = BackgroundTaskState.COMPLETED
            del self._background_tasks["rag_search"]
            status_panel.update_status()

            logger.info(
                "rag_search_complete",
                pages_loaded=len(self.page_contents)
            )

            # Start LLM decisions worker now that page_contents is populated
            # This allows decisions to stream in during Phase 2, so they're
            # ready when user reaches Phase 3
            # Only start if not already running (user might have pressed 'n' already)
            from logsqueak.tui.app import LogsqueakApp
            task_exists = False
            if isinstance(self.app, LogsqueakApp):
                task_exists = "llm_decisions" in self.app.background_tasks

            if self.llm_client and not task_exists:
                # Save any unsaved edits before starting integration worker
                self._save_all_content()
                logger.info("saved_current_edits_before_integration")

                self.run_worker(self._llm_decisions_worker(), name="llm_decisions")
                logger.info("llm_decisions_worker_started_after_rag")

        except Exception as e:
            # Mark failed
            self._background_tasks["rag_search"].status = "failed"
            self._background_tasks["rag_search"].error_message = str(e)
            self.rag_search_state = BackgroundTaskState.FAILED
            self.rag_search_error = str(e)
            status_panel.update_status()

            logger.error(
                "rag_search_error",
                error=str(e),
                exc_info=True
            )
