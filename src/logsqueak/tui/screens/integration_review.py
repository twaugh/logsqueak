"""Phase 3 Screen: Integration Review.

This screen allows users to review integration decisions for each knowledge block,
see target page previews, and accept/skip decisions for writing to pages.
"""

from typing import Optional, List
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
from logsqueak.models.background_task import BackgroundTaskState
from logsqueak.tui.widgets.target_page_preview import TargetPagePreview
from logsqueak.tui.widgets.decision_list import DecisionList
from logsqueak.tui.widgets.status_panel import StatusPanel
from logsqueak.services.file_operations import write_integration_atomic
from logsqueak.services.file_monitor import FileMonitor
import structlog

logger = structlog.get_logger()


class Phase3Screen(Screen):
    """Phase 3: Integration Review screen.

    Displays integration decisions for each knowledge block with:
    1. Journal context panel (top-left)
    2. Refined content panel (top-right)
    3. Decision list (bottom-left)
    4. Target page preview (bottom-right)
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
        decisions: list[IntegrationDecision],
        journal_date: str,
        graph_paths: Optional[GraphPaths] = None,
        file_monitor: Optional[FileMonitor] = None,
        auto_start_workers: bool = True,
        **kwargs
    ):
        """Initialize Phase 3 screen.

        Args:
            journal_blocks: Original journal blocks (for context)
            edited_content: Refined content from Phase 2
            decisions: List of all integration decisions (for all blocks)
            journal_date: Journal date (YYYY-MM-DD)
            graph_paths: GraphPaths instance for file operations
            file_monitor: FileMonitor for concurrent modification detection
            auto_start_workers: Whether to auto-start background workers (default True)
        """
        super().__init__(**kwargs)
        self.journal_blocks = journal_blocks
        self.edited_content = edited_content
        self.decisions = decisions
        self.journal_date = journal_date
        self.graph_paths = graph_paths
        self.file_monitor = file_monitor or FileMonitor()
        self.auto_start_workers = auto_start_workers

        # Map block_id to EditedContent for quick lookup
        self.edited_content_map = {ec.block_id: ec for ec in edited_content}

        # Map block_id to decisions for batching
        self.decisions_by_block = self._group_decisions_by_block(decisions)

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

            # Four-panel layout (2x2 grid)
            with Vertical(id="content-panels"):
                # Top row: Journal context and refined content
                with Container(id="top-row"):
                    # Top-left: Journal context
                    with Container(id="journal-context-panel"):
                        yield Label("Journal Context", classes="panel-header")
                        with VerticalScroll():
                            yield Static("", id="journal-context")

                    # Top-right: Refined content
                    with Container(id="refined-content-panel"):
                        yield Label("Refined Content", classes="panel-header")
                        with VerticalScroll():
                            yield Static("", id="refined-content")

                # Bottom row: Decision list and target page preview
                with Container(id="bottom-row"):
                    # Bottom-left: Decision list
                    with Container(id="decision-list-panel"):
                        yield Label("Integration Decisions", classes="panel-header")
                        yield DecisionList()

                    # Bottom-right: Target page preview
                    with Container(id="preview-panel"):
                        yield Label("Target Page Preview", classes="panel-header")
                        yield TargetPagePreview()

            # Status panel for background tasks
            # For Phase 3, we track LLM decision generation
            yield StatusPanel({})

        # Footer with keyboard shortcuts
        yield Footer()

    def on_mount(self) -> None:
        """Initialize screen when mounted."""
        # Display first block
        self._display_current_block()

        # Start background workers if enabled
        if self.auto_start_workers:
            # Background workers would generate decisions via LLM
            # For now, we're using pre-generated decisions
            pass

    def _display_current_block(self) -> None:
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

        # Display journal context (hierarchical)
        from logseq_outline.context import generate_full_context
        # For journal context, we need parent blocks
        # For simplicity, just show the block itself
        context_widget = self.query_one("#journal-context", Static)
        context_widget.update(current_block.get_full_content())

        # Display refined content
        edited = self.edited_content_map.get(block_id)
        refined_widget = self.query_one("#refined-content", Static)
        if edited:
            refined_widget.update(edited.current_content)
        else:
            refined_widget.update("No refined content available")

        # Get decisions for this block
        block_decisions = self.decisions_by_block.get(block_id, [])

        # Display decisions in list
        decision_list = self.query_one(DecisionList)
        decision_list.load_decisions(block_decisions, self.current_decision_index)

        # Display preview for current decision
        self._update_preview()

    def _update_preview(self) -> None:
        """Update target page preview with current decision."""
        block_id = self.journal_blocks[self.current_block_index].block_id
        block_decisions = self.decisions_by_block.get(block_id, [])

        if not block_decisions or self.current_decision_index >= len(block_decisions):
            preview = self.query_one(TargetPagePreview)
            preview.clear()
            return

        decision = block_decisions[self.current_decision_index]

        # Generate preview content with integrated block
        preview_text, highlight_block_id = self._generate_preview_with_integration(decision)

        preview = self.query_one(TargetPagePreview)
        preview.load_preview(preview_text, highlight_block_id=highlight_block_id)

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
            # Find the target block and add as child
            target_found = self._add_block_under(outline, target_block_id, new_block_content, new_block_id)
            if not target_found:
                logger.warning("target_block_not_found", target_id=target_block_id)
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
        self, outline: LogseqOutline, target_id: str, content: str, block_id: str
    ) -> bool:
        """Add a new block under the target block.

        Args:
            outline: Outline to modify
            target_id: ID of target block to add under
            content: Content for new block
            block_id: ID for new block

        Returns:
            True if target was found and block added
        """
        def search_and_add(blocks: list[LogseqBlock], parents: list[LogseqBlock]) -> bool:
            for parent_block in blocks:
                # Check if this is the target (by explicit ID or content hash)
                block_matches = False
                if parent_block.block_id == target_id:
                    block_matches = True
                else:
                    # Try content hash
                    full_context = generate_full_context(parent_block, parents)
                    if generate_content_hash(full_context) == target_id:
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
            self._update_decision_display()

    def action_navigate_previous_decision(self) -> None:
        """Navigate to previous decision for current block."""
        if self.current_decision_index > 0:
            self.current_decision_index -= 1
            self._update_decision_display()

    def _update_decision_display(self) -> None:
        """Update decision list and preview after navigation."""
        block_id = self.journal_blocks[self.current_block_index].block_id
        block_decisions = self.decisions_by_block.get(block_id, [])

        # Update decision list highlight
        decision_list = self.query_one(DecisionList)
        decision_list.set_current_index(self.current_decision_index)

        # Update preview
        self._update_preview()

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
            await write_integration_atomic(
                decision=decision,
                journal_date=self.journal_date,
                graph_paths=self.graph_paths,
                file_monitor=self.file_monitor
            )

    def action_next_block(self) -> None:
        """Advance to next knowledge block."""
        if self.current_block_index < len(self.journal_blocks) - 1:
            self.current_block_index += 1
            self.current_decision_index = 0
            self._display_current_block()

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
        preview = self.query_one(TargetPagePreview)
        preview.focus()

    def action_back(self) -> None:
        """Return to previous screen."""
        self.dismiss()
