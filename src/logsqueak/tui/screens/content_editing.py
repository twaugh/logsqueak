"""Phase 2 Screen: Content Editing.

This screen allows users to review and refine knowledge blocks before integration.
Users can see LLM-suggested rewordings, manually edit content, and review RAG search results.
"""

from typing import Optional
from textual.app import ComposeResult
from textual.screen import Screen
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.widgets import Static, Footer, Label
from textual.reactive import reactive
from logseq_outline.parser import LogseqBlock
from logsqueak.models.edited_content import EditedContent
from logsqueak.models.background_task import BackgroundTaskState
from logsqueak.tui.widgets.content_editor import ContentEditor
from logsqueak.tui.widgets.status_panel import StatusPanel
import structlog

logger = structlog.get_logger()


class Phase2Screen(Screen):
    """Phase 2: Content Editing screen.

    Displays selected knowledge blocks with three vertical panels (top to bottom):
    1. Top: Original hierarchical context (read-only)
    2. Middle: LLM reworded version (when available)
    3. Bottom: Current editable content (TextArea)
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
        auto_start_workers: bool = True,
        **kwargs
    ):
        """Initialize Phase 2 screen.

        Args:
            blocks: List of selected knowledge blocks
            edited_content: List of EditedContent for each block
            auto_start_workers: Whether to auto-start background workers (default True)
        """
        super().__init__(**kwargs)
        self.blocks = blocks
        self.edited_content = edited_content
        self.auto_start_workers = auto_start_workers

        # Map block_id to EditedContent for quick lookup
        self.edited_content_map = {ec.block_id: ec for ec in edited_content}

    def compose(self) -> ComposeResult:
        """Compose the Phase 2 screen layout."""
        with Container(id="phase2-container"):
            # Header showing block progress
            yield Label("", id="block-counter")

            # Three-panel layout (vertical - top to bottom)
            with Vertical(id="content-panels"):
                # Top panel: Original context
                with Container(id="original-panel"):
                    yield Label("Original Context", classes="panel-header")
                    with VerticalScroll():
                        yield Static("", id="original-context")

                # Middle panel: LLM reworded version
                with Container(id="llm-panel"):
                    yield Label("LLM Reworded", classes="panel-header")
                    with VerticalScroll():
                        yield Static("", id="llm-reworded")

                # Bottom panel: Editable current content
                with Container(id="editor-panel"):
                    yield Label("Current Content (Editable)", classes="panel-header")
                    yield ContentEditor()

            # Status panel for background tasks
            # TODO: Pass actual background tasks
            yield StatusPanel(background_tasks={})

            # Footer with keyboard shortcuts
            yield Footer()

    def on_mount(self) -> None:
        """Handle screen mount event."""
        # Update initial display
        self._update_display()

        # Start background workers if enabled
        if self.auto_start_workers:
            self._start_background_workers()

    def _start_background_workers(self) -> None:
        """Start background workers for LLM rewording and RAG search."""
        # TODO: Implement worker startup
        # self.run_worker(self._llm_rewording_worker(), name="llm_rewording")
        # self.run_worker(self._page_indexing_worker(), name="page_indexing")
        pass

    def _update_display(self) -> None:
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

        # Update original context panel
        original_context = self.query_one("#original-context", Static)
        formatted_context = self._format_hierarchical_context(current_ec.original_content)
        original_context.update(formatted_context)

        # Update LLM reworded panel
        llm_reworded = self.query_one("#llm-reworded", Static)
        if current_ec.rewording_complete and current_ec.reworded_content:
            llm_reworded.update(current_ec.reworded_content)
        else:
            llm_reworded.update("⏳ Waiting for LLM rewording...")

        # Update editor with current content
        editor = self.query_one(ContentEditor)
        editor.load_content(current_ec.current_content)

    def action_navigate_next(self) -> None:
        """Navigate to next block (j or down arrow)."""
        # Only navigate if editor is not focused
        editor = self.query_one(ContentEditor)
        if editor.has_focus:
            return

        # Save current content before navigating
        self._save_current_content()

        # Move to next block
        if self.current_block_index < len(self.edited_content) - 1:
            self.current_block_index += 1
            self._update_display()

    def action_navigate_previous(self) -> None:
        """Navigate to previous block (k or up arrow)."""
        # Only navigate if editor is not focused
        editor = self.query_one(ContentEditor)
        if editor.has_focus:
            return

        # Save current content before navigating
        self._save_current_content()

        # Move to previous block
        if self.current_block_index > 0:
            self.current_block_index -= 1
            self._update_display()

    def action_toggle_focus(self) -> None:
        """Toggle focus on/off the editor (Tab key)."""
        editor = self.query_one(ContentEditor)

        if editor.has_focus:
            # Unfocus the editor
            self.set_focus(None)
        else:
            # Focus the editor
            editor.focus()

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

        # TODO: Transition to Phase 3
        logger.info("phase2_complete", blocks_edited=len(self.edited_content))

    def action_back(self) -> None:
        """Go back to Phase 1 ('q' key)."""
        # Save current content
        self._save_current_content()

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

    def _format_hierarchical_context(self, context: str, width: int = 80) -> str:
        """Format hierarchical context with bullets and hanging indent.

        Each block (separated by hierarchy) gets ONE bullet on its first line.
        All subsequent lines of that block use hanging indent.

        Takes context like:
            "Projects\n  Python Learning\n    Block content\n    Second line\n    Third line"

        Returns formatted with one bullet per block:
            "• Projects\n  • Python Learning\n    • Block content\n      Second line\n      Third line"

        Args:
            context: The hierarchical context string
            width: Maximum line width (default 80, will be overridden by widget width)
        """
        if not context:
            return ""

        lines = context.split('\n')
        formatted_lines = []
        previous_indent = -1

        for line in lines:
            if not line.strip():  # Empty line
                formatted_lines.append("")
                continue

            # Count leading spaces to determine indent level
            stripped = line.lstrip(' ')
            indent_count = len(line) - len(stripped)

            # Determine if this is a new block (indent decreased or at same level as new block)
            # First line of a block gets a bullet, continuation lines don't
            is_new_block = indent_count != previous_indent

            if is_new_block:
                # First line of block gets bullet
                first_line_prefix = ' ' * indent_count + '• '
                hanging_indent_prefix = ' ' * (indent_count + 2)
                previous_indent = indent_count
            else:
                # Continuation line of same block - no bullet, just hanging indent
                first_line_prefix = ' ' * (indent_count + 2)
                hanging_indent_prefix = ' ' * (indent_count + 2)

            # Calculate available width for text
            available_width = max(40, width - len(first_line_prefix))

            # Wrap the text manually
            wrapped = self._wrap_text(stripped, available_width)

            # Add the first wrapped line (with or without bullet)
            if wrapped:
                formatted_lines.append(first_line_prefix + wrapped[0])

                # Add subsequent wrapped lines with hanging indent
                for wrapped_line in wrapped[1:]:
                    formatted_lines.append(hanging_indent_prefix + wrapped_line)

        return '\n'.join(formatted_lines)

    def _wrap_text(self, text: str, width: int) -> list[str]:
        """Wrap text to specified width, breaking on word boundaries.

        Args:
            text: Text to wrap
            width: Maximum width per line

        Returns:
            List of wrapped lines
        """
        if len(text) <= width:
            return [text]

        words = text.split()
        lines = []
        current_line = []
        current_length = 0

        for word in words:
            word_length = len(word)

            # Check if adding this word would exceed width
            # Account for space before word (except first word)
            space_needed = 1 if current_line else 0

            if current_length + space_needed + word_length <= width:
                current_line.append(word)
                current_length += space_needed + word_length
            else:
                # Start new line
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = word_length

        # Add remaining words
        if current_line:
            lines.append(' '.join(current_line))

        return lines if lines else [text]

    # Background worker methods (stubs for now)

    async def _llm_rewording_worker(self) -> None:
        """Worker: Generate LLM reworded versions."""
        # TODO: Implement LLM rewording streaming
        pass

    async def _page_indexing_worker(self) -> None:
        """Worker: Build page index with progress."""
        # TODO: Implement page indexing
        pass

    async def _rag_search_worker(self) -> None:
        """Worker: Perform RAG search for candidate pages."""
        # TODO: Implement RAG search with progress
        pass
