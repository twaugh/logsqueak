"""Phase 2 Screen: Content Editing.

This screen allows users to review and refine knowledge blocks before integration.
Users can see LLM-suggested rewordings, manually edit content, and review RAG search results.
"""

from typing import Optional
from textual.app import ComposeResult
from textual.screen import Screen
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Footer, Label
from textual.reactive import reactive
from logseq_outline.parser import LogseqBlock
from logsqueak.models.edited_content import EditedContent
from logsqueak.models.background_task import BackgroundTaskState
from logsqueak.tui.widgets.content_editor import ContentEditor
from logsqueak.tui.widgets.status_panel import StatusPanel
from logsqueak.tui.widgets.target_page_preview import TargetPagePreview
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
                # Top panel: Original context (using TargetPagePreview for syntax highlighting)
                with Container(id="original-panel"):
                    yield TargetPagePreview(id="original-context")

                # Middle panel: LLM reworded version (using TargetPagePreview for syntax highlighting)
                with Container(id="llm-panel"):
                    yield TargetPagePreview(id="llm-reworded")

                # Bottom panel: Editable current content
                with Container(id="editor-panel"):
                    yield ContentEditor()

            # Status panel for background tasks
            # TODO: Pass actual background tasks
            yield StatusPanel(background_tasks={})

            # Footer with keyboard shortcuts
            yield Footer()

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
        # TODO: Implement worker startup
        # self.run_worker(self._llm_rewording_worker(), name="llm_rewording")
        # self.run_worker(self._page_indexing_worker(), name="page_indexing")
        pass

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

            # Count leading spaces
            stripped = line.lstrip(' ')
            indent_count = len(line) - len(stripped)
            indent = ' ' * indent_count

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
        # Convert hierarchical context to Logseq bullets if needed
        context_with_bullets = self._convert_to_logseq_bullets(current_ec.hierarchical_context)
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
            self.current_block_index += 1
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
            self.current_block_index -= 1
            await self._update_display()

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
