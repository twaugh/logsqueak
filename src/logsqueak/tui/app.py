"""Main Logsqueak TUI Application.

This module defines the main Textual App class that manages screen transitions
between Phase 1 (Block Selection), Phase 2 (Content Editing), and Phase 3
(Integration Review).
"""

from typing import Dict, List, Optional
from pathlib import Path

from textual.app import App
from textual.binding import Binding
from textual.widgets import Footer, Header
import structlog

from logseq_outline.parser import LogseqOutline
from logsqueak.models.config import Config
from logsqueak.models.block_state import BlockState
from logsqueak.models.edited_content import EditedContent
from logsqueak.services.llm_client import LLMClient
from logsqueak.services.page_indexer import PageIndexer
from logsqueak.services.rag_search import RAGSearch
from logsqueak.services.file_monitor import FileMonitor
from logsqueak.tui.screens import Phase1Screen, Phase2Screen, Phase3Screen

logger = structlog.get_logger()


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
        journal_outline: LogseqOutline,
        journal_date: str,
        config: Config,
        llm_client: LLMClient,
        page_indexer: PageIndexer,
        rag_search: RAGSearch,
        file_monitor: FileMonitor,
    ):
        """Initialize the Logsqueak app.

        Args:
            journal_outline: Parsed journal entry
            journal_date: Date of journal entry (YYYY-MM-DD format)
            config: Application configuration
            llm_client: LLM client for streaming responses
            page_indexer: Page indexing service for RAG
            rag_search: Semantic search service
            file_monitor: File modification tracking service
        """
        super().__init__()

        # Store journal data
        self.journal_outline = journal_outline
        self.journal_date = journal_date

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

        logger.info(
            "app_initialized",
            journal_date=journal_date,
            num_blocks=len(journal_outline.blocks),
        )

    def on_mount(self) -> None:
        """Called when app is mounted. Start with Phase 1."""
        logger.info("app_on_mount_started")

        # Create and push Phase 1 screen FIRST (so it shows immediately)
        logger.info("creating_phase1_screen")
        phase1_screen = Phase1Screen(
            blocks=self.journal_outline.blocks,
            journal_date=self.journal_date,
            journal_outline=self.journal_outline,
            llm_client=self.llm_client,
            auto_start_workers=True,  # Auto-start LLM classification
            name="phase1",
        )
        logger.info("pushing_phase1_screen")
        self.push_screen(phase1_screen)
        logger.info("phase1_screen_pushed")

        # Defer model preloading with a delay so UI renders first
        # Use set_timer instead of call_later to ensure UI is visible
        self.set_timer(0.1, self._start_model_preload)
        logger.info("app_on_mount_finished")

    def _start_model_preload(self) -> None:
        """Start the model preload worker (called after screen is visible)."""
        self.run_worker(self._preload_embedding_model(), name="model_preload")

    async def _preload_embedding_model(self) -> None:
        """Preload SentenceTransformer model in background during Phase 1.

        This triggers lazy loading of the embedding model so it's ready
        by the time user transitions to Phase 2, avoiding UI delays.

        Uses run_in_executor to prevent blocking the async event loop.
        """
        import asyncio

        try:
            logger.info("preloading_embedding_model", phase="phase1")

            # Run the model loading in a thread pool to avoid blocking
            # the async event loop (SentenceTransformer import is CPU-heavy)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,  # Use default executor
                lambda: self.rag_search.encoder  # Trigger lazy load
            )

            logger.info("embedding_model_preloaded", phase="phase1")
        except Exception as e:
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

        # Store selected blocks
        self.selected_blocks = selected_blocks

        # Initialize edited content from selected blocks
        self.edited_content = []
        for block_state in selected_blocks:
            # Find the actual block and its context using generate_chunks
            from logseq_outline.context import generate_chunks

            block = None
            hierarchical_context = ""

            for found_block, context, hybrid_id in generate_chunks(self.journal_outline):
                if hybrid_id == block_state.block_id:
                    block = found_block
                    hierarchical_context = context
                    break

            if block:
                # Create EditedContent
                edited = EditedContent(
                    block_id=block_state.block_id,
                    original_content=block.get_full_content(),
                    hierarchical_context=hierarchical_context,
                    current_content=block.get_full_content(),
                )
                self.edited_content.append(edited)

        # Extract the selected blocks (LogseqBlock objects)
        from logseq_outline.graph import GraphPaths
        from pathlib import Path

        selected_logseq_blocks = []
        for block_state in selected_blocks:
            block = self.journal_outline.find_block_by_id(block_state.block_id)
            if block:
                selected_logseq_blocks.append(block)

        # Create GraphPaths from config
        graph_paths = GraphPaths(Path(self.config.logseq.graph_path))

        # Create and push Phase 2 screen
        phase2_screen = Phase2Screen(
            blocks=selected_logseq_blocks,
            edited_content=self.edited_content,
            journal_outline=self.journal_outline,
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
        page_contents: Dict[str, str],
    ) -> None:
        """Transition from Phase 2 to Phase 3.

        Args:
            edited_content: List of EditedContent from Phase 2
            candidate_pages: List of candidate page names from RAG search
            page_contents: Dict mapping page names to their content
        """
        logger.info(
            "transitioning_to_phase3",
            num_blocks=len(edited_content),
            num_candidates=len(candidate_pages),
        )

        # Store Phase 2 outputs
        self.edited_content = edited_content
        self.candidate_pages = candidate_pages
        self.page_contents = page_contents

        # Generate original contexts for each block
        self.original_contexts = {}
        for content in edited_content:
            self.original_contexts[content.block_id] = content.hierarchical_context

        # Create and push Phase 3 screen
        phase3_screen = Phase3Screen(
            edited_content_list=edited_content,
            candidate_pages=candidate_pages,
            page_contents=page_contents,
            original_contexts=self.original_contexts,
            llm_client=self.llm_client,
            journal_date=self.journal_date,
            journal_outline=self.journal_outline,
            file_monitor=self.file_monitor,
            name="phase3",
        )
        self.push_screen(phase3_screen)

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
