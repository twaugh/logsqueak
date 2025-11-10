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

    SCREENS = {
        "phase1": Phase1Screen,
        "phase2": Phase2Screen,
        "phase3": Phase3Screen,
    }

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
        logger.info("app_mounted", starting_phase="phase1")

        # Install and push Phase 1 screen
        self.push_screen("phase1")

    def action_quit(self) -> None:
        """Handle quit action."""
        logger.info("app_quit_requested")
        self.exit()
