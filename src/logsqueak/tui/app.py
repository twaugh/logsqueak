"""Main Textual application for interactive knowledge extraction.

This module defines the top-level ExtractionApp that coordinates the entire
interactive TUI workflow across all phases (Phase 1-4).
"""

from datetime import date
from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Footer, Header

from logsqueak.models.config import Configuration
from logsqueak.llm.client import LLMClient
from logsqueak.llm.providers.openai_compat import OpenAICompatibleProvider
from logsqueak.llm.prompt_logger import PromptLogger
from logsqueak.logseq.graph import GraphPaths
from logsqueak.logseq.parser import LogseqOutline
from logsqueak.models.journal import JournalEntry
from logsqueak.tui.models import ScreenState


class ExtractionApp(App):
    """
    Interactive TUI application for knowledge extraction workflow.

    Architecture:
    - Single ScreenState shared across all screens (Phase 1-4)
    - Screens pushed/popped from stack for navigation
    - LLM client initialized once, reused for all streaming operations
    - Config and journal entry loaded on mount, passed to all screens

    Lifecycle:
    1. on_mount(): Initialize state, load journal, create LLM client
    2. Push Phase1Screen (knowledge classification)
    3. User navigates through Phase 2-4 via screen.push()
    4. on_unmount(): Cleanup resources (prompt logger, etc.)

    Keyboard Bindings:
    - q: Quit application (at any phase)
    - ctrl+c: Force quit
    - Individual screens define their own bindings
    """

    CSS = """
    Screen {
        background: $surface;
    }

    Header {
        background: $primary;
        color: $text;
        height: 3;
    }

    Footer {
        background: $panel;
        color: $text;
    }

    Container {
        height: 100%;
        overflow-y: auto;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
    ]

    def __init__(
        self,
        journal_date: date,
        config: Configuration,
        prompt_log_file: Path | None = None,
    ):
        """
        Initialize the ExtractionApp.

        Args:
            journal_date: Date of journal entry to extract from
            config: User configuration (from ~/.config/logsqueak/config.yaml)
            prompt_log_file: Optional path to log prompts/responses (for debugging)
        """
        super().__init__()
        self.journal_date = journal_date
        self.config = config
        self.prompt_log_file = prompt_log_file
        self.state: ScreenState | None = None

    def compose(self) -> ComposeResult:
        """Compose the main application layout."""
        yield Header(show_clock=True)
        yield Container()
        yield Footer()

    async def on_mount(self) -> None:
        """
        Initialize application state and push Phase 1 screen.

        Steps:
        1. Load journal entry from disk
        2. Initialize LLM client with optional prompt logging
        3. Create ScreenState with empty collections
        4. Push Phase1Screen to start workflow

        Error Handling:
        - Journal file not found → Show error modal, exit
        - Invalid config (missing API key, etc.) → Show error modal, exit
        - LLM connection failure → Continue to Phase 1 (fail during streaming)
        """
        # Load journal entry
        graph = GraphPaths(self.config.logseq.graph_path)
        journal_path = graph.get_journal_path(self.journal_date.strftime("%Y_%m_%d"))

        if not journal_path.exists():
            # TODO: Show error modal when we have modal widget
            self.exit(message=f"Journal entry not found: {journal_path}")
            return

        # Parse journal entry
        with open(journal_path, "r", encoding="utf-8") as f:
            content = f.read()

        outline = LogseqOutline.parse(content)
        line_count = content.count('\n') + 1
        journal_entry = JournalEntry(
            date=self.journal_date,
            file_path=journal_path,
            raw_content=content,
            outline=outline,
            line_count=line_count
        )

        # Initialize LLM client with optional prompt logging
        prompt_logger = None
        if self.prompt_log_file:
            prompt_logger = PromptLogger(log_file=self.prompt_log_file)

        llm_client: LLMClient = OpenAICompatibleProvider(
            endpoint=str(self.config.llm.endpoint),
            api_key=self.config.llm.api_key,
            model=self.config.llm.model,
            prompt_logger=prompt_logger,
            num_ctx=self.config.llm.num_ctx,
        )

        # Initialize shared state
        self.state = ScreenState(
            current_phase=1,
            journal_entry=journal_entry,
            block_states={},  # Populated by Phase1Screen on mount
            candidates={},  # Populated by Phase2Screen
            decisions={},  # Populated by Phase3Screen
            config=self.config,
            llm_client=llm_client,
        )

        # Push Phase1Screen to start workflow
        from logsqueak.tui.screens.phase1 import Phase1Screen
        await self.push_screen(Phase1Screen(self.state))

        self.title = f"Logsqueak TUI - {self.journal_date}"

    def action_quit(self) -> None:
        """Quit the application."""
        self.exit()
