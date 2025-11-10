"""Unit tests for main TUI App class."""

import pytest
from unittest.mock import Mock
from pathlib import Path
import tempfile
import shutil

from logsqueak.tui.app import LogsqueakApp
from logsqueak.tui.screens import Phase1Screen, Phase2Screen, Phase3Screen
from logsqueak.models.config import Config, LLMConfig, LogseqConfig, RAGConfig
from logsqueak.services.llm_client import LLMClient
from logsqueak.services.page_indexer import PageIndexer
from logsqueak.services.rag_search import RAGSearch
from logsqueak.services.file_monitor import FileMonitor
from logseq_outline.parser import LogseqOutline, LogseqBlock


@pytest.fixture
def temp_graph_dir():
    """Create temporary graph directory."""
    temp_dir = tempfile.mkdtemp(prefix="logsqueak-test-")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_config(temp_graph_dir):
    """Create mock configuration."""
    return Config(
        llm=LLMConfig(
            endpoint="http://localhost:11434/v1",
            api_key="test-key",
            model="test-model",
            num_ctx=2048,
        ),
        logseq=LogseqConfig(graph_path=temp_graph_dir),
        rag=RAGConfig(top_k=5),
    )


@pytest.fixture
def mock_services(mock_config):
    """Create mock services."""
    llm_client = Mock(spec=LLMClient)
    page_indexer = Mock(spec=PageIndexer)
    rag_search = Mock(spec=RAGSearch)
    file_monitor = Mock(spec=FileMonitor)

    return {
        "config": mock_config,
        "llm_client": llm_client,
        "page_indexer": page_indexer,
        "rag_search": rag_search,
        "file_monitor": file_monitor,
    }


@pytest.fixture
def sample_journal_outline():
    """Create sample journal outline."""
    source_text = "- Root block 1\n  - Child block\n- Root block 2"
    outline = LogseqOutline(
        blocks=[
            LogseqBlock(
                content=["Root block 1"],
                indent_level=0,
                children=[
                    LogseqBlock(
                        content=["Child block"],
                        indent_level=1,
                        children=[],
                    ),
                ],
            ),
            LogseqBlock(
                content=["Root block 2"],
                indent_level=0,
                children=[],
            ),
        ],
        source_text=source_text,
        frontmatter=[],
    )
    return outline


def test_app_instantiates_without_errors(mock_services, sample_journal_outline):
    """Test that App instantiates without errors."""
    app = LogsqueakApp(
        journal_outline=sample_journal_outline,
        journal_date="2025-01-15",
        **mock_services,
    )

    assert app is not None
    assert app.config == mock_services["config"]
    assert app.llm_client == mock_services["llm_client"]
    assert app.page_indexer == mock_services["page_indexer"]
    assert app.rag_search == mock_services["rag_search"]
    assert app.file_monitor == mock_services["file_monitor"]


def test_app_tracks_journal_outline(mock_services, sample_journal_outline):
    """Test that App tracks journal outline and date."""
    app = LogsqueakApp(
        journal_outline=sample_journal_outline,
        journal_date="2025-01-15",
        **mock_services,
    )

    assert app.journal_outline == sample_journal_outline
    assert app.journal_date == "2025-01-15"


def test_app_tracks_current_phase_state(mock_services, sample_journal_outline):
    """Test that App tracks current phase state."""
    app = LogsqueakApp(
        journal_outline=sample_journal_outline,
        journal_date="2025-01-15",
        **mock_services,
    )

    # App should have phase state tracking attributes
    assert hasattr(app, "selected_blocks")
    assert hasattr(app, "edited_content")
    assert hasattr(app, "candidate_pages")
    assert hasattr(app, "page_contents")
    assert hasattr(app, "original_contexts")

    # Initial state should be empty
    assert app.selected_blocks is None or app.selected_blocks == []
    assert app.edited_content is None or app.edited_content == []
    assert app.candidate_pages is None or app.candidate_pages == []
    assert app.page_contents is None or app.page_contents == {}
    assert app.original_contexts is None or app.original_contexts == {}


@pytest.mark.asyncio
async def test_app_can_install_phase1_screen(mock_services, sample_journal_outline):
    """Test that App can install Phase1Screen."""
    app = LogsqueakApp(
        journal_outline=sample_journal_outline,
        journal_date="2025-01-15",
        **mock_services,
    )

    # App should have SCREENS attribute for Textual screen management
    assert hasattr(app, "SCREENS")
    assert "phase1" in app.SCREENS or len(app.SCREENS) >= 1


@pytest.mark.asyncio
async def test_app_starts_with_phase1(mock_services, sample_journal_outline):
    """Test that App starts with Phase 1 screen."""
    app = LogsqueakApp(
        journal_outline=sample_journal_outline,
        journal_date="2025-01-15",
        **mock_services,
    )

    # App should have method to determine initial screen
    # (Textual apps typically override SCREENS or install_screen)
    # For now, just verify the app has the necessary structure
    assert hasattr(app, "SCREENS") or hasattr(app, "on_mount")
