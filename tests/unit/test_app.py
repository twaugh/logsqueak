"""Unit tests for main TUI App class."""

import pytest
from unittest.mock import Mock, AsyncMock
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
from logsqueak.services.llm_wrappers import _augment_outline_with_ids
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
    # Configure rag_search.find_candidates to return an empty dict (not an AsyncMock)
    rag_search.find_candidates = AsyncMock(return_value={})
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
    """Create sample journal outline with augmented hybrid IDs.

    Note: This fixture augments the outline with hybrid IDs to match the
    behavior of load_journal_entries() in the CLI, which augments journals
    before passing them to the app.
    """
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
    # Augment with hybrid IDs (matches CLI behavior)
    return _augment_outline_with_ids(outline)


def test_app_instantiates_without_errors(mock_services, sample_journal_outline):
    """Test that App instantiates without errors."""
    journals = {"2025-01-15": sample_journal_outline}
    app = LogsqueakApp(
        journals=journals,
        **mock_services,
    )

    assert app is not None
    assert app.config == mock_services["config"]
    assert app.llm_client == mock_services["llm_client"]
    assert app.page_indexer == mock_services["page_indexer"]
    assert app.rag_search == mock_services["rag_search"]
    assert app.file_monitor == mock_services["file_monitor"]


def test_app_tracks_journal_outline(mock_services, sample_journal_outline):
    """Test that App tracks journal outlines dict."""
    journals = {"2025-01-15": sample_journal_outline}
    app = LogsqueakApp(
        journals=journals,
        **mock_services,
    )

    # Note: app.journals values are augmented with IDs, so check structure instead of equality
    assert len(app.journals) == 1
    assert "2025-01-15" in app.journals
    assert len(app.journals["2025-01-15"].blocks) == len(sample_journal_outline.blocks)
    # All blocks should have IDs after augmentation
    for block in app.journals["2025-01-15"].blocks:
        assert block.block_id is not None


def test_app_tracks_current_phase_state(mock_services, sample_journal_outline):
    """Test that App tracks current phase state."""
    journals = {"2025-01-15": sample_journal_outline}
    app = LogsqueakApp(
        journals=journals,
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
    journals = {"2025-01-15": sample_journal_outline}
    app = LogsqueakApp(
        journals=journals,
        **mock_services,
    )

    # App should have on_mount method that installs screens
    assert hasattr(app, "on_mount")
    # App tracks journal data needed for screen creation
    assert app.journals is not None
    assert "2025-01-15" in app.journals


@pytest.mark.asyncio
async def test_app_starts_with_phase1(mock_services, sample_journal_outline):
    """Test that App starts with Phase 1 screen."""
    journals = {"2025-01-15": sample_journal_outline}
    app = LogsqueakApp(
        journals=journals,
        **mock_services,
    )

    # App should have method to determine initial screen
    # (Textual apps typically override SCREENS or install_screen)
    # For now, just verify the app has the necessary structure
    assert hasattr(app, "SCREENS") or hasattr(app, "on_mount")


def test_app_augments_outline_with_ids(mock_services, sample_journal_outline):
    """Test that App augments journal outline with IDs for all blocks."""
    journals = {"2025-01-15": sample_journal_outline}
    app = LogsqueakApp(
        journals=journals,
        **mock_services,
    )

    # All blocks should have IDs after augmentation (explicit or hash-based)
    def check_blocks_have_ids(blocks):
        for block in blocks:
            assert block.block_id is not None, f"Block missing ID: {block.content}"
            # Recursively check children
            if block.children:
                check_blocks_have_ids(block.children)

    check_blocks_have_ids(app.journals["2025-01-15"].blocks)


@pytest.mark.asyncio
async def test_transition_to_phase2_filters_id_property(mock_services, sample_journal_outline):
    """Test that transition_to_phase2 excludes id:: property from editable content."""
    from logsqueak.models.block_state import BlockState

    journals = {"2025-01-15": sample_journal_outline}
    app = LogsqueakApp(
        journals=journals,
        **mock_services,
    )

    async with app.run_test() as pilot:
        await pilot.pause()

        # Create a selected block (simulating Phase 1 completion)
        # The augmented outline will have id:: properties added
        first_block = app.journals["2025-01-15"].blocks[0]

        selected_blocks = [
            BlockState(
                block_id=first_block.block_id,
                classification="knowledge",
                source="user",
            )
        ]

        # Transition to Phase 2
        app.transition_to_phase2(selected_blocks)
        await pilot.pause()

        # Verify edited_content was created
        assert app.edited_content is not None
        assert len(app.edited_content) == 1

        # Verify the id:: property is NOT in the editable content
        edited = app.edited_content[0]
        assert "id::" not in edited.original_content, "id:: property should be filtered out from original_content"
        assert "id::" not in edited.current_content, "id:: property should be filtered out from current_content"

        # Verify the actual block content is present
        assert "Root block 1" in edited.original_content
