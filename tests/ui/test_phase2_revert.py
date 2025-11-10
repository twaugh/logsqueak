"""UI tests for Phase 2 revert to original ('r' key).

Tests for 'r' key to revert edited content back to the original
block content from the journal.
"""

import pytest
from textual.app import App
from textual.pilot import Pilot
from logsqueak.tui.screens.content_editing import Phase2Screen
from logsqueak.models.edited_content import EditedContent
from logseq_outline.parser import LogseqBlock, LogseqOutline
from logseq_outline.graph import GraphPaths


class Phase2TestApp(App):
    """Test app wrapper for Phase2Screen."""

    def __init__(self, screen: Phase2Screen):
        super().__init__()
        self.test_screen = screen

    def on_mount(self) -> None:
        """Push the test screen on mount."""
        self.push_screen(self.test_screen)


@pytest.fixture
def sample_blocks():
    """Create sample knowledge blocks for testing."""
    blocks = [
        LogseqBlock(
            content=["Original content from journal"],
            indent_level=0,
            block_id="block-1",
            children=[]
        ),
    ]
    return blocks


@pytest.fixture
def sample_edited_content_modified():
    """Create EditedContent with modified current_content."""
    return [
        EditedContent(
            block_id="block-1",
            original_content="Original content from journal",
            hierarchical_context="Original content from journal",
            current_content="User edited this content",
            reworded_content="LLM reworded version",
            rewording_complete=True
        ),
    ]


@pytest.fixture
def sample_edited_content_with_llm():
    """Create EditedContent where user accepted LLM version."""
    return [
        EditedContent(
            block_id="block-1",
            original_content="Original content from journal",
            hierarchical_context="Original content from journal",
            current_content="LLM reworded version",
            reworded_content="LLM reworded version",
            rewording_complete=True
        ),
    ]


@pytest.fixture
def journal_outline(sample_blocks):
    """Create a LogseqOutline from sample blocks."""
    return LogseqOutline(blocks=sample_blocks, source_text="", frontmatter=[])


@pytest.fixture
def graph_paths(tmp_path):
    """Create a temporary GraphPaths instance."""
    graph_dir = tmp_path / "test-graph"
    graph_dir.mkdir()
    (graph_dir / "pages").mkdir()
    (graph_dir / "journals").mkdir()
    return GraphPaths(graph_dir)


@pytest.mark.asyncio
async def test_revert_to_original_after_manual_edit(sample_blocks, sample_edited_content_modified, journal_outline, graph_paths):
    """Test 'r' key reverts manually edited content to original."""
    screen = Phase2Screen(
        blocks=sample_blocks,
        edited_content=sample_edited_content_modified,
        journal_outline=journal_outline,
        graph_paths=graph_paths,
        auto_start_workers=False
    )
    app = Phase2TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        editor = screen.query_one("#content-editor")

        # Initially showing modified content
        assert editor.text == "User edited this content"

        # Press 'r' to revert to original
        await pilot.press("r")
        await pilot.pause()

        # Editor should now show original content
        assert editor.text == "Original content from journal"

        # current_content should also be updated
        assert screen.edited_content[0].current_content == "Original content from journal"


@pytest.mark.asyncio
async def test_revert_to_original_after_accepting_llm(sample_blocks, sample_edited_content_with_llm, journal_outline, graph_paths):
    """Test 'r' key reverts LLM-accepted content to original."""
    screen = Phase2Screen(
        blocks=sample_blocks,
        edited_content=sample_edited_content_with_llm,
        journal_outline=journal_outline,
        graph_paths=graph_paths,
        auto_start_workers=False
    )
    app = Phase2TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        editor = screen.query_one("#content-editor")

        # Initially showing LLM reworded version
        assert editor.text == "LLM reworded version"

        # Press 'r' to revert to original
        await pilot.press("r")
        await pilot.pause()

        # Editor should now show original content
        assert editor.text == "Original content from journal"


@pytest.mark.asyncio
async def test_revert_only_works_when_unfocused(sample_blocks, sample_edited_content_modified, journal_outline, graph_paths):
    """Test 'r' key only works when editor is unfocused."""
    screen = Phase2Screen(
        blocks=sample_blocks,
        edited_content=sample_edited_content_modified,
        journal_outline=journal_outline,
        graph_paths=graph_paths,
        auto_start_workers=False
    )
    app = Phase2TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        editor = screen.query_one("#content-editor")

        # Focus the editor
        await pilot.press("tab")
        await pilot.pause()
        assert editor.has_focus

        editor.text = "test"
        original_len = len(editor.text)

        # Try to press 'r' while focused - should type 'r' not trigger shortcut
        await pilot.press("r")
        await pilot.pause()

        # 'r' should be typed (length increases)
        assert "r" in editor.text

        # Unfocus the editor
        await pilot.press("tab")
        await pilot.pause()
        assert not editor.has_focus

        # Now 'r' should work as shortcut
        await pilot.press("r")
        await pilot.pause()

        # Text should now be original version (shortcut worked)
        assert editor.text == "Original content from journal"


@pytest.mark.asyncio
async def test_revert_when_already_original(sample_blocks, journal_outline, graph_paths):
    """Test 'r' key when content is already original (no-op)."""
    # Content that hasn't been modified
    edited_content = [
        EditedContent(
            block_id="block-1",
            original_content="Original content from journal",
            hierarchical_context="Original content from journal",
            current_content="Original content from journal",  # Same as original
            reworded_content=None,
            rewording_complete=False
        ),
    ]

    screen = Phase2Screen(
        blocks=sample_blocks,
        edited_content=edited_content,
        journal_outline=journal_outline,
        graph_paths=graph_paths,
        auto_start_workers=False
    )
    app = Phase2TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        editor = screen.query_one("#content-editor")

        # Content is already original
        assert editor.text == "Original content from journal"

        # Press 'r' - should still work but have no visible effect
        await pilot.press("r")
        await pilot.pause()

        # Content should remain the same
        assert editor.text == "Original content from journal"


@pytest.mark.asyncio
async def test_revert_updates_indicator(sample_blocks, sample_edited_content_modified, journal_outline, graph_paths):
    """Test that reverting updates visual indicator."""
    screen = Phase2Screen(
        blocks=sample_blocks,
        edited_content=sample_edited_content_modified,
        journal_outline=journal_outline,
        graph_paths=graph_paths,
        auto_start_workers=False
    )
    app = Phase2TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        # Press 'r' to revert
        await pilot.press("r")
        await pilot.pause()

        # Some visual indicator should show that content was reverted
        # (Implementation detail: may be a status message or highlight change)
        # For now, just verify the content was updated correctly
        assert screen.edited_content[0].current_content == screen.edited_content[0].original_content


@pytest.mark.asyncio
async def test_can_revert_multiple_times(sample_blocks, sample_edited_content_modified, journal_outline, graph_paths):
    """Test that user can revert, edit, and revert again."""
    screen = Phase2Screen(
        blocks=sample_blocks,
        edited_content=sample_edited_content_modified,
        journal_outline=journal_outline,
        graph_paths=graph_paths,
        auto_start_workers=False
    )
    app = Phase2TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        editor = screen.query_one("#content-editor")

        # First revert
        await pilot.press("r")
        await pilot.pause()
        assert editor.text == "Original content from journal"

        # Focus and edit again
        await pilot.press("tab")
        await pilot.pause()
        editor.text = "New edit after revert"
        await pilot.pause()

        # Unfocus
        await pilot.press("tab")
        await pilot.pause()

        # Verify new edit is shown
        assert editor.text == "New edit after revert"

        # Revert again
        await pilot.press("r")
        await pilot.pause()

        # Should go back to original again
        assert editor.text == "Original content from journal"


@pytest.mark.asyncio
async def test_revert_then_accept_llm(sample_blocks, sample_edited_content_modified, journal_outline, graph_paths):
    """Test workflow: revert to original, then accept LLM version."""
    screen = Phase2Screen(
        blocks=sample_blocks,
        edited_content=sample_edited_content_modified,
        journal_outline=journal_outline,
        graph_paths=graph_paths,
        auto_start_workers=False
    )
    app = Phase2TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        editor = screen.query_one("#content-editor")

        # Start with modified content
        assert editor.text == "User edited this content"

        # Revert to original
        await pilot.press("r")
        await pilot.pause()
        assert editor.text == "Original content from journal"

        # Now accept LLM version
        await pilot.press("a")
        await pilot.pause()
        assert editor.text == "LLM reworded version"

        # And revert one more time
        await pilot.press("r")
        await pilot.pause()
        assert editor.text == "Original content from journal"
