"""UI tests for Phase 2 block navigation with auto-save.

Tests for j/k keys to navigate between knowledge blocks with
automatic saving of editor content before moving to next block.
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
            content=["First knowledge block with some content"],
            indent_level=0,
            block_id="block-1",
            children=[]
        ),
        LogseqBlock(
            content=["Second knowledge block about Python"],
            indent_level=0,
            block_id="block-2",
            children=[]
        ),
        LogseqBlock(
            content=["Third knowledge block with technical notes"],
            indent_level=0,
            block_id="block-3",
            children=[]
        ),
    ]
    return blocks


@pytest.fixture
def sample_edited_content(sample_blocks):
    """Create sample EditedContent for testing."""
    return [
        EditedContent(
            block_id="block-1",
            original_content="First knowledge block with some content",
            hierarchical_context="First knowledge block with some content",
            current_content="First knowledge block with some content"
        ),
        EditedContent(
            block_id="block-2",
            original_content="Second knowledge block about Python",
            hierarchical_context="Second knowledge block about Python",
            current_content="Second knowledge block about Python"
        ),
        EditedContent(
            block_id="block-3",
            original_content="Third knowledge block with technical notes",
            hierarchical_context="Third knowledge block with technical notes",
            current_content="Third knowledge block with technical notes"
        ),
    ]


@pytest.fixture
def journal_outline(sample_blocks):
    """Create a LogseqOutline from sample blocks."""
    return LogseqOutline(blocks=sample_blocks, source_text="", frontmatter=[])


@pytest.fixture
def journals(journal_outline):
    """Create journals dict for Phase2Screen."""
    return {"2025-01-15": journal_outline}


@pytest.fixture
def graph_paths(tmp_path):
    """Create a temporary GraphPaths instance."""
    graph_dir = tmp_path / "test-graph"
    graph_dir.mkdir()
    (graph_dir / "pages").mkdir()
    (graph_dir / "journals").mkdir()
    return GraphPaths(graph_dir)


@pytest.mark.asyncio
async def test_navigate_with_j_key(sample_blocks, sample_edited_content, journal_outline, graph_paths):
    """Test j key navigates to next block."""
    screen = Phase2Screen(
        blocks=sample_blocks,
        edited_content=sample_edited_content,
        journals=journals,
        graph_paths=graph_paths,
        auto_start_workers=False
    )
    app = Phase2TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        # Should start at first block (index 0)
        assert screen.current_block_index == 0

        # Press j to move to next block
        await pilot.press("j")
        await pilot.pause()

        # Should now be at second block (index 1)
        assert screen.current_block_index == 1


@pytest.mark.asyncio
async def test_navigate_with_k_key(sample_blocks, sample_edited_content, journal_outline, graph_paths):
    """Test k key navigates to previous block."""
    screen = Phase2Screen(
        blocks=sample_blocks,
        edited_content=sample_edited_content,
        journals=journals,
        graph_paths=graph_paths,
        auto_start_workers=False
    )
    app = Phase2TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        # Move to second block first
        await pilot.press("j")
        await pilot.pause()

        assert screen.current_block_index == 1

        # Press k to move back to first block
        await pilot.press("k")
        await pilot.pause()

        # Should be back at first block
        assert screen.current_block_index == 0


@pytest.mark.asyncio
async def test_navigate_with_arrow_keys(sample_blocks, sample_edited_content, journal_outline, graph_paths):
    """Test arrow keys also work for navigation."""
    screen = Phase2Screen(
        blocks=sample_blocks,
        edited_content=sample_edited_content,
        journals=journals,
        graph_paths=graph_paths,
        auto_start_workers=False
    )
    app = Phase2TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        # Initially at first block
        assert screen.current_block_index == 0

        # Press down arrow to move forward
        await pilot.press("down")
        await pilot.pause()

        assert screen.current_block_index == 1

        # Press up arrow to move back
        await pilot.press("up")
        await pilot.pause()

        assert screen.current_block_index == 0


@pytest.mark.asyncio
async def test_auto_save_on_navigation(sample_blocks, sample_edited_content, journal_outline, graph_paths):
    """Test that navigating saves editor content automatically."""
    screen = Phase2Screen(
        blocks=sample_blocks,
        edited_content=sample_edited_content,
        journals=journals,
        graph_paths=graph_paths,
        auto_start_workers=False
    )
    app = Phase2TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        # Get the editor widget
        editor = screen.query_one("#content-editor")

        # Edit the content of first block
        editor.text = "Modified content for first block"
        await pilot.pause()

        # Navigate to next block (should trigger auto-save)
        await pilot.press("j")
        await pilot.pause()

        # Check that first block's current_content was updated
        assert screen.edited_content[0].current_content == "Modified content for first block"


@pytest.mark.asyncio
async def test_navigation_only_works_when_editor_unfocused(sample_blocks, sample_edited_content, journal_outline, graph_paths):
    """Test that j/k keys only navigate when editor is unfocused."""
    screen = Phase2Screen(
        blocks=sample_blocks,
        edited_content=sample_edited_content,
        journals=journals,
        graph_paths=graph_paths,
        auto_start_workers=False
    )
    app = Phase2TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        # Focus the editor
        await pilot.press("tab")
        await pilot.pause()

        editor = screen.query_one("#content-editor")
        assert editor.has_focus

        # Try to navigate with j while focused - should not navigate
        await pilot.press("j")
        await pilot.pause()

        # Should still be at first block
        assert screen.current_block_index == 0

        # Unfocus the editor
        await pilot.press("tab")
        await pilot.pause()

        assert not editor.has_focus

        # Now j should work
        await pilot.press("j")
        await pilot.pause()

        assert screen.current_block_index == 1


@pytest.mark.asyncio
async def test_navigation_wraps_at_boundaries(sample_blocks, sample_edited_content, journal_outline, graph_paths):
    """Test that navigation stops at first and last blocks."""
    screen = Phase2Screen(
        blocks=sample_blocks,
        edited_content=sample_edited_content,
        journals=journals,
        graph_paths=graph_paths,
        auto_start_workers=False
    )
    app = Phase2TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        # At first block, pressing k should do nothing
        assert screen.current_block_index == 0
        await pilot.press("k")
        await pilot.pause()
        assert screen.current_block_index == 0

        # Navigate to last block
        await pilot.press("j", "j")
        await pilot.pause()
        assert screen.current_block_index == 2

        # Pressing j at last block should do nothing
        await pilot.press("j")
        await pilot.pause()
        assert screen.current_block_index == 2


@pytest.mark.asyncio
async def test_block_indicator_updates_on_navigation(sample_blocks, sample_edited_content, journal_outline, graph_paths):
    """Test that block count indicator updates when navigating."""
    screen = Phase2Screen(
        blocks=sample_blocks,
        edited_content=sample_edited_content,
        journals=journals,
        graph_paths=graph_paths,
        auto_start_workers=False
    )
    app = Phase2TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        # Check initial indicator (should show "Block 1 of 3")
        # This assumes there's a widget or label showing this info
        # The actual implementation may vary

        # Navigate to second block
        await pilot.press("j")
        await pilot.pause()

        # Indicator should update to "Block 2 of 3"
        # Verify the screen state
        assert screen.current_block_index == 1
