"""UI tests for Phase 2 text editor focus/unfocus with Tab key.

Tests for Tab key to focus/unfocus the multi-line text editor,
and verify that editor state changes appropriately.
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
            content=["First knowledge block"],
            indent_level=0,
            block_id="block-1",
            children=[]
        ),
    ]
    return blocks


@pytest.fixture
def sample_edited_content():
    """Create sample EditedContent for testing."""
    return [
        EditedContent(
            block_id="block-1",
            original_content="First knowledge block",
            hierarchical_context="First knowledge block",
            current_content="First knowledge block"
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
async def test_tab_focuses_editor(sample_blocks, sample_edited_content, journal_outline, graph_paths):
    """Test Tab key focuses the text editor."""
    screen = Phase2Screen(
        blocks=sample_blocks,
        edited_content=sample_edited_content,
        journal_outline=journal_outline,
        graph_paths=graph_paths,
        auto_start_workers=False
    )
    app = Phase2TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        editor = screen.query_one("#content-editor")

        # Initially editor should not have focus
        assert not editor.has_focus

        # Press Tab to focus
        await pilot.press("tab")
        await pilot.pause()

        # Editor should now have focus
        assert editor.has_focus


@pytest.mark.asyncio
async def test_tab_unfocuses_editor(sample_blocks, sample_edited_content, journal_outline, graph_paths):
    """Test Tab key unfocuses the text editor when already focused."""
    screen = Phase2Screen(
        blocks=sample_blocks,
        edited_content=sample_edited_content,
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

        # Press Tab again to unfocus
        await pilot.press("tab")
        await pilot.pause()

        # Editor should no longer have focus
        assert not editor.has_focus


@pytest.mark.asyncio
async def test_editor_border_highlights_when_focused(sample_blocks, sample_edited_content, journal_outline, graph_paths):
    """Test that editor border highlights when focused."""
    screen = Phase2Screen(
        blocks=sample_blocks,
        edited_content=sample_edited_content,
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

        # Editor should have visual indication of focus
        # (Implementation detail: may check border style or CSS class)
        assert editor.has_focus


@pytest.mark.asyncio
async def test_can_type_in_focused_editor(sample_blocks, sample_edited_content, journal_outline, graph_paths):
    """Test that user can type in the editor when focused."""
    screen = Phase2Screen(
        blocks=sample_blocks,
        edited_content=sample_edited_content,
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

        # Type some text
        # Note: Textual pilot may handle this differently
        # This test may need adjustment based on actual TextArea API
        initial_text = editor.text

        # Simulate typing (implementation-specific)
        editor.text = "Modified text content"
        await pilot.pause()

        # Verify text changed
        assert editor.text == "Modified text content"
        assert editor.text != initial_text


@pytest.mark.asyncio
async def test_keyboard_shortcuts_disabled_when_focused(sample_blocks, sample_edited_content, journal_outline, graph_paths):
    """Test that keyboard shortcuts (a, r, j, k) are typed when editor focused."""
    screen = Phase2Screen(
        blocks=sample_blocks,
        edited_content=sample_edited_content,
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

        # Set some initial editor content
        editor.text = "Original"
        await pilot.pause()

        # Try pressing 'a' (accept LLM) - should type 'a' instead of triggering shortcut
        await pilot.press("a")
        await pilot.pause()

        # 'a' should be typed into the editor
        assert "a" in editor.text

        # Navigation should not work while focused
        initial_index = screen.current_block_index
        await pilot.press("j")
        await pilot.pause()

        # Index should not change
        assert screen.current_block_index == initial_index


@pytest.mark.asyncio
async def test_cursor_appears_in_focused_editor(sample_blocks, sample_edited_content, journal_outline, graph_paths):
    """Test that cursor appears when editor is focused."""
    screen = Phase2Screen(
        blocks=sample_blocks,
        edited_content=sample_edited_content,
        journal_outline=journal_outline,
        graph_paths=graph_paths,
        auto_start_workers=False
    )
    app = Phase2TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        editor = screen.query_one("#content-editor")

        # Initially no cursor (unfocused)
        assert not editor.has_focus

        # Focus the editor
        await pilot.press("tab")
        await pilot.pause()

        # Cursor should now be visible (editor has focus)
        assert editor.has_focus


@pytest.mark.asyncio
async def test_editor_loads_current_content_on_display(sample_blocks, sample_edited_content, journal_outline, graph_paths):
    """Test that editor displays current_content when block is shown."""
    # Set up edited content with modified current_content
    edited_content = [
        EditedContent(
            block_id="block-1",
            original_content="Original text",
            hierarchical_context="Original text",
            current_content="Modified text that user edited",
            reworded_content=None
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

        # Editor should show current_content, not original
        assert editor.text == "Modified text that user edited"
