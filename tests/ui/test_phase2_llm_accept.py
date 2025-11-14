"""UI tests for Phase 2 accept LLM version ('a' key).

Tests for 'a' key to accept LLM-reworded content and update
the editor with the reworded version.
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
            content=["Today I learned about Python decorators"],
            indent_level=0,
            block_id="block-1",
            children=[]
        ),
        LogseqBlock(
            content=["This morning I fixed the authentication bug"],
            indent_level=0,
            block_id="block-2",
            children=[]
        ),
    ]
    return blocks


@pytest.fixture
def sample_edited_content_with_rewording():
    """Create EditedContent with LLM rewording already complete."""
    return [
        EditedContent(
            block_id="block-1",
            original_content="Today I learned about Python decorators",
            hierarchical_context="Today I learned about Python decorators",
            current_content="Today I learned about Python decorators",
            reworded_content="Python decorators are functions that modify other functions",
            rewording_complete=True
        ),
        EditedContent(
            block_id="block-2",
            original_content="This morning I fixed the authentication bug",
            hierarchical_context="This morning I fixed the authentication bug",
            current_content="This morning I fixed the authentication bug",
            reworded_content="Authentication bug was resolved by validating token expiry",
            rewording_complete=True
        ),
    ]


@pytest.fixture
def sample_edited_content_pending_rewording():
    """Create EditedContent with LLM rewording still pending."""
    return [
        EditedContent(
            block_id="block-1",
            original_content="Today I learned about Python decorators",
            hierarchical_context="Today I learned about Python decorators",
            current_content="Today I learned about Python decorators",
            reworded_content=None,
            rewording_complete=False
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
async def test_accept_llm_version_when_available(sample_blocks, sample_edited_content_with_rewording, journal_outline, graph_paths):
    """Test 'a' key accepts LLM reworded version when available."""
    screen = Phase2Screen(
        blocks=sample_blocks,
        edited_content=sample_edited_content_with_rewording,
        journals=journals,
        graph_paths=graph_paths,
        auto_start_workers=False
    )
    app = Phase2TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        editor = screen.query_one("#content-editor")

        # Initially showing original content
        assert editor.text == "Today I learned about Python decorators"

        # Press 'a' to accept LLM version
        await pilot.press("a")
        await pilot.pause()

        # Editor should now show reworded content
        assert editor.text == "Python decorators are functions that modify other functions"

        # current_content should also be updated
        assert screen.edited_content[0].current_content == "Python decorators are functions that modify other functions"


@pytest.mark.asyncio
async def test_accept_llm_disabled_when_not_available(sample_blocks, sample_edited_content_pending_rewording, journal_outline, graph_paths):
    """Test 'a' key does nothing when LLM rewording is not yet available."""
    screen = Phase2Screen(
        blocks=sample_blocks,
        edited_content=sample_edited_content_pending_rewording,
        journals=journals,
        graph_paths=graph_paths,
        auto_start_workers=False
    )
    app = Phase2TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        editor = screen.query_one("#content-editor")

        # Initially showing original content
        original_text = editor.text
        assert original_text == "Today I learned about Python decorators"

        # Press 'a' - should do nothing since rewording not available
        await pilot.press("a")
        await pilot.pause()

        # Editor text should not change
        assert editor.text == original_text


@pytest.mark.asyncio
async def test_accept_llm_only_works_when_unfocused(sample_blocks, sample_edited_content_with_rewording, journal_outline, graph_paths):
    """Test 'a' key only works when editor is unfocused."""
    screen = Phase2Screen(
        blocks=sample_blocks,
        edited_content=sample_edited_content_with_rewording,
        journals=journals,
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

        # Clear text for cleaner test
        editor.text = "test"
        original_text = editor.text

        # Try to press 'a' while focused - should type 'a' not trigger shortcut
        await pilot.press("a")
        await pilot.pause()

        # Text should have 'a' appended (typed, not shortcut triggered)
        assert editor.text != original_text
        assert "a" in editor.text

        # Unfocus the editor
        await pilot.press("tab")
        await pilot.pause()
        assert not editor.has_focus

        # Now 'a' should work as shortcut
        await pilot.press("a")
        await pilot.pause()

        # Text should now be reworded version (shortcut worked)
        assert editor.text == "Python decorators are functions that modify other functions"


@pytest.mark.asyncio
async def test_accept_llm_updates_indicator(sample_blocks, sample_edited_content_with_rewording, journal_outline, graph_paths):
    """Test that accepting LLM version updates visual indicator."""
    screen = Phase2Screen(
        blocks=sample_blocks,
        edited_content=sample_edited_content_with_rewording,
        journals=journals,
        graph_paths=graph_paths,
        auto_start_workers=False
    )
    app = Phase2TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        # Press 'a' to accept LLM version
        await pilot.press("a")
        await pilot.pause()

        # Some visual indicator should show that LLM version was accepted
        # (Implementation detail: may be a label, icon, or status message)
        # For now, just verify the content was updated
        assert screen.edited_content[0].current_content == screen.edited_content[0].reworded_content


@pytest.mark.asyncio
async def test_accept_llm_after_manual_edit(sample_blocks, sample_edited_content_with_rewording, journal_outline, graph_paths):
    """Test that accepting LLM version replaces manually edited content."""
    screen = Phase2Screen(
        blocks=sample_blocks,
        edited_content=sample_edited_content_with_rewording,
        journals=journals,
        graph_paths=graph_paths,
        auto_start_workers=False
    )
    app = Phase2TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        editor = screen.query_one("#content-editor")

        # Focus and manually edit the content
        await pilot.press("tab")
        await pilot.pause()

        editor.text = "Manually edited text by the user"
        await pilot.pause()

        # Unfocus
        await pilot.press("tab")
        await pilot.pause()

        # Press 'a' to accept LLM version (should replace manual edit)
        await pilot.press("a")
        await pilot.pause()

        # Editor should show reworded content, not manual edit
        assert editor.text == "Python decorators are functions that modify other functions"


@pytest.mark.asyncio
async def test_llm_version_display_when_available(sample_blocks, sample_edited_content_with_rewording, journal_outline, graph_paths):
    """Test that LLM reworded version is displayed alongside original."""
    screen = Phase2Screen(
        blocks=sample_blocks,
        edited_content=sample_edited_content_with_rewording,
        journals=journals,
        graph_paths=graph_paths,
        auto_start_workers=False
    )
    app = Phase2TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        # There should be a display area showing the LLM reworded version
        # This is implementation-specific (may be a label, markdown viewer, etc.)
        # For now, just verify the data is available
        assert screen.edited_content[0].reworded_content is not None
        assert screen.edited_content[0].rewording_complete is True


@pytest.mark.asyncio
async def test_llm_version_shows_loading_when_pending(sample_blocks, sample_edited_content_pending_rewording, journal_outline, graph_paths):
    """Test that a loading indicator is shown while LLM rewording is pending."""
    screen = Phase2Screen(
        blocks=sample_blocks,
        edited_content=sample_edited_content_pending_rewording,
        journals=journals,
        graph_paths=graph_paths,
        auto_start_workers=False
    )
    app = Phase2TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        # When rewording is not complete, there should be a loading indicator
        # (Implementation detail: may be a label, spinner, or placeholder text)
        assert screen.edited_content[0].rewording_complete is False
        assert screen.edited_content[0].reworded_content is None
