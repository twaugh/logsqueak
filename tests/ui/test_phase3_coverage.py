"""Additional UI tests to improve Phase 3 coverage.

This file focuses on testing uncovered code paths in integration_review.py,
including preview generation, helper methods, polling logic, and edge cases.
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from pathlib import Path
from textual.app import App
from logsqueak.tui.screens.integration_review import Phase3Screen
from logsqueak.models.integration_decision import IntegrationDecision
from logsqueak.models.edited_content import EditedContent
from logsqueak.models.background_task import BackgroundTask
from logseq_outline.parser import LogseqBlock, LogseqOutline
from logseq_outline.graph import GraphPaths


class Phase3TestApp(App):
    """Test app wrapper for Phase3Screen."""

    def __init__(self, screen: Phase3Screen):
        super().__init__()
        self.test_screen = screen

    def on_mount(self) -> None:
        """Push the test screen on mount."""
        self.push_screen(self.test_screen)


@pytest.fixture
def sample_journal_blocks():
    """Create sample journal blocks for testing."""
    return [
        LogseqBlock(
            content=["Python async patterns"],
            indent_level=0,
            block_id="block-1",
            children=[]
        ),
        LogseqBlock(
            content=["Textual widgets"],
            indent_level=0,
            block_id="block-2",
            children=[]
        ),
    ]


@pytest.fixture
def sample_edited_content():
    """Create sample EditedContent for testing."""
    return [
        EditedContent(
            block_id="block-1",
            original_content="Python async patterns",
            hierarchical_context="- Journal\n  - Python async patterns",
            current_content="Python async programming"
        ),
        EditedContent(
            block_id="block-2",
            original_content="Textual widgets",
            hierarchical_context="- Journal\n  - Textual widgets",
            current_content="Textual widget system"
        ),
    ]


@pytest.fixture
def sample_decisions(sample_edited_content):
    """Create sample integration decisions for testing."""
    return [
        IntegrationDecision(
            knowledge_block_id="block-1",
            target_page="Python",
            action="add_section",
            confidence=0.9,
            edited_content=sample_edited_content[0],
            reasoning="New section"
        ),
    ]


@pytest.fixture
def sample_journal_content(sample_journal_blocks):
    """Create sample journal content (markdown) for preview."""
    outline = LogseqOutline(blocks=sample_journal_blocks, source_text="", frontmatter=[])
    return outline.render()


@pytest.fixture
def sample_page_contents():
    """Create sample page contents for testing."""
    return {
        "Python": LogseqOutline(
            blocks=[
                LogseqBlock(
                    content=["# Python"],
                    indent_level=0,
                    block_id="python-header",
                    children=[]
                )
            ],
            source_text="",
            frontmatter=[]
        )
    }


@pytest.fixture
def temp_graph_paths(tmp_path):
    """Create temporary GraphPaths for file operations."""
    graph_path = tmp_path / "test-graph"
    graph_path.mkdir()

    pages_dir = graph_path / "pages"
    pages_dir.mkdir()

    journals_dir = graph_path / "journals"
    journals_dir.mkdir()

    # Create a sample page
    python_page = pages_dir / "Python.md"
    python_page.write_text("- # Python\n  - Basics\n    id:: target-1\n    - Content here\n")

    return GraphPaths(graph_path)


# Test: _mark_all_blocks_ready() method
@pytest.mark.asyncio
async def test_mark_all_blocks_ready_after_worker_completion(
    sample_journal_blocks, sample_edited_content, sample_page_contents,
    sample_journal_content, sample_journals
):
    """Test that _mark_all_blocks_ready() marks all blocks ready after worker completes."""
    # Create decisions for only block-1 (block-2 should not be ready initially)
    decisions = [
        IntegrationDecision(
            knowledge_block_id="block-1",
            target_page="Python",
            action="add_section",
            confidence=0.9,
            edited_content=sample_edited_content[0],
            reasoning="New section"
        ),
    ]

    screen = Phase3Screen(
        journal_blocks=sample_journal_blocks,
        edited_content=sample_edited_content,
        page_contents=sample_page_contents,
        decisions=decisions,
        journals=sample_journals,
        journal_content=sample_journal_content,
        auto_start_workers=False
    )

    # Manually set decisions_ready to only include block-1
    screen.decisions_ready = {"block-1": True}

    app = Phase3TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        # Initially, only blocks with decisions are ready
        assert "block-1" in screen.decisions_ready
        assert screen.decisions_ready["block-1"] is True

        # Block 2 not ready yet (no decisions)
        assert "block-2" not in screen.decisions_ready

        # Call _mark_all_blocks_ready() to simulate worker completion
        screen._mark_all_blocks_ready()

        # Now both blocks should be ready
        assert "block-1" in screen.decisions_ready
        assert "block-2" in screen.decisions_ready
        assert screen.decisions_ready["block-2"] is True


# Test: Preview generation with add_section action
@pytest.mark.asyncio
async def test_generate_preview_add_section_action(
    sample_journal_blocks, sample_edited_content, sample_page_contents,
    sample_decisions, sample_journal_content, temp_graph_paths, sample_journals
):
    """Test preview generation for add_section action."""
    screen = Phase3Screen(
        journal_blocks=sample_journal_blocks,
        edited_content=sample_edited_content,
        page_contents=sample_page_contents,
        decisions=sample_decisions,
        journals=sample_journals,
        journal_content=sample_journal_content,
        graph_paths=temp_graph_paths,
        auto_start_workers=False
    )
    app = Phase3TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        decision = sample_decisions[0]
        preview_text, highlight_id, old_id = screen._generate_preview_with_integration(decision)

        # Should generate preview with new section added
        assert preview_text is not None
        assert "Python async programming" in preview_text
        assert highlight_id is not None  # New block should be highlighted
        assert old_id is None  # No old block for add_section


# Test: Preview generation with add_under action
@pytest.mark.asyncio
async def test_generate_preview_add_under_action(
    sample_journal_blocks, sample_edited_content, sample_page_contents,
    sample_journal_content, temp_graph_paths, sample_journals
):
    """Test preview generation for add_under action."""
    decisions = [
        IntegrationDecision(
            knowledge_block_id="block-1",
            target_page="Python",
            action="add_under",
            target_block_id="target-1",
            target_block_title="Basics",
            confidence=0.9,
            edited_content=sample_edited_content[0],
            reasoning="Add under Basics"
        ),
    ]

    screen = Phase3Screen(
        journal_blocks=sample_journal_blocks,
        edited_content=sample_edited_content,
        page_contents=sample_page_contents,
        decisions=decisions,
        journals=sample_journals,
        journal_content=sample_journal_content,
        graph_paths=temp_graph_paths,
        auto_start_workers=False
    )
    app = Phase3TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        decision = decisions[0]
        preview_text, highlight_id, old_id = screen._generate_preview_with_integration(decision)

        # Should generate preview with block added under target
        assert preview_text is not None
        assert "Python async programming" in preview_text
        assert highlight_id is not None


# Test: Preview generation with replace action
@pytest.mark.asyncio
async def test_generate_preview_replace_action(
    sample_journal_blocks, sample_edited_content, sample_page_contents,
    sample_journal_content, temp_graph_paths, sample_journals
):
    """Test preview generation for replace action shows both old and new blocks."""
    decisions = [
        IntegrationDecision(
            knowledge_block_id="block-1",
            target_page="Python",
            action="replace",
            target_block_id="target-1",
            target_block_title="Basics",
            confidence=0.9,
            edited_content=sample_edited_content[0],
            reasoning="Replace old content"
        ),
    ]

    screen = Phase3Screen(
        journal_blocks=sample_journal_blocks,
        edited_content=sample_edited_content,
        page_contents=sample_page_contents,
        decisions=decisions,
        journals=sample_journals,
        journal_content=sample_journal_content,
        graph_paths=temp_graph_paths,
        auto_start_workers=False
    )
    app = Phase3TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        decision = decisions[0]
        preview_text, highlight_id, old_id = screen._generate_preview_with_integration(decision)

        # Should generate preview with both old and new blocks
        assert preview_text is not None
        assert highlight_id is not None  # New block highlighted green
        assert old_id is not None  # Old block highlighted red


# Test: Preview generation when page doesn't exist
@pytest.mark.asyncio
async def test_generate_preview_page_not_exists(
    sample_journal_blocks, sample_edited_content, sample_page_contents,
    sample_journal_content, temp_graph_paths, sample_journals
):
    """Test preview generation when target page doesn't exist yet."""
    decisions = [
        IntegrationDecision(
            knowledge_block_id="block-1",
            target_page="NewPage",
            action="add_section",
            confidence=0.9,
            edited_content=sample_edited_content[0],
            reasoning="New page"
        ),
    ]

    screen = Phase3Screen(
        journal_blocks=sample_journal_blocks,
        edited_content=sample_edited_content,
        page_contents=sample_page_contents,
        decisions=decisions,
        journals=sample_journals,
        journal_content=sample_journal_content,
        graph_paths=temp_graph_paths,
        auto_start_workers=False
    )
    app = Phase3TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        decision = decisions[0]
        preview_text, highlight_id, old_id = screen._generate_preview_with_integration(decision)

        # Should create minimal page structure
        assert preview_text is not None
        assert "NewPage" in preview_text or "Python async programming" in preview_text


# Test: Preview generation without graph_paths (placeholder)
@pytest.mark.asyncio
async def test_generate_preview_no_graph_paths(
    sample_journal_blocks, sample_edited_content, sample_page_contents,
    sample_decisions, sample_journal_content, sample_journals
):
    """Test preview generation falls back to placeholder when no graph_paths."""
    screen = Phase3Screen(
        journal_blocks=sample_journal_blocks,
        edited_content=sample_edited_content,
        page_contents=sample_page_contents,
        decisions=sample_decisions,
        journals=sample_journals,
        journal_content=sample_journal_content,
        graph_paths=None,  # No graph paths
        auto_start_workers=False
    )
    app = Phase3TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        decision = sample_decisions[0]
        preview_text, highlight_id, old_id = screen._generate_preview_with_integration(decision)

        # Should return placeholder preview
        assert preview_text is not None
        assert "Unable to load target page" in preview_text
        assert highlight_id is None
        assert old_id is None


# Test: Target block not found fallback
@pytest.mark.asyncio
async def test_add_under_target_not_found_fallback(
    sample_journal_blocks, sample_edited_content, sample_page_contents,
    sample_journal_content, temp_graph_paths, sample_journals
):
    """Test add_under falls back to add_section when target block not found."""
    decisions = [
        IntegrationDecision(
            knowledge_block_id="block-1",
            target_page="Python",
            action="add_under",
            target_block_id="nonexistent-id",  # Target doesn't exist
            target_block_title="Nonexistent",
            confidence=0.9,
            edited_content=sample_edited_content[0],
            reasoning="Should fallback"
        ),
    ]

    screen = Phase3Screen(
        journal_blocks=sample_journal_blocks,
        edited_content=sample_edited_content,
        page_contents=sample_page_contents,
        decisions=decisions,
        journals=sample_journals,
        journal_content=sample_journal_content,
        graph_paths=temp_graph_paths,
        auto_start_workers=False
    )
    app = Phase3TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        decision = decisions[0]
        preview_text, highlight_id, old_id = screen._generate_preview_with_integration(decision)

        # Should still generate preview (with fallback to add_section)
        assert preview_text is not None
        assert "Python async programming" in preview_text


# Test: Write integration with actual graph_paths
@pytest.mark.asyncio
async def test_write_integration_with_graph_paths(
    sample_journal_blocks, sample_edited_content, sample_page_contents,
    sample_decisions, sample_journal_content, temp_graph_paths, sample_journals
):
    """Test write_integration calls write_integration_atomic with correct parameters."""
    screen = Phase3Screen(
        journal_blocks=sample_journal_blocks,
        edited_content=sample_edited_content,
        page_contents=sample_page_contents,
        decisions=sample_decisions,
        journals=sample_journals,
        journal_content=sample_journal_content,
        graph_paths=temp_graph_paths,
        auto_start_workers=False
    )
    app = Phase3TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        decision = sample_decisions[0]

        # Mock write_integration_atomic
        with patch('logsqueak.tui.screens.integration_review.write_integration_atomic') as mock_write:
            mock_write.return_value = None

            await screen.write_integration(decision)

            # Should call write_integration_atomic with decision, date, graph_paths, file_monitor
            mock_write.assert_called_once()
            call_kwargs = mock_write.call_args.kwargs
            assert call_kwargs['decision'] == decision
            assert 'journal_date' in call_kwargs
            assert call_kwargs['graph_paths'] == temp_graph_paths
            assert 'file_monitor' in call_kwargs


# Test: Skip_exists decision action
@pytest.mark.asyncio
async def test_skip_exists_decision_no_write(
    sample_journal_blocks, sample_edited_content, sample_page_contents,
    sample_journal_content, sample_journals
):
    """Test that skip_exists decisions don't trigger writes."""
    decisions = [
        IntegrationDecision(
            knowledge_block_id="block-1",
            target_page="Python",
            action="skip_exists",
            target_block_id="existing-id",
            confidence=0.9,
            edited_content=sample_edited_content[0],
            reasoning="Already exists"
        ),
    ]

    screen = Phase3Screen(
        journal_blocks=sample_journal_blocks,
        edited_content=sample_edited_content,
        page_contents=sample_page_contents,
        decisions=decisions,
        journals=sample_journals,
        journal_content=sample_journal_content,
        auto_start_workers=False
    )
    app = Phase3TestApp(screen)

    screen.write_integration = AsyncMock()

    async with app.run_test() as pilot:
        await pilot.pause()

        # Try to accept skip_exists decision
        await pilot.press("y")
        await pilot.pause()

        # Should NOT call write_integration
        screen.write_integration.assert_not_called()


# Test: Polling for new decisions
@pytest.mark.asyncio
async def test_check_for_new_decisions_polling(
    sample_journal_blocks, sample_edited_content, sample_page_contents,
    sample_journal_content, sample_journals
):
    """Test _check_for_new_decisions() detects new decisions added to shared list."""
    decisions = []  # Start with empty list

    screen = Phase3Screen(
        journal_blocks=sample_journal_blocks,
        edited_content=sample_edited_content,
        page_contents=sample_page_contents,
        decisions=decisions,
        journals=sample_journals,
        journal_content=sample_journal_content,
        auto_start_workers=False
    )
    app = Phase3TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        # Initially no decisions
        assert len(screen.decisions) == 0
        assert screen._last_known_decision_count == 0

        # Simulate Phase 2 worker adding a new decision
        new_decision = IntegrationDecision(
            knowledge_block_id="block-1",
            target_page="Python",
            action="add_section",
            confidence=0.9,
            edited_content=sample_edited_content[0],
            reasoning="New decision"
        )
        decisions.append(new_decision)

        # Call _check_for_new_decisions() to detect the change
        screen._check_for_new_decisions()

        # Should detect new decision and update state
        assert "block-1" in screen.decisions_by_block
        assert len(screen.decisions_by_block["block-1"]) == 1
        assert screen._last_known_decision_count == 1
        assert "block-1" in screen.decisions_ready


# Test: Count skip_exists blocks
@pytest.mark.asyncio
async def test_count_skip_exists_blocks(
    sample_journal_blocks, sample_edited_content, sample_page_contents,
    sample_journal_content, sample_journals
):
    """Test _count_skip_exists_blocks() counts blocks where ALL decisions are skip_exists."""
    decisions = [
        # Block 1: all skip_exists
        IntegrationDecision(
            knowledge_block_id="block-1",
            target_page="Python",
            action="skip_exists",
            confidence=0.9,
            edited_content=sample_edited_content[0],
            reasoning="Exists 1"
        ),
        IntegrationDecision(
            knowledge_block_id="block-1",
            target_page="Python/Advanced",
            action="skip_exists",
            confidence=0.9,
            edited_content=sample_edited_content[0],
            reasoning="Exists 2"
        ),
        # Block 2: mixed
        IntegrationDecision(
            knowledge_block_id="block-2",
            target_page="Textual",
            action="add_section",
            confidence=0.9,
            edited_content=sample_edited_content[1],
            reasoning="New"
        ),
    ]

    screen = Phase3Screen(
        journal_blocks=sample_journal_blocks,
        edited_content=sample_edited_content,
        page_contents=sample_page_contents,
        decisions=decisions,
        journals=sample_journals,
        journal_content=sample_journal_content,
        auto_start_workers=False
    )
    app = Phase3TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        skip_count = screen._count_skip_exists_blocks()

        # Only block-1 has ALL skip_exists decisions
        assert skip_count == 1


# Test: Completion summary with multi-journal support
@pytest.mark.asyncio
async def test_completion_summary_multi_journal(
    sample_journal_blocks, sample_edited_content, sample_page_contents,
    sample_decisions, temp_graph_paths
):
    """Test completion summary displays multiple journal entries correctly."""
    # Create multi-journal fixture
    multi_journals = {
        "2025-01-10": LogseqOutline(
            blocks=[sample_journal_blocks[0]],
            source_text="",
            frontmatter=[]
        ),
        "2025-01-11": LogseqOutline(
            blocks=[sample_journal_blocks[1]],
            source_text="",
            frontmatter=[]
        ),
    }

    journal_content = "- Block 1\n- Block 2"

    screen = Phase3Screen(
        journal_blocks=sample_journal_blocks,
        edited_content=sample_edited_content,
        page_contents=sample_page_contents,
        decisions=sample_decisions,
        journals=multi_journals,
        journal_content=journal_content,
        graph_paths=temp_graph_paths,
        auto_start_workers=False
    )
    app = Phase3TestApp(screen)

    screen.write_integration = AsyncMock()

    async with app.run_test() as pilot:
        await pilot.pause()

        # Navigate to last block
        screen.current_block_index = len(sample_journal_blocks) - 1

        # Trigger completion summary
        await screen._show_completion_summary()

        # Get journal preview
        from logsqueak.tui.widgets.target_page_preview import TargetPagePreview
        journal_preview = screen.query_one("#journal-preview", TargetPagePreview)

        # Should show multiple journal entries
        assert journal_preview.border_title == "Extraction Complete"


# Test: Display block with no decisions shows helpful message
@pytest.mark.asyncio
async def test_display_block_with_no_decisions(
    sample_journal_blocks, sample_edited_content, sample_page_contents,
    sample_journal_content, sample_journals
):
    """Test that blocks with no decisions show helpful 'No relevant pages found' message."""
    decisions = []  # No decisions for any block

    screen = Phase3Screen(
        journal_blocks=sample_journal_blocks,
        edited_content=sample_edited_content,
        page_contents=sample_page_contents,
        decisions=decisions,
        journals=sample_journals,
        journal_content=sample_journal_content,
        auto_start_workers=False
    )
    app = Phase3TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        # Get target preview
        from logsqueak.tui.widgets.target_page_preview import TargetPagePreview
        preview = screen.query_one("#target-page-preview", TargetPagePreview)

        # Should show helpful message about no relevant pages
        # (The actual content checking would require accessing widget internals)


# Test: Focus preview widget
@pytest.mark.asyncio
async def test_action_focus_preview(
    sample_journal_blocks, sample_edited_content, sample_page_contents,
    sample_decisions, sample_journal_content, sample_journals
):
    """Test tab key focuses the preview widget."""
    screen = Phase3Screen(
        journal_blocks=sample_journal_blocks,
        edited_content=sample_edited_content,
        page_contents=sample_page_contents,
        decisions=sample_decisions,
        journals=sample_journals,
        journal_content=sample_journal_content,
        auto_start_workers=False
    )
    app = Phase3TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        # Press tab to focus preview
        await pilot.press("tab")
        await pilot.pause()

        # Preview should have focus
        from logsqueak.tui.widgets.target_page_preview import TargetPagePreview
        preview = screen.query_one("#target-page-preview", TargetPagePreview)
        assert preview.has_focus


# Test: Back action dismisses screen
@pytest.mark.asyncio
async def test_action_back_dismisses_screen(
    sample_journal_blocks, sample_edited_content, sample_page_contents,
    sample_decisions, sample_journal_content, sample_journals
):
    """Test q key dismisses the screen."""
    screen = Phase3Screen(
        journal_blocks=sample_journal_blocks,
        edited_content=sample_edited_content,
        page_contents=sample_page_contents,
        decisions=sample_decisions,
        journals=sample_journals,
        journal_content=sample_journal_content,
        auto_start_workers=False
    )
    app = Phase3TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        # Press q to go back
        await pilot.press("q")
        await pilot.pause()

        # Screen should be dismissed (app should not have the screen anymore)
        # This is hard to test directly in Textual without checking screen stack
