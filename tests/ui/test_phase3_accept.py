"""UI tests for Phase 3 decision acceptance (y key triggers write).

Tests for accepting integration decisions with 'y' key, which triggers
immediate write operations with provenance markers.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from textual.app import App
from logsqueak.tui.screens.integration_review import Phase3Screen
from logsqueak.models.integration_decision import IntegrationDecision
from logsqueak.models.edited_content import EditedContent
from logseq_outline.parser import LogseqBlock, LogseqOutline


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
            content=["Knowledge block about Python async"],
            indent_level=0,
            block_id="journal-block-1",
            children=[]
        ),
    ]


@pytest.fixture
def sample_edited_content():
    """Create sample EditedContent for testing."""
    return [
        EditedContent(
            block_id="journal-block-1",
            original_content="Knowledge block about Python async",
            hierarchical_context="- Daily journal\n  - Programming notes\n    - Knowledge block about Python async",
            current_content="Python async programming patterns"
        ),
    ]


@pytest.fixture
def sample_decisions():
    """Create sample integration decisions for testing."""
    return [
        IntegrationDecision(
            knowledge_block_id="journal-block-1",
            target_page="Python/Concurrency",
            action="add_section",
            confidence=0.85,
            refined_text="Python async programming patterns",
            reasoning="Relevant to async programming concepts"
        ),
        IntegrationDecision(
            knowledge_block_id="journal-block-1",
            target_page="Python/Best Practices",
            action="add_under",
            target_block_id="target-1",
            target_block_title="Code Patterns",
            confidence=0.75,
            refined_text="Python async programming patterns",
            reasoning="Relevant to coding best practices"
        ),
    ]


@pytest.fixture
def sample_journal_content(sample_journal_blocks):
    """Create sample journal content (markdown) for preview."""
    from logseq_outline.parser import LogseqOutline
    outline = LogseqOutline(blocks=sample_journal_blocks, source_text="", frontmatter=[])
    return outline.render()


@pytest.fixture
def sample_page_contents():
    """Create sample page contents for testing."""
    return {
        "Python/Concurrency": LogseqOutline(
            blocks=[
                LogseqBlock(
                    content=["# Python Concurrency"],
                    indent_level=0,
                    block_id="concurrency-page-block-1",
                    children=[]
                )
            ],
            source_text="",
            frontmatter=[]
        ),
        "Python/Best Practices": LogseqOutline(
            blocks=[
                LogseqBlock(
                    content=["# Python Best Practices"],
                    indent_level=0,
                    block_id="best-practices-page-block-1",
                    children=[]
                )
            ],
            source_text="",
            frontmatter=[]
        )
    }


@pytest.mark.asyncio
async def test_y_key_accepts_current_decision(
    sample_journal_blocks, sample_edited_content, sample_page_contents, sample_decisions, sample_journal_content, sample_journals
):
    """Test y key accepts current decision and triggers write."""
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

    # Mock the write operation
    screen.write_integration = AsyncMock(return_value=True)

    async with app.run_test() as pilot:
        await pilot.pause()

        # Verify initial state - decision is pending
        assert sample_decisions[0].write_status == "pending"

        # Press y to accept current decision
        await pilot.press("y")
        await pilot.pause()

        # Verify write was triggered
        screen.write_integration.assert_called_once()

        # Verify decision status changed to completed
        assert sample_decisions[0].write_status == "completed"


@pytest.mark.asyncio
async def test_accept_decision_shows_checkmark(
    sample_journal_blocks, sample_edited_content, sample_page_contents, sample_decisions, sample_journal_content, sample_journals
):
    """Test that accepted decision shows checkmark (✓) indicator."""
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

    screen.write_integration = AsyncMock(return_value=True)

    async with app.run_test() as pilot:
        await pilot.pause()

        # Accept first decision
        await pilot.press("y")
        await pilot.pause()

        # Decision list should show ✓ indicator for completed decision
        decision_list = screen.query_one("#decision-list")
        # (Exact visual representation depends on widget implementation)


@pytest.mark.asyncio
async def test_accept_multiple_decisions_for_same_block(
    sample_journal_blocks, sample_edited_content, sample_page_contents, sample_decisions, sample_journal_content, sample_journals
):
    """Test that user can accept multiple decisions for the same block."""
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

    screen.write_integration = AsyncMock(return_value=True)

    async with app.run_test() as pilot:
        await pilot.pause()

        # Accept first decision
        await pilot.press("y")
        await pilot.pause()

        assert sample_decisions[0].write_status == "completed"

        # Navigate to second decision
        await pilot.press("j")
        await pilot.pause()

        # Accept second decision
        await pilot.press("y")
        await pilot.pause()

        assert sample_decisions[1].write_status == "completed"

        # Both decisions should be marked as completed
        # User can integrate same block to multiple pages


@pytest.mark.asyncio
async def test_write_failure_marks_decision_as_failed(
    sample_journal_blocks, sample_edited_content, sample_page_contents, sample_decisions, sample_journal_content, sample_journals
):
    """Test that write failure marks decision as failed with error indicator."""
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

    # Mock write operation to fail
    screen.write_integration = AsyncMock(
        side_effect=ValueError("Target block not found")
    )

    async with app.run_test() as pilot:
        await pilot.pause()

        # Press y to accept current decision
        await pilot.press("y")
        await pilot.pause()

        # Verify decision status changed to failed
        assert sample_decisions[0].write_status == "failed"
        assert sample_decisions[0].error_message is not None
        assert "Target block not found" in sample_decisions[0].error_message


@pytest.mark.asyncio
async def test_failed_decision_shows_warning_indicator(
    sample_journal_blocks, sample_edited_content, sample_page_contents, sample_decisions, sample_journal_content, sample_journals
):
    """Test that failed decision shows warning (⚠) indicator."""
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

    screen.write_integration = AsyncMock(
        side_effect=ValueError("Write failed")
    )

    async with app.run_test() as pilot:
        await pilot.pause()

        # Accept decision (will fail)
        await pilot.press("y")
        await pilot.pause()

        # Decision list should show ⚠ indicator for failed decision
        # Error details should be visible


@pytest.mark.asyncio
async def test_continue_after_write_failure(
    sample_journal_blocks, sample_edited_content, sample_page_contents, sample_decisions, sample_journal_content, sample_journals
):
    """Test that user can continue with remaining decisions after a failure."""
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

    # First write fails, second succeeds
    screen.write_integration = AsyncMock(
        side_effect=[ValueError("Write failed"), True]
    )

    async with app.run_test() as pilot:
        await pilot.pause()

        # Accept first decision (will fail)
        await pilot.press("y")
        await pilot.pause()

        assert sample_decisions[0].write_status == "failed"

        # Navigate to second decision
        await pilot.press("j")
        await pilot.pause()

        # Accept second decision (will succeed)
        await pilot.press("y")
        await pilot.pause()

        assert sample_decisions[1].write_status == "completed"


@pytest.mark.asyncio
async def test_cannot_accept_already_completed_decision(
    sample_journal_blocks, sample_edited_content, sample_page_contents, sample_decisions, sample_journal_content, sample_journals
):
    """Test that pressing y on a completed decision does nothing."""
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

    screen.write_integration = AsyncMock(return_value=True)

    async with app.run_test() as pilot:
        await pilot.pause()

        # Accept decision
        await pilot.press("y")
        await pilot.pause()

        call_count_after_first = screen.write_integration.call_count

        # Try to accept again
        await pilot.press("y")
        await pilot.pause()

        # Write should not be called again
        assert screen.write_integration.call_count == call_count_after_first
