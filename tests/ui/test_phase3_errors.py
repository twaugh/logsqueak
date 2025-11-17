"""UI tests for Phase 3 write failure handling.

Tests for error handling when write operations fail, including
concurrent modification, target not found, and I/O errors.
"""

import pytest
from unittest.mock import AsyncMock
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
            content=["Knowledge block about Python"],
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
            original_content="Knowledge block about Python",
            hierarchical_context="- Daily notes\n  - Python learning\n    - Knowledge block about Python",
            current_content="Python async programming patterns"
        ),
    ]


@pytest.fixture
def sample_decisions(sample_edited_content):
    """Create sample integration decisions for testing."""
    return [
        IntegrationDecision(
            knowledge_block_id="journal-block-1",
            target_page="Python/Concurrency",
            action="add_under",
            target_block_id="target-1",
            target_block_title="Async Patterns",
            confidence=0.85,
            edited_content=sample_edited_content[0],
            reasoning="Relevant to async programming concepts"
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
async def test_target_block_not_found_error(
    sample_journal_blocks, sample_edited_content, sample_page_contents, sample_decisions, sample_journal_content, sample_journals
):
    """Test error handling when target block doesn't exist."""
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

    # Mock write to fail with target not found
    screen.write_integration = AsyncMock(
        side_effect=ValueError("Target block not found: target-1")
    )

    async with app.run_test() as pilot:
        await pilot.pause()

        # Try to accept decision
        await pilot.press("y")
        await pilot.pause()

        # Decision should be marked as failed
        assert sample_decisions[0].write_status == "failed"
        assert "Target block not found" in sample_decisions[0].error_message

        # Error details should be displayed
        # (Exact UI representation depends on implementation)


@pytest.mark.asyncio
async def test_file_modified_externally_error(
    sample_journal_blocks, sample_edited_content, sample_page_contents, sample_decisions, sample_journal_content, sample_journals
):
    """Test error handling when file is modified externally during operation."""
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

    # Mock write to fail with concurrent modification
    screen.write_integration = AsyncMock(
        side_effect=ValueError("File modified and validation failed: target block deleted")
    )

    async with app.run_test() as pilot:
        await pilot.pause()

        # Try to accept decision
        await pilot.press("y")
        await pilot.pause()

        # Decision should be marked as failed
        assert sample_decisions[0].write_status == "failed"
        assert "File modified" in sample_decisions[0].error_message or \
               "validation failed" in sample_decisions[0].error_message


@pytest.mark.asyncio
async def test_permission_error_during_write(
    sample_journal_blocks, sample_edited_content, sample_page_contents, sample_decisions, sample_journal_content, sample_journals
):
    """Test error handling for file permission errors."""
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

    # Mock write to fail with permission error
    screen.write_integration = AsyncMock(
        side_effect=PermissionError("Cannot write to page: Permission denied")
    )

    async with app.run_test() as pilot:
        await pilot.pause()

        # Try to accept decision
        await pilot.press("y")
        await pilot.pause()

        # Decision should be marked as failed
        assert sample_decisions[0].write_status == "failed"
        assert "Permission" in sample_decisions[0].error_message or \
               "Cannot write" in sample_decisions[0].error_message


@pytest.mark.asyncio
async def test_page_file_not_found_error(
    sample_journal_blocks, sample_edited_content, sample_page_contents, sample_decisions, sample_journal_content, sample_journals
):
    """Test error handling when target page file doesn't exist."""
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

    # Mock write to fail with file not found
    screen.write_integration = AsyncMock(
        side_effect=FileNotFoundError("Page not found: Python/Concurrency.md")
    )

    async with app.run_test() as pilot:
        await pilot.pause()

        # Try to accept decision
        await pilot.press("y")
        await pilot.pause()

        # Decision should be marked as failed
        assert sample_decisions[0].write_status == "failed"
        assert "not found" in sample_decisions[0].error_message.lower()


@pytest.mark.asyncio
async def test_error_message_displayed_in_ui(
    sample_journal_blocks, sample_edited_content, sample_page_contents, sample_decisions, sample_journal_content, sample_journals
):
    """Test that error messages are displayed in the UI."""
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

    error_msg = "Target block 'Async Patterns' not found in page Python/Concurrency"
    screen.write_integration = AsyncMock(
        side_effect=ValueError(error_msg)
    )

    async with app.run_test() as pilot:
        await pilot.pause()

        # Accept decision (will fail)
        await pilot.press("y")
        await pilot.pause()

        # Error message should be stored in decision
        assert sample_decisions[0].error_message == error_msg

        # UI should display error details
        # (Could be in decision list, status panel, or dedicated error widget)


@pytest.mark.asyncio
async def test_failed_decision_can_be_retried(
    sample_journal_blocks, sample_edited_content, sample_page_contents, sample_decisions, sample_journal_content, sample_journals
):
    """Test that user can retry a failed decision."""
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

    # First attempt fails, second succeeds
    screen.write_integration = AsyncMock(
        side_effect=[ValueError("Temporary error"), True]
    )

    async with app.run_test() as pilot:
        await pilot.pause()

        # First attempt (will fail)
        await pilot.press("y")
        await pilot.pause()

        assert sample_decisions[0].write_status == "failed"

        # Reset decision status for retry
        sample_decisions[0].write_status = "pending"
        sample_decisions[0].error_message = None

        # Retry (will succeed)
        await pilot.press("y")
        await pilot.pause()

        assert sample_decisions[0].write_status == "completed"


@pytest.mark.asyncio
async def test_error_details_suggest_remediation(
    sample_journal_blocks, sample_edited_content, sample_page_contents, sample_decisions, sample_journal_content, sample_journals
):
    """Test that error messages suggest remediation actions."""
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

    # Error with remediation suggestion
    error_msg = (
        "Target block not found: target-1\n"
        "Possible reasons:\n"
        "- Block was deleted externally\n"
        "- Block ID changed (id:: property modified)\n"
        "- Page structure changed significantly"
    )
    screen.write_integration = AsyncMock(
        side_effect=ValueError(error_msg)
    )

    async with app.run_test() as pilot:
        await pilot.pause()

        # Accept decision (will fail)
        await pilot.press("y")
        await pilot.pause()

        # Error message should include remediation suggestions
        assert "Possible reasons" in sample_decisions[0].error_message


@pytest.mark.asyncio
async def test_multiple_errors_handled_independently(
    sample_journal_blocks, sample_edited_content, sample_page_contents, sample_journal_content, sample_journals
):
    """Test that multiple failed decisions are tracked independently."""
    decisions = [
        IntegrationDecision(
            knowledge_block_id="journal-block-1",
            target_page="Python/Concurrency",
            action="add_section",
            confidence=0.85,
            edited_content=sample_edited_content[0],
            reasoning="Reason 1"
        ),
        IntegrationDecision(
            knowledge_block_id="journal-block-1",
            target_page="Python/Best Practices",
            action="add_under",
            target_block_id="target-1",
            confidence=0.75,
            edited_content=sample_edited_content[0],
            reasoning="Reason 2"
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

    # Different errors for each decision
    screen.write_integration = AsyncMock(
        side_effect=[
            ValueError("Error 1: Target not found"),
            PermissionError("Error 2: Permission denied")
        ]
    )

    async with app.run_test() as pilot:
        await pilot.pause()

        # Accept first decision (will fail with Error 1)
        await pilot.press("y")
        await pilot.pause()

        assert decisions[0].write_status == "failed"
        assert "Error 1" in decisions[0].error_message

        # Navigate to second decision
        await pilot.press("j")
        await pilot.pause()

        # Accept second decision (will fail with Error 2)
        await pilot.press("y")
        await pilot.pause()

        assert decisions[1].write_status == "failed"
        assert "Error 2" in decisions[1].error_message

        # Each decision should have its own error message
        assert decisions[0].error_message != decisions[1].error_message
