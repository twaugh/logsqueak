"""UI tests for Phase 3 batch accept (a key writes all pending).

Tests for accepting all pending decisions for current knowledge block
using 'a' key, which triggers writes for all pending decisions and
then advances to the next block.
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
            content=["First knowledge block about Python"],
            indent_level=0,
            block_id="journal-block-1",
            children=[]
        ),
        LogseqBlock(
            content=["Second knowledge block about Textual"],
            indent_level=0,
            block_id="journal-block-2",
            children=[]
        ),
    ]


@pytest.fixture
def sample_edited_content():
    """Create sample EditedContent for testing."""
    return [
        EditedContent(
            block_id="journal-block-1",
            original_content="First knowledge block about Python",
            hierarchical_context="- Journal 2025-01-15\n  - Learning notes\n    - First knowledge block about Python",
            current_content="Python async programming patterns"
        ),
        EditedContent(
            block_id="journal-block-2",
            original_content="Second knowledge block about Textual",
            hierarchical_context="- Journal 2025-01-15\n  - TUI development\n    - Second knowledge block about Textual",
            current_content="Textual TUI framework architecture"
        ),
    ]


@pytest.fixture
def sample_decisions():
    """Create sample integration decisions for testing."""
    # Block 1 has 3 decisions
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
            target_page="Python/Concurrency",
            action="add_under",
            target_block_id="target-1",
            target_block_title="Async Patterns",
            confidence=0.90,
            refined_text="Python async programming patterns",
            reasoning="Fits under async patterns section"
        ),
        IntegrationDecision(
            knowledge_block_id="journal-block-1",
            target_page="Python/Best Practices",
            action="add_under",
            target_block_id="target-2",
            target_block_title="Code Patterns",
            confidence=0.75,
            refined_text="Python async programming patterns",
            reasoning="Relevant to coding best practices"
        ),
        # Block 2 has 2 decisions
        IntegrationDecision(
            knowledge_block_id="journal-block-2",
            target_page="Textual/Architecture",
            action="add_section",
            confidence=0.88,
            refined_text="Textual TUI framework architecture",
            reasoning="Core architectural concepts"
        ),
        IntegrationDecision(
            knowledge_block_id="journal-block-2",
            target_page="Textual/Widgets",
            action="add_under",
            target_block_id="target-3",
            target_block_title="Custom Widgets",
            confidence=0.82,
            refined_text="Textual TUI framework architecture",
            reasoning="Relevant to widget design"
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
        ),
        "Textual/Architecture": LogseqOutline(
            blocks=[
                LogseqBlock(
                    content=["# Textual Architecture"],
                    indent_level=0,
                    block_id="textual-arch-page-block-1",
                    children=[]
                )
            ],
            source_text="",
            frontmatter=[]
        ),
        "Textual/Widgets": LogseqOutline(
            blocks=[
                LogseqBlock(
                    content=["# Textual Widgets"],
                    indent_level=0,
                    block_id="textual-widgets-page-block-1",
                    children=[]
                )
            ],
            source_text="",
            frontmatter=[]
        )
    }


@pytest.mark.asyncio
async def test_a_key_accepts_all_pending_decisions_for_block(
    sample_journal_blocks, sample_edited_content, sample_page_contents, sample_decisions, sample_journal_content, sample_journals
):
    """Test a key accepts all pending decisions for current block."""
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

        # All decisions start as pending
        assert sample_decisions[0].write_status == "pending"
        assert sample_decisions[1].write_status == "pending"
        assert sample_decisions[2].write_status == "pending"

        # Press a to accept all decisions for block 1
        await pilot.press("a")
        await pilot.pause()

        # All decisions for block 1 should be completed
        assert sample_decisions[0].write_status == "completed"
        assert sample_decisions[1].write_status == "completed"
        assert sample_decisions[2].write_status == "completed"

        # Write should be called 3 times (once per decision)
        assert screen.write_integration.call_count == 3


@pytest.mark.asyncio
async def test_a_key_advances_to_next_block_after_accepting(
    sample_journal_blocks, sample_edited_content, sample_page_contents, sample_decisions, sample_journal_content, sample_journals
):
    """Test a key advances to next block after accepting all decisions."""
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

        # Should start at block 1
        assert screen.current_block_index == 0

        # Press a to accept all and advance
        await pilot.press("a")
        await pilot.pause()

        # Should advance to block 2
        assert screen.current_block_index == 1


@pytest.mark.asyncio
async def test_a_key_skips_already_completed_decisions(
    sample_journal_blocks, sample_edited_content, sample_page_contents, sample_decisions, sample_journal_content, sample_journals
):
    """Test a key only writes pending decisions, skips completed ones."""
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

        # Accept first decision manually
        await pilot.press("y")
        await pilot.pause()

        assert sample_decisions[0].write_status == "completed"
        initial_call_count = screen.write_integration.call_count

        # Press a to accept remaining decisions
        await pilot.press("a")
        await pilot.pause()

        # Should only write the 2 remaining pending decisions
        assert screen.write_integration.call_count == initial_call_count + 2

        # All decisions should be completed
        assert sample_decisions[0].write_status == "completed"
        assert sample_decisions[1].write_status == "completed"
        assert sample_decisions[2].write_status == "completed"


@pytest.mark.asyncio
async def test_a_key_continues_if_some_writes_fail(
    sample_journal_blocks, sample_edited_content, sample_page_contents, sample_decisions, sample_journal_content, sample_journals
):
    """Test a key continues writing remaining decisions if some fail."""
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

    # Second write fails, others succeed
    screen.write_integration = AsyncMock(
        side_effect=[True, ValueError("Write failed"), True]
    )

    async with app.run_test() as pilot:
        await pilot.pause()

        # Press a to accept all
        await pilot.press("a")
        await pilot.pause()

        # First and third should succeed, second should fail
        assert sample_decisions[0].write_status == "completed"
        assert sample_decisions[1].write_status == "failed"
        assert sample_decisions[2].write_status == "completed"

        # All three writes should have been attempted
        assert screen.write_integration.call_count == 3


@pytest.mark.asyncio
async def test_a_key_shows_progress_during_batch_write(
    sample_journal_blocks, sample_edited_content, sample_page_contents, sample_decisions, sample_journal_content, sample_journals
):
    """Test a key shows progress status during batch write operation."""
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

        # Press a to accept all
        await pilot.press("a")
        await pilot.pause()

        # Status should show progress (e.g., "Writing 1/3", "Writing 2/3", etc.)
        # (Exact implementation depends on status widget)


@pytest.mark.asyncio
async def test_a_key_at_last_block_shows_completion_summary(
    sample_journal_blocks, sample_edited_content, sample_page_contents, sample_decisions, sample_journal_content, sample_journals
):
    """Test a key at last block shows completion summary after writing."""
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

        # Navigate to last block
        await pilot.press("n")
        await pilot.pause()

        assert screen.current_block_index == 1

        # Press a to accept all at last block
        await pilot.press("a")
        await pilot.pause()

        # Should show completion summary
        # (Implementation may show summary screen with statistics)


@pytest.mark.asyncio
async def test_a_key_with_no_pending_decisions_advances_immediately(
    sample_journal_blocks, sample_edited_content, sample_page_contents, sample_decisions, sample_journal_content, sample_journals
):
    """Test a key with all decisions completed just advances to next block."""
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

        # Accept all decisions manually first
        await pilot.press("y", "j", "y", "j", "y")
        await pilot.pause()

        # All decisions completed
        assert sample_decisions[0].write_status == "completed"
        assert sample_decisions[1].write_status == "completed"
        assert sample_decisions[2].write_status == "completed"

        initial_call_count = screen.write_integration.call_count

        # Press a (no pending decisions to write)
        await pilot.press("a")
        await pilot.pause()

        # Should NOT call write again
        assert screen.write_integration.call_count == initial_call_count

        # Should advance to next block
        assert screen.current_block_index == 1
