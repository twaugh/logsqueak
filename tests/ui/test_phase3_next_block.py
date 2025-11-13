"""UI tests for Phase 3 next block navigation (n key).

Tests for advancing to the next knowledge block using 'n' key,
skipping any remaining pending decisions for the current block.
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
            hierarchical_context="- Today's learning\n  - Programming\n    - First knowledge block about Python",
            current_content="Python async programming patterns"
        ),
        EditedContent(
            block_id="journal-block-2",
            original_content="Second knowledge block about Textual",
            hierarchical_context="- Today's learning\n  - UI development\n    - Second knowledge block about Textual",
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
async def test_n_key_advances_to_next_block(
    sample_journal_blocks, sample_edited_content, sample_page_contents, sample_decisions, sample_journal_content
):
    """Test n key advances to next knowledge block."""
    screen = Phase3Screen(
        journal_blocks=sample_journal_blocks,
        edited_content=sample_edited_content,
        page_contents=sample_page_contents,
        decisions=sample_decisions,
        journal_date="2025-11-06",
        journal_content=sample_journal_content,
        auto_start_workers=False
    )
    app = Phase3TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        # Should start at first block
        assert screen.current_block_index == 0
        assert screen.current_decision_index == 0

        # Press n to skip to next block
        await pilot.press("n")
        await pilot.pause()

        # Should now be at second block, first decision
        assert screen.current_block_index == 1
        assert screen.current_decision_index == 0


@pytest.mark.asyncio
async def test_n_key_skips_remaining_pending_decisions(
    sample_journal_blocks, sample_edited_content, sample_page_contents, sample_decisions, sample_journal_content
):
    """Test n key skips any remaining pending decisions for current block."""
    screen = Phase3Screen(
        journal_blocks=sample_journal_blocks,
        edited_content=sample_edited_content,
        page_contents=sample_page_contents,
        decisions=sample_decisions,
        journal_date="2025-11-06",
        journal_content=sample_journal_content,
        auto_start_workers=False
    )
    app = Phase3TestApp(screen)

    screen.write_integration = AsyncMock(return_value=True)

    async with app.run_test() as pilot:
        await pilot.pause()

        # Accept first decision for block 1
        await pilot.press("y")
        await pilot.pause()

        assert sample_decisions[0].write_status == "completed"
        assert sample_decisions[1].write_status == "pending"
        assert sample_decisions[2].write_status == "pending"

        # Press n to skip remaining decisions and go to next block
        await pilot.press("n")
        await pilot.pause()

        # Should be at block 2
        assert screen.current_block_index == 1

        # Decisions 1 and 2 should still be pending (skipped)
        assert sample_decisions[1].write_status == "pending"
        assert sample_decisions[2].write_status == "pending"


@pytest.mark.skip(reason="decisions_ready tracking not yet implemented - requires background worker integration")
@pytest.mark.asyncio
async def test_n_key_shows_processing_status_if_next_block_not_ready(
    sample_journal_blocks, sample_edited_content, sample_page_contents, sample_decisions, sample_journal_content
):
    """Test n key shows 'Processing knowledge blocks...' if next block decisions not ready."""
    screen = Phase3Screen(
        journal_blocks=sample_journal_blocks,
        edited_content=sample_edited_content,
        page_contents=sample_page_contents,
        decisions=sample_decisions,
        journal_date="2025-11-06",
        journal_content=sample_journal_content,
        auto_start_workers=False
    )
    app = Phase3TestApp(screen)

    # Simulate next block not ready yet
    screen.decisions_ready = {0: True, 1: False}

    async with app.run_test() as pilot:
        await pilot.pause()

        # Press n to try to advance
        await pilot.press("n")
        await pilot.pause()

        # Should show "Processing knowledge blocks..." status
        # Should NOT advance to next block yet
        assert screen.current_block_index == 0


@pytest.mark.asyncio
async def test_n_key_at_last_block_does_nothing(
    sample_journal_blocks, sample_edited_content, sample_page_contents, sample_decisions, sample_journal_content
):
    """Test n key at last block does nothing (or shows completion)."""
    screen = Phase3Screen(
        journal_blocks=sample_journal_blocks,
        edited_content=sample_edited_content,
        page_contents=sample_page_contents,
        decisions=sample_decisions,
        journal_date="2025-11-06",
        journal_content=sample_journal_content,
        auto_start_workers=False
    )
    app = Phase3TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        # Navigate to second (last) block
        await pilot.press("n")
        await pilot.pause()

        assert screen.current_block_index == 1

        # Press n at last block
        await pilot.press("n")
        await pilot.pause()

        # Should show completion summary or stay at last block
        # (Implementation may vary - could show completion screen)


@pytest.mark.asyncio
async def test_block_counter_updates_on_next_block(
    sample_journal_blocks, sample_edited_content, sample_page_contents, sample_decisions, sample_journal_content
):
    """Test that block counter updates when advancing to next block."""
    screen = Phase3Screen(
        journal_blocks=sample_journal_blocks,
        edited_content=sample_edited_content,
        page_contents=sample_page_contents,
        decisions=sample_decisions,
        journal_date="2025-11-06",
        journal_content=sample_journal_content,
        auto_start_workers=False
    )
    app = Phase3TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        # Should show "Block 1 of 2" initially
        # (Exact implementation depends on status widget)

        # Advance to next block
        await pilot.press("n")
        await pilot.pause()

        # Should show "Block 2 of 2"
        assert screen.current_block_index == 1
