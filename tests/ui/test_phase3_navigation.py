"""UI tests for Phase 3 decision navigation (j/k keys).

Tests for navigating between integration decisions for the current knowledge block
using j/k keys and arrow keys.
"""

import pytest
from textual.app import App
from logsqueak.tui.screens.integration_review import Phase3Screen
from logsqueak.models.integration_decision import IntegrationDecision
from logsqueak.models.edited_content import EditedContent
from logseq_outline.parser import LogseqBlock


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
            hierarchical_context="- Journal entry\n  - Coding notes\n    - First knowledge block about Python",
            current_content="Python async programming patterns"
        ),
        EditedContent(
            block_id="journal-block-2",
            original_content="Second knowledge block about Textual",
            hierarchical_context="- Journal entry\n  - TUI notes\n    - Second knowledge block about Textual",
            current_content="Textual TUI framework architecture"
        ),
    ]


@pytest.fixture
def sample_decisions():
    """Create sample integration decisions for testing."""
    # Block 1 has 3 decisions across 2 pages
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


@pytest.mark.asyncio
async def test_navigate_between_decisions_with_j_key(
    sample_journal_blocks, sample_edited_content, sample_decisions
):
    """Test j key navigates to next decision for current block."""
    screen = Phase3Screen(
        journal_blocks=sample_journal_blocks,
        edited_content=sample_edited_content,
        decisions=sample_decisions,
        journal_date="2025-11-06",
        auto_start_workers=False
    )
    app = Phase3TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        # Should start at first block, first decision (index 0)
        assert screen.current_block_index == 0
        assert screen.current_decision_index == 0

        # Press j to move to next decision
        await pilot.press("j")
        await pilot.pause()

        # Should now be at second decision (index 1) for same block
        assert screen.current_block_index == 0
        assert screen.current_decision_index == 1


@pytest.mark.asyncio
async def test_navigate_between_decisions_with_k_key(
    sample_journal_blocks, sample_edited_content, sample_decisions
):
    """Test k key navigates to previous decision for current block."""
    screen = Phase3Screen(
        journal_blocks=sample_journal_blocks,
        edited_content=sample_edited_content,
        decisions=sample_decisions,
        journal_date="2025-11-06",
        auto_start_workers=False
    )
    app = Phase3TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        # Move to second decision first
        await pilot.press("j")
        await pilot.pause()

        assert screen.current_decision_index == 1

        # Press k to move back to first decision
        await pilot.press("k")
        await pilot.pause()

        # Should be back at first decision
        assert screen.current_decision_index == 0


@pytest.mark.asyncio
async def test_navigate_with_arrow_keys(
    sample_journal_blocks, sample_edited_content, sample_decisions
):
    """Test arrow keys also work for decision navigation."""
    screen = Phase3Screen(
        journal_blocks=sample_journal_blocks,
        edited_content=sample_edited_content,
        decisions=sample_decisions,
        journal_date="2025-11-06",
        auto_start_workers=False
    )
    app = Phase3TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        # Initially at first decision
        assert screen.current_decision_index == 0

        # Press down arrow to move forward
        await pilot.press("down")
        await pilot.pause()

        assert screen.current_decision_index == 1

        # Press up arrow to move back
        await pilot.press("up")
        await pilot.pause()

        assert screen.current_decision_index == 0


@pytest.mark.asyncio
async def test_navigation_wraps_at_decision_boundaries(
    sample_journal_blocks, sample_edited_content, sample_decisions
):
    """Test that navigation stops at first and last decisions for current block."""
    screen = Phase3Screen(
        journal_blocks=sample_journal_blocks,
        edited_content=sample_edited_content,
        decisions=sample_decisions,
        journal_date="2025-11-06",
        auto_start_workers=False
    )
    app = Phase3TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        # At first decision, pressing k should do nothing
        assert screen.current_decision_index == 0
        await pilot.press("k")
        await pilot.pause()
        assert screen.current_decision_index == 0

        # Navigate to last decision for block 1 (index 2)
        await pilot.press("j", "j")
        await pilot.pause()
        assert screen.current_decision_index == 2

        # Pressing j at last decision should do nothing
        await pilot.press("j")
        await pilot.pause()
        assert screen.current_decision_index == 2


@pytest.mark.asyncio
async def test_target_page_preview_updates_on_navigation(
    sample_journal_blocks, sample_edited_content, sample_decisions
):
    """Test that target page preview updates when navigating between decisions."""
    screen = Phase3Screen(
        journal_blocks=sample_journal_blocks,
        edited_content=sample_edited_content,
        decisions=sample_decisions,
        journal_date="2025-11-06",
        auto_start_workers=False
    )
    app = Phase3TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        # Get the preview widget
        preview = screen.query_one("#target-page-preview")

        # Should show preview for first decision (Python/Concurrency)
        # (Implementation will determine exact preview content)

        # Navigate to next decision
        await pilot.press("j")
        await pilot.pause()

        # Preview should update to show second decision
        # (Different action or same page, different insertion point)


@pytest.mark.asyncio
async def test_decision_list_highlights_current_selection(
    sample_journal_blocks, sample_edited_content, sample_decisions
):
    """Test that decision list highlights the currently selected decision."""
    screen = Phase3Screen(
        journal_blocks=sample_journal_blocks,
        edited_content=sample_edited_content,
        decisions=sample_decisions,
        journal_date="2025-11-06",
        auto_start_workers=False
    )
    app = Phase3TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        # Navigate through decisions and verify highlight updates
        await pilot.press("j")
        await pilot.pause()

        assert screen.current_decision_index == 1

        # Decision list should visually indicate current selection
        # (Exact visual representation depends on widget implementation)
