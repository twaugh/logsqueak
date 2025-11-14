"""Snapshot tests for Phase 3 Integration Review screen.

Tests for visual rendering of Phase 3 screen layout, including
journal context, refined content, decision list, and target page preview.
"""

import pytest
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
    """Create sample journal blocks with hierarchical context."""
    parent = LogseqBlock(
        content=["Parent context: Daily standup"],
        indent_level=0,
        block_id="parent-1",
        children=[]
    )

    knowledge_block = LogseqBlock(
        content=["Discovered async/await pattern for handling concurrent file I/O"],
        indent_level=1,
        block_id="journal-block-1",
        children=[]
    )

    parent.children.append(knowledge_block)

    return [parent]


@pytest.fixture
def sample_edited_content():
    """Create sample EditedContent for testing."""
    return [
        EditedContent(
            block_id="journal-block-1",
            original_content="Discovered async/await pattern for handling concurrent file I/O",
            hierarchical_context="- Work notes\n  - Python patterns\n    - Discovered async/await pattern for handling concurrent file I/O",
            current_content="Async/await pattern for concurrent file I/O operations"
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
            refined_text="Async/await pattern for concurrent file I/O operations",
            reasoning="This knowledge is directly relevant to Python concurrency patterns. "
                      "Adding as a new section allows it to stand independently."
        ),
        IntegrationDecision(
            knowledge_block_id="journal-block-1",
            target_page="Python/Concurrency",
            action="add_under",
            target_block_id="async-patterns-block",
            target_block_title="Async Patterns",
            confidence=0.90,
            refined_text="Async/await pattern for concurrent file I/O operations",
            reasoning="Best fit under existing 'Async Patterns' section. "
                      "Complements existing async documentation."
        ),
        IntegrationDecision(
            knowledge_block_id="journal-block-1",
            target_page="Python/File Operations",
            action="add_under",
            target_block_id="file-io-block",
            target_block_title="File I/O Best Practices",
            confidence=0.75,
            refined_text="Async/await pattern for concurrent file I/O operations",
            reasoning="Relevant to file operations best practices. "
                      "Lower confidence due to less direct semantic match."
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
        "Python/File Operations": LogseqOutline(
            blocks=[
                LogseqBlock(
                    content=["# Python File Operations"],
                    indent_level=0,
                    block_id="file-ops-page-block-1",
                    children=[]
                )
            ],
            source_text="",
            frontmatter=[]
        )
    }


@pytest.mark.asyncio
async def test_phase3_initial_render_snapshot(
    snapshot, sample_journal_blocks, sample_edited_content, sample_page_contents, sample_decisions, sample_journal_content, sample_journals
):
    """Snapshot test for initial Phase 3 screen render."""
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

        # Capture snapshot of initial render
        assert snapshot == pilot.app.screen


@pytest.mark.asyncio
async def test_phase3_decision_list_snapshot(
    snapshot, sample_journal_blocks, sample_edited_content, sample_page_contents, sample_decisions, sample_journal_content, sample_journals
):
    """Snapshot test for decision list display."""
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

        # Get decision list widget
        decision_list = screen.query_one("#decision-list")

        # Capture snapshot of decision list
        assert snapshot == decision_list


@pytest.mark.asyncio
async def test_phase3_with_completed_decision_snapshot(
    snapshot, sample_journal_blocks, sample_edited_content, sample_page_contents, sample_decisions, sample_journal_content, sample_journals
):
    """Snapshot test for screen with a completed decision (checkmark indicator)."""
    from unittest.mock import AsyncMock

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

        # Capture snapshot showing completed decision with ✓ indicator
        assert snapshot == pilot.app.screen


@pytest.mark.asyncio
async def test_phase3_with_failed_decision_snapshot(
    snapshot, sample_journal_blocks, sample_edited_content, sample_page_contents, sample_decisions, sample_journal_content, sample_journals
):
    """Snapshot test for screen with a failed decision (warning indicator)."""
    from unittest.mock import AsyncMock

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
        side_effect=ValueError("Target block not found: async-patterns-block")
    )

    async with app.run_test() as pilot:
        await pilot.pause()

        # Try to accept decision (will fail)
        await pilot.press("y")
        await pilot.pause()

        # Capture snapshot showing failed decision with ⚠ indicator
        assert snapshot == pilot.app.screen


@pytest.mark.asyncio
async def test_phase3_target_preview_snapshot(
    snapshot, sample_journal_blocks, sample_edited_content, sample_page_contents, sample_decisions, sample_journal_content, sample_journals
):
    """Snapshot test for target page preview display."""
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

        # Get target page preview widget
        preview = screen.query_one("#target-page-preview")

        # Capture snapshot of preview panel
        assert snapshot == preview


@pytest.mark.asyncio
async def test_phase3_footer_snapshot(
    snapshot, sample_journal_blocks, sample_edited_content, sample_page_contents, sample_decisions, sample_journal_content, sample_journals
):
    """Snapshot test for Phase 3 footer with keyboard shortcuts."""
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

        # Get footer widget
        footer = screen.query_one("Footer")

        # Capture snapshot of footer
        assert snapshot == footer
