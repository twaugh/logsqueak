"""UI tests for Phase 1 block tree navigation.

These tests use Textual pilot to simulate keyboard interactions
and verify navigation behavior works correctly.
"""

import pytest
from textual.app import App
from textual.pilot import Pilot
from logsqueak.tui.screens.block_selection import Phase1Screen
from logsqueak.models.block_state import BlockState
from logseq_outline.parser import LogseqBlock


class Phase1TestApp(App):
    """Test app wrapper for Phase1Screen."""

    def __init__(self, screen: Phase1Screen):
        super().__init__()
        self.test_screen = screen

    def on_mount(self) -> None:
        """Push the test screen on mount."""
        self.push_screen(self.test_screen)


@pytest.fixture
def sample_blocks():
    """Create sample blocks for testing."""
    blocks = [
        LogseqBlock(
            content=["Block 1: First root block"],
            indent_level=0,
            block_id="block-1",
            children=[]
        ),
        LogseqBlock(
            content=["Block 2: Second root block"],
            indent_level=0,
            block_id="block-2",
            children=[
                LogseqBlock(
                    content=["Block 2.1: Child of block 2"],
                    indent_level=1,
                    block_id="block-2-1",
                    children=[]
                )
            ]
        ),
        LogseqBlock(
            content=["Block 3: Third root block"],
            indent_level=0,
            block_id="block-3",
            children=[]
        ),
    ]
    return blocks


@pytest.mark.asyncio
async def test_navigate_with_j_k_keys(sample_blocks):
    """Test j/k keys navigate through block tree."""
    screen = Phase1Screen(blocks=sample_blocks, journal_date="2025-01-15")
    app = Phase1TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        tree = screen.query_one("#block-tree")

        # Initially at line 1 (first actual block)
        assert tree.cursor_line == 1

        # Press j twice
        await pilot.press("j", "j")
        assert tree.cursor_line == 3

        # Press k once
        await pilot.press("k")
        assert tree.cursor_line == 2


@pytest.mark.asyncio
async def test_navigate_with_arrow_keys(sample_blocks):
    """Test arrow keys also work for navigation."""
    screen = Phase1Screen(blocks=sample_blocks, journal_date="2025-01-15")
    app = Phase1TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        tree = screen.query_one("#block-tree")

        # Initially at line 1 (first actual block)
        assert tree.cursor_line == 1

        # Press down arrow twice
        await pilot.press("down", "down")
        assert tree.cursor_line == 3

        # Press up arrow once
        await pilot.press("up")
        assert tree.cursor_line == 2


@pytest.mark.asyncio
async def test_jump_to_next_knowledge_block(sample_blocks):
    """Test Shift+j jumps to next LLM-suggested knowledge block."""
    # Set up block states with some marked as knowledge
    block_states = {
        "block-1": BlockState(
            block_id="block-1",
            classification="knowledge",
            confidence=0.92,
            source="llm",
            llm_classification="knowledge",
            llm_confidence=0.92,
            reason="Test reason"
        ),
        "block-2": BlockState(
            block_id="block-2",
            classification="pending",
            source="user",
        ),
        "block-3": BlockState(
            block_id="block-3",
            classification="knowledge",
            confidence=0.85,
            source="llm",
            llm_classification="knowledge",
            llm_confidence=0.85,
            reason="Another test reason"
        ),
    }

    screen = Phase1Screen(
        blocks=sample_blocks,
        journal_date="2025-01-15",
        initial_block_states=block_states
    )
    app = Phase1TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        tree = screen.query_one("#block-tree")

        # Initially at line 1 (block-1, which is knowledge)
        assert tree.cursor_line == 1

        # Press J to jump to next knowledge block
        await pilot.press("J")

        # Should jump to block-3 (need to check data, not line number)
        cursor_node = tree.get_node_at_line(tree.cursor_line)
        assert cursor_node is not None
        assert cursor_node.data == "block-3"


@pytest.mark.asyncio
async def test_jump_to_previous_knowledge_block(sample_blocks):
    """Test Shift+k jumps to previous LLM-suggested knowledge block."""
    # Set up block states with some marked as knowledge
    block_states = {
        "block-1": BlockState(
            block_id="block-1",
            classification="knowledge",
            confidence=0.92,
            source="llm",
            llm_classification="knowledge",
            llm_confidence=0.92,
            reason="Test reason"
        ),
        "block-2": BlockState(
            block_id="block-2",
            classification="pending",
            source="user",
        ),
        "block-3": BlockState(
            block_id="block-3",
            classification="knowledge",
            confidence=0.85,
            source="llm",
            llm_classification="knowledge",
            llm_confidence=0.85,
            reason="Another test reason"
        ),
    }

    screen = Phase1Screen(
        blocks=sample_blocks,
        journal_date="2025-01-15",
        initial_block_states=block_states
    )
    app = Phase1TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        tree = screen.query_one("#block-tree")

        # Start at block-3 (move down to it)
        # Tree starts at line 1 (block-1), so j,j should get to block-3
        await pilot.press("j", "j")

        # Press K to jump to previous knowledge block
        await pilot.press("K")

        # Should jump back to block-1
        cursor_node = tree.get_node_at_line(tree.cursor_line)
        assert cursor_node is not None
        assert cursor_node.data == "block-1"


@pytest.mark.asyncio
async def test_bottom_panel_updates_on_navigation(sample_blocks):
    """Test bottom panel shows selected block details."""
    screen = Phase1Screen(blocks=sample_blocks, journal_date="2025-01-15")
    app = Phase1TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        # Initially showing block 1
        block_detail_panel = screen.query_one("#block-detail-panel")
        assert screen.current_block_id == "block-1"

        # Navigate to block 2
        await pilot.press("j")
        await pilot.pause()

        # Bottom panel should update to block 2
        assert screen.current_block_id == "block-2"
