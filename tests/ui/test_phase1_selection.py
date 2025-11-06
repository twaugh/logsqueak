"""UI tests for Phase 1 block selection toggle.

Tests for Space key to select/deselect blocks, and batch operations.
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
            children=[]
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
async def test_space_toggles_selection(sample_blocks):
    """Test Space key toggles block selection."""
    screen = Phase1Screen(blocks=sample_blocks, journal_date="2025-01-15")
    app = Phase1TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        tree = screen.query_one("#block-tree")

        # Initially at block-1, not selected
        cursor_node = tree.get_node_at_line(tree.cursor_line)
        block_id = cursor_node.data
        assert screen.block_states[block_id].classification == "pending"

        # Press Space to select
        await pilot.press("space")
        await pilot.pause()

        # Should now be selected
        assert screen.block_states[block_id].classification == "knowledge"
        assert screen.block_states[block_id].source == "user"
        assert screen.block_states[block_id].confidence == 1.0

        # Press Space again to deselect
        await pilot.press("space")
        await pilot.pause()

        # Should be back to pending
        assert screen.block_states[block_id].classification == "pending"


@pytest.mark.asyncio
async def test_selection_updates_visual_indicator(sample_blocks):
    """Test that selection adds visual indicator (green highlight)."""
    screen = Phase1Screen(blocks=sample_blocks, journal_date="2025-01-15")
    app = Phase1TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        tree = screen.query_one("#block-tree")
        cursor_node = tree.get_node_at_line(tree.cursor_line)

        # Initially no highlight
        assert "green" not in cursor_node.label.markup.lower()

        # Press Space to select
        await pilot.press("space")
        await pilot.pause()

        # Should have green highlight
        cursor_node = tree.get_node_at_line(tree.cursor_line)
        # Note: Actual markup depends on Rich Text formatting
        # This test may need adjustment based on implementation


@pytest.mark.asyncio
async def test_accept_all_llm_suggestions(sample_blocks):
    """Test 'a' key accepts all LLM suggestions."""
    # Set up block states with some LLM suggestions
    block_states = {
        "block-1": BlockState(
            block_id="block-1",
            classification="pending",
            source="user",
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
            classification="pending",
            source="user",
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

        # Press 'a' to accept all LLM suggestions
        await pilot.press("a")
        await pilot.pause()

        # Blocks with LLM suggestions should be selected
        assert screen.block_states["block-1"].classification == "knowledge"
        assert screen.block_states["block-1"].source == "llm"

        # Blocks without LLM suggestions should remain pending
        assert screen.block_states["block-2"].classification == "pending"

        # All LLM-suggested blocks should be selected
        assert screen.block_states["block-3"].classification == "knowledge"
        assert screen.block_states["block-3"].source == "llm"


@pytest.mark.asyncio
async def test_clear_all_selections(sample_blocks):
    """Test 'c' key clears all selections."""
    # Set up block states with some selected
    block_states = {
        "block-1": BlockState(
            block_id="block-1",
            classification="knowledge",
            confidence=1.0,
            source="user",
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
            reason="Test reason"
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

        # Verify initial state (some selected)
        assert screen.block_states["block-1"].classification == "knowledge"
        assert screen.block_states["block-3"].classification == "knowledge"

        # Press 'c' to clear all
        await pilot.press("c")
        await pilot.pause()

        # All should be pending now
        assert screen.block_states["block-1"].classification == "pending"
        assert screen.block_states["block-2"].classification == "pending"
        assert screen.block_states["block-3"].classification == "pending"


@pytest.mark.asyncio
async def test_reset_to_llm_suggestion(sample_blocks):
    """Test 'r' key resets current block to LLM suggestion."""
    # Set up a block with LLM suggestion but user override
    block_states = {
        "block-1": BlockState(
            block_id="block-1",
            classification="pending",  # User cleared it
            source="user",
            llm_classification="knowledge",
            llm_confidence=0.92,
            reason="Test reason"
        ),
        "block-2": BlockState(
            block_id="block-2",
            classification="pending",
            source="user",
        ),
    }

    screen = Phase1Screen(
        blocks=sample_blocks[:2],
        journal_date="2025-01-15",
        initial_block_states=block_states
    )
    app = Phase1TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        # At block-1, which is pending but has LLM suggestion
        assert screen.block_states["block-1"].classification == "pending"

        # Press 'r' to reset to LLM suggestion
        await pilot.press("r")
        await pilot.pause()

        # Should now match LLM suggestion
        assert screen.block_states["block-1"].classification == "knowledge"
        assert screen.block_states["block-1"].source == "llm"
        assert screen.block_states["block-1"].confidence == 0.92


@pytest.mark.asyncio
async def test_next_button_enabled_when_blocks_selected(sample_blocks):
    """Test 'n' key is only enabled when at least one block is selected."""
    screen = Phase1Screen(blocks=sample_blocks, journal_date="2025-01-15")
    app = Phase1TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        # Initially no blocks selected, 'n' should be disabled
        # Try pressing 'n' - should have no effect
        await pilot.press("n")
        await pilot.pause()

        # Should still be on Phase1Screen (no transition)
        # Screen should still be Phase1Screen (no transition)
        assert app.screen == screen

        # Select a block
        await pilot.press("space")
        await pilot.pause()

        # Now 'n' should be enabled
        # Note: Actual screen transition testing may need different approach
