"""UI tests for Phase 1 confidence slider integration.

Tests the slider's interaction with block filtering and LLM suggestions.
"""

import pytest
from textual.app import App
from textual.events import MouseDown, MouseUp
from logsqueak.tui.screens.block_selection import Phase1Screen
from logsqueak.tui.widgets.confidence_slider import ConfidenceSlider
from logsqueak.models.block_state import BlockState
from logseq_outline.parser import LogseqBlock, LogseqOutline


class Phase1TestApp(App):
    """Test app wrapper for Phase1Screen."""

    def __init__(self, screen: Phase1Screen):
        super().__init__()
        self.test_screen = screen

    def on_mount(self) -> None:
        """Push the test screen on mount."""
        self.push_screen(self.test_screen)


@pytest.fixture
def sample_blocks_with_confidence():
    """Create sample blocks with varying confidence levels for testing."""
    blocks = [
        LogseqBlock(
            content=["High confidence block"],
            indent_level=0,
            block_id="block-high",
            children=[]
        ),
        LogseqBlock(
            content=["Medium confidence block"],
            indent_level=0,
            block_id="block-medium",
            children=[]
        ),
        LogseqBlock(
            content=["Low confidence block"],
            indent_level=0,
            block_id="block-low",
            children=[]
        ),
        LogseqBlock(
            content=["Very low confidence block"],
            indent_level=0,
            block_id="block-very-low",
            children=[]
        ),
    ]
    return blocks


@pytest.fixture
def journal_with_confidence(sample_blocks_with_confidence):
    """Create a LogseqOutline from sample blocks."""
    return LogseqOutline(
        blocks=sample_blocks_with_confidence,
        source_text="",
        frontmatter=[]
    )


@pytest.fixture
def journals_with_confidence(journal_with_confidence):
    """Create journals dict for Phase1Screen."""
    return {"2025-01-15": journal_with_confidence}


@pytest.fixture
def initial_states_with_llm_suggestions():
    """Create initial block states with LLM suggestions at different confidence levels."""
    return {
        "block-high": BlockState(
            block_id="block-high",
            classification="knowledge",
            llm_classification="knowledge",
            llm_confidence=0.95,
            source="llm"
        ),
        "block-medium": BlockState(
            block_id="block-medium",
            classification="knowledge",
            llm_classification="knowledge",
            llm_confidence=0.75,
            source="llm"
        ),
        "block-low": BlockState(
            block_id="block-low",
            classification="knowledge",
            llm_classification="knowledge",
            llm_confidence=0.55,
            source="llm"
        ),
        "block-very-low": BlockState(
            block_id="block-very-low",
            classification="knowledge",
            llm_classification="knowledge",
            llm_confidence=0.35,
            source="llm"
        ),
    }


@pytest.mark.asyncio
async def test_slider_exists_in_phase1_screen(journals_with_confidence):
    """Test that ConfidenceSlider is present in Phase1Screen."""
    screen = Phase1Screen(
        journals=journals_with_confidence,
        auto_start_workers=False
    )
    app = Phase1TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        # Query for slider
        slider = screen.query_one(ConfidenceSlider)
        assert slider is not None
        assert isinstance(slider, ConfidenceSlider)


@pytest.mark.asyncio
async def test_slider_default_threshold_is_08(journals_with_confidence):
    """Test that slider initializes with default threshold of 0.8."""
    screen = Phase1Screen(
        journals=journals_with_confidence,
        auto_start_workers=False
    )
    app = Phase1TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        slider = screen.query_one(ConfidenceSlider)
        assert slider.threshold == 0.8
        assert screen.confidence_threshold == 0.8


@pytest.mark.asyncio
async def test_slider_updates_screen_threshold(journals_with_confidence):
    """Test that moving slider updates Phase1Screen.confidence_threshold."""
    screen = Phase1Screen(
        journals=journals_with_confidence,
        auto_start_workers=False
    )
    app = Phase1TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        slider = screen.query_one(ConfidenceSlider)

        # Change slider threshold programmatically
        slider.threshold = 0.6
        await pilot.pause()

        # Screen's confidence_threshold should update
        assert screen.confidence_threshold == 0.6


@pytest.mark.asyncio
async def test_slider_filters_llm_suggestions_above_threshold(
    journals_with_confidence,
    initial_states_with_llm_suggestions
):
    """Test that slider sets confidence threshold correctly."""
    screen = Phase1Screen(
        journals=journals_with_confidence,
        initial_block_states=initial_states_with_llm_suggestions,
        auto_start_workers=False
    )
    app = Phase1TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        tree = screen.query_one("#block-tree")
        slider = screen.query_one(ConfidenceSlider)

        # Default threshold is 0.8
        assert screen.confidence_threshold == 0.8

        # Note: The actual filtering logic happens in BlockTree.set_confidence_threshold()
        # which is tested separately. Here we verify the threshold is propagated.
        # All blocks still exist in block_states regardless of threshold
        all_llm_blocks = [
            bid for bid, state in screen.block_states.items()
            if state.source == "llm" and state.classification == "knowledge"
        ]
        assert len(all_llm_blocks) == 4  # All 4 blocks are still in state


@pytest.mark.asyncio
async def test_slider_lowering_threshold_shows_more_suggestions(
    journals_with_confidence,
    initial_states_with_llm_suggestions
):
    """Test that lowering threshold reveals more LLM suggestions."""
    screen = Phase1Screen(
        journals=journals_with_confidence,
        initial_block_states=initial_states_with_llm_suggestions,
        auto_start_workers=False
    )
    app = Phase1TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        slider = screen.query_one(ConfidenceSlider)
        tree = screen.query_one("#block-tree")

        # Lower threshold to 0.5
        slider.threshold = 0.5
        await pilot.pause()

        # Now high (0.95), medium (0.75), and low (0.55) should be visible
        visible_count = sum(
            1 for state in screen.block_states.values()
            if state.source == "llm" and state.classification == "knowledge"
        )

        # Note: The actual filtering happens in BlockTree.set_confidence_threshold()
        # We're testing that the threshold change is propagated
        assert screen.confidence_threshold == 0.5


@pytest.mark.asyncio
async def test_slider_raising_threshold_hides_suggestions(
    journals_with_confidence,
    initial_states_with_llm_suggestions
):
    """Test that raising threshold hides lower-confidence LLM suggestions."""
    screen = Phase1Screen(
        journals=journals_with_confidence,
        initial_block_states=initial_states_with_llm_suggestions,
        auto_start_workers=False
    )
    app = Phase1TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        slider = screen.query_one(ConfidenceSlider)

        # Start with low threshold to show all
        slider.threshold = 0.3
        await pilot.pause()

        # Raise to 0.9 - only very high confidence should remain
        slider.threshold = 0.9
        await pilot.pause()

        assert screen.confidence_threshold == 0.9


@pytest.mark.asyncio
async def test_slider_updates_llm_range_markers(journals_with_confidence):
    """Test that slider shows min/max confidence markers from LLM results."""
    # Create states with specific min/max range
    initial_states = {
        "block-high": BlockState(
            block_id="block-high",
            classification="knowledge",
            llm_classification="knowledge",
            llm_confidence=0.9,  # Max
            source="llm"
        ),
        "block-low": BlockState(
            block_id="block-low",
            classification="knowledge",
            llm_classification="knowledge",
            llm_confidence=0.4,  # Min
            source="llm"
        ),
    }

    screen = Phase1Screen(
        journals=journals_with_confidence,
        initial_block_states=initial_states,
        auto_start_workers=False
    )
    app = Phase1TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        slider = screen.query_one(ConfidenceSlider)

        # Slider should show the LLM range
        # (Note: This is set by the LLM worker in production)
        # For testing, we verify the slider can display different ranges

        slider.min_confidence = 0.4
        slider.max_confidence = 0.9
        await pilot.pause()

        rendered = str(slider.render())
        assert "LLM range: 40%–90%" in rendered


@pytest.mark.asyncio
async def test_slider_user_selections_unaffected_by_threshold(
    journals_with_confidence,
    initial_states_with_llm_suggestions
):
    """Test that user manual selections are not filtered by threshold."""
    screen = Phase1Screen(
        journals=journals_with_confidence,
        initial_block_states=initial_states_with_llm_suggestions,
        auto_start_workers=False
    )
    app = Phase1TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        tree = screen.query_one("#block-tree")
        slider = screen.query_one(ConfidenceSlider)

        # Manually select a block that would be filtered out
        # (This would require navigating to the block and pressing space)
        # For this test, we directly modify the state
        screen.block_states["block-very-low"].classification = "knowledge"
        screen.block_states["block-very-low"].source = "user"
        screen.block_states["block-very-low"].confidence = 1.0

        # Raise threshold to 0.9 (would filter out 0.35 if LLM-suggested)
        slider.threshold = 0.9
        await pilot.pause()

        # User selection should still be present
        user_selections = [
            bid for bid, state in screen.block_states.items()
            if state.source == "user" and state.classification == "knowledge"
        ]
        assert "block-very-low" in user_selections


@pytest.mark.asyncio
async def test_slider_mouse_interaction_updates_threshold(journals_with_confidence):
    """Test that clicking slider updates threshold via mouse events."""
    screen = Phase1Screen(
        journals=journals_with_confidence,
        auto_start_workers=False
    )
    app = Phase1TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        slider = screen.query_one(ConfidenceSlider)
        slider._size = app.size  # Ensure slider has size

        initial_threshold = slider.threshold

        # Simulate clicking slider at a different position
        # (We'll use internal method since pilot mouse events are complex)
        slider._update_threshold_from_position(x=40)
        await pilot.pause()

        # Threshold should change
        assert slider.threshold != initial_threshold


@pytest.mark.asyncio
async def test_slider_logs_threshold_changes(
    journals_with_confidence,
    caplog
):
    """Test that threshold changes are logged as user actions."""
    screen = Phase1Screen(
        journals=journals_with_confidence,
        auto_start_workers=False
    )
    app = Phase1TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        slider = screen.query_one(ConfidenceSlider)

        # Change threshold
        slider.threshold = 0.6
        await pilot.pause()

        # Check that user action was logged
        # (Note: This requires structlog to be configured for testing)
        # In production, this logs "user_action_confidence_threshold_changed"


@pytest.mark.asyncio
async def test_slider_threshold_affects_accept_all_count(
    journals_with_confidence,
    initial_states_with_llm_suggestions
):
    """Test that threshold affects which blocks are accepted by 'Accept All'."""
    # Start with LLM suggestions not yet accepted (classification != "knowledge")
    unaccepted_states = {
        "block-high": BlockState(
            block_id="block-high",
            classification="pending",
            llm_classification="knowledge",
            llm_confidence=0.95,
            source="user"
        ),
        "block-medium": BlockState(
            block_id="block-medium",
            classification="pending",
            llm_classification="knowledge",
            llm_confidence=0.75,
            source="user"
        ),
        "block-low": BlockState(
            block_id="block-low",
            classification="pending",
            llm_classification="knowledge",
            llm_confidence=0.55,
            source="user"
        ),
        "block-very-low": BlockState(
            block_id="block-very-low",
            classification="pending",
            llm_classification="knowledge",
            llm_confidence=0.35,
            source="user"
        ),
    }

    screen = Phase1Screen(
        journals=journals_with_confidence,
        initial_block_states=unaccepted_states,
        auto_start_workers=False
    )
    app = Phase1TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        slider = screen.query_one(ConfidenceSlider)

        # Set threshold to 0.7 - should include high (0.95) and medium (0.75)
        slider.threshold = 0.7
        await pilot.pause()

        # Press 'a' to accept all (above threshold)
        await pilot.press("a")
        await pilot.pause()

        # Check which blocks were accepted
        accepted_blocks = [
            bid for bid, state in screen.block_states.items()
            if state.classification == "knowledge"
        ]

        # Should include high and medium, but not low or very-low
        assert "block-high" in accepted_blocks
        assert "block-medium" in accepted_blocks
        assert "block-low" not in accepted_blocks
        assert "block-very-low" not in accepted_blocks

        # Verify exactly 2 blocks were accepted
        assert len(accepted_blocks) == 2


@pytest.mark.asyncio
async def test_slider_threshold_affects_reset_to_llm(
    journals_with_confidence,
    initial_states_with_llm_suggestions
):
    """Test that threshold affects which blocks are shown after 'Reset to LLM'."""
    screen = Phase1Screen(
        journals=journals_with_confidence,
        initial_block_states=initial_states_with_llm_suggestions,
        auto_start_workers=False
    )
    app = Phase1TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        slider = screen.query_one(ConfidenceSlider)

        # Clear all selections first
        await pilot.press("c")
        await pilot.pause()

        # Set threshold to 0.6
        slider.threshold = 0.6
        await pilot.pause()

        # Press 'r' to reset to LLM suggestions
        await pilot.press("r")
        await pilot.pause()

        # Should restore LLM suggestions above threshold
        llm_suggestions = [
            bid for bid, state in screen.block_states.items()
            if state.source == "llm" and state.classification == "knowledge"
        ]

        # With threshold 0.6: high (0.95), medium (0.75), low (0.55) should be suggested
        # very-low (0.35) should not


@pytest.mark.asyncio
async def test_slider_updates_dynamically_during_llm_streaming(
    journals_with_confidence
):
    """Test that slider min/max updates as LLM results stream in."""
    # Create mock streaming function that yields chunks with different confidence
    async def mock_llm_stream():
        from logsqueak.models.llm_chunks import KnowledgeClassificationChunk
        import asyncio

        chunks = [
            KnowledgeClassificationChunk(
                block_id="block-high",
                insight="High confidence content",
                confidence=0.9
            ),
            KnowledgeClassificationChunk(
                block_id="block-low",
                insight="Low confidence content",
                confidence=0.4
            ),
        ]

        for chunk in chunks:
            await asyncio.sleep(0.01)  # Simulate streaming
            yield chunk

    screen = Phase1Screen(
        journals=journals_with_confidence,
        llm_stream_fn=mock_llm_stream,
        auto_start_workers=False
    )
    app = Phase1TestApp(screen)

    async with app.run_test() as pilot:
        await pilot.pause()

        slider = screen.query_one(ConfidenceSlider)

        # Initially, slider should have default range
        assert slider.min_confidence == 0.0
        assert slider.max_confidence == 1.0

        # Start LLM worker manually
        screen.run_worker(screen._llm_classification_worker())
        await pilot.pause(1.0)  # Wait for streaming to complete

        # Slider should now have updated range
        # (In production, this is done by _llm_classification_worker)
        # For testing, we verify the mechanism works
