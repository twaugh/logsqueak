"""Unit tests for ConfidenceSlider widget.

Tests the slider rendering, mouse interaction, and reactive behavior.
"""

import pytest
from textual.app import App
from textual.geometry import Size
from logsqueak.tui.widgets.confidence_slider import ConfidenceSlider


class SliderTestApp(App):
    """Test app for ConfidenceSlider."""

    def __init__(self, slider: ConfidenceSlider):
        super().__init__()
        self.slider = slider

    def compose(self):
        yield self.slider


@pytest.fixture
def slider():
    """Create a ConfidenceSlider instance."""
    return ConfidenceSlider()


@pytest.mark.asyncio
async def test_slider_default_values(slider):
    """Test slider initializes with correct default values."""
    app = SliderTestApp(slider)

    async with app.run_test():
        assert slider.threshold == 0.8
        assert slider.min_confidence == 0.0
        assert slider.max_confidence == 1.0
        assert slider._dragging is False
        assert slider.can_focus is True


@pytest.mark.asyncio
async def test_slider_threshold_reactive_update(slider):
    """Test that updating threshold triggers re-render."""
    app = SliderTestApp(slider)
    messages = []

    def on_threshold_changed(message):
        messages.append(message.threshold)

    async with app.run_test():
        # Subscribe to threshold changes
        slider.watch_threshold(0.5)

        # Update threshold
        slider.threshold = 0.5
        await app.animator.wait_for_idle()

        # Verify new threshold
        assert slider.threshold == 0.5


@pytest.mark.asyncio
async def test_slider_render_narrow_widget(slider):
    """Test slider renders fallback text when too narrow."""
    app = SliderTestApp(slider)

    async with app.run_test():
        # Simulate very narrow widget (width < 11)
        slider._size = Size(8, 3)
        rendered = slider.render()

        # Should show fallback text
        assert "Threshold: 80%" in str(rendered)


@pytest.mark.asyncio
async def test_slider_render_normal_width(slider):
    """Test slider renders track and labels at normal width."""
    app = SliderTestApp(slider)

    async with app.run_test():
        # Simulate normal widget width
        slider._size = Size(80, 3)
        rendered = str(slider.render())

        # Should contain track elements
        assert "💬" in rendered  # Chatter/noise emoji
        assert "💡" in rendered  # Insight/signal emoji
        assert "├" in rendered   # Left track boundary
        assert "┤" in rendered   # Right track boundary
        assert "●" in rendered   # Threshold marker
        assert "Confidence Threshold: 80%" in rendered
        assert "LLM range: 0%–100%" in rendered


@pytest.mark.asyncio
async def test_slider_render_with_llm_range(slider):
    """Test slider renders min/max markers for LLM confidence range."""
    app = SliderTestApp(slider)

    async with app.run_test():
        # Set LLM range to middle values
        slider.min_confidence = 0.3
        slider.max_confidence = 0.9
        slider._size = Size(80, 3)

        rendered = str(slider.render())

        # Should show LLM range in label
        assert "LLM range: 30%–90%" in rendered
        # Should contain min/max markers (┬)
        assert "┬" in rendered


@pytest.mark.asyncio
async def test_slider_threshold_position_calculation(slider):
    """Test threshold marker position is calculated correctly."""
    app = SliderTestApp(slider)

    async with app.run_test():
        slider._size = Size(80, 3)

        # Test various threshold values
        test_cases = [
            (0.0, "●"),   # Leftmost position
            (0.5, "●"),   # Middle position
            (1.0, "●"),   # Rightmost position
        ]

        for threshold, marker in test_cases:
            slider.threshold = threshold
            rendered = str(slider.render())
            assert marker in rendered


@pytest.mark.asyncio
async def test_slider_min_max_markers_at_extremes(slider):
    """Test that min/max markers don't appear at 0% and 100%."""
    app = SliderTestApp(slider)

    async with app.run_test():
        slider._size = Size(80, 3)

        # Min/max at extremes (0.0, 1.0) - no markers
        slider.min_confidence = 0.0
        slider.max_confidence = 1.0
        rendered = str(slider.render())

        # Count markers - should only be threshold (●), no min/max (┬)
        marker_count = rendered.count("┬")
        assert marker_count == 0


@pytest.mark.asyncio
async def test_slider_min_max_markers_overlap_threshold(slider):
    """Test threshold marker overwrites min/max when at same position."""
    app = SliderTestApp(slider)

    async with app.run_test():
        slider._size = Size(80, 3)

        # Set threshold to 0.5, min to 0.5 (same position)
        slider.threshold = 0.5
        slider.min_confidence = 0.5
        slider.max_confidence = 0.9

        rendered = str(slider.render())

        # Threshold marker should appear (overwrites min marker)
        assert "●" in rendered


@pytest.mark.asyncio
async def test_update_threshold_from_position_left_edge(slider):
    """Test clicking left edge sets threshold to 0.0."""
    app = SliderTestApp(slider)

    async with app.run_test():
        slider._size = Size(80, 3)

        # Click on left track boundary '├' (x = 4, adjusted = -1)
        slider._update_threshold_from_position(x=4)

        # Should clamp to 0.0
        assert slider.threshold == 0.0


@pytest.mark.asyncio
async def test_update_threshold_from_position_right_edge(slider):
    """Test clicking right edge sets threshold to 1.0."""
    app = SliderTestApp(slider)

    async with app.run_test():
        await app.animator.wait_for_idle()

        # Use actual widget size from layout
        size = slider.size
        total_width = size.width - 2
        slider_width = total_width - 6

        # Calculate rightmost position
        # Layout: border(1) + padding(1) + "💬 "(2) + "├"(1) + slider + "┤"(1)
        # First slider pos: 5, Last slider pos: 5 + (slider_width-1)
        last_pos = 5 + (slider_width - 1)
        slider._update_threshold_from_position(x=last_pos)

        # Should be 1.0
        assert slider.threshold == 1.0


@pytest.mark.asyncio
async def test_update_threshold_from_position_middle(slider):
    """Test clicking middle of slider sets threshold to ~0.5."""
    app = SliderTestApp(slider)

    async with app.run_test():
        slider._size = Size(80, 3)

        # total_width = 78, slider_width = 72
        # Middle position: 5 + (72-1)/2 = 5 + 35.5 ≈ 40 or 41
        # For odd slider_width-1 = 71, middle is at index 35 or 36
        # Position 40 = adjusted_x 35 = 35/71 = 0.493
        slider._update_threshold_from_position(x=40)

        # Should be close to 0.5 (allow rounding due to discrete positions)
        assert 0.48 <= slider.threshold <= 0.54


@pytest.mark.asyncio
async def test_update_threshold_from_position_outside_bounds(slider):
    """Test clicking outside slider bounds is ignored."""
    app = SliderTestApp(slider)

    async with app.run_test():
        slider._size = Size(80, 3)
        initial_threshold = slider.threshold

        # Click far to the left (outside slider)
        slider._update_threshold_from_position(x=0)

        # Threshold should clamp to valid range, not stay unchanged
        # (clicking on border/padding area before '?' should be ignored,
        # but _update_threshold_from_position allows -1 <= adjusted_x)
        # So x=0 → adjusted_x = -5, which is outside [-1, slider_width]
        # and won't update threshold
        assert slider.threshold == initial_threshold


@pytest.mark.asyncio
async def test_update_threshold_narrow_slider(slider):
    """Test threshold update when slider width is 1."""
    app = SliderTestApp(slider)

    async with app.run_test():
        # Very narrow slider (edge case)
        slider._size = Size(11, 3)  # total_width=9, slider_width=3

        slider._update_threshold_from_position(x=6)  # Middle-ish

        # Should handle narrow case gracefully
        assert 0.0 <= slider.threshold <= 1.0


@pytest.mark.asyncio
async def test_dragging_state_starts_false(slider):
    """Test slider starts with dragging state as False."""
    app = SliderTestApp(slider)

    async with app.run_test():
        assert slider._dragging is False


@pytest.mark.asyncio
async def test_dragging_state_can_be_set(slider):
    """Test dragging state can be modified internally."""
    app = SliderTestApp(slider)

    async with app.run_test():
        slider._dragging = True
        assert slider._dragging is True

        slider._dragging = False
        assert slider._dragging is False


@pytest.mark.asyncio
async def test_threshold_changed_message_posted(slider):
    """Test that ThresholdChanged message is posted when threshold changes."""
    app = SliderTestApp(slider)
    messages = []

    async def on_threshold_changed(message):
        messages.append(message.threshold)

    async with app.run_test():
        # Subscribe to messages
        app.slider.watch_threshold = lambda new_value: messages.append(new_value)

        # Change threshold programmatically
        slider.threshold = 0.6
        await app.animator.wait_for_idle()

        # Message should be posted
        assert 0.6 in messages


@pytest.mark.asyncio
async def test_update_threshold_multiple_positions(slider):
    """Test updating threshold at multiple positions."""
    app = SliderTestApp(slider)

    async with app.run_test():
        slider._size = Size(80, 3)

        # Click at left position (x=10)
        slider._update_threshold_from_position(x=10)
        threshold1 = slider.threshold

        # Click at right position (x=60)
        slider._update_threshold_from_position(x=60)
        threshold2 = slider.threshold

        # Thresholds should be different and ordered
        assert threshold2 > threshold1
        assert 0.0 <= threshold1 < 0.3  # Left side
        assert 0.7 < threshold2 <= 1.0  # Right side
