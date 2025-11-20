"""ConfidenceSlider widget for adjusting LLM confidence threshold.

This widget provides a visual slider for filtering LLM suggestions by confidence level,
with markers showing the actual min/max confidence range from LLM results.
"""

from textual.widget import Widget
from textual.reactive import reactive
from textual.message import Message
from textual.events import Click, MouseMove, MouseDown, MouseUp, Leave
from rich.console import RenderableType


class ConfidenceSlider(Widget):
    """Interactive slider for confidence threshold with min/max markers.

    Mouse-only interaction: click anywhere on the track to set threshold.
    Reactive attributes automatically trigger re-render when changed.
    """

    DEFAULT_CSS = """
    ConfidenceSlider {
        height: 3;
        border: solid $primary;
        padding: 0 1;
    }
    """

    can_focus = False

    threshold = reactive(0.8)
    min_confidence = reactive(0.0)
    max_confidence = reactive(1.0)
    _dragging = False  # Track if user is dragging

    def render(self) -> RenderableType:
        """Render the slider bar with markers.

        Returns:
            Rendered slider with track, markers, and labels
        """
        # Total width available for content (inside border and padding)
        # We need to account for: emoji(1) + space(1) + '├'(1) + slider + '┤'(1) + space(1) + emoji(1)
        # So slider width = total_width - 6
        total_width = self.size.width - 2  # Subtract border+padding on each side

        # Reserve space for decorations: "? ├" (3) + "┤ ✓" (3) = 6 characters
        slider_width = total_width - 6

        if slider_width < 5:
            return f"Threshold: {self.threshold:.0%}"

        # Calculate positions (0 to slider_width-1)
        threshold_pos = int(self.threshold * (slider_width - 1))
        min_pos = int(self.min_confidence * (slider_width - 1)) if self.min_confidence > 0.0 else 0
        max_pos = int(self.max_confidence * (slider_width - 1)) if self.max_confidence < 1.0 else slider_width - 1

        # Build slider string
        slider = ['─'] * slider_width

        # Mark min/max confidence positions (only if not at extremes and different from each other)
        if 0 < min_pos < slider_width - 1:
            slider[min_pos] = '┬'
        if 0 < max_pos < slider_width - 1 and max_pos != min_pos:
            slider[max_pos] = '┬'

        # Mark current threshold (overwrites min/max if same position)
        slider[threshold_pos] = '●'

        # Build the track line with emoji indicators
        track = '❓ ├' + ''.join(slider) + '┤ ✅'

        # Build labels line
        label = f"Confidence Threshold: {self.threshold:.0%}  (LLM range: {self.min_confidence:.0%}–{self.max_confidence:.0%})"

        return f"{track}\n{label}"

    def on_mouse_down(self, event: MouseDown) -> None:
        """Handle mouse press to start dragging.

        Args:
            event: MouseDown event with x coordinate
        """
        self._dragging = True
        self.capture_mouse()
        self._update_threshold_from_position(event.x)
        event.prevent_default()
        event.stop()

    def on_mouse_move(self, event: MouseMove) -> None:
        """Handle mouse drag to update threshold.

        Args:
            event: MouseMove event with x coordinate
        """
        if self._dragging:
            self._update_threshold_from_position(event.x)
            event.prevent_default()
            event.stop()

    def on_mouse_up(self, event: MouseUp) -> None:
        """Handle mouse release to stop dragging.

        Args:
            event: Mouse up event
        """
        if self._dragging:
            self._dragging = False
            self.release_mouse()
            event.prevent_default()
            event.stop()

    def on_leave(self, event: Leave) -> None:
        """Handle mouse leaving widget to stop dragging.

        Args:
            event: Leave event
        """
        if self._dragging:
            self._dragging = False
            self.release_mouse()

    def _update_threshold_from_position(self, x: int) -> None:
        """Update threshold based on x position.

        Args:
            x: X coordinate of mouse (relative to widget)
        """
        # Calculate slider width (same as in render())
        total_width = self.size.width - 2
        slider_width = total_width - 6  # Reserve space for "? ├" (3) + "┤ ✓" (3)

        # Layout: border(1) + padding(1) + "? "(2) + "├"(1) + slider + "┤"(1) + " ✓"(2) + padding(1) + border(1)
        # The first slider position is at: border(1) + padding(1) + "? "(2) + "├"(1) = 5
        adjusted_x = x - 5

        if -1 <= adjusted_x <= slider_width:
            # Allow clicking on '├' (adjusted_x = -1) to mean 0.0
            # and clicking on '┤' (adjusted_x = slider_width-1) to mean 1.0
            clamped_x = max(0, min(slider_width - 1, adjusted_x))
            if slider_width > 1:
                self.threshold = clamped_x / (slider_width - 1)
            else:
                self.threshold = 0.5

    def watch_threshold(self, new_value: float) -> None:
        """Reactive callback when threshold changes.

        Args:
            new_value: New threshold value
        """
        self.post_message(self.ThresholdChanged(new_value))

    class ThresholdChanged(Message):
        """Posted when threshold changes."""

        def __init__(self, threshold: float) -> None:
            """Initialize message.

            Args:
                threshold: New threshold value
            """
            super().__init__()
            self.threshold = threshold
