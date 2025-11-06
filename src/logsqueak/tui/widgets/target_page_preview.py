"""TargetPagePreview widget for showing integration preview with green bar indicator.

This widget displays the target page content with the new knowledge block
visually integrated at the insertion point, marked with a green bar.
"""

from textual.widgets import Static
from textual.reactive import reactive
from rich.text import Text
from rich.console import RenderableType


class TargetPagePreview(Static):
    """Scrollable preview of target page with integration point marked."""

    # Reactive attribute for preview content
    preview_content = reactive("")
    insertion_line = reactive(-1)  # Line number where insertion happens

    def __init__(self, *args, **kwargs):
        """Initialize TargetPagePreview."""
        super().__init__("", *args, id="target-page-preview", **kwargs)
        self.can_focus = True

    def render(self) -> RenderableType:
        """Render the preview with green bar indicator at insertion point.

        Returns:
            Rich Text object with formatted preview
        """
        if not self.preview_content:
            return Text("No preview available", style="dim")

        lines = self.preview_content.split("\n")
        result = Text()

        for i, line in enumerate(lines):
            # Check if this line is at the insertion point
            if i == self.insertion_line:
                # Add green bar indicator and highlight
                result.append("┃ ", style="bold green")
                result.append(line, style="green")
            else:
                result.append(line)

            # Add newline except for last line
            if i < len(lines) - 1:
                result.append("\n")

        return result

    def load_preview(self, content: str, insertion_line: int = -1) -> None:
        """Load preview content and mark insertion point.

        Args:
            content: Full page content with integrated block
            insertion_line: Line number where new content is inserted (-1 for none)
        """
        self.preview_content = content
        self.insertion_line = insertion_line

    def clear(self) -> None:
        """Clear preview content."""
        self.preview_content = ""
        self.insertion_line = -1

    def on_focus(self) -> None:
        """Handle focus event."""
        self.styles.border = ("heavy", "blue")

    def on_blur(self) -> None:
        """Handle blur event."""
        self.styles.border = ("solid", "white")
