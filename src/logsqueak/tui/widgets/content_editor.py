"""ContentEditor widget for editing knowledge block content.

This widget provides a multi-line text editor for refining knowledge blocks
in Phase 2 of the workflow.
"""

from textual.widgets import TextArea
from textual.reactive import reactive


class ContentEditor(TextArea):
    """Multi-line text editor for knowledge block content."""

    # Reactive attribute to track if editor has focus
    editor_has_focus = reactive(False)

    def __init__(
        self,
        *args,
        **kwargs
    ):
        """Initialize ContentEditor."""
        super().__init__("", *args, id="content-editor", **kwargs)
        self.can_focus = True
        self.show_line_numbers = False

    def on_focus(self) -> None:
        """Handle focus event."""
        self.editor_has_focus = True
        # Add visual indication of focus
        self.styles.border = ("heavy", "blue")

    def on_blur(self) -> None:
        """Handle blur event."""
        self.editor_has_focus = False
        # Remove visual indication
        self.styles.border = ("solid", "white")

    def load_content(self, content: str) -> None:
        """Load content into the editor.

        Args:
            content: Text content to load
        """
        self.text = content

    def get_content(self) -> str:
        """Get current content from editor.

        Returns:
            Current text content
        """
        return self.text

    def clear(self) -> None:
        """Clear editor content."""
        self.text = ""
