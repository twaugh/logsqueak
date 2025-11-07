"""BlockDetailPanel widget for displaying block details in Phase 1.

This widget combines:
1. TargetPagePreview for beautifully rendered Logseq markdown
2. StatusInfoPanel for LLM analysis and selection status

The layout is horizontal: block content on the left, metadata on the right.
"""

from typing import Optional
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widget import Widget
from textual.widgets import Static
from rich.text import Text

from logseq_outline.parser import LogseqBlock
from logsqueak.models.block_state import BlockState
from logsqueak.tui.widgets.target_page_preview import TargetPagePreview


class StatusInfoPanel(Static):
    """Widget for displaying LLM analysis and selection status.

    Shows:
    - Selection status (selected by user, suggested by LLM, or pending)
    - LLM confidence score
    - LLM reasoning
    """

    DEFAULT_CSS = """
    StatusInfoPanel {
        width: 35;
        border: solid $accent;
        background: $surface;
        padding: 1;
        height: 100%;
    }
    """

    def __init__(self, *args, **kwargs):
        """Initialize StatusInfoPanel."""
        super().__init__("", *args, id="status-info-panel", **kwargs)

    def show_status(self, state: BlockState) -> None:
        """Display status information for a block.

        Args:
            state: BlockState for the current block
        """
        lines = []

        # Selection status header
        lines.append(Text("Status", style="bold underline"))
        lines.append(Text(""))

        # Show selection status
        if state.classification == "knowledge":
            # Block is selected
            if state.source == "user":
                lines.append(Text("✓ Selected by you", style="bold green"))
            else:
                lines.append(Text("✓ Selected (LLM)", style="bold green"))
        elif state.llm_classification == "knowledge":
            # LLM suggested but not selected
            lines.append(Text("🤖 Suggested by LLM", style="bold yellow"))
            lines.append(Text("(not yet selected)", style="dim"))
        else:
            # Not selected, no LLM suggestion
            lines.append(Text("Not selected", style="dim"))

        # LLM analysis section (if available)
        if state.llm_classification == "knowledge":
            lines.append(Text(""))
            lines.append(Text("LLM Analysis", style="bold underline"))
            lines.append(Text(""))

            # Confidence score
            if state.llm_confidence is not None:
                confidence_pct = int(state.llm_confidence * 100)
                confidence_text = Text("Confidence: ", style="bold")
                confidence_text.append(f"{confidence_pct}%", style="cyan")
                lines.append(confidence_text)
                lines.append(Text(""))

            # Reasoning
            if state.reason:
                lines.append(Text("Reasoning:", style="bold"))
                # Wrap reasoning text to fit panel width
                reason_words = state.reason.split()
                current_line = []
                current_length = 0
                max_width = 30  # Leave some margin

                for word in reason_words:
                    word_len = len(word) + 1  # +1 for space
                    if current_length + word_len > max_width and current_line:
                        lines.append(Text(" ".join(current_line), style="italic"))
                        current_line = [word]
                        current_length = word_len
                    else:
                        current_line.append(word)
                        current_length += word_len

                if current_line:
                    lines.append(Text(" ".join(current_line), style="italic"))

        # Combine all lines
        combined = Text()
        for i, line in enumerate(lines):
            if i > 0:
                combined.append("\n")
            combined.append_text(line)

        self.update(combined)

    def show_empty(self) -> None:
        """Display empty state message."""
        self.update(Text("Select a block to view details", style="dim italic"))


class BlockDetailPanel(Widget):
    """Combined panel showing block content and status information.

    Layout:
    - Left: TargetPagePreview (renders Logseq markdown beautifully)
    - Right: StatusInfoPanel (LLM analysis and selection status)
    """

    DEFAULT_CSS = """
    BlockDetailPanel {
        height: 16;
        background: $surface;
    }

    BlockDetailPanel Horizontal {
        height: 100%;
        width: 100%;
    }

    BlockDetailPanel TargetPagePreview {
        width: 1fr;
        height: 100%;
    }
    """

    def __init__(self, *args, **kwargs):
        """Initialize BlockDetailPanel."""
        super().__init__(*args, id="block-detail-panel", **kwargs)
        self._preview: Optional[TargetPagePreview] = None
        self._status: Optional[StatusInfoPanel] = None

    def compose(self) -> ComposeResult:
        """Compose the horizontal layout."""
        with Horizontal():
            self._preview = TargetPagePreview()
            yield self._preview
            self._status = StatusInfoPanel()
            yield self._status

    async def show_block(
        self,
        block: LogseqBlock,
        state: BlockState,
        parent_blocks: Optional[list[LogseqBlock]] = None
    ) -> None:
        """Display a block with its details.

        Args:
            block: LogseqBlock to display
            state: BlockState for this block
            parent_blocks: Optional list of parent blocks for context
        """
        if not self._preview or not self._status:
            return

        # Build full context markdown for the preview
        lines = []

        # Include parent blocks for context (if provided)
        if parent_blocks:
            for parent in parent_blocks:
                # Render parent as Logseq markdown
                parent_content = parent.get_full_content(normalize_whitespace=True)
                lines.append(f"{'  ' * parent.indent_level}- {parent_content}")

        # Add the current block (only the block itself, not its children)
        block_content = block.get_full_content(normalize_whitespace=True)
        lines.append(f"{'  ' * block.indent_level}- {block_content}")

        # Combine into markdown text
        markdown_text = "\n".join(lines)

        # Load preview (with block highlighted)
        await self._preview.load_preview(
            content=markdown_text,
            highlight_block_id=block.block_id
        )

        # Update status panel
        self._status.show_status(state)

    def show_empty(self) -> None:
        """Display empty state message."""
        if self._preview:
            self._preview.clear()
        if self._status:
            self._status.show_empty()
