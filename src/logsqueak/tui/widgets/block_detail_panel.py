"""BlockDetailPanel widget for displaying block details in Phase 1.

This widget combines:
1. TargetPagePreview for beautifully rendered Logseq markdown
2. StatusInfoPanel for LLM analysis and selection status

The layout is horizontal: block content on the left, metadata on the right.
"""

from typing import Optional
from pathlib import Path
from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widget import Widget
from textual.widgets import Static
from rich.text import Text
from rich.style import Style
from rich.color import Color

from logseq_outline.parser import LogseqBlock
from logsqueak.models.block_state import BlockState
from logsqueak.tui.widgets.target_page_preview import TargetPagePreview


class StatusInfoPanel(Widget):
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
        height: 100%;
    }

    StatusInfoPanel VerticalScroll {
        width: 100%;
        height: 100%;
        padding: 0;
    }
    """

    def __init__(self, graph_path: Optional[Path] = None, *args, **kwargs):
        """Initialize StatusInfoPanel.

        Args:
            graph_path: Path to Logseq graph (for creating clickable links)
        """
        super().__init__(*args, id="status-info-panel", **kwargs)
        self.graph_path = graph_path
        self._content_widget: Optional[Static] = None
        self._scroll_container: Optional[VerticalScroll] = None
        self.can_focus = True

    def compose(self) -> ComposeResult:
        """Compose the scrollable content area."""
        self._scroll_container = VerticalScroll()
        # Prevent VerticalScroll from being in tab order
        self._scroll_container.can_focus = False
        with self._scroll_container:
            self._content_widget = Static("")
            yield self._content_widget

    def on_focus(self) -> None:
        """Handle focus event - change border to indicate focus using theme accent color."""
        accent_color = self.app.get_css_variables().get("accent", "#FEA62B")
        self.styles.border = ("heavy", accent_color)

    def on_blur(self) -> None:
        """Handle blur event - restore normal border."""
        self.styles.border = ("solid", "white")

    async def _on_key(self, event) -> None:
        """Handle keyboard events for scrolling.

        Only handle keys when scrolling is actually possible, otherwise let them
        bubble up for screen-level navigation.
        """
        if not self._scroll_container:
            return

        # Check if scrolling is possible before intercepting keys
        # If content fits in the viewport, let arrow keys bubble up
        can_scroll_up = self._scroll_container.scroll_offset.y > 0
        can_scroll_down = (
            self._scroll_container.scroll_offset.y <
            self._scroll_container.max_scroll_y
        )

        # Only handle arrow keys if scrolling is actually possible in that direction
        if event.key == "up":
            if can_scroll_up:
                self._scroll_container.scroll_up()
                event.stop()
        elif event.key == "down":
            if can_scroll_down:
                self._scroll_container.scroll_down()
                event.stop()
        elif event.key == "pageup":
            if can_scroll_up:
                self._scroll_container.scroll_page_up()
                event.stop()
        elif event.key == "pagedown":
            if can_scroll_down:
                self._scroll_container.scroll_page_down()
                event.stop()
        elif event.key == "home":
            if can_scroll_up:
                self._scroll_container.scroll_home()
                event.stop()
        elif event.key == "end":
            if can_scroll_down:
                self._scroll_container.scroll_end()
                event.stop()

    def show_status(self, state: BlockState, block: Optional[LogseqBlock] = None) -> None:
        """Display status information for a block.

        Args:
            state: BlockState for the current block
            block: Optional LogseqBlock (for extracting extracted-to:: property)
        """
        lines = []

        # Show selection status (no header, just the status)
        if state.classification == "knowledge":
            # Block is selected
            if state.source == "user":
                lines.append(Text("✓ Selected by you", style="bold green"))
            else:
                lines.append(Text("✓ Selected (LLM)", style="bold green"))
        elif state.llm_classification == "knowledge":
            # LLM suggested but not selected
            lines.append(Text("🤖 Suggested by LLM", style="bold yellow"))
        else:
            # Not selected, no LLM suggestion
            lines.append(Text("Not selected", style="dim"))

        # LLM analysis section (if available)
        if state.llm_classification == "knowledge":
            # Confidence score
            if state.llm_confidence is not None:
                confidence_pct = int(state.llm_confidence * 100)
                confidence_text = Text("Confidence: ", style="bold")
                confidence_text.append(f"{confidence_pct}%", style="cyan")
                lines.append(confidence_text)

            # Reasoning
            if state.reason:
                lines.append(Text("Reasoning:", style="bold"))
                # Manually wrap reasoning text to fit panel width
                # Panel width is 35, minus 2 for border, minus 1 for spacing = 32 characters
                max_width = 32
                reason_words = state.reason.split()
                current_line = []
                current_length = 0

                for word in reason_words:
                    word_len = len(word)
                    # Check if adding this word would exceed max width
                    if current_line and current_length + 1 + word_len > max_width:
                        # Flush current line and start new one
                        lines.append(Text(" ".join(current_line), style="italic"))
                        current_line = [word]
                        current_length = word_len
                    else:
                        # Add word to current line
                        if current_line:
                            current_length += 1  # Space before word
                        current_line.append(word)
                        current_length += word_len

                # Flush remaining words
                if current_line:
                    lines.append(Text(" ".join(current_line), style="italic"))

        # Show extracted-to:: links (if block has been previously integrated)
        if block:
            extracted_to = block.get_property("extracted-to")
            if extracted_to:
                lines.append(Text(""))
                lines.append(Text("Previously integrated:", style="bold"))

                # Parse extracted-to:: property
                # Format: [Page Name](((uuid))), [Another Page](((uuid2)))
                self._add_extracted_to_links(lines, extracted_to)

        # Combine all lines
        combined = Text()
        for i, line in enumerate(lines):
            if i > 0:
                combined.append("\n")
            combined.append_text(line)

        if self._content_widget:
            self._content_widget.update(combined)

    def _add_extracted_to_links(self, lines: list[Text], extracted_to: str) -> None:
        """Parse and add extracted-to:: links to lines list.

        Args:
            lines: List to append formatted link lines to
            extracted_to: Raw extracted-to:: property value
        """
        import re

        # Pattern to match [Page Name](((block-id)))
        pattern = r'\[([^\]]+)\]\(\(\(([^)]+)\)\)\)'
        matches = re.findall(pattern, extracted_to)

        for page_name, block_id in matches:
            link_text = Text()
            link_text.append("  • ", style="dim")

            if self.graph_path:
                # Create logseq:// URL with block reference
                from logsqueak.utils.logseq_urls import create_logseq_url
                logseq_url = create_logseq_url(page_name, self.graph_path, block_id=block_id)

                # Use Rich Style with link parameter for clickable links
                page_style = Style(
                    bold=True,
                    color=Color.from_rgb(100, 149, 237),  # Cornflower blue
                    link=logseq_url
                )
                link_text.append(page_name, style=page_style)
            else:
                link_text.append(page_name, style="bold blue")

            lines.append(link_text)

    def show_empty(self) -> None:
        """Display empty state message."""
        if self._content_widget:
            self._content_widget.update(Text("Select a block to view details", style="dim italic"))


class BlockDetailPanel(Widget):
    """Combined panel showing block content and status information.

    Layout:
    - Left: TargetPagePreview (renders Logseq markdown beautifully)
    - Right: StatusInfoPanel (LLM analysis and selection status)
    """

    DEFAULT_CSS = """
    BlockDetailPanel {
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

    def __init__(self, graph_path: Optional[Path] = None, *args, **kwargs):
        """Initialize BlockDetailPanel.

        Args:
            graph_path: Path to Logseq graph (for creating clickable links)
        """
        super().__init__(*args, id="block-detail-panel", **kwargs)
        self.graph_path = graph_path
        self._preview: Optional[TargetPagePreview] = None
        self._status: Optional[StatusInfoPanel] = None

    def compose(self) -> ComposeResult:
        """Compose the horizontal layout."""
        with Horizontal():
            self._preview = TargetPagePreview()
            yield self._preview
            self._status = StatusInfoPanel(graph_path=self.graph_path)
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
        # Note: Always render at indent level 0 in the detail panel
        lines = []

        # Include parent blocks for context (if provided)
        if parent_blocks:
            for parent in parent_blocks:
                # Render parent as Logseq markdown (no indentation)
                # Use get_user_content_lines() to exclude id:: properties
                # First line gets the bullet, continuation lines get proper indent
                parent_lines = parent.get_user_content_lines()
                for i, content_line in enumerate(parent_lines):
                    if i == 0:
                        lines.append(f"- {content_line}")
                    else:
                        lines.append(f"  {content_line}")

        # Add the current block (only the block itself, not its children)
        # Always render at indent level 0
        # Use get_user_content_lines() to exclude id:: properties
        # First line gets the bullet, continuation lines get hanging indent
        block_lines = block.get_user_content_lines()
        for i, content_line in enumerate(block_lines):
            if i == 0:
                lines.append(f"- {content_line}")
            else:
                lines.append(f"  {content_line}")

        # Combine into markdown text
        markdown_text = "\n".join(lines)

        # Load preview (no highlighting - just display the content)
        await self._preview.load_preview(markdown_text)

        # Update status panel
        self._status.show_status(state, block)

    def show_empty(self) -> None:
        """Display empty state message."""
        if self._preview:
            self._preview.clear()
        if self._status:
            self._status.show_empty()

    def show_no_knowledge_blocks(self) -> None:
        """Display message when no knowledge blocks are identified by LLM."""
        if self._preview:
            self._preview.clear()
        if self._status and self._status._content_widget:
            message = Text()
            message.append("No knowledge blocks identified\n\n", style="bold yellow")
            message.append("The AI did not find any blocks containing lasting knowledge in this journal entry.\n\n", style="italic")
            message.append("You can still manually select blocks using Space if needed.", style="dim")
            self._status._content_widget.update(message)
