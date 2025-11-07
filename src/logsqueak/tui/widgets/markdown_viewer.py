"""MarkdownViewer widget for displaying block details with markdown rendering.

This widget displays the full content of the selected block along with
LLM reasoning, confidence scores, and hierarchical context.
"""

from typing import Optional
from textual.widgets import Static
from textual.containers import VerticalScroll
from rich.markdown import Markdown
from rich.text import Text
from logseq_outline.parser import LogseqBlock
from logsqueak.models.block_state import BlockState


class MarkdownViewer(Static):
    """Widget for displaying block content with markdown rendering."""

    def __init__(
        self,
        *args,
        **kwargs
    ):
        """Initialize MarkdownViewer."""
        super().__init__("", *args, id="markdown-viewer", **kwargs)
        self.current_block: Optional[LogseqBlock] = None
        self.current_state: Optional[BlockState] = None
        self._markdown_content: str = ""

    def show_block(
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
        self.current_block = block
        self.current_state = state

        # Build markdown content
        content_parts = []

        # Show hierarchical context if parents exist
        if parent_blocks:
            content_parts.append("**Context:**\n")
            for i, parent in enumerate(parent_blocks):
                indent = "  " * i
                parent_content = parent.get_full_content(normalize_whitespace=True)
                content_parts.append(f"{indent}- {parent_content}\n")

        # Show block content
        content_parts.append("\n**Content:**\n")
        block_content = block.get_full_content(normalize_whitespace=True)
        content_parts.append(f"{block_content}\n")

        # Show LLM information if available
        if state.llm_classification == "knowledge":
            content_parts.append("\n---\n")
            content_parts.append("**LLM Analysis:**\n")

            if state.llm_confidence is not None:
                confidence_pct = int(state.llm_confidence * 100)
                content_parts.append(f"- Confidence: {confidence_pct}%\n")

            if state.reason:
                content_parts.append(f"- Reasoning: {state.reason}\n")

        # Show selection status
        content_parts.append("\n---\n")
        if state.classification == "knowledge":
            # Block is selected (will be processed in Phase 2)
            content_parts.append("✓ **Selected by you**\n")
        elif state.llm_classification == "knowledge":
            # Block is suggested but not selected
            content_parts.append("🤖 **Suggested by LLM** (not yet selected)\n")

        # Combine and render
        markdown_text = "".join(content_parts)
        self._markdown_content = markdown_text
        self.update(markdown_text)

    def show_empty(self) -> None:
        """Display empty state message."""
        self.update("*Select a block to view details*")

    def show_no_selection(self) -> None:
        """Display no selection message."""
        self.update("*No block selected*")

    @property
    def markdown(self) -> str:
        """Get current markdown content as string.

        Returns:
            Current markdown content
        """
        return self._markdown_content
