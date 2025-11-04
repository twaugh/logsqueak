"""BlockList widget for selection-focused knowledge classification.

This widget displays journal blocks in a flat, scrollable list optimized for selection.
Unlike the tree widget, this focuses on making selection feel natural and intuitive.
"""

from typing import Optional

from rich.text import Text
from textual.widgets import ListView, ListItem, Static
from textual.containers import Horizontal

from logseq_outline import LogseqBlock
from logsqueak.tui.models import BlockState
from logsqueak.tui.markdown_renderer import render_block_for_display
from logsqueak.tui.utils import generate_content_hash


class BlockListItem(ListItem):
    """
    A single block in the selection list.

    Shows:
    - Selection indicator (✓ for knowledge, ✗ for activity, ⊙ for pending)
    - Block content (rendered markdown, excluding 'id' property)
    - Confidence score (if available)
    - Visual highlight for knowledge blocks
    """

    def __init__(
        self,
        block: LogseqBlock,
        block_state: BlockState,
        indent_str: str = "  ",
    ):
        """
        Initialize a block list item.

        Args:
            block: The Logseq block to display
            block_state: Current classification state
            indent_str: Indentation string for rendering
        """
        super().__init__()
        self.block = block
        self.block_state = block_state
        self.indent_str = indent_str

        # Generate block ID
        self.block_id = block.block_id or generate_content_hash(block)

    def compose(self):
        """Compose the list item layout."""
        # Render block content (exclude 'id' property)
        content = render_block_for_display(
            self.block,
            indent_str=self.indent_str,
            exclude_properties={"id"},
            show_children=False,
        )

        # Format selection indicator
        icon = self._get_classification_icon()
        confidence_str = ""
        if self.block_state.confidence is not None:
            confidence_pct = int(self.block_state.confidence * 100)
            confidence_str = f" ({confidence_pct}%)"

        # Create rich text with styling
        text = Text()
        text.append(f"{icon} ", style=self._get_icon_style())
        text.append(content)
        text.append(confidence_str, style="dim")

        yield Static(text)

    def _get_classification_icon(self) -> str:
        """Get icon for current classification."""
        icons = {
            "knowledge": "✓",
            "activity": "✗",
            "pending": "⊙",
        }
        return icons.get(self.block_state.classification, "?")

    def _get_icon_style(self) -> str:
        """Get style for classification icon."""
        styles = {
            "knowledge": "bold green",
            "activity": "dim red",
            "pending": "yellow",
        }
        return styles.get(self.block_state.classification, "")

    def update_state(self, new_state: BlockState) -> None:
        """Update the block state and refresh display."""
        self.block_state = new_state
        # Trigger re-compose
        self.refresh()


class BlockList(ListView):
    """
    Flat list widget for block selection.

    Displays all journal blocks in a scrollable list with:
    - Visual indentation to show hierarchy
    - Highlight-style selection (knowledge blocks are visually distinct)
    - Smooth keyboard navigation (j/k, arrows)
    - Real-time classification updates from LLM streaming
    """

    def __init__(
        self,
        blocks: list[LogseqBlock],
        block_states: dict[str, BlockState],
        indent_str: str = "  ",
    ):
        """
        Initialize block list.

        Args:
            blocks: Root-level journal blocks
            block_states: Map of block_id → BlockState
            indent_str: Indentation string
        """
        super().__init__()
        self.block_states = block_states
        self.indent_str = indent_str

        # Build flat list of all blocks
        self.flat_blocks: list[tuple[LogseqBlock, BlockState]] = []
        for root_block in blocks:
            self._add_block_recursive(root_block)

    def _add_block_recursive(self, block: LogseqBlock) -> None:
        """Recursively add block and its children to flat list."""
        block_id = block.block_id or generate_content_hash(block)
        block_state = self.block_states.get(
            block_id,
            BlockState(
                block_id=block_id,
                classification="pending",
                confidence=None,
                source="llm",
            ),
        )

        self.flat_blocks.append((block, block_state))

        # Add children
        for child in block.children:
            self._add_block_recursive(child)

    def on_mount(self) -> None:
        """Populate list with block items."""
        for block, block_state in self.flat_blocks:
            item = BlockListItem(block, block_state, self.indent_str)
            self.append(item)

    def get_current_block_id(self) -> Optional[str]:
        """Get the block ID of the currently highlighted item."""
        if self.index is None or self.index < 0:
            return None

        if self.index >= len(self.flat_blocks):
            return None

        block, _ = self.flat_blocks[self.index]
        return block.block_id or generate_content_hash(block)

    def update_block_state(self, block_id: str, new_state: BlockState) -> None:
        """Update a block's state and refresh its display."""
        # Find the block in flat_blocks
        for i, (block, state) in enumerate(self.flat_blocks):
            current_id = block.block_id or generate_content_hash(block)
            if current_id == block_id:
                # Update state
                self.flat_blocks[i] = (block, new_state)

                # Update the list item widget
                if i < len(self.children):
                    item = self.children[i]
                    if isinstance(item, BlockListItem):
                        item.update_state(new_state)
                break
