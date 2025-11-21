"""BlockTree widget for displaying hierarchical Logseq blocks.

This widget uses Textual's Tree component to display journal blocks
in a hierarchical structure with visual indicators for LLM suggestions
and user selections.
"""

from typing import Dict, Optional
from textual import events
from textual.widgets import Tree
from textual.widgets.tree import TreeNode
from textual.message import Message
from rich.text import Text
from logseq_outline.parser import LogseqBlock, LogseqOutline
from logsqueak.models.block_state import BlockState


class BlockTree(Tree):
    """Tree widget for displaying Logseq blocks with selection state."""

    class BlockClicked(Message):
        """Message posted when a block is clicked for selection toggle.

        Args:
            block_id: The ID of the clicked block
        """
        def __init__(self, block_id: str) -> None:
            super().__init__()
            self.block_id = block_id

    def __init__(
        self,
        label: str,
        journals: Dict[str, 'LogseqOutline'],
        block_states: Dict[str, BlockState],
        confidence_threshold: float = 0.8,
        *args,
        **kwargs
    ):
        """Initialize BlockTree.

        Args:
            label: Root label for the tree
            journals: Dictionary mapping date string (YYYY-MM-DD) to LogseqOutline
            block_states: Dictionary mapping block_id to BlockState
            confidence_threshold: Confidence threshold for highlighting LLM suggestions
        """
        super().__init__(label, *args, id="block-tree", **kwargs)
        self.journals = journals
        self.block_states = block_states
        self.confidence_threshold = confidence_threshold

    def on_mount(self) -> None:
        """Build tree structure when widget is mounted."""
        self.root.expand()

        # Sort journals by date and add date headers if multiple journals
        sorted_dates = sorted(self.journals.keys())

        if len(sorted_dates) == 1:
            # Single journal - add blocks directly without date header
            date = sorted_dates[0]
            for block in self.journals[date].blocks:
                self._add_block_to_tree(block, self.root)
        else:
            # Multiple journals - add date headers
            for date in sorted_dates:
                # Create date header node
                from rich.text import Text
                date_header = Text(f"📅 {date}", style="bold cyan")
                date_node = self.root.add(date_header, data=None, expand=True)

                # Add blocks under date header
                for block in self.journals[date].blocks:
                    self._add_block_to_tree(block, date_node)

        # Set cursor to first actual block (line 1, since line 0 is root)
        if self.root.children:
            self.cursor_line = 1

    def _add_block_to_tree(
        self,
        block: LogseqBlock,
        parent_node: TreeNode
    ) -> TreeNode:
        """Recursively add block and children to tree.

        Args:
            block: LogseqBlock to add
            parent_node: Parent tree node

        Returns:
            The created tree node
        """
        # Get block content (first line only for tree display)
        content = block.get_full_content(normalize_whitespace=True)
        first_line = content.split('\n')[0] if content else ""

        # Create label with visual indicators
        label = self._create_block_label(block.block_id, first_line)

        # Add node with block_id as data
        node = parent_node.add(label, data=block.block_id)

        # Expand node to show children by default
        node.expand()

        # Recursively add children
        for child in block.children:
            self._add_block_to_tree(child, node)

        return node

    def _create_block_label(self, block_id: str, content: str) -> Text:
        """Create Rich Text label with visual indicators.

        Uses emoji indicators to show selection and processing state:
        Priority order (first match wins):
        1. ✅ = User selected this session (green background)
        2. 📌 = Previously processed (blue if LLM suggested above threshold, default if not)
        3. 🤖 = LLM suggested above threshold (blue background)
        4. Invisible padding = Not selected (includes below-threshold LLM suggestions)

        Uses invisible Braille Pattern Blank characters (U+2800) to ensure
        consistent alignment when no emoji is shown.

        Args:
            block_id: Block identifier
            content: Block content text

        Returns:
            Rich Text object with formatting and indicators
        """
        # Get block state
        state = self.block_states.get(block_id)

        # Find the actual LogseqBlock to check for extracted-to:: property
        block = self._find_block_in_journals(block_id)
        is_processed = block and block.get_property("extracted-to") is not None

        # Check if LLM suggestion is above threshold
        is_above_threshold = (
            state and
            state.llm_classification == "knowledge" and
            state.llm_confidence is not None and
            state.llm_confidence >= self.confidence_threshold
        )

        label = Text()

        # Strategy: Emoji OR invisible characters, then content
        # Using Braille Pattern Blank (U+2800) - an invisible char that takes up space
        invisible_padding = "\u2800\u2800"  # 2 invisible characters = 2 cells

        # Determine emoji and style based on priority
        if state and state.classification == "knowledge":
            # Priority 1: User selected this session - checkmark + green
            label.append("✅", style="")
            label.append(content, style="on dark_green")
        elif is_processed:
            # Priority 2: Previously processed - pushpin
            # Blue background if LLM suggested above threshold, default if not
            label.append("📌", style="")
            if is_above_threshold:
                label.append(content, style="on dark_blue")
            else:
                label.append(content, style="")
        elif is_above_threshold:
            # Priority 3: LLM suggested above threshold (not selected, not processed) - robot + blue
            label.append("🤖", style="")
            label.append(content, style="on dark_blue")
        else:
            # Priority 4: No special state (includes below-threshold LLM suggestions)
            label.append(invisible_padding + content, style="")

        return label

    def update_block_label(self, block_id: str) -> None:
        """Update label for a specific block after state change.

        Args:
            block_id: Block identifier to update
        """
        # Find the node with this block_id
        node = self._find_node_by_block_id(self.root, block_id)
        if node:
            # Get block content from journals
            block = self._find_block_in_journals(block_id)
            if block:
                content = block.get_full_content(normalize_whitespace=True)
                first_line = content.split('\n')[0] if content else ""
                node.label = self._create_block_label(block_id, first_line)

    def set_confidence_threshold(self, threshold: float) -> None:
        """Update confidence threshold and refresh all labels.

        Args:
            threshold: New confidence threshold value
        """
        self.confidence_threshold = threshold
        self._refresh_all_labels()

    def _refresh_all_labels(self) -> None:
        """Refresh labels for all nodes to reflect new threshold."""
        self._refresh_labels_recursive(self.root)

    def _refresh_labels_recursive(self, node: TreeNode) -> None:
        """Recursively refresh labels for a node and its children.

        Args:
            node: Tree node to refresh
        """
        if node.data:  # Has block_id
            block = self._find_block_in_journals(node.data)
            if block:
                content = block.get_full_content(normalize_whitespace=True)
                first_line = content.split('\n')[0] if content else ""
                node.label = self._create_block_label(node.data, first_line)

        # Recursively refresh children
        for child in node.children:
            self._refresh_labels_recursive(child)

    def _find_node_by_block_id(
        self,
        node: TreeNode,
        block_id: str
    ) -> Optional[TreeNode]:
        """Recursively find tree node by block_id.

        Args:
            node: Starting node for search
            block_id: Block identifier to find

        Returns:
            TreeNode if found, None otherwise
        """
        if node.data == block_id:
            return node

        for child in node.children:
            result = self._find_node_by_block_id(child, block_id)
            if result:
                return result

        return None

    def _find_block_in_journals(self, block_id: str) -> Optional[LogseqBlock]:
        """Find a LogseqBlock across all journals by id.

        Args:
            block_id: Block identifier to find

        Returns:
            LogseqBlock if found, None otherwise
        """
        for outline in self.journals.values():
            result = self._find_block_by_id(outline.blocks, block_id)
            if result:
                return result
        return None

    def _find_block_by_id(
        self,
        blocks: list[LogseqBlock],
        block_id: str
    ) -> Optional[LogseqBlock]:
        """Recursively find LogseqBlock by id.

        Args:
            blocks: List of blocks to search
            block_id: Block identifier to find

        Returns:
            LogseqBlock if found, None otherwise
        """
        for block in blocks:
            if block.block_id == block_id:
                return block

            result = self._find_block_by_id(block.children, block_id)
            if result:
                return result

        return None

    def get_current_block_id(self) -> Optional[str]:
        """Get block_id of currently selected node.

        Returns:
            Block ID string or None if no selection
        """
        if self.cursor_node:
            return self.cursor_node.data
        return None

    async def _on_click(self, event: events.Click) -> None:
        """Override default click behavior to toggle selection instead of expand/collapse.

        Args:
            event: Click event
        """
        # Prevent default expand/collapse behavior
        event.prevent_default()
        event.stop()

        async with self.lock:
            meta = event.style.meta
            if "line" in meta:
                cursor_line = meta["line"]
                # Move cursor to clicked line
                self.cursor_line = cursor_line

                # Get the block at this line and trigger selection toggle
                node = self.get_node_at_line(cursor_line)
                if node and node.data:
                    # Post message to parent screen to handle selection toggle
                    self.post_message(self.BlockClicked(node.data))

    def find_next_knowledge_block(self, from_line: int) -> Optional[int]:
        """Find next LLM-suggested knowledge block above threshold (wraps around to start).

        Args:
            from_line: Start searching from this line

        Returns:
            Line number of next knowledge block above threshold, or None if none exist
        """
        # Try to search forward through reasonable number of lines
        # (tree doesn't expose total line count directly)
        current_line = from_line + 1
        max_attempts = 100  # Reasonable upper bound
        found_end = False

        for _ in range(max_attempts):
            try:
                node = self.get_node_at_line(current_line)
                if node is None:
                    found_end = True
                    break
                if node.data:
                    state = self.block_states.get(node.data)
                    if (state and
                        state.llm_classification == "knowledge" and
                        state.llm_confidence is not None and
                        state.llm_confidence >= self.confidence_threshold):
                        return current_line
                current_line += 1
            except Exception:
                found_end = True
                break

        # If we reached the end, wrap around and search from the beginning
        if found_end:
            current_line = 0
            while current_line <= from_line:
                try:
                    node = self.get_node_at_line(current_line)
                    if node and node.data:
                        state = self.block_states.get(node.data)
                        if (state and
                            state.llm_classification == "knowledge" and
                            state.llm_confidence is not None and
                            state.llm_confidence >= self.confidence_threshold):
                            return current_line
                    current_line += 1
                except Exception:
                    break

        return None

    def find_previous_knowledge_block(self, from_line: int) -> Optional[int]:
        """Find previous LLM-suggested knowledge block above threshold (wraps around to end).

        Args:
            from_line: Start searching from this line

        Returns:
            Line number of previous knowledge block above threshold, or None if none exist
        """
        # Iterate backwards through lines
        current_line = from_line - 1

        while current_line >= 0:
            try:
                node = self.get_node_at_line(current_line)
                if node and node.data:
                    state = self.block_states.get(node.data)
                    if (state and
                        state.llm_classification == "knowledge" and
                        state.llm_confidence is not None and
                        state.llm_confidence >= self.confidence_threshold):
                        return current_line
                current_line -= 1
            except Exception:
                break

        # If we reached the beginning, wrap around and search from the end
        # Find the last line
        current_line = from_line + 1
        max_attempts = 100
        last_valid_line = from_line

        for _ in range(max_attempts):
            try:
                node = self.get_node_at_line(current_line)
                if node is None:
                    break
                last_valid_line = current_line
                current_line += 1
            except Exception:
                break

        # Now search backwards from the end to from_line
        current_line = last_valid_line
        while current_line > from_line:
            try:
                node = self.get_node_at_line(current_line)
                if node and node.data:
                    state = self.block_states.get(node.data)
                    if (state and
                        state.llm_classification == "knowledge" and
                        state.llm_confidence is not None and
                        state.llm_confidence >= self.confidence_threshold):
                        return current_line
                current_line -= 1
            except Exception:
                break

        return None
