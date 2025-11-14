"""BlockTree widget for displaying hierarchical Logseq blocks.

This widget uses Textual's Tree component to display journal blocks
in a hierarchical structure with visual indicators for LLM suggestions
and user selections.
"""

from typing import Dict, Optional
from textual.widgets import Tree
from textual.widgets.tree import TreeNode
from rich.text import Text
from logseq_outline.parser import LogseqBlock
from logsqueak.models.block_state import BlockState


class BlockTree(Tree):
    """Tree widget for displaying Logseq blocks with selection state."""

    def __init__(
        self,
        label: str,
        blocks: list[LogseqBlock],
        block_states: Dict[str, BlockState],
        *args,
        **kwargs
    ):
        """Initialize BlockTree.

        Args:
            label: Root label for the tree
            blocks: List of LogseqBlock objects to display
            block_states: Dictionary mapping block_id to BlockState
        """
        super().__init__(label, *args, id="block-tree", **kwargs)
        self.blocks = blocks
        self.block_states = block_states

    def on_mount(self) -> None:
        """Build tree structure when widget is mounted."""
        self.root.expand()
        for block in self.blocks:
            self._add_block_to_tree(block, self.root)

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
        1. ✓ = User selected this session (green background)
        2. 📌 = Previously processed (blue if LLM suggested, default if not)
        3. 🤖 = LLM suggested (blue background)
        4. Invisible padding = Not selected

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

        # Find the actual LogseqBlock to check for processed:: property
        block = self._find_block_by_id(self.blocks, block_id)
        is_processed = block and block.get_property("processed") is not None

        label = Text()

        # Strategy: Emoji OR invisible characters, then content
        # Using Braille Pattern Blank (U+2800) - an invisible char that takes up space
        invisible_padding = "\u2800\u2800"  # 2 invisible characters = 2 cells

        # Determine emoji and style based on priority
        if state and state.classification == "knowledge":
            # Priority 1: User selected this session - checkmark + green
            label.append("✓", style="")
            label.append(content, style="on dark_green")
        elif is_processed:
            # Priority 2: Previously processed - pushpin
            # Blue background if LLM suggested, default if not
            label.append("📌", style="")
            if state and state.llm_classification == "knowledge":
                label.append(content, style="on dark_blue")
            else:
                label.append(content, style="")
        elif state and state.llm_classification == "knowledge":
            # Priority 3: LLM suggested (not selected, not processed) - robot + blue
            label.append("🤖", style="")
            label.append(content, style="on dark_blue")
        else:
            # Priority 4: No special state - invisible padding
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
            # Get block content from original blocks list
            block = self._find_block_by_id(self.blocks, block_id)
            if block:
                content = block.get_full_content(normalize_whitespace=True)
                first_line = content.split('\n')[0] if content else ""
                node.label = self._create_block_label(block_id, first_line)

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

    def find_next_knowledge_block(self, from_line: int) -> Optional[int]:
        """Find next LLM-suggested knowledge block (wraps around to start).

        Args:
            from_line: Start searching from this line

        Returns:
            Line number of next knowledge block, or None if no knowledge blocks exist
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
                    if state and state.llm_classification == "knowledge":
                        return current_line
                current_line += 1
            except:
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
                        if state and state.llm_classification == "knowledge":
                            return current_line
                    current_line += 1
                except:
                    break

        return None

    def find_previous_knowledge_block(self, from_line: int) -> Optional[int]:
        """Find previous LLM-suggested knowledge block (wraps around to end).

        Args:
            from_line: Start searching from this line

        Returns:
            Line number of previous knowledge block, or None if no knowledge blocks exist
        """
        # Iterate backwards through lines
        current_line = from_line - 1

        while current_line >= 0:
            try:
                node = self.get_node_at_line(current_line)
                if node and node.data:
                    state = self.block_states.get(node.data)
                    if state and state.llm_classification == "knowledge":
                        return current_line
                current_line -= 1
            except:
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
            except:
                break

        # Now search backwards from the end to from_line
        current_line = last_valid_line
        while current_line > from_line:
            try:
                node = self.get_node_at_line(current_line)
                if node and node.data:
                    state = self.block_states.get(node.data)
                    if state and state.llm_classification == "knowledge":
                        return current_line
                current_line -= 1
            except:
                break

        return None
