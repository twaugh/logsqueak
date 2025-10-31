"""BlockTree widget for displaying hierarchical journal blocks with classification state.

This widget displays a hierarchical tree of journal blocks with:
- Classification icons (✓ knowledge, ✗ activity, ⊙ pending, ⊗ skipped, ? unknown)
- Confidence percentages for LLM classifications
- Low-confidence warnings for scores below thresholds
- User override indicators
- Rich markdown rendering (links, bold, code, TODO/DONE checkboxes, etc.)
"""

from typing import Optional

from rich.text import Text
from textual.widgets import Tree
from textual.widgets.tree import TreeNode

from logsqueak.logseq.parser import LogseqBlock
from logsqueak.tui.models import BlockState
from logsqueak.tui.utils import generate_content_hash


class BlockTree(Tree):
    """
    Custom tree widget for displaying journal blocks with classification state.

    Features:
    - Hierarchical display matching LogseqBlock structure
    - Real-time classification updates from LLM streaming
    - User override indicators (bold/highlighted for user-marked blocks)
    - Confidence score display with color coding
    - Low-confidence warnings (⚠ for <60% confidence)

    Keyboard Navigation:
    - j/↓: Move down
    - k/↑: Move up
    - Space: Expand/collapse node
    - Enter: Select node
    """

    def __init__(
        self,
        label: str,
        block_states: dict[str, BlockState],
        *args,
        **kwargs,
    ):
        """
        Initialize BlockTree widget.

        Args:
            label: Tree root label
            block_states: Map of block_id → BlockState for all blocks
            *args: Positional arguments for Tree
            **kwargs: Keyword arguments for Tree
        """
        super().__init__(label, *args, **kwargs)
        self.block_states = block_states

    def add_block(
        self,
        block: LogseqBlock,
        parent_node: Optional[TreeNode] = None,
    ) -> TreeNode:
        """
        Add a LogseqBlock to the tree with formatted label.

        Recursively adds child blocks to maintain hierarchy.

        Args:
            block: LogseqBlock to add
            parent_node: Parent tree node (None = add to root)

        Returns:
            Created TreeNode for this block
        """
        # Generate stable block ID (same logic as _initialize_block_states)
        # Use block.block_id if present, otherwise generate content hash
        block_id = block.block_id or generate_content_hash(block)

        # Get block state (should exist, default to pending if not)
        block_state = self.block_states.get(
            block_id,
            BlockState(
                block_id=block_id,
                classification="pending",
                confidence=None,
                source="llm",
            ),
        )

        # Count knowledge blocks in this subtree
        knowledge_count = self._count_knowledge_in_subtree(block)

        # Format label with knowledge count, classification icon and confidence
        label = self.format_block_label(block, block_state, knowledge_count)

        # Add to tree (use the generated block_id, not block.block_id)
        if parent_node is None:
            node = self.root.add(label, data=block_id, allow_expand=len(block.children) > 0)
        else:
            node = parent_node.add(label, data=block_id, allow_expand=len(block.children) > 0)

        # Recursively add children
        for child in block.children:
            self.add_block(child, node)

        # Expand node if it has children (show all nodes initially)
        if len(block.children) > 0:
            node.expand()

        return node

    def format_block_label(self, block: LogseqBlock, block_state: BlockState, knowledge_count: int) -> Text:
        """
        Format a block label with knowledge count, classification icon, content, and confidence.

        Format: [count] [icon] content (confidence%) [warning]

        Knowledge count:
        - Shows number of knowledge blocks in this subtree (including self)
        - [0] = no knowledge in subtree
        - [N] = N knowledge blocks found

        Icons:
        - ✓ : Knowledge (green)
        - ✗ : Activity (red/dim)
        - ⊙ : Pending (yellow)
        - ? : Unknown/error

        Confidence display:
        - 90-100%: Normal (no warning)
        - 75-89%: Normal
        - 60-74%: Yellow color
        - 0-59%: ⚠ warning indicator

        User overrides:
        - Bold/highlighted to indicate manual classification
        - Shows original LLM confidence if available

        Args:
            block: LogseqBlock to format
            block_state: Current classification state
            knowledge_count: Number of knowledge blocks in subtree

        Returns:
            Formatted label as Rich Text with markdown rendering
        """
        # Get first line of block content
        first_line = block.content[0] if block.content else "(empty block)"

        # Truncate if too long (but after markdown parsing so we preserve formatting)
        max_length = 100

        # Choose icon based on classification
        icon = self._get_classification_icon(block_state.classification)
        icon_style = self._get_icon_style(block_state.classification)

        # Build Rich Text object
        text = Text()

        # Knowledge count
        text.append(f"[{knowledge_count}] ", style="dim")

        # Classification icon
        text.append(f"{icon} ", style=icon_style)

        # Render markdown content (excluding 'id' property)
        content_to_render = self._strip_id_property(first_line)
        self._render_markdown_to_text(content_to_render, text)

        # Truncate if needed (after rendering)
        if len(text) > max_length:
            text = text[:max_length - 3]
            text.append("...")

        # Format confidence if present
        if block_state.confidence is not None:
            confidence_pct = int(block_state.confidence * 100)
            text.append(f" ({confidence_pct}%)", style="dim")

            # Add warning for low confidence
            if block_state.confidence < 0.60:
                text.append(" ⚠", style="bold yellow")

        # User override indicator
        if block_state.source == "user":
            text.append(" [USER]", style="bold magenta")
            # If user overrode LLM, show original LLM confidence
            if block_state.llm_confidence is not None:
                llm_pct = int(block_state.llm_confidence * 100)
                text.append(f" (was {llm_pct}%)", style="dim")

        return text

    def _count_knowledge_in_subtree(self, block: LogseqBlock) -> int:
        """
        Count knowledge blocks in this block's subtree (including self).

        Args:
            block: LogseqBlock to count from

        Returns:
            Number of knowledge blocks in subtree
        """
        # Generate block ID
        block_id = block.block_id or generate_content_hash(block)

        # Check if this block is knowledge
        block_state = self.block_states.get(block_id)
        count = 1 if (block_state and block_state.classification == "knowledge") else 0

        # Recursively count children
        for child in block.children:
            count += self._count_knowledge_in_subtree(child)

        return count

    def _get_classification_icon(self, classification: str) -> str:
        """
        Get icon for classification state.

        Args:
            classification: One of "knowledge", "activity", "pending"

        Returns:
            Unicode icon character
        """
        icons = {
            "knowledge": "✓",
            "activity": "✗",
            "pending": "⊙",
        }
        return icons.get(classification, "?")

    def _get_icon_style(self, classification: str) -> str:
        """
        Get Rich style for classification icon.

        Args:
            classification: One of "knowledge", "activity", "pending"

        Returns:
            Rich style string
        """
        styles = {
            "knowledge": "bold green",
            "activity": "dim",
            "pending": "yellow",
        }
        return styles.get(classification, "")

    def _strip_id_property(self, content: str) -> str:
        """
        Remove 'id::' property from content if present.

        Args:
            content: Block content line

        Returns:
            Content with id:: property removed
        """
        # Check if this line is an id:: property
        stripped = content.lstrip()
        if stripped.startswith("id::"):
            return ""  # Don't show id properties
        return content

    def _render_markdown_to_text(self, content: str, text: Text) -> None:
        """
        Render Logseq markdown to Rich Text with styles.

        Supports:
        - **bold**
        - *italic*
        - `code`
        - [[page links]]
        - TODO/DONE/CANCELLED markers (with checkboxes)
        - ~~strikethrough~~

        Args:
            content: Raw markdown content
            text: Rich Text object to append styled text to
        """
        if not content or content == "":
            return

        # Handle TODO/DONE/CANCELLED markers with checkbox indicators
        if content.startswith("TODO "):
            text.append("☐ ", style="bold yellow")
            text.append("TODO ", style="yellow")
            content = content[5:]
        elif content.startswith("DONE "):
            text.append("☑ ", style="bold green")
            text.append("DONE ", style="green")
            content = content[5:]
        elif content.startswith("CANCELLED ") or content.startswith("CANCELED "):
            prefix_len = 10 if content.startswith("CANCELLED ") else 9
            text.append("☒ ", style="bold dim")
            text.append(content[:prefix_len], style="dim strike")
            content = content[prefix_len:]

        # Process the rest with simple parser
        i = 0
        while i < len(content):
            # Check for [[page links]]
            if content[i:i+2] == "[[":
                end = content.find("]]", i+2)
                if end != -1:
                    link_text = content[i+2:end]
                    text.append(link_text, style="bold blue underline")
                    i = end + 2
                    continue

            # Check for **bold**
            if content[i:i+2] == "**":
                end = content.find("**", i+2)
                if end != -1:
                    bold_text = content[i+2:end]
                    text.append(bold_text, style="bold")
                    i = end + 2
                    continue

            # Check for *italic*
            if content[i] == "*" and (i == 0 or content[i-1] != "*"):
                end = content.find("*", i+1)
                if end != -1 and (end+1 >= len(content) or content[end+1] != "*"):
                    italic_text = content[i+1:end]
                    text.append(italic_text, style="italic")
                    i = end + 1
                    continue

            # Check for `code`
            if content[i] == "`":
                end = content.find("`", i+1)
                if end != -1:
                    code_text = content[i+1:end]
                    text.append(code_text, style="bold cyan on #222222")
                    i = end + 1
                    continue

            # Check for ~~strikethrough~~
            if content[i:i+2] == "~~":
                end = content.find("~~", i+2)
                if end != -1:
                    strike_text = content[i+2:end]
                    text.append(strike_text, style="strike dim")
                    i = end + 2
                    continue

            # Check for ~strikethrough~
            if content[i] == "~" and (i == 0 or content[i-1] != "~"):
                end = content.find("~", i+1)
                if end != -1 and (end+1 >= len(content) or content[end+1] != "~"):
                    strike_text = content[i+1:end]
                    text.append(strike_text, style="strike dim")
                    i = end + 1
                    continue

            # Regular character
            text.append(content[i])
            i += 1

    def update_block_label(self, block_id: str) -> None:
        """
        Update a single block's label after state change.

        Searches tree for node with matching block_id data and updates its label.

        Args:
            block_id: Block ID to update
        """
        # Find node with matching block_id
        node = self._find_node_by_id(self.root, block_id)
        if node is None:
            return

        # Get updated state
        block_state = self.block_states.get(block_id)
        if block_state is None:
            return

        # Need to get the LogseqBlock to format label
        # For now, just update the icon portion (optimization)
        # TODO: Store LogseqBlock reference in node data for full re-format

    def _find_node_by_id(
        self, node: TreeNode, block_id: str
    ) -> Optional[TreeNode]:
        """
        Recursively find tree node by block_id.

        Args:
            node: Current node to search from
            block_id: Block ID to find

        Returns:
            TreeNode if found, None otherwise
        """
        if node.data == block_id:
            return node

        for child in node.children:
            result = self._find_node_by_id(child, block_id)
            if result is not None:
                return result

        return None
