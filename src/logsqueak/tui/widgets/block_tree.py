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
from textual import events
from textual.widgets import Tree
from textual.widgets.tree import TreeNode

from logsqueak.logseq.parser import LogseqBlock
from logsqueak.tui.models import BlockState
from logsqueak.tui.utils import generate_content_hash
from logsqueak.tui.markdown import render_markdown


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

    Note: Space and Enter are handled by the parent screen (Phase1Screen)
    for toggle selection and continue actions. Tree expand/collapse is
    disabled since the tree starts fully expanded.
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

        Background colors:
        - LLM-suggested knowledge (pending): Dark blue background
        - User-selected knowledge: Dark green background

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

        # Determine background color based on state
        # Priority: User-selected knowledge > LLM-suggested knowledge > No background
        background_style = ""
        if block_state.classification == "knowledge":
            # User-selected knowledge blocks get green background
            if block_state.source == "user":
                background_style = "on dark_green"
            # LLM-classified knowledge blocks get lighter green background
            else:
                background_style = "on #004400"  # Very dark green
        elif block_state.classification == "pending" and block_state.llm_classification == "knowledge":
            # LLM-suggested but not yet accepted knowledge blocks get blue background
            background_style = "on dark_blue"

        # Build Rich Text object
        text = Text()

        # Knowledge count
        text.append(f"[{knowledge_count}] ", style=f"dim {background_style}".strip())

        # Render markdown content (excluding 'id' property)
        rendered = render_markdown(first_line, strip_id=True)
        # Apply background to rendered content
        if background_style:
            rendered.stylize(background_style)
        text.append(rendered)

        # Truncate if needed (after rendering)
        if len(text) > max_length:
            text = text[:max_length - 3]
            text.append("...", style=background_style if background_style else "")

        # Format confidence if present
        if block_state.confidence is not None:
            confidence_pct = int(block_state.confidence * 100)
            text.append(f" ({confidence_pct}%)", style=f"dim {background_style}".strip())

            # Add warning for low confidence
            if block_state.confidence < 0.60:
                text.append(" ⚠", style=f"bold yellow {background_style}".strip())

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
