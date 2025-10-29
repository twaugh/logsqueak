"""Block locator for reliably finding journal blocks across re-parses."""

from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from logsqueak.logseq.parser import LogseqBlock, LogseqOutline


@dataclass
class BlockLocator:
    """Stores the logical position of a block for reliable re-finding.

    Instead of storing a direct object reference to a LogseqBlock (which
    becomes invalid when the AST is re-parsed), this stores enough information
    to locate the same block after a re-parse.

    Attributes:
        parent_path: List of parent block contents (from root to immediate parent)
                     Example: ["Project X", "Milestone 1", "Task A"]
        position_in_parent: Index among siblings at this level (0-based)
        content_preview: First 100 chars of block content (for verification)
        block_id: The id:: property if present (most reliable locator)
    """

    parent_path: List[str]
    position_in_parent: int
    content_preview: str
    block_id: Optional[str] = None

    @classmethod
    def from_block(
        cls, block: "LogseqBlock", parents: List["LogseqBlock"], root_blocks: Optional[List["LogseqBlock"]] = None
    ) -> "BlockLocator":
        """Create a BlockLocator from a block and its parent chain.

        Args:
            block: The block to create a locator for
            parents: List of parent blocks from root to immediate parent
            root_blocks: List of root-level blocks (required if parents is empty)

        Returns:
            BlockLocator that can be used to re-find this block

        Example:
            >>> locator = BlockLocator.from_block(block, [root, parent])
            >>> # Later, after re-parsing:
            >>> found_block = locator.find_in_outline(new_outline)
        """
        # Build parent path using first content line of each parent
        parent_path = [p.content[0] if p.content else "" for p in parents]

        # Find position among siblings
        if parents:
            siblings = parents[-1].children
        else:
            # Root level - need the root_blocks list
            siblings = root_blocks or []

        position_in_parent = -1
        for i, sibling in enumerate(siblings):
            if sibling is block:
                position_in_parent = i
                break

        # Get content preview (first 100 chars of first content line)
        content_preview = (block.content[0][:100] if block.content else "")

        return cls(
            parent_path=parent_path,
            position_in_parent=position_in_parent,
            content_preview=content_preview,
            block_id=block.block_id,
        )

    def find_in_outline(self, outline: "LogseqOutline") -> Optional["LogseqBlock"]:
        """Find the block in a (possibly re-parsed) outline.

        Uses multiple strategies in order of reliability:
        1. If block_id is present, search for it (most reliable)
        2. Follow parent_path and position_in_parent
        3. Verify with content_preview as sanity check

        Args:
            outline: The outline to search in

        Returns:
            The block if found, None otherwise

        Example:
            >>> outline = LogseqOutline.parse(journal_path.read_text())
            >>> block = locator.find_in_outline(outline)
        """
        # Strategy 1: Use explicit block ID if available
        if self.block_id:
            found = outline.find_block_by_id(self.block_id)
            if found:
                return found

        # Strategy 2: Navigate via parent path and position
        current_blocks = outline.blocks

        # Navigate down the parent path
        for parent_content in self.parent_path:
            # Find the parent block at this level
            parent_block = None
            for block in current_blocks:
                if block.content and block.content[0] == parent_content:
                    parent_block = block
                    break

            if not parent_block:
                # Parent path broken - block structure changed
                return None

            # Move down to this parent's children
            current_blocks = parent_block.children

        # Now we're at the right level - find by position
        if self.position_in_parent < 0 or self.position_in_parent >= len(current_blocks):
            # Position out of bounds - structure changed
            return None

        candidate = current_blocks[self.position_in_parent]

        # Strategy 3: Verify with content preview
        candidate_preview = candidate.content[0][:100] if candidate.content else ""
        if candidate_preview != self.content_preview:
            # Content mismatch - warn but return anyway
            # (content might have been modified by processed:: marker)
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(
                f"Content mismatch at position {self.position_in_parent}: "
                f"expected '{self.content_preview[:50]}...', "
                f"found '{candidate_preview[:50]}...'"
            )

        return candidate

    def __repr__(self) -> str:
        """Human-readable representation."""
        path_str = " > ".join(self.parent_path) if self.parent_path else "(root)"
        return f"BlockLocator({path_str} [{self.position_in_parent}]: {self.content_preview[:30]}...)"
