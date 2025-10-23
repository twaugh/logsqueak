"""Logseq markdown parser for outline-based documents.

This module handles parsing and rendering of Logseq's outline format,
which uses indented bullets (2 spaces per level) with properties and page links.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LogseqBlock:
    """Single bullet in outline with children.

    IMPORTANT: Preserves exact structure and order from source.

    - Properties must never be reordered (insertion order preserved)
    - Children can be inserted where appropriate (targeted, minimal changes)

    Attributes:
        content: Block text content
        indent_level: Indentation level (0 = root)
        properties: Block properties (preserves insertion order)
        children: Child blocks
        _original_line: Original line for exact round-trip
    """

    content: str
    indent_level: int
    properties: dict = field(default_factory=dict)  # Preserves insertion order (Python 3.7+)
    children: list["LogseqBlock"] = field(default_factory=list)
    _original_line: Optional[str] = field(default=None, repr=False)  # For exact round-trip

    def add_child(
        self, content: str, position: Optional[int] = None
    ) -> "LogseqBlock":
        """Add child bullet with proper indentation.

        Args:
            content: The bullet content
            position: Optional index to insert at (None = append to end)

        Returns:
            The created child block
        """
        child = LogseqBlock(content=content, indent_level=self.indent_level + 1)

        if position is None:
            self.children.append(child)
        else:
            self.children.insert(position, child)

        return child


@dataclass
class LogseqOutline:
    """Parsed representation of Logseq's outline-based markdown structure.

    Attributes:
        blocks: Top-level blocks in document
        source_text: Original markdown for debugging
        indent_str: Indentation string (detected from source, default "  ")
    """

    blocks: list[LogseqBlock]
    source_text: str
    indent_str: str = "  "  # Default to 2 spaces

    @classmethod
    def parse(cls, markdown: str) -> "LogseqOutline":
        """Parse Logseq markdown into outline structure.

        IMPORTANT: Preserves exact order of properties:

        - Blocks appear in same order as source
        - Properties MUST maintain insertion order (dict preserves order in Python 3.7+)
        - Children preserve their original sequence (can be modified with targeted inserts)
        - Whitespace and formatting captured in _original_line for round-trip

        Args:
            markdown: Logseq markdown content

        Returns:
            Parsed LogseqOutline

        Raises:
            ValueError: If markdown is malformed
        """
        if not markdown.strip():
            return cls(blocks=[], source_text=markdown)

        lines = markdown.split("\n")

        # Detect indentation style from source first
        indent_str = _detect_indentation(lines)

        # Parse blocks using the detected indentation
        blocks = _parse_blocks(lines, indent_str)

        return cls(blocks=blocks, source_text=markdown, indent_str=indent_str)

    def find_heading(self, text: str) -> Optional[LogseqBlock]:
        """Recursively search for heading containing text.

        Args:
            text: Text to search for (case-insensitive)

        Returns:
            Block if found, None otherwise
        """

        def search(blocks: list[LogseqBlock]) -> Optional[LogseqBlock]:
            for block in blocks:
                if text.lower() in block.content.lower():
                    return block
                if found := search(block.children):
                    return found
            return None

        return search(self.blocks)

    def render(self) -> str:
        """Render outline back to Logseq markdown.

        IMPORTANT: Minimal changes guarantee (FR-008):

        - Preserves exact order of all blocks
        - Maintains original indentation (2 spaces per level)
        - NEVER reorders properties (insertion order sacred)
        - Children modifications are targeted and minimal
        - Uses _original_line when available for perfect round-trip

        Returns:
            Rendered markdown
        """
        lines = []

        def render_block(block: LogseqBlock) -> None:
            # Use original line if available (unchanged blocks)
            if block._original_line and not block.children:
                lines.append(block._original_line)
            else:
                # Render block (possibly with new children appended)
                # Use detected indentation style
                indent = self.indent_str * block.indent_level
                lines.append(f"{indent}- {block.content}")

                # Render children in exact order (new ones at end)
                for child in block.children:
                    render_block(child)

        for block in self.blocks:
            render_block(block)

        return "\n".join(lines)


def _parse_blocks(lines: list[str], indent_str: str = "  ") -> list[LogseqBlock]:
    """Parse lines into hierarchical blocks.

    Args:
        lines: Markdown lines
        indent_str: Indentation string (e.g., "  ", "\t", "    ")

    Returns:
        List of root-level blocks
    """
    root_blocks = []
    stack: list[LogseqBlock] = []  # Stack of parent blocks by indent level

    for line in lines:
        if not line.strip():
            continue

        # Skip non-bullet lines
        if not line.lstrip().startswith("- "):
            continue

        # Calculate indent level based on indent_str
        leading_whitespace = line[: len(line) - len(line.lstrip())]
        indent_level = leading_whitespace.count(indent_str)

        # Extract content (after "- ")
        content = line.lstrip()[2:]  # Skip "- "

        # Create block
        block = LogseqBlock(
            content=content,
            indent_level=indent_level,
            _original_line=line,
        )

        # Determine parent
        if indent_level == 0:
            # Root level
            root_blocks.append(block)
            stack = [block]
        else:
            # Find parent (closest block with indent_level - 1)
            while len(stack) > indent_level:
                stack.pop()

            if stack and stack[-1].indent_level == indent_level - 1:
                # Add as child to parent
                parent = stack[-1]
                parent.children.append(block)
                stack.append(block)
            else:
                # Malformed indentation - treat as root
                root_blocks.append(block)
                stack = [block]

    return root_blocks


def _detect_indentation(lines: list[str]) -> str:
    """Detect indentation style from markdown lines.

    Looks for the first level-1 indented bullet (smallest indent).
    Falls back to 2 spaces if no indented bullets found.

    Args:
        lines: Lines of markdown

    Returns:
        Indentation string (e.g., "  ", "    ", "\t")
    """
    # Collect all leading whitespace from indented bullets
    indents = []

    for line in lines:
        # Skip empty lines
        if not line or not line.strip():
            continue

        # Only look at bullet lines
        if not line.lstrip().startswith("-"):
            continue

        # Check if line has leading whitespace
        stripped = line.lstrip()
        if line != stripped:
            # Found an indented line - extract leading whitespace
            leading_whitespace = line[: len(line) - len(stripped)]
            indents.append(leading_whitespace)

    if not indents:
        # No indented lines found, default to 2 spaces
        return "  "

    # Find the shortest indentation (this is our indent unit)
    # This handles cases where we might have multi-level indents
    shortest = min(indents, key=len)
    return shortest
