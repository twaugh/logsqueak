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
    - Continuation lines are preserved exactly for round-trip

    Attributes:
        content: Block text content (first line after "- ")
        indent_level: Indentation level (0 = root)
        properties: Block properties (preserves insertion order)
        block_id: Persistent block ID from id:: property (None if not present)
        children: Child blocks
        continuation_lines: Lines that continue this block's content (preserves exact formatting)
        _original_lines: All original lines for exact round-trip (bullet + continuations)
    """

    content: str
    indent_level: int
    properties: dict = field(default_factory=dict)  # Preserves insertion order (Python 3.7+)
    block_id: Optional[str] = None  # Persistent ID from id:: property
    children: list["LogseqBlock"] = field(default_factory=list)
    continuation_lines: list[str] = field(default_factory=list)  # Multi-line content
    _original_lines: list[str] = field(default_factory=list, repr=False)  # For exact round-trip

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

    def get_hybrid_id(self, parents: list["LogseqBlock"] = None, indent_str: str = "  ") -> str:
        """Get hybrid ID for this block.

        Returns the persistent id:: property if present, otherwise generates
        a content hash based on full context (block + all parents).

        Args:
            parents: List of parent blocks from root to immediate parent
            indent_str: Indentation string (default: "  ")

        Returns:
            Hybrid ID (id:: property or MD5 hash of full context)

        Examples:
            >>> # Block with id:: property
            >>> block.get_hybrid_id()
            '65f3a8e0-1234-5678-9abc-def012345678'

            >>> # Block without id:: (generates hash)
            >>> block.get_hybrid_id(parents=[root, parent])
            'a1b2c3d4e5f6...'
        """
        if self.block_id:
            # Has explicit id:: property
            return self.block_id

        # Generate hash from full context
        from logsqueak.logseq.context import generate_full_context, generate_content_hash

        parents = parents or []
        full_context = generate_full_context(self, parents, indent_str)
        return generate_content_hash(full_context)


@dataclass
class LogseqOutline:
    """Parsed representation of Logseq's outline-based markdown structure.

    Attributes:
        blocks: Top-level blocks in document
        source_text: Original markdown for debugging
        indent_str: Indentation string (detected from source, default "  ")
        frontmatter: Lines before first bullet (page-level properties, etc.)
    """

    blocks: list[LogseqBlock]
    source_text: str
    indent_str: str = "  "  # Default to 2 spaces
    frontmatter: list[str] = field(default_factory=list)  # Page-level content

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

        # Parse blocks and frontmatter using the detected indentation
        frontmatter, blocks = _parse_blocks(lines, indent_str)

        return cls(blocks=blocks, source_text=markdown, indent_str=indent_str, frontmatter=frontmatter)

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

    def find_block_by_id(self, target_id: str) -> Optional[LogseqBlock]:
        """Find block by hybrid ID (id:: property or content hash).

        Uses generate_chunks() for efficient one-pass traversal with pre-computed IDs.

        Args:
            target_id: Hybrid ID to search for

        Returns:
            LogseqBlock if found, None otherwise

        Examples:
            >>> outline.find_block_by_id("explicit-id-123")
            <LogseqBlock with id:: explicit-id-123>

            >>> outline.find_block_by_id("a1b2c3d4...")  # content hash
            <LogseqBlock matching that hash>
        """
        from logsqueak.logseq.context import generate_chunks

        # Generate all chunks with pre-computed hybrid IDs (single pass)
        chunks = generate_chunks(self)

        # Search for matching ID
        for block, _context, hybrid_id in chunks:
            if hybrid_id == target_id:
                return block

        return None

    def render(self) -> str:
        """Render outline back to Logseq markdown.

        IMPORTANT: Minimal changes guarantee (FR-008):

        - Preserves exact order of all blocks
        - Maintains original indentation (2 spaces per level)
        - NEVER reorders properties (insertion order sacred)
        - Children modifications are targeted and minimal
        - Uses _original_lines when available for perfect round-trip
        - Preserves continuation lines exactly (code blocks, properties, etc.)
        - Preserves frontmatter (page-level properties)

        Returns:
            Rendered markdown
        """
        lines = []

        # Render frontmatter first (page-level properties)
        if self.frontmatter:
            lines.extend(self.frontmatter)

        def render_block(block: LogseqBlock) -> None:
            # Use original lines if available and no children added (perfect round-trip)
            if block._original_lines and not block.children:
                lines.extend(block._original_lines)
            else:
                # Render block (possibly with new children appended)
                # Use detected indentation style
                indent = self.indent_str * block.indent_level
                lines.append(f"{indent}- {block.content}")

                # Render continuation lines exactly as they were
                if block.continuation_lines:
                    lines.extend(block.continuation_lines)

                # Render children in exact order (new ones at end)
                for child in block.children:
                    render_block(child)

        for block in self.blocks:
            render_block(block)

        return "\n".join(lines)


def _parse_blocks(lines: list[str], indent_str: str = "  ") -> tuple[list[str], list[LogseqBlock]]:
    """Parse lines into hierarchical blocks with continuation lines.

    A block consists of:
    - A bullet line (starts with "- " after leading whitespace)
    - All continuation lines until the next bullet (or end of file)

    Continuation lines are any non-bullet lines that follow a bullet.
    They are preserved exactly to maintain formatting of code blocks,
    properties, and multi-line content.

    Lines before the first bullet are captured as frontmatter (page-level properties).

    Args:
        lines: Markdown lines
        indent_str: Indentation string (e.g., "  ", "\t", "    ")

    Returns:
        Tuple of (frontmatter_lines, root_blocks)
    """
    frontmatter = []
    root_blocks = []
    stack: list[LogseqBlock] = []  # Stack of parent blocks by indent level
    found_first_bullet = False

    i = 0
    while i < len(lines):
        line = lines[i]

        # Check if this is a bullet line
        is_bullet = line.lstrip().startswith("- ")

        if is_bullet:
            found_first_bullet = True
            # Calculate indent level based on indent_str
            leading_whitespace = line[: len(line) - len(line.lstrip())]
            indent_level = leading_whitespace.count(indent_str)

            # Extract content (after "- ")
            content = line.lstrip()[2:]  # Skip "- "

            # Collect all original lines for this block (bullet + continuations)
            original_lines = [line]
            continuation_lines = []

            # Look ahead for continuation lines
            j = i + 1
            while j < len(lines):
                next_line = lines[j]

                # Check if next line is a bullet
                if next_line.lstrip().startswith("- "):
                    # This is the next bullet, stop collecting
                    break

                # This is a continuation line
                original_lines.append(next_line)
                continuation_lines.append(next_line)
                j += 1

            # Parse properties from continuation lines
            properties = _parse_properties(continuation_lines)

            # Extract block_id from id:: property if present
            block_id = properties.get("id")

            # Create block with continuation lines and properties
            block = LogseqBlock(
                content=content,
                indent_level=indent_level,
                properties=properties,
                block_id=block_id,
                continuation_lines=continuation_lines,
                _original_lines=original_lines,
            )

            # Determine parent based on indent level
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

            # Skip past all the lines we just processed
            i = j
        else:
            # Non-bullet line
            if not found_first_bullet:
                # Before first bullet - this is frontmatter
                frontmatter.append(line)
            # else: orphan line after blocks started - skip it
            i += 1

    return frontmatter, root_blocks


def _parse_properties(continuation_lines: list[str]) -> dict[str, str]:
    """Parse properties from continuation lines.

    Properties have the format: `key:: value` (with optional leading whitespace).

    Args:
        continuation_lines: Lines following a bullet

    Returns:
        Dictionary of properties (preserves insertion order)

    Examples:
        >>> _parse_properties(["  id:: 123", "  tags:: foo, bar"])
        {'id': '123', 'tags': 'foo, bar'}
    """
    properties = {}

    for line in continuation_lines:
        stripped = line.strip()

        # Property lines have format: key:: value
        if "::" in stripped:
            # Split on first "::" only
            parts = stripped.split("::", 1)
            if len(parts) == 2:
                key = parts[0].strip()
                value = parts[1].strip()
                properties[key] = value

    return properties


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
