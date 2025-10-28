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

    - All block lines stored in unified content list (first line, continuation, properties)
    - Properties accessed via get_property/set_property (preserves location in content)
    - Children can be inserted where appropriate (targeted, minimal changes)

    Attributes:
        content: All block lines (first line, continuation lines, properties)
                 Example: ["First line", "continuation", "key:: value"]
        indent_level: Indentation level (0 = root)
        block_id: Persistent block ID from id:: property (None if not present)
        children: Child blocks
    """

    content: list[str]
    indent_level: int
    block_id: Optional[str] = None  # Persistent ID from id:: property
    children: list["LogseqBlock"] = field(default_factory=list)

    def __post_init__(self):
        """Initialize block_id from id:: property if not explicitly set."""
        if self.block_id is None:
            self.block_id = self.get_property("id")

    def get_full_content(self, normalize_whitespace: bool = False) -> str:
        """Get full block content as a single string.

        Joins all content lines (first line, continuation lines, properties)
        with newlines.

        Args:
            normalize_whitespace: If True, strips leading/trailing whitespace from each line
                                 before joining (useful for embeddings/matching).
                                 If False, preserves original line content.

        Returns:
            Full block content joined with newlines

        Examples:
            >>> block.get_full_content()
            "First line\\ncontinuation\\nid:: 12345"
            >>> block.get_full_content(normalize_whitespace=True)
            "First line\\ncontinuation\\nid:: 12345"  # Each line stripped
        """
        if normalize_whitespace:
            # Strip each line individually, then join
            return "\n".join(line.strip() for line in self.content)
        return "\n".join(self.content)

    def add_child(
        self, content: str, position: Optional[int] = None
    ) -> "LogseqBlock":
        """Add child bullet with proper indentation.

        Args:
            content: The bullet content (first line)
            position: Optional index to insert at (None = append to end)

        Returns:
            The created child block
        """
        child = LogseqBlock(content=[content], indent_level=self.indent_level + 1)

        if position is None:
            self.children.append(child)
        else:
            self.children.insert(position, child)

        return child

    def get_property(self, key: str) -> Optional[str]:
        """Get property value by key.

        Scans content for lines matching 'key:: value' pattern.

        Args:
            key: Property key to search for

        Returns:
            Property value if found, None otherwise

        Examples:
            >>> block.get_property("id")
            "12345"
            >>> block.get_property("missing")
            None
        """
        for line in self.content:
            stripped = line.strip()
            if "::" in stripped:
                parts = stripped.split("::", 1)
                if len(parts) == 2 and parts[0].strip() == key:
                    return parts[1].strip()
        return None

    def set_property(self, key: str, value: str) -> None:
        """Set property value, preserving location or appending.

        If the property exists, updates it in-place.
        If the property doesn't exist, appends it to the end of content.

        Args:
            key: Property key
            value: Property value

        Examples:
            >>> block.set_property("id", "12345")  # Appends: "id:: 12345"
            >>> block.set_property("id", "67890")  # Updates existing in-place
        """
        # Search for existing property
        for i, line in enumerate(self.content):
            stripped = line.strip()
            if "::" in stripped:
                parts = stripped.split("::", 1)
                if len(parts) == 2 and parts[0].strip() == key:
                    # Update in-place, preserving indentation style
                    leading_space = line[: len(line) - len(line.lstrip())]
                    self.content[i] = f"{leading_space}{key}:: {value}"

                    # Update block_id if this is the id property
                    if key == "id":
                        self.block_id = value
                    return

        # Property not found, append to end
        self.content.append(f"{key}:: {value}")

        # Update block_id if this is the id property
        if key == "id":
            self.block_id = value

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

        IMPORTANT: Preserves exact order:

        - Blocks appear in same order as source
        - Properties MUST maintain insertion order (preserved in content list)
        - Children preserve their original sequence (can be modified with targeted inserts)
        - Continuation lines stored in content list (whitespace normalized during parse)

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
                # Search in first line (main block content)
                if text.lower() in block.content[0].lower():
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
        - Maintains original indentation (uses detected indent_str)
        - NEVER reorders properties (location preserved in content)
        - Children modifications are targeted and minimal
        - Preserves frontmatter (page-level properties)

        Returns:
            Rendered markdown
        """
        lines = []

        # Render frontmatter first (page-level properties)
        if self.frontmatter:
            lines.extend(self.frontmatter)

        def render_block(block: LogseqBlock) -> None:
            # Calculate base indentation
            indent = self.indent_str * block.indent_level

            # Render first line with bullet
            lines.append(f"{indent}- {block.content[0]}")

            # Render subsequent lines (continuation/properties) with "  " instead of "- "
            for line in block.content[1:]:
                lines.append(f"{indent}  {line}")

            # Render children in exact order (new ones at end)
            for child in block.children:
                render_block(child)

        for block in self.blocks:
            render_block(block)

        return "\n".join(lines)


def _parse_blocks(lines: list[str], indent_str: str = "  ") -> tuple[list[str], list[LogseqBlock]]:
    """Parse lines into hierarchical blocks.

    A block consists of:
    - A bullet line (starts with "- " after leading whitespace)
    - All continuation lines until the next bullet (or end of file)

    All lines are stored in the block's content list. Continuation line
    whitespace is normalized (stripped) during parsing and re-added during rendering.

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

            # Extract first line content (after "- ")
            first_line = line.lstrip()[2:]  # Skip "- "

            # Build unified content list (first line + continuation lines)
            content = [first_line]

            # Calculate expected base indentation for continuation lines
            # Continuation lines should be indented at least as much as the bullet + 2 spaces
            continuation_base_indent = leading_whitespace + "  "

            # Look ahead for continuation lines
            j = i + 1
            while j < len(lines):
                next_line = lines[j]

                # Check if next line is a bullet
                if next_line.lstrip().startswith("- "):
                    # This is the next bullet, stop collecting
                    break

                # This is a continuation line - strip only the base indentation
                # to preserve any extra indentation (e.g., code block indentation)
                if next_line.startswith(continuation_base_indent):
                    # Strip the base indentation, keep any extra
                    continuation_content = next_line[len(continuation_base_indent):]
                else:
                    # Line doesn't have the expected indentation, only strip leading
                    # (preserve trailing whitespace)
                    continuation_content = next_line.lstrip()

                content.append(continuation_content)
                j += 1

            # Create block with unified content
            # block_id will be extracted automatically in __post_init__
            block = LogseqBlock(
                content=content,
                indent_level=indent_level,
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
