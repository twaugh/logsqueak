"""Helper functions for LLM decision processing and formatting."""

from collections import defaultdict
from logseq_outline.parser import LogseqOutline
import re


def format_chunks_for_llm(
    chunks: list[tuple[str, str, str]],
    page_contents: dict[str, LogseqOutline]
) -> str:
    """
    Format RAG search result chunks into XML for LLM prompt.

    This function takes hierarchical chunks from RAG search (already formatted
    as context strings from ChromaDB) and formats them as XML with page properties
    and block IDs.

    Args:
        chunks: List of (page_name, block_id, hierarchical_context) tuples from RAG search
        page_contents: Mapping of page_name to LogseqOutline (for page properties)

    Returns:
        XML string in format:
        <pages>
        <page name="foo/bar">
        <properties>
        area:: [[Ideas]]
        type:: Best Practice
        </properties>
        <block id="block-hash-or-id">
        - full hierarchical context
          - goes here
            - and this is the deepest block
        </block>
        ...
        </page>
        </pages>

    Example:
        >>> chunks = [
        ...     ("Page1", "abc123", "- Parent\\n  - Block content"),
        ...     ("Page1", "def456", "- Block content"),
        ...     ("Page2", "ghi789", "- Grandparent\\n  - Parent\\n    - Block content")
        ... ]
        >>> xml = format_chunks_for_llm(chunks, page_contents)
        >>> print(xml)
        <pages>
        <page name="Page1">
        <properties>
        title:: Page 1
        </properties>
        <block id="abc123">
        - Parent
          - Block content
        </block>
        <block id="def456">
        - Block content
        </block>
        </page>
        <page name="Page2">
        <block id="ghi789">
        - Grandparent
          - Parent
            - Block content
        </block>
        </page>
        </pages>
    """
    # Group chunks by page
    chunks_by_page = defaultdict(list)
    for page_name, block_id, context in chunks:
        chunks_by_page[page_name].append((block_id, context))

    # Build XML
    xml_parts = ["<pages>"]

    for page_name in sorted(chunks_by_page.keys()):
        xml_parts.append(f'<page name="{xml_escape(page_name)}">')

        # Add page properties if available
        outline = page_contents.get(page_name)
        if outline and outline.frontmatter:
            # frontmatter is a list of strings - filter out blank lines
            non_blank_properties = [line for line in outline.frontmatter if line.strip()]
            if non_blank_properties:
                xml_parts.append("<properties>")
                xml_parts.extend(non_blank_properties)
                xml_parts.append("</properties>")

        # Add blocks
        for block_id, context in chunks_by_page[page_name]:
            xml_parts.append(f'<block id="{xml_escape(block_id)}">')

            # Strip id:: properties and page-level properties from content
            # (ChromaDB may include them in root-level blocks)
            context_without_ids = strip_id_properties(context)

            # Strip page properties if they appear at the start of context
            if outline and outline.frontmatter:
                context_clean = strip_page_properties(context_without_ids, outline.frontmatter)
            else:
                context_clean = context_without_ids

            xml_parts.append(context_clean)
            xml_parts.append("</block>")

        xml_parts.append("</page>")

    xml_parts.append("</pages>")

    return "\n".join(xml_parts)


def xml_escape(text: str) -> str:
    """Escape XML special characters.

    Args:
        text: Text to escape

    Returns:
        Escaped text safe for XML attributes/content
    """
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def strip_id_properties(content: str) -> str:
    """Strip id:: properties from block content.

    This removes lines matching the pattern 'id:: <value>' (with proper
    indentation for continuation lines).

    Args:
        content: Block content with possible id:: properties

    Returns:
        Content with id:: properties removed
    """
    lines = content.split('\n')
    filtered_lines = []

    for line in lines:
        # Match id:: property lines (with any indentation)
        # Pattern: optional spaces/tabs + "id::" + space + value
        if not re.match(r'^\s*id::\s+\S', line):
            filtered_lines.append(line)

    return '\n'.join(filtered_lines)


def strip_page_properties(content: str, page_properties: list[str]) -> str:
    """Strip page-level properties from block content.

    This removes page property lines that appear at the start of block content
    (before the first bullet). This happens when ChromaDB stores root-level blocks.

    Args:
        content: Block content that may include page properties
        page_properties: List of page property lines to remove

    Returns:
        Content with page properties removed
    """
    lines = content.split('\n')

    # Create set of property lines (stripped) for fast lookup
    property_set = {prop.strip() for prop in page_properties if prop.strip()}

    filtered_lines = []
    found_first_bullet = False

    for line in lines:
        stripped = line.strip()

        # Once we find the first bullet, keep everything
        if stripped.startswith('- '):
            found_first_bullet = True

        # Skip property lines before first bullet
        if not found_first_bullet and stripped in property_set:
            continue

        # Skip blank lines before first bullet
        if not found_first_bullet and not stripped:
            continue

        filtered_lines.append(line)

    return '\n'.join(filtered_lines)
