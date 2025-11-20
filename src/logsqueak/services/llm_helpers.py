"""Helper functions for LLM decision processing and formatting."""

from collections import defaultdict
from logseq_outline.parser import LogseqOutline


def format_chunks_for_llm(
    chunks: list[tuple[str, str, str]],
    page_contents: dict[str, LogseqOutline],
    id_mapper
) -> str:
    """
    Format RAG search result chunks into XML for LLM prompt.

    This function takes hierarchical chunks from RAG search (pre-cleaned during
    indexing to remove id:: and page properties) and formats them as XML with
    page properties and short block IDs.

    Args:
        chunks: List of (page_name, block_id, hierarchical_context) tuples from RAG search
        page_contents: Mapping of page_name to LogseqOutline (for page properties)
        id_mapper: LLMIDMapper for translating hybrid IDs to short IDs

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

        # Add blocks with short IDs
        for block_id, context in chunks_by_page[page_name]:
            short_id = id_mapper.to_short(block_id)
            xml_parts.append(f'<block id="{xml_escape(short_id)}">')
            xml_parts.append(context)
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
