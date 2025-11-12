"""Helper functions for LLM decision processing and batching."""

from typing import AsyncIterator
from collections import defaultdict
from logsqueak.models.integration_decision import IntegrationDecision
from logseq_outline.parser import LogseqOutline, LogseqBlock
from logseq_outline.context import generate_full_context
import re


async def batch_decisions_by_block(
    decision_stream: AsyncIterator[IntegrationDecision]
) -> AsyncIterator[list[IntegrationDecision]]:
    """
    Batch consecutive decisions by knowledge_block_id.

    Takes a stream of IntegrationDecision objects and groups consecutive
    decisions with the same knowledge_block_id into batches.

    Args:
        decision_stream: Async iterator yielding IntegrationDecision objects

    Yields:
        Lists of IntegrationDecision objects grouped by knowledge_block_id

    Example:
        Input stream: [A1, A2, B1, C1, C2, C3]
        Output batches: [[A1, A2], [B1], [C1, C2, C3]]
    """
    current_batch: list[IntegrationDecision] = []
    current_block_id: str | None = None

    async for decision in decision_stream:
        # If this is a new block, yield the current batch and start a new one
        if current_block_id is not None and decision.knowledge_block_id != current_block_id:
            if current_batch:
                yield current_batch
            current_batch = []

        # Add decision to current batch
        current_batch.append(decision)
        current_block_id = decision.knowledge_block_id

    # Yield final batch if not empty
    if current_batch:
        yield current_batch


class FilteredStreamWithCount:
    """
    Wrapper that filters decisions and tracks skipped count.

    This class provides both an async iterator interface and a skipped_count
    property that can be accessed after iteration completes.
    """

    def __init__(self, stream: AsyncIterator[IntegrationDecision]):
        self.stream = stream
        self.skipped_count = 0
        self._skip_block_ids: set[str] = set()
        self._current_block_id: str | None = None
        self._current_batch: list[IntegrationDecision] = []

    def __aiter__(self):
        """Async iterator protocol."""
        return self._iterate()

    async def _iterate(self):
        """Async iterator implementation."""
        async for decision in self.stream:
            # Check if we've moved to a new block
            if (
                self._current_block_id is not None
                and decision.knowledge_block_id != self._current_block_id
            ):
                # Process completed batch - yield decisions immediately
                async for yielded_decision in self._process_batch():
                    yield yielded_decision

            # Add to current batch
            self._current_batch.append(decision)
            self._current_block_id = decision.knowledge_block_id

        # Process final batch
        if self._current_batch:
            async for yielded_decision in self._process_batch():
                yield yielded_decision

    async def _process_batch(self):
        """Process a completed batch of decisions for one block.

        Yields decisions from the batch if the block should not be skipped.
        """
        # Check if any decision in batch has skip_exists
        has_skip = any(d.action == "skip_exists" for d in self._current_batch)

        if has_skip:
            # Skip entire block
            self.skipped_count += 1
            self._skip_block_ids.add(self._current_block_id)
        else:
            # Yield all decisions from this block immediately
            for decision in self._current_batch:
                yield decision

        # Clear batch
        self._current_batch = []


def filter_skip_exists_blocks(
    decision_stream: AsyncIterator[IntegrationDecision]
) -> FilteredStreamWithCount:
    """
    Filter out entire knowledge blocks that have ANY skip_exists decision.

    This function wraps the decision stream and filters out all decisions
    for blocks where at least one decision has action="skip_exists".
    It also tracks the count of skipped blocks.

    Args:
        decision_stream: Async iterator yielding IntegrationDecision objects

    Returns:
        FilteredStreamWithCount object that provides:
        - Async iterator protocol for filtered decisions
        - skipped_count property (accessible after iteration)

    Example:
        Input: [A1(add), A2(skip_exists), B1(add)]
        Output: FilteredStreamWithCount where iteration yields [B1] and .skipped_count == 1

    Usage:
        filtered_stream = filter_skip_exists_blocks(decision_stream())
        filtered = []
        async for decision in filtered_stream:
            filtered.append(decision)
        print(f"Skipped {filtered_stream.skipped_count} blocks")
    """
    return FilteredStreamWithCount(decision_stream)


def format_chunks_for_llm(
    chunks: list[tuple[str, LogseqBlock, list[LogseqBlock]]],
    page_contents: dict[str, LogseqOutline]
) -> str:
    """
    Format RAG search result chunks into XML for LLM prompt.

    This function takes hierarchical chunks from RAG search and formats them
    as XML with page properties and block context. Block id:: properties are
    stripped from content and included only in the XML id attribute.

    Args:
        chunks: List of (page_name, block, parents) tuples from RAG search
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
            - and this is the deepest block (id stripped from content)
        </block>
        ...
        </page>
        </pages>

    Example:
        >>> chunks = [
        ...     ("Page1", block1, [parent1]),
        ...     ("Page1", block2, []),
        ...     ("Page2", block3, [parent2, parent3])
        ... ]
        >>> xml = format_chunks_for_llm(chunks, page_contents)
        >>> print(xml)
        <pages>
        <page name="Page1">
        <properties>
        title:: Page 1
        </properties>
        <block id="abc123">
        - Parent content
          - Block content
        </block>
        <block id="def456">
        - Block content
        </block>
        </page>
        <page name="Page2">
        <block id="ghi789">
        - Grandparent content
          - Parent content
            - Block content
        </block>
        </page>
        </pages>
    """
    # Group chunks by page
    chunks_by_page = defaultdict(list)
    for page_name, block, parents in chunks:
        chunks_by_page[page_name].append((block, parents))

    # Build XML
    xml_parts = ["<pages>"]

    for page_name in sorted(chunks_by_page.keys()):
        xml_parts.append(f'<page name="{_xml_escape(page_name)}">')

        # Add page properties if available
        outline = page_contents.get(page_name)
        if outline and outline.frontmatter:
            # frontmatter is a list of strings
            if any(line.strip() for line in outline.frontmatter):
                xml_parts.append("<properties>")
                xml_parts.extend(outline.frontmatter)
                xml_parts.append("</properties>")

        # Add blocks
        for block, parents in chunks_by_page[page_name]:
            # Get hybrid ID (explicit id:: or content hash)
            block_id = block.block_id if block.block_id else f"hash-{id(block)}"

            xml_parts.append(f'<block id="{_xml_escape(block_id)}">')

            # Generate full hierarchical context
            full_context = generate_full_context(block, parents)

            # Strip id:: properties from content
            context_without_ids = _strip_id_properties(full_context)

            xml_parts.append(context_without_ids)
            xml_parts.append("</block>")

        xml_parts.append("</page>")

    xml_parts.append("</pages>")

    return "\n".join(xml_parts)


def _xml_escape(text: str) -> str:
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


def _strip_id_properties(content: str) -> str:
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
