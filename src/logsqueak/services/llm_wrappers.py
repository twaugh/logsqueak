"""LLM wrapper functions that wrap LLMClient.stream_ndjson with specific prompts."""

from typing import AsyncIterator
import copy
from logseq_outline.parser import LogseqOutline, LogseqBlock
from logseq_outline.context import generate_full_context, generate_content_hash
from logsqueak.services.llm_client import LLMClient
from logsqueak.models.llm_chunks import (
    KnowledgeClassificationChunk,
    ContentRewordingChunk,
    IntegrationDecisionChunk
)
from logsqueak.models.edited_content import EditedContent


def _detect_indent_style(outline: LogseqOutline) -> str:
    """
    Detect indentation style from Logseq outline.

    Returns:
        Human-readable description of indentation (e.g., "2 spaces per level")
    """
    # Logseq uses 2 spaces by default
    return "2 spaces per level"


def _augment_outline_with_ids(outline: LogseqOutline) -> LogseqOutline:
    """
    Augment outline blocks with temporary IDs using the hybrid ID system.

    For blocks without explicit id:: properties, adds a temporary ID based on
    content hash. This allows the LLM to reference specific blocks in its response.

    Args:
        outline: Original outline

    Returns:
        Deep copy of outline with all blocks having id:: properties
    """
    # Deep copy to avoid modifying the original
    augmented = copy.deepcopy(outline)

    def augment_block(block: LogseqBlock, parents: list[LogseqBlock]) -> LogseqBlock:
        """Recursively augment a block and its children with IDs."""
        # If block already has an explicit ID, keep it
        if block.block_id is not None:
            block_id = block.block_id
        else:
            # Generate content hash as temporary ID
            # Use generate_full_context to get hierarchical context, then hash it
            full_context = generate_full_context(block, parents)
            block_id = generate_content_hash(full_context)

        # Ensure block has id:: property
        if block.get_property("id") is None:
            block.set_property("id", block_id)

        # Recursively augment children
        new_parents = parents + [block]
        for child in block.children:
            augment_block(child, new_parents)

        return block

    # Augment all root blocks
    for block in augmented.blocks:
        augment_block(block, [])

    return augmented


def _generate_xml_blocks_for_rewording(
    edited_contents: list[EditedContent],
    outline: LogseqOutline
) -> str:
    """
    Generate XML-formatted blocks with hierarchical context for rewording.

    Args:
        edited_contents: List of knowledge blocks to reword
        outline: Original journal outline for looking up page properties

    Returns:
        XML string with <blocks> containing each block with full context
    """
    xml_lines = ["<blocks>"]

    for edited in edited_contents:
        # Add block with full hierarchical context
        xml_lines.append(f'  <block id="{edited.block_id}">')

        # Add hierarchical context (already formatted with proper indentation)
        for line in edited.hierarchical_context.split('\n'):
            xml_lines.append(line)

        xml_lines.append('  </block>')
        xml_lines.append('')  # Empty line between blocks

    xml_lines.append('</blocks>')
    return '\n'.join(xml_lines)


def _generate_xml_formatted_content(
    edited_contents: list[EditedContent],
    page_contents: dict[str, LogseqOutline]
) -> str:
    """
    Generate XML-formatted content for integration decision planning.

    Args:
        edited_contents: Knowledge blocks with refined content
        page_contents: Dict mapping page names to full page outlines

    Returns:
        XML string with <knowledge_blocks> and <candidate_pages>
    """
    xml_lines = ["<knowledge_blocks>"]

    # Add knowledge blocks with original context and refined content
    for edited in edited_contents:
        xml_lines.append(f'  <block id="{edited.block_id}">')
        xml_lines.append('    <original_journal_context>')

        # Add original context with proper indentation
        for line in edited.hierarchical_context.split('\n'):
            xml_lines.append('      ' + line if line.strip() else '')

        xml_lines.append('    </original_journal_context>')
        xml_lines.append(f'    <refined_content>{edited.current_content}</refined_content>')
        xml_lines.append('  </block>')
        xml_lines.append('')

    xml_lines.append('</knowledge_blocks>')
    xml_lines.append('')
    xml_lines.append('<candidate_pages>')

    # Add candidate pages with structure
    for page_name, page_outline in page_contents.items():
        xml_lines.append(f'  <page name="{page_name}">')

        # Add page properties if present
        if page_outline.frontmatter:
            xml_lines.append('    <page_properties>')
            for line in page_outline.frontmatter:
                if line.strip():
                    xml_lines.append('      ' + line)
            xml_lines.append('    </page_properties>')

        # Add existing blocks
        xml_lines.append('    <existing_blocks>')

        def render_block(block: LogseqBlock, indent: str = "- ") -> list[str]:
            """Recursively render a block and its children."""
            lines = []
            # Render block content
            for i, content_line in enumerate(block.content):
                if i == 0:
                    lines.append(f"{indent}{content_line}")
                else:
                    lines.append(f"  {content_line}")
            # Render children
            for child in block.children:
                child_lines = render_block(child, indent + "  ")
                lines.extend(child_lines)
            return lines

        for block in page_outline.blocks:
            xml_lines.append('      <block>')
            block_lines = render_block(block)
            for line in block_lines:
                if line.strip():
                    xml_lines.append('        ' + line)
            xml_lines.append('      </block>')
        xml_lines.append('    </existing_blocks>')

        xml_lines.append('  </page>')
        xml_lines.append('')

    xml_lines.append('</candidate_pages>')
    return '\n'.join(xml_lines)


async def classify_blocks(
    llm_client: LLMClient,
    journal_outline: LogseqOutline
) -> AsyncIterator[KnowledgeClassificationChunk]:
    """
    Classify journal blocks as knowledge or activity using LLM.

    Wraps LLMClient.stream_ndjson with Phase 1 knowledge classification prompt.

    Args:
        llm_client: LLM client instance
        journal_outline: Parsed journal entry outline (should be pre-augmented with IDs)

    Yields:
        KnowledgeClassificationChunk for each block classified as knowledge

    Example:
        >>> async for chunk in classify_blocks(client, outline):
        ...     print(f"Block {chunk.block_id}: {chunk.confidence}")
    """
    indent_style = _detect_indent_style(journal_outline)

    # Render the outline (expecting it to already have IDs from _augment_outline_with_ids)
    journal_content = journal_outline.render()

    system_prompt = (
        f"Identify lasting knowledge in Logseq journal entries.\n\n"
        f"Format notes:\n"
        f"- Markdown bullets with {indent_style}\n"
        f"- Each block has id:: property\n"
        f"- Properties format: key:: value\n\n"
        f"Task: Return ONLY blocks with lasting insights (skip events/tasks).\n"
        f"If child block needs parent context, return ONLY the child block ID.\n\n"
        f"Output: One JSON per line (NDJSON):\n"
        f'{{"block_id": "...", "confidence": 0.85, "reason": "..."}}\n\n'
        f"Output nothing for activity blocks."
    )

    prompt = f"Analyze this Logseq journal entry:\n\n{journal_content}"

    async for chunk in llm_client.stream_ndjson(
        prompt=prompt,
        system_prompt=system_prompt,
        chunk_model=KnowledgeClassificationChunk,
        temperature=0.3  # Low temperature for deterministic classification
    ):
        yield chunk


async def reword_content(
    llm_client: LLMClient,
    edited_contents: list[EditedContent],
    journal_outline: LogseqOutline
) -> AsyncIterator[ContentRewordingChunk]:
    """
    Reword knowledge blocks to remove temporal context.

    Wraps LLMClient.stream_ndjson with Phase 2 content rewording prompt.

    Args:
        llm_client: LLM client instance
        edited_contents: List of knowledge blocks to reword
        journal_outline: Original journal outline (for indent detection)

    Yields:
        ContentRewordingChunk for each reworded block

    Example:
        >>> async for chunk in reword_content(client, edited_contents, outline):
        ...     print(f"Block {chunk.block_id}: {chunk.reworded_content}")
    """
    indent_style = _detect_indent_style(journal_outline)
    xml_blocks = _generate_xml_blocks_for_rewording(edited_contents, journal_outline)

    system_prompt = (
        f"Transform journal content to evergreen knowledge.\n\n"
        f"Input: XML blocks with parent context ({indent_style})\n\n"
        f"Rules:\n"
        f"1. Remove temporal context (today, yesterday, dates)\n"
        f"2. Convert first-person → third-person or neutral\n"
        f"3. Resolve pronouns using parent context\n"
        f"4. Keep technical details intact\n"
        f"5. Reword ONLY the deepest block, not parents\n\n"
        f"Example pronoun resolution:\n"
        f'Parent: "Tried Textual framework"\n'
        f'Child: "This is Python-specific"\n'
        f'→ Reword: "The Textual framework is Python-specific"\n\n'
        f"Output: One JSON per line:\n"
        f'{{"block_id": "...", "reworded_content": "..."}}'
    )

    prompt = xml_blocks

    async for chunk in llm_client.stream_ndjson(
        prompt=prompt,
        system_prompt=system_prompt,
        chunk_model=ContentRewordingChunk,
        temperature=0.5  # Moderate temperature for creative rewording
    ):
        yield chunk


async def plan_integrations(
    llm_client: LLMClient,
    edited_contents: list[EditedContent],
    page_contents: dict[str, LogseqOutline]
) -> AsyncIterator[IntegrationDecisionChunk]:
    """
    Plan integration decisions for knowledge blocks using LLM.

    Wraps LLMClient.stream_ndjson with Phase 3 integration decision prompt.
    Returns RAW stream (not batched or filtered).

    Args:
        llm_client: LLM client instance
        edited_contents: Knowledge blocks with refined content
        page_contents: Dict mapping page names to full page outlines

    Yields:
        IntegrationDecisionChunk for each integration decision (raw stream)

    Note:
        The returned stream is RAW and includes all decisions (including skip_exists).
        Caller should use batch_decisions_by_block() and filter_skip_exists_blocks()
        to process the stream before displaying to user.

    Example:
        >>> async for chunk in plan_integrations(client, edited_contents, pages):
        ...     print(f"Decision: {chunk.knowledge_block_id} → {chunk.target_page}")
    """
    xml_formatted_content = _generate_xml_formatted_content(
        edited_contents,
        page_contents
    )

    system_prompt = (
        "Decide where to integrate knowledge into Logseq pages.\n\n"
        "Input: Knowledge blocks + candidate pages with structure\n\n"
        "FILTERING:\n"
        "• Confidence ≥ 0.30 only\n"
        "• Max 2 decisions per (block, page) pair\n"
        "• Omit irrelevant pages\n\n"
        "ACTIONS:\n"
        "• add_section: New top-level section\n"
        "• add_under: Child of existing block (needs target_block_id)\n"
        "• replace: Replace existing block (needs target_block_id)\n"
        "• skip_exists: Duplicate found (needs target_block_id)\n\n"
        "DUPLICATE CHECK:\n"
        "First check if knowledge exists in page. If yes → skip_exists\n\n"
        "OUTPUT ORDER (CRITICAL):\n"
        "All decisions for blockA, then blockB, then blockC\n"
        "✓ [A1, A2, B1, C1, C2]\n"
        "✗ [A1, B1, A2]\n\n"
        "Format: One JSON per line:\n"
        '{"knowledge_block_id": "...", "target_page": "...", "action": "...", '
        '"target_block_id": null|"...", "target_block_title": null|"...", '
        '"confidence": 0.85, "reasoning": "..."}'
    )

    prompt = xml_formatted_content

    async for chunk in llm_client.stream_ndjson(
        prompt=prompt,
        system_prompt=system_prompt,
        chunk_model=IntegrationDecisionChunk,
        temperature=0.4  # Balanced temperature for structured decisions
    ):
        yield chunk
