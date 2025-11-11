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
        # Add block with the target block ID in XML attribute
        xml_lines.append(f'  <block id="{edited.block_id}">')

        # Clean hierarchical context to avoid confusing the LLM
        # The LLM should use the XML attribute, not id:: properties in content
        for line in edited.hierarchical_context.split('\n'):
            # Skip id:: property lines (LLM should use XML attribute instead)
            stripped = line.strip()
            if stripped.startswith('id::'):
                continue

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

        # Clean and add original context (remove id:: properties)
        for line in edited.hierarchical_context.split('\n'):
            # Skip id:: property lines (block ID is in XML attribute)
            stripped = line.strip()
            if stripped.startswith('id::'):
                continue

            xml_lines.append('      ' + line if line.strip() else '')

        xml_lines.append('    </original_journal_context>')
        xml_lines.append(f'    <refined_content>{edited.current_content}</refined_content>')
        xml_lines.append('  </block>')
        xml_lines.append('')

    xml_lines.append('</knowledge_blocks>')
    xml_lines.append('')
    xml_lines.append('<candidate_pages>')

    # Add candidate pages as Logseq markdown with hybrid IDs
    for page_name, page_outline in page_contents.items():
        xml_lines.append(f'<page name="{page_name}">')
        xml_lines.append('')

        # Augment page with hybrid IDs (preserves existing IDs, adds hashes for blocks without)
        augmented_page = _augment_outline_with_ids(page_outline)

        # Render full page as clean Logseq markdown
        page_markdown = augmented_page.render()
        xml_lines.append(page_markdown)

        xml_lines.append('')
        xml_lines.append('</page>')

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
        f"You are a JSON-only knowledge classifier. Output ONLY valid JSON lines (NDJSON format).\n"
        f"NO markdown. NO explanations. NO conversational text. ONLY JSON objects.\n\n"
        f"Task: Identify lasting knowledge in Logseq journal entries (skip activity logs/events/tasks).\n\n"
        f"Input format:\n"
        f"- Markdown bullets with {indent_style}\n"
        f"- Each block has id:: property\n"
        f"- Properties format: key:: value\n\n"
        f"Rules:\n"
        f"1. Return ONLY blocks with lasting insights/lessons/concepts\n"
        f"2. If child block needs parent context, return ONLY the child block ID (not parent)\n"
        f"3. Skip activity logs, events, tasks, temporal notes\n"
        f"4. Output nothing for non-knowledge blocks\n\n"
        f"Output format (STRICT NDJSON - one JSON object per line):\n"
        f'{{"block_id": "abc123", "confidence": 0.85, "reason": "Technical insight about testing"}}\n'
        f'{{"block_id": "def456", "confidence": 0.70, "reason": "Conceptual understanding of frameworks"}}\n\n'
        f"CRITICAL: Output ONLY JSON lines. No markdown, no prose, no preamble."
    )

    # User prompt: Instruction BEFORE data (critical for attention)
    prompt = (
        f"Classify knowledge blocks and output as NDJSON (one JSON object per line).\n"
        f"Output first JSON object now:\n\n"
        f"{journal_content}"
    )

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
        f"You are a JSON-only content rewriter. Output ONLY valid JSON lines (NDJSON format).\n"
        f"NO markdown. NO explanations. NO conversational text. ONLY JSON objects.\n\n"
        f"Task: Transform journal content to evergreen knowledge (remove temporal context).\n\n"
        f"Input format:\n"
        f"- XML blocks with hierarchical context ({indent_style})\n"
        f"- The deepest (most indented) block is the target to reword\n"
        f"- Parent blocks provide context for pronoun resolution\n\n"
        f"Rules:\n"
        f"1. Remove temporal context (today, yesterday, dates, times)\n"
        f"2. Convert first-person → third-person or neutral voice\n"
        f"3. Resolve pronouns using parent block context\n"
        f"4. Keep technical details, links, and code intact\n"
        f"5. Reword ONLY the deepest block (target), not parent blocks\n\n"
        f"Example pronoun resolution:\n"
        f'Parent: "Tried Textual framework"\n'
        f'Child: "This is Python-specific"\n'
        f'→ Reword child to: "The Textual framework is Python-specific"\n\n'
        f"Output format (STRICT NDJSON - one JSON object per line):\n"
        f'{{"block_id": "abc123", "reworded_content": "PyTest supports fixture dependency injection"}}\n'
        f'{{"block_id": "def456", "reworded_content": "The Textual framework is Python-specific"}}\n\n'
        f"CRITICAL: Output ONLY JSON lines. Use block_id from XML attribute. No markdown, no prose."
    )

    # User prompt: Instruction BEFORE data (critical for attention)
    prompt = (
        f"Reword the deepest block in each XML block and output as NDJSON (one JSON object per line).\n"
        f"Output first JSON object now:\n\n"
        f"{xml_blocks}"
    )

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
        "CRITICAL CONSTRAINTS:\n"
        "- NO conversational responses (\"Great\", \"Here's\", \"I'll\", \"Let me\", etc.)\n"
        "- NO markdown formatting (no bullets, no headers, no bold)\n"
        "- NO preambles, explanations, or summaries\n"
        "- Output ONLY raw NDJSON (newline-delimited JSON objects)\n"
        "- Start output immediately with first JSON object\n\n"
        "TASK: Generate integration decisions for knowledge blocks.\n\n"
        "INPUT:\n"
        "- <knowledge_blocks>: Content to integrate (with block IDs)\n"
        "- <candidate_pages>: Target pages (Logseq markdown with id:: properties)\n\n"
        "DECISION RULES:\n"
        "1. Check if duplicate exists → use skip_exists\n"
        "2. Confidence ≥ 0.30 only\n"
        "3. Max 2 decisions per (knowledge_block, page)\n"
        "4. Skip irrelevant pages\n"
        "5. Group by knowledge_block_id\n\n"
        "ACTIONS:\n"
        "- add_section: New top-level (target_block_id: null)\n"
        "- add_under: Child of block (target_block_id: from id::)\n"
        "- replace: Replace block (target_block_id: from id::)\n"
        "- skip_exists: Duplicate (target_block_id: from id::)\n\n"
        "REQUIRED JSON SCHEMA (one object per line):\n"
        '{"knowledge_block_id": "string", "target_page": "string", "action": "add_section|add_under|replace|skip_exists", '
        '"target_block_id": "string|null", "target_block_title": "string|null", '
        '"confidence": 0.0-1.0, "reasoning": "string"}\n\n'
        "EXAMPLE OUTPUT:\n"
        '{"knowledge_block_id": "abc123", "target_page": "Software/Python", "action": "add_under", '
        '"target_block_id": "def456", "target_block_title": "Testing", "confidence": 0.85, "reasoning": "Fits under testing"}\n'
        '{"knowledge_block_id": "abc123", "target_page": "Tools/TDD", "action": "add_section", '
        '"target_block_id": null, "target_block_title": null, "confidence": 0.70, "reasoning": "New section"}\n\n'
        "START OUTPUT NOW (first character must be '{'):"
    )

    # User prompt: Instruction BEFORE data (critical for attention)
    prompt = (
        f"Generate integration decisions as NDJSON (one JSON object per line).\n"
        f"Output first JSON object now:\n\n"
        f"{xml_formatted_content}"
    )

    async for chunk in llm_client.stream_ndjson(
        prompt=prompt,
        system_prompt=system_prompt,
        chunk_model=IntegrationDecisionChunk,
        temperature=0.2  # Low temperature to reduce conversational responses
        # Note: json_mode not used because it expects single JSON object, not NDJSON
    ):
        yield chunk
