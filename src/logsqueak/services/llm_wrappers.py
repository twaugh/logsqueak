"""LLM wrapper functions that wrap LLMClient.stream_ndjson with specific prompts."""

from typing import AsyncIterator, Optional
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

    # Get frontmatter and indent_str to match generate_chunks() behavior
    frontmatter = augmented.frontmatter if augmented.frontmatter else None
    indent_str = augmented.indent_str

    def augment_block(block: LogseqBlock, parents: list[LogseqBlock]) -> LogseqBlock:
        """Recursively augment a block and its children with IDs."""
        # If block already has an explicit ID, keep it
        if block.block_id is not None:
            block_id = block.block_id
        else:
            # Generate content hash as temporary ID
            # IMPORTANT: Must match generate_chunks() behavior exactly:
            # - Include frontmatter (for journals with properties)
            # - Include indent_str (for correct formatting)
            # - Do NOT include page_name (journals don't use page_name in hashes)
            full_context = generate_full_context(block, parents, indent_str, frontmatter)
            block_id = generate_content_hash(full_context)

        # CRITICAL: Recursively augment children BEFORE adding id:: property to this block
        # This ensures child hashes don't include parent's augmented id:: property
        new_parents = parents + [block]
        for child in block.children:
            augment_block(child, new_parents)

        # NOW set the id:: property (after children are processed)
        if block.get_property("id") is None:
            block.set_property("id", block_id)

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
        f"Task: Identify reusable knowledge in Logseq journal entries.\n\n"
        f"Input format:\n"
        f"- Markdown bullets with {indent_style}\n"
        f"- Each block has id:: property\n"
        f"- Properties format: key:: value\n\n"
        f"What is knowledge?\n"
        f"Knowledge is content that would be useful in the future, even without the journal context.\n"
        f"It contains insights, explanations, techniques, or understanding that can stand alone.\n\n"
        f"What is NOT knowledge?\n"
        f"Metadata about learning or doing (\"Learned X\", \"Read Y\", \"Attended Z\").\n"
        f"Events, tasks, or temporal records (\"Today I...\", \"Meeting at...\").\n\n"
        f"Rules:\n"
        f"1. Return ONLY blocks containing reusable insights or understanding\n"
        f"2. If child block needs parent context, return ONLY the child block ID (not parent)\n"
        f"3. Skip activity metadata and temporal records\n"
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
        temperature=0.3,  # Low temperature for deterministic classification
        request_id="classify_blocks",
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
        temperature=0.5,  # Moderate temperature for creative rewording
        request_id="reword_content",
    ):
        yield chunk


async def plan_integration_for_block(
    llm_client: LLMClient,
    edited_content: EditedContent,
    candidate_chunks: list[tuple[str, str, str]],
    page_contents: dict[str, LogseqOutline]
) -> AsyncIterator[IntegrationDecisionChunk]:
    """
    Generate integration decisions for a SINGLE knowledge block.

    Uses hierarchical chunks instead of full pages to drastically reduce prompt size.

    Args:
        llm_client: LLM client instance
        edited_content: Single knowledge block with refined content
        candidate_chunks: List of (page_name, block_id, hierarchical_context) tuples from RAG
        page_contents: Dict mapping page names to LogseqOutline (for frontmatter/properties)

    Yields:
        IntegrationDecisionChunk for each integration decision

    Example:
        >>> async for chunk in plan_integration_for_block(
        ...     client, block, rag_chunks, page_contents
        ... ):
        ...     print(f"Decision: {chunk.target_page}")
    """
    from logsqueak.services.llm_helpers import format_chunks_for_llm, strip_id_properties

    # Generate XML for single block (no indentation, no refined_content, no id:: properties)
    clean_context = strip_id_properties(edited_content.hierarchical_context)
    knowledge_block_xml = (
        f"<block id=\"{edited_content.block_id}\">\n"
        f"{clean_context}\n"
        f"</block>"
    )

    # Format hierarchical chunks using helper function
    pages_xml = format_chunks_for_llm(candidate_chunks, page_contents)

    xml_formatted_content = (
        f"<knowledge_blocks>\n"
        f"{knowledge_block_xml}\n"
        f"</knowledge_blocks>\n"
        f"{pages_xml}"
    )

    system_prompt = (
        "CRITICAL CONSTRAINTS:\n"
        "- NO conversational responses (\"Great\", \"Here's\", \"I'll\", \"Let me\", etc.)\n"
        "- NO markdown formatting (no bullets, no headers, no bold)\n"
        "- NO preambles, explanations, or summaries\n"
        "- Output ONLY raw NDJSON (newline-delimited JSON objects)\n"
        "- Start output immediately with first JSON object\n\n"
        "TASK: Find where knowledge belongs in existing pages.\n\n"
        "INPUT:\n"
        "- <knowledge_blocks>: Content to integrate (block ID in XML attribute)\n"
        "- <pages>: Hierarchical page structure (block IDs in XML attributes)\n\n"
        "HOW TO FIND THE BEST LOCATION:\n"
        "Pages show hierarchical structure (parent → child → grandchild).\n"
        "Look for semantic relationships in the hierarchy.\n"
        "Prefer specific, deeper locations over generic top-level sections.\n"
        "Return ONLY the best 1-2 locations per page.\n\n"
        "DECISION RULES:\n"
        "1. If duplicate exists → use skip_exists\n"
        "2. Prefer add_under semantically related parent blocks\n"
        "3. Confidence ≥ 0.30 only\n\n"
        "ACTIONS:\n"
        "- add_under: Best - add as child of semantically related block\n"
        "- add_section: Add as new top-level section if no good parent exists\n"
        "- replace: Replace existing block with updated knowledge\n"
        "- skip_exists: Duplicate already exists\n\n"
        "REQUIRED JSON SCHEMA (one object per line):\n"
        '{"knowledge_block_id": "string", "target_page": "string", "action": "add_section|add_under|replace|skip_exists", '
        '"target_block_id": "string|null", "target_block_title": "string|null", '
        '"confidence": 0.0-1.0, "reasoning": "string"}\n\n'
        "START OUTPUT NOW (first character must be '{'):"
    )

    # User prompt
    prompt = (
        f"Generate integration decisions as NDJSON (one JSON object per line).\n"
        f"Output first JSON object now:\n\n"
        f"{xml_formatted_content}"
    )

    # Create unique request ID using truncated block ID (first 8 chars)
    block_id_short = edited_content.block_id[:8] if len(edited_content.block_id) >= 8 else edited_content.block_id

    async for chunk in llm_client.stream_ndjson(
        prompt=prompt,
        system_prompt=system_prompt,
        chunk_model=IntegrationDecisionChunk,
        temperature=0.2,
        request_id=f"plan_integration_{block_id_short}",
    ):
        yield chunk


async def plan_integrations(
    llm_client: LLMClient,
    edited_contents: list[EditedContent],
    page_contents: dict[str, LogseqOutline],
    candidate_chunks: dict[str, list[tuple[str, str, str]]]
) -> AsyncIterator[IntegrationDecisionChunk]:
    """
    Plan integration decisions for knowledge blocks using LLM.

    Wrapper that iterates over blocks and calls plan_integration_for_block()
    for each one. Uses hierarchical chunks instead of full pages.

    Args:
        llm_client: LLM client instance
        edited_contents: Knowledge blocks with refined content
        page_contents: Dict mapping page names to LogseqOutline (for frontmatter/properties)
        candidate_chunks: Dict mapping block_id to list of (page_name, block_id, hierarchical_context) tuples

    Yields:
        IntegrationDecisionChunk for each integration decision (raw stream)

    Note:
        The returned stream is RAW and includes all decisions (including skip_exists).
        Decisions are naturally grouped by block (one LLM call per block).
        Caller should use filter_skip_exists_blocks() to process the stream before
        displaying to user.

    Example:
        >>> async for chunk in plan_integrations(client, edited_contents, pages, chunks):
        ...     print(f"Decision: {chunk.knowledge_block_id} → {chunk.target_page}")
    """
    # Process each block individually
    for edited_content in edited_contents:
        block_chunks = candidate_chunks.get(edited_content.block_id, [])

        # Skip if no candidates
        if not block_chunks:
            continue

        async for chunk in plan_integration_for_block(
            llm_client,
            edited_content,
            block_chunks,
            page_contents
        ):
            yield chunk
