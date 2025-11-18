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

    Performance optimization: Caches hierarchical contexts during augmentation to avoid
    regenerating them 3-5 times per block during the workflow.

    Args:
        outline: Original outline

    Returns:
        Deep copy of outline with all blocks having id:: properties and cached contexts
    """
    # Deep copy to avoid modifying the original
    augmented = copy.deepcopy(outline)

    # Get frontmatter and indent_str to match generate_chunks() behavior
    frontmatter = augmented.frontmatter if augmented.frontmatter else None
    indent_str = augmented.indent_str

    def augment_block(block: LogseqBlock, parents: list[LogseqBlock]) -> LogseqBlock:
        """Recursively augment a block and its children with IDs and cached contexts."""
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

        # PERFORMANCE OPTIMIZATION: Cache hierarchical contexts to avoid regeneration
        # Generate both versions (with and without frontmatter) once here
        block._cached_context = generate_full_context(block, parents, indent_str, frontmatter)
        block._cached_context_no_frontmatter = generate_full_context(block, parents, indent_str, None)

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
    # IMPORTANT: Exclude frontmatter from LLM prompt (it's page-level metadata, not knowledge)
    # Temporarily clear frontmatter, render, then restore it
    original_frontmatter = journal_outline.frontmatter
    journal_outline.frontmatter = []
    journal_content = journal_outline.render()
    journal_outline.frontmatter = original_frontmatter

    system_prompt = (
        f"You are a JSON-only knowledge extractor. Output ONLY valid JSON lines (NDJSON format).\n"
        f"NO markdown. NO explanations. NO conversational text. ONLY JSON objects.\n\n"
        f"Task: Extract lasting insights from Logseq journal entries.\n\n"
        f"FOCUS ON THE INSIGHT FIRST, NOT THE BLOCK:\n"
        f"1. Identify valuable knowledge/insights in individual journal blocks\n"
        f"2. Reword each insight to be timeless (remove temporal context, dates, activity verbs)\n"
        f"3. Associate each insight with exactly ONE journal block ID\n\n"
        f"CRITICAL CONSTRAINT:\n"
        f"- Each block contains at most one insight\n"
        f"- If an insight spans multiple blocks, choose the MOST SPECIFIC block that contains it\n"
        f"- Output each block_id at most once\n\n"
        f"Input format:\n"
        f"- Markdown bullets with {indent_style}\n"
        f"- Each block has id:: property\n"
        f"- Properties format: key:: value\n"
        f"- Indented blocks have parent context (ancestors are included when selected)\n\n"
        f"What is an insight?\n"
        f"An insight is content that would be valuable long-term, even without journal context.\n"
        f"It contains learnings, discoveries, best practices, solutions, patterns, or principles.\n"
        f"Focus on the DEEPEST block that contains the complete insight.\n"
        f"Child blocks with pronouns ('this', 'it') are valid - resolve them using parent context.\n\n"
        f"What is NOT an insight?\n"
        f"Routine tasks, meeting notes, status updates, temporal activities.\n"
        f"Activity metadata (\"Learned X\", \"Read Y\", \"Reviewed Z\").\n"
        f"Generic parent blocks that just introduce context (extract the specific child instead).\n\n"
        f"REWORDING RULES:\n"
        f"1. Extract the knowledge itself, not the activity that led to it\n"
        f"   - BAD: \"Reviewing Python docs reveals type hints are powerful\"\n"
        f"   - GOOD: \"Python type hints are becoming more powerful\"\n"
        f"2. Remove activity verbs (reviewing, learned, discovered, found, tried, tested)\n"
        f"3. Remove temporal context (today, yesterday, dates, times)\n"
        f"4. Convert first-person → third-person or neutral voice\n"
        f"5. Resolve pronouns using parent block context\n"
        f"6. Preserve ALL technical details EXACTLY:\n"
        f"   - Copy links character-for-character: [text](url) or [id|text](url)\n"
        f"   - NEVER drop URLs or convert links to plain text\n"
        f"   - Code snippets, commands, file paths unchanged\n"
        f"   - Specific names, IDs, version numbers preserved\n\n"
        f"Output format (STRICT NDJSON - one JSON object per line):\n"
        f'{{"block_id": "abc123", "insight": "PyTest supports fixture dependency injection", "confidence": 0.85}}\n'
        f'{{"block_id": "def456", "insight": "The Textual framework uses CSS for styling", "confidence": 0.70}}\n\n'
        f"FIELD DEFINITIONS:\n"
        f"- block_id: String containing the journal block ID for this insight\n"
        f"- insight: The reworded insight suitable for knowledge base (timeless, no temporal context)\n"
        f"- confidence: How well this insight will remain useful over time (0.0-1.0)\n"
        f"  High confidence (0.8-1.0): Fundamental concepts, proven patterns, universal principles\n"
        f"  Medium confidence (0.5-0.7): Context-specific solutions, emerging patterns, tool-specific knowledge\n"
        f"  Low confidence (0.0-0.4): Speculative ideas, highly contextual notes, rapidly changing details\n\n"
        f"CRITICAL: Output ONLY JSON lines. No markdown, no prose, no preamble."
    )

    # User prompt: Instruction BEFORE data (critical for attention)
    prompt = (
        f"Extract insights and output as NDJSON (one JSON object per line).\n"
        f"Each insight must be reworded and associated with exactly ONE block ID.\n"
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
        f"You are a JSON-only content rewriter. Output ONLY valid NDJSON (newline-delimited JSON).\n"
        f"CRITICAL FORMAT RULES:\n"
        f"- Each line is a separate JSON object (NOT an array)\n"
        f"- NO opening [ or closing ] brackets\n"
        f"- NO commas between objects\n"
        f"- NO markdown, explanations, or conversational text\n"
        f"- Start output immediately with first {{\n\n"
        f"Task: Transform journal content to evergreen knowledge (remove temporal context).\n\n"
        f"Input format:\n"
        f"- XML blocks with hierarchical context ({indent_style})\n"
        f"- The deepest (most indented) block is the target to reword\n"
        f"- Parent blocks provide context for pronoun resolution\n\n"
        f"Rules:\n"
        f"1. Extract the knowledge/insight itself, not the activity that led to it\n"
        f"   - BAD: \"Reviewing Python docs reveals type hints are powerful\"\n"
        f"   - GOOD: \"Python type hints are becoming more powerful\"\n"
        f"2. Remove activity verbs (reviewing, learned, discovered, found, tried, tested)\n"
        f"3. Remove temporal context (today, yesterday, dates, times)\n"
        f"4. Convert first-person → third-person or neutral voice\n"
        f"5. Resolve pronouns using parent block context\n"
        f"6. Preserve ALL technical details EXACTLY as written:\n"
        f"   - Copy links character-for-character: [text](url) or [id|text](url)\n"
        f"   - NEVER drop URLs or convert links to plain text\n"
        f"   - Code snippets, commands, file paths unchanged\n"
        f"   - Specific names, IDs, version numbers preserved\n"
        f"   - Do NOT summarize, paraphrase, or simplify technical content\n"
        f"7. Reword ONLY the deepest block (target), not parent blocks\n\n"
        f"Example transformations:\n"
        f'Parent: "DONE Review [[Python]] documentation for new features"\n'
        f'Child: "Type hints are becoming more powerful"\n'
        f'Grandchild: "PEP 692 adds TypedDict to function signatures"\n'
        f'→ Reword grandchild to: "PEP 692 adds TypedDict support to function signatures in Python"\n'
        f'(NOT: "Reviewing Python documentation reveals that PEP 692...")\n\n'
        f'Parent: "Tested asyncio.gather() behavior"\n'
        f'Child: "It preserves execution order"\n'
        f'→ Reword child to: "asyncio.gather() preserves execution order"\n'
        f'(NOT: "Testing revealed that it preserves order")\n\n'
        f'Parent: "Database optimization work"\n'
        f'Child: "Applied fix from [PERF-1234|query performance](https://tickets.example.com/PERF-1234)"\n'
        f'→ Reword child to: "Database query performance improved via [PERF-1234|query performance](https://tickets.example.com/PERF-1234)"\n'
        f'(NOT: "Applied fix from PERF-1234" - URL must be preserved exactly)\n\n'
        f"Output format (STRICT NDJSON - one JSON object per line, NO arrays):\n"
        f'{{"block_id": "abc123", "reworded_content": "PyTest supports fixture dependency injection"}}\n'
        f'{{"block_id": "def456", "reworded_content": "The Textual framework is Python-specific"}}\n\n'
        f"CRITICAL:\n"
        f"- Output ONLY JSON objects, one per line\n"
        f"- NO array brackets [ ]\n"
        f"- Use block_id from XML attribute\n"
        f"- First character of output must be {{"
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
    from logsqueak.services.llm_helpers import format_chunks_for_llm

    # Generate XML for single block
    knowledge_block_xml = (
        f"<block id=\"{edited_content.block_id}\">\n"
        f"{edited_content.hierarchical_context}\n"
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
        "INPUT FORMAT:\n"
        "- <knowledge_blocks>: Content to integrate (block ID in XML attribute)\n"
        "- <pages>: Existing page structure\n"
        "  Each <page name=\"PageName\"> contains blocks with id attributes\n"
        "  CRITICAL: Use the page NAME (e.g. \"TDD\") NOT the block id (e.g. \"abc123\")\n\n"
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
        "FIELD DEFINITIONS:\n"
        "- knowledge_block_id: The block id from <knowledge_blocks>\n"
        "- target_page: The PAGE NAME from <page name=\"...\"> (example: \"TDD\" not \"abc123\")\n"
        "- target_block_id: The block id from <block id=\"...\"> within the page\n\n"
        "REQUIRED JSON SCHEMA (one object per line):\n"
        '{"knowledge_block_id": "string", "target_page": "string", "action": "add_section|add_under|replace|skip_exists", '
        '"target_block_id": "string|null", "target_block_title": "string|null", '
        '"confidence": 0.0-1.0, "reasoning": "string"}\n\n'
        'EXAMPLE: {"knowledge_block_id": "xyz789", "target_page": "TDD", "action": "add_under", '
        '"target_block_id": "abc123", "target_block_title": "Core concepts", "confidence": 0.85, '
        '"reasoning": "Fits under existing section"}\n\n'
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
        The returned stream includes all decisions (including skip_exists).
        Decisions are naturally grouped by block (one LLM call per block).
        Skip_exists decisions are displayed to the user for transparency.

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
