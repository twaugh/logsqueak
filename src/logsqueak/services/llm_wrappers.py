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
    outline: LogseqOutline,
    id_mapper
) -> str:
    """
    Generate XML-formatted blocks with hierarchical context for rewording.

    Args:
        edited_contents: List of knowledge blocks to reword
        outline: Original journal outline for looking up page properties
        id_mapper: LLMIDMapper for translating hybrid IDs to short IDs

    Returns:
        XML string with <blocks> containing each block with full context
    """
    xml_lines = ["<blocks>"]

    for edited in edited_contents:
        # Get short ID for this block
        short_id = id_mapper.to_short(edited.block_id)

        # Add block with the short ID in XML attribute
        xml_lines.append(f'  <block id="{short_id}">')

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
    from logsqueak.utils.llm_id_mapper import LLMIDMapper
    import structlog
    import copy

    logger = structlog.get_logger()
    indent_style = _detect_indent_style(journal_outline)

    # Build ID mapper for all blocks (hybrid ID → short ID)
    id_mapper = LLMIDMapper()

    def collect_block_ids(blocks: list[LogseqBlock]) -> None:
        """Recursively collect all block IDs for mapping."""
        for block in blocks:
            if block.block_id:
                id_mapper.add(block.block_id)
            collect_block_ids(block.children)

    collect_block_ids(journal_outline.blocks)

    # IMPORTANT: Deep copy the outline to avoid mutating the original
    outline_copy = copy.deepcopy(journal_outline)

    # Replace hybrid IDs with short IDs in the COPY
    def replace_ids_in_blocks(blocks: list[LogseqBlock]) -> None:
        """Recursively replace id:: properties with short IDs."""
        for block in blocks:
            if block.block_id:
                short_id = id_mapper.to_short(block.block_id)
                block.set_property("id", short_id)
            replace_ids_in_blocks(block.children)

    replace_ids_in_blocks(outline_copy.blocks)

    # Render the outline copy (expecting it to already have IDs from _augment_outline_with_ids)
    # IMPORTANT: Exclude frontmatter from LLM prompt (it's page-level metadata, not knowledge)
    # Temporarily clear frontmatter, render, then restore it
    original_frontmatter = outline_copy.frontmatter
    outline_copy.frontmatter = []
    journal_content = outline_copy.render()
    outline_copy.frontmatter = original_frontmatter

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
        f"- Each block's content is on the bullet line (starts with -)\n"
        f"- The id:: property appears on the NEXT indented line below the content\n"
        f"- Other properties (time-estimate::, etc.) also appear below the content\n\n"
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
        f'{{"block_id": "1", "insight": "PyTest supports fixture dependency injection", "confidence": 0.85}}\n'
        f'{{"block_id": "2", "insight": "The Textual framework uses CSS for styling", "confidence": 0.70}}\n\n'
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
    # Few-shot example to show id:: property location (critical for parsing)
    prompt = (
        f"Extract insights and output as NDJSON (one JSON object per line).\n"
        f"Each insight must be reworded and associated with exactly ONE block ID.\n\n"
        f"EXAMPLE (shows how to find the id:: property):\n"
        f"Input:\n"
        f"- tags::\n"
        f"  id:: 1\n"
        f"- Learned that Python type hints improve code quality\n"
        f"  id:: 2\n\n"
        f"Output:\n"
        f'{{\"block_id\": \"2\", \"insight\": \"Python type hints improve code quality\", \"confidence\": 0.85}}\n\n'
        f"Note: Block 1 (\"tags::\") has id:: 1 but no insight.\n"
        f"Block 2 (\"Learned that...\") has id:: 2 - the id:: is on the indented line below.\n\n"
        f"Now extract from the following journal entries:\n\n"
        f"{journal_content}"
    )

    async for chunk in llm_client.stream_ndjson(
        prompt=prompt,
        system_prompt=system_prompt,
        chunk_model=KnowledgeClassificationChunk,
        temperature=0.3,  # Low temperature for deterministic classification
        request_id="classify_blocks",
    ):
        # Translate short ID back to hybrid ID
        hybrid_id = id_mapper.try_to_hybrid(chunk.block_id)
        if hybrid_id is None:
            logger.warning(
                "llm_invalid_block_id",
                short_id=chunk.block_id,
                phase="classification"
            )
            continue  # Skip this chunk

        # Return chunk with hybrid ID
        chunk.block_id = hybrid_id
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
    from logsqueak.utils.llm_id_mapper import LLMIDMapper
    import structlog

    logger = structlog.get_logger()
    indent_style = _detect_indent_style(journal_outline)

    # Build ID mapper for edited content blocks
    id_mapper = LLMIDMapper()
    for edited in edited_contents:
        id_mapper.add(edited.block_id)

    xml_blocks = _generate_xml_blocks_for_rewording(edited_contents, journal_outline, id_mapper)

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
        f'{{"block_id": "1", "reworded_content": "PyTest supports fixture dependency injection"}}\n'
        f'{{"block_id": "2", "reworded_content": "The Textual framework is Python-specific"}}\n\n'
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
        # Translate short ID back to hybrid ID
        hybrid_id = id_mapper.try_to_hybrid(chunk.block_id)
        if hybrid_id is None:
            logger.warning(
                "llm_invalid_block_id",
                short_id=chunk.block_id,
                phase="rewording"
            )
            continue  # Skip this chunk

        # Return chunk with hybrid ID
        chunk.block_id = hybrid_id
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
    from logsqueak.utils.llm_id_mapper import LLMIDMapper
    from logsqueak.utils.context_cleaning import strip_id_property
    import structlog

    logger = structlog.get_logger()

    # Build ID mapper for both knowledge blocks and RAG target blocks
    id_mapper = LLMIDMapper()

    # Map the knowledge block
    id_mapper.add(edited_content.block_id)

    # Map all RAG candidate blocks
    for page_name, block_id, context in candidate_chunks:
        id_mapper.add(block_id)

    # Generate XML for single block with short ID
    # CRITICAL: Strip id:: property to avoid conflict with XML attribute
    knowledge_short_id = id_mapper.to_short(edited_content.block_id)
    cleaned_context = strip_id_property(edited_content.hierarchical_context)
    knowledge_block_xml = (
        f"<block id=\"{knowledge_short_id}\">\n"
        f"{cleaned_context}\n"
        f"</block>"
    )

    # Format hierarchical chunks using helper function
    pages_xml = format_chunks_for_llm(candidate_chunks, page_contents, id_mapper)

    xml_formatted_content = (
        f"<knowledge_blocks>\n"
        f"{knowledge_block_xml}\n"
        f"</knowledge_blocks>\n"
        f"{pages_xml}"
    )

    system_prompt = (
        "You are a JSON-only integration planner. Output ONLY valid NDJSON (one JSON object per line).\n"
        "Start output immediately with first JSON object (first character: '{').\n\n"
        "TASK: Find where knowledge blocks belong in existing pages.\n\n"
        "INPUT:\n"
        "- <knowledge_blocks>: Content to integrate (block id in XML attribute)\n"
        "- <pages>: Existing page structure\n"
        "  - Each <page name=\"PageName\"> contains blocks\n"
        "  - Blocks have id attributes\n"
        "  - Use page NAME (e.g. \"TDD\") as target_page, NOT block id\n\n"
        "DECISION TREE (follow in order):\n"
        "1. Is the candidate block semantically related to the knowledge?\n"
        "   NO → Skip this candidate (create no decision for it)\n"
        "   YES → Continue to step 2\n\n"
        "2. Does this block already contain this exact knowledge?\n"
        "   YES → action: skip_exists\n"
        "   NO → Continue to step 3\n\n"
        "3. Is confidence ≥ 0.30?\n"
        "   NO → Skip this candidate (create no decision for it)\n"
        "   YES → action: add_under (or add_section if no good parent)\n\n"
        "ACTIONS:\n"
        "- add_under: Add knowledge as child of this block\n"
        "- add_section: Add as new top-level section\n"
        "- skip_exists: Knowledge already exists here (duplicate)\n"
        "- replace: Update existing block with new knowledge\n\n"
        "OUTPUT SCHEMA (one object per line):\n"
        '{"knowledge_block_id": "1", "target_page": "PageName", "action": "add_under", '
        '"target_block_id": "2", "target_block_title": "Section title", '
        '"confidence": 0.85, "reasoning": "Why this location fits"}'
    )

    # User prompt with few-shot examples (leverages recency bias for Mistral-7B)
    prompt = (
        f"Generate integration decisions as NDJSON (one JSON object per line).\n\n"
        f"EXAMPLES:\n\n"
        f"Example 1 - skip_exists (duplicate already exists):\n"
        f"Knowledge: \"Python type hints improve code quality\"\n"
        f"Candidate: \"- Type hints are essential for maintainable Python code\"\n"
        f"Decision: {{\"knowledge_block_id\": \"1\", \"target_page\": \"Python\", \"action\": \"skip_exists\", "
        f"\"target_block_id\": \"2\", \"target_block_title\": \"Type hints\", \"confidence\": 0.95, "
        f"\"reasoning\": \"Duplicate content already exists\"}}\n\n"
        f"Example 2 - add_under (good semantic match):\n"
        f"Knowledge: \"Redis supports pub/sub messaging patterns\"\n"
        f"Candidate: \"- Database systems (parent block)\"\n"
        f"Decision: {{\"knowledge_block_id\": \"1\", \"target_page\": \"Databases\", \"action\": \"add_under\", "
        f"\"target_block_id\": \"3\", \"target_block_title\": \"Database systems\", \"confidence\": 0.80, "
        f"\"reasoning\": \"Redis is a database system\"}}\n\n"
        f"Example 3 - omit (not semantically related):\n"
        f"Knowledge: \"Kubernetes uses YAML for configuration\"\n"
        f"Candidate: \"- Python testing frameworks\"\n"
        f"Decision: (NO DECISION - candidate not related to Kubernetes)\n\n"
        f"Now generate decisions for the following:\n\n"
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
        # Translate short IDs back to hybrid IDs
        knowledge_hybrid_id = id_mapper.try_to_hybrid(chunk.knowledge_block_id)
        if knowledge_hybrid_id is None:
            logger.warning(
                "llm_invalid_knowledge_block_id",
                short_id=chunk.knowledge_block_id,
                phase="integration"
            )
            continue  # Skip this chunk

        # Translate target_block_id if present
        target_hybrid_id = None
        if chunk.target_block_id:
            target_hybrid_id = id_mapper.try_to_hybrid(chunk.target_block_id)
            if target_hybrid_id is None:
                logger.warning(
                    "llm_invalid_target_block_id",
                    short_id=chunk.target_block_id,
                    phase="integration"
                )
                continue  # Skip this chunk

        # Return chunk with hybrid IDs
        chunk.knowledge_block_id = knowledge_hybrid_id
        chunk.target_block_id = target_hybrid_id
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
