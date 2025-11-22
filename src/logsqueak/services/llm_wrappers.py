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
        f"You are a knowledge classifier. Output ONLY NDJSON (one JSON object per line).\n"
        f"NO markdown code blocks. NO explanations. First character must be '{{'.\n\n"
        f"TASK: Classify journal blocks as knowledge (valuable long-term) or not.\n\n"
        f"OUTPUT FORMAT (reasoning BEFORE confidence):\n"
        f'{{"block_id": "ID", "reasoning": "explanation here", "confidence": 0.85}}\n\n'
        f"CRITICAL RULES:\n"
        f"1. Provide reasoning FIRST (chain-of-thought), confidence second\n"
        f"2. Output each block_id at most once\n"
        f"3. Only output blocks containing lasting knowledge\n"
        f"4. Skip blocks that are just activities, meetings, or status updates\n\n"
        f"KNOWLEDGE DEFINITION:\n"
        f"Content valuable long-term: learnings, best practices, solutions, patterns, insights.\n"
        f"NOT knowledge: meetings, tasks, status updates, activity logs, temporary notes.\n\n"
        f"REASONING GUIDELINES:\n"
        f"Explain WHY the block contains lasting knowledge:\n"
        f"- What makes this valuable beyond today?\n"
        f"- What fundamental concept or pattern does it capture?\n"
        f"- How does it connect to broader understanding?\n\n"
        f"INPUT FORMAT:\n"
        f"Markdown bullets with {indent_style}. The id:: property appears indented below each block.\n"
        f"Example:\n"
        f"- Block content here\n"
        f"  id:: 123\n\n"
        f"CONFIDENCE SCORING:\n"
        f"0.9 = fundamental knowledge, 0.7 = tool-specific, 0.5 = contextual"
    )

    # User prompt: Instruction BEFORE data (critical for attention)
    # Structural markup helps Mistral's attention mechanism
    prompt = (
        f"Classify journal blocks as knowledge and output as NDJSON (one JSON object per line).\n\n"
        f"<examples>\n\n"
        f"Example 1 - Basic classification:\n"
        f"Input:\n"
        f"- Database indexes on foreign keys significantly speed up JOIN queries\n"
        f"  id:: 1\n\n"
        f"Output:\n"
        f'{{\"block_id\": \"1\", \"reasoning\": \"Captures a fundamental database optimization principle with broad applicability to relational databases. Technical fact with lasting value.\", \"confidence\": 0.85}}\n\n'
        f"Example 2 - Context from parent block:\n"
        f"Input:\n"
        f"- Explored asyncio.gather() behavior\n"
        f"  id:: 2\n"
        f"  - It preserves execution order\n"
        f"    id:: 3\n\n"
        f"Output:\n"
        f'{{\"block_id\": \"3\", \"reasoning\": \"Documents specific asyncio behavior that is fundamental to concurrent programming patterns. The parent context shows this came from exploration, but the insight itself is durable.\", \"confidence\": 0.90}}\n\n'
        f"Example 3 - Multiple knowledge blocks:\n"
        f"Input:\n"
        f"- Docker containers should be stateless\n"
        f"  id:: 4\n"
        f"  - This makes horizontal scaling easier\n"
        f"    id:: 5\n"
        f"  - Enables zero-downtime rolling updates\n"
        f"    id:: 6\n\n"
        f"Output:\n"
        f'{{\"block_id\": \"4\", \"reasoning\": \"Core architectural principle for container design with lasting relevance across cloud platforms\", \"confidence\": 0.95}}\n'
        f'{{\"block_id\": \"5\", \"reasoning\": \"Explains a key operational benefit of stateless design - essential DevOps knowledge\", \"confidence\": 0.90}}\n'
        f'{{\"block_id\": \"6\", \"reasoning\": \"Connects stateless architecture to deployment strategy - practical infrastructure knowledge\", \"confidence\": 0.90}}\n\n'
        f"Example 4 - TODO/task (not knowledge):\n"
        f"Input:\n"
        f"- TODO: Research Git rebase vs merge best practices\n"
        f"  id:: 7\n\n"
        f"Output:\n"
        f"(no output - future intention, not captured knowledge)\n\n"
        f"Example 5 - Temporal framing with knowledge:\n"
        f"Input:\n"
        f"- Today while debugging a shell script, discovered that set -euo pipefail catches errors early\n"
        f"  id:: 8\n\n"
        f"Output:\n"
        f'{{\"block_id\": \"8\", \"reasoning\": \"Documents a bash best practice for error handling. The temporal framing (\\\"today\\\", \\\"discovered\\\") describes how the knowledge was acquired, but the insight about set -euo pipefail is durable and valuable.\", \"confidence\": 0.78}}\n\n'
        f"</examples>\n\n"
        f"CRITICAL: Reasoning first, then confidence. Output NDJSON only.\n\n"
        f"<task>\n"
        f"{journal_content}\n"
        f"</task>\n\n"
        f"Output first JSON object now (start with '{{'): "
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
        f"   - Wiki links: Keep [[Page]] double-bracket syntax EXACTLY (do NOT convert to plain text)\n"
        f"   - Markdown links: Copy [text](url) or [id|text](url) character-for-character\n"
        f"   - NEVER drop URLs or convert links to plain text\n"
        f"   - Code snippets, commands, file paths unchanged\n"
        f"   - Specific names, IDs, version numbers preserved\n"
        f"   - Do NOT summarize, paraphrase, or simplify technical content\n"
        f"7. Reword ONLY the deepest block (target), not parent blocks\n\n"
        f"Example transformations:\n\n"
        f"Example 1 - Preserve [[Wiki]] links:\n"
        f'Parent: "DONE Review [[Elixir]] documentation for new features"\n'
        f'Child: "[[Elixir]] 1.15 introduced some interesting improvements"\n'
        f'→ Reword child to: "[[Elixir]] 1.15 introduces several improvements to type system and pattern matching"\n'
        f'(CRITICAL: Keep [[Elixir]] wiki link exactly as-is, do NOT convert to "Elixir")\n\n'
        f"Example 2 - Multiple wiki links:\n"
        f'Parent: "Learned about [[Svelte]] best practices"\n'
        f'Child: "[[Svelte]] reactive declarations are fundamental"\n'
        f'→ Reword child to: "[[Svelte]] reactive declarations form the foundation of state management"\n'
        f'(CRITICAL: Keep [[Svelte]] wiki link intact)\n\n'
        f"Example 3 - Pronoun resolution:\n"
        f'Parent: "Tested Array.reduce() behavior"\n'
        f'Child: "It accumulates values from left to right"\n'
        f'→ Reword child to: "Array.reduce() accumulates values from left to right in JavaScript"\n'
        f'(NOT: "Testing revealed that it accumulates values")\n\n'
        f"Example 4 - Markdown links:\n"
        f'Parent: "API optimization work"\n'
        f'Child: "Applied fix from [PERF-5678|response caching](https://issues.example.com/PERF-5678)"\n'
        f'→ Reword child to: "API response caching improved via [PERF-5678|response caching](https://issues.example.com/PERF-5678)"\n'
        f'(CRITICAL: Preserve entire [text](URL) exactly)\n\n'
        f"Output format (STRICT NDJSON - one JSON object per line, NO arrays):\n"
        f'{{"block_id": "1", "reworded_content": "[[MongoDB]] supports aggregation pipelines for complex queries"}}\n'
        f'{{"block_id": "2", "reworded_content": "The [[Terraform]] state file tracks [[Infrastructure]] resources"}}\n\n'
        f"CRITICAL:\n"
        f"- Output ONLY JSON objects, one per line\n"
        f"- NO array brackets [ ]\n"
        f"- Use block_id from XML attribute\n"
        f"- First character of output must be {{"
    )

    # User prompt: Instruction BEFORE data (critical for attention)
    prompt = (
        f"Reword the deepest block in each XML block and output as NDJSON (one JSON object per line).\n\n"
        f"CRITICAL REMINDERS:\n"
        f"1. Keep ALL [[Page]] wiki links EXACTLY as written - do NOT convert to plain text\n"
        f"2. Keep ALL [text](url) markdown links character-for-character\n"
        f"3. Remove temporal words (today, learned, discovered) but keep ALL links intact\n\n"
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


def _extract_block_title_from_context(hierarchical_context: str) -> str:
    """
    Extract a human-readable title from hierarchical context.

    Takes the first line of the context (deepest block's first line).

    Args:
        hierarchical_context: Full hierarchical context string from RAG

    Returns:
        First line of content, truncated to 50 chars
    """
    # Get first non-empty line
    lines = [line.strip() for line in hierarchical_context.split('\n') if line.strip()]
    if not lines:
        return "Unknown block"

    first_line = lines[0]

    # Strip markdown bullet and whitespace
    first_line = first_line.lstrip("- \t")

    # Truncate to 50 chars
    if len(first_line) > 50:
        return first_line[:47] + "..."

    return first_line


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

    # Build ID mapper for RAG candidate blocks only
    # (knowledge block ID is not sent to LLM - it's set in the handler)
    id_mapper = LLMIDMapper()

    # Build lookup: block_id -> hierarchical_context (for title extraction)
    context_by_block_id: dict[str, str] = {}

    # Map all RAG candidate blocks
    for page_name, block_id, context in candidate_chunks:
        id_mapper.add(block_id)
        context_by_block_id[block_id] = context

    # Generate XML for single block (no ID needed - not asked from LLM)
    # CRITICAL: Strip id:: property to keep prompt clean
    cleaned_context = strip_id_property(edited_content.hierarchical_context)

    # Format hierarchical chunks using helper function
    pages_xml = format_chunks_for_llm(candidate_chunks, page_contents, id_mapper)

    xml_formatted_content = (
        f"<knowledge_block>\n"
        f"{cleaned_context}\n"
        f"</knowledge_block>\n\n"
        f"{pages_xml}"
    )

    system_prompt = (
        "You are a knowledge organization expert specializing in Logseq integration.\n"
        "Output ONLY NDJSON (one JSON object per line).\n"
        "NO markdown code blocks. NO explanations. First character must be '{'.\n\n"
        "TASK: Decide where a knowledge block belongs in existing pages.\n\n"
        "OUTPUT FORMAT (reasoning BEFORE action and confidence):\n"
        '{"target_page": "PageName", "reasoning": "explanation here", '
        '"action": "add_under", "target_block_id": "2", "confidence": 0.85}\n\n'
        "CRITICAL RULES:\n"
        "1. Provide reasoning FIRST (chain-of-thought), then action and confidence\n"
        "2. ALWAYS output at least ONE decision (find the best match even if confidence is low)\n"
        "3. PREFER outputting 2-3 decisions when multiple valid placements exist\n"
        "4. Output decisions for semantically related candidates (confidence ≥ 0.30)\n"
        "5. If ALL candidates are below 0.30, still output the BEST match with its actual confidence\n\n"
        "INPUT:\n"
        "- <knowledge_block>: Single block to integrate\n"
        "- <pages>: Existing page structure (each <page name=\"PageName\"> contains blocks with id attributes)\n"
        "- Use page NAME as target_page, NOT block id\n\n"
        "REASONING GUIDELINES:\n"
        "For each candidate, think through:\n"
        "1. Semantic relationship: How does the knowledge block relate to this page/section?\n"
        "2. Information comparison: Does existing content cover the same points?\n"
        "   - Same information → skip_exists\n"
        "   - More information in new block → replace\n"
        "   - Complementary information → add_under or add_section\n"
        "3. Placement quality: Why is this the right location?\n"
        "4. Confidence assessment: How strong is the semantic match?\n\n"
        "ACTIONS:\n"
        "- add_under: Add as child of existing block (requires target_block_id)\n"
        "- add_section: Add as new top-level section (no target_block_id)\n"
        "- skip_exists: Existing block has SAME OR MORE information (requires target_block_id)\n"
        "- replace: New knowledge has MORE information than existing (requires target_block_id)\n\n"
        "CONFIDENCE SCORING (0.0-1.0):\n"
        "- 0.85-1.0: Perfect semantic match (same topic/subtopic)\n"
        "- 0.60-0.84: Strong thematic fit (related concepts)\n"
        "- 0.30-0.59: Weak but valid connection\n"
        "- Below 0.30: Not related enough (skip this candidate)"
    )

    # User prompt with few-shot examples (leverages recency bias for Mistral-7B)
    prompt = (
        f"Decide where to integrate this knowledge block. Think through each candidate systematically.\n\n"
        f"<examples>\n\n"
        f"Example 1 - skip_exists (existing has equivalent information):\n"
        f"Knowledge: Python type hints improve code quality and catch bugs early\n"
        f"Candidate Page: Python\n"
        f"Existing Block: Type hints are essential for maintainable Python code and early error detection\n\n"
        f"Output:\n"
        f'{{"target_page": "Python", '
        f'"reasoning": "Existing block already covers the core insight about type hints improving code quality and catching bugs. The knowledge is semantically equivalent - both emphasize maintainability and error prevention.", '
        f'"action": "skip_exists", "target_block_id": "2", "confidence": 0.95}}\n\n'
        f"Example 2 - replace (new has MORE specific information):\n"
        f"Knowledge: Redis supports pub/sub messaging with channel patterns, wildcards, and pattern-based subscriptions\n"
        f"Candidate Page: Databases\n"
        f"Existing Block: Redis has pub/sub support\n\n"
        f"Output:\n"
        f'{{"target_page": "Databases", '
        f'"reasoning": "The existing block mentions pub/sub but lacks crucial implementation details. The new knowledge adds valuable specifics about channel patterns, wildcards, and pattern-based subscriptions. This is a clear case where the new content provides more actionable information.", '
        f'"action": "replace", "target_block_id": "3", "confidence": 0.90}}\n\n'
        f"Example 3 - add_under (complementary detail for existing concept):\n"
        f"Knowledge: Redis sorted sets enable leaderboard implementations with O(log n) insertion and range queries\n"
        f"Candidate Page: Databases\n"
        f"Existing Block: Redis data structures provide specialized operations\n\n"
        f"Output:\n"
        f'{{"target_page": "Databases", '
        f'"reasoning": "The existing block discusses Redis data structures broadly. The new knowledge provides a specific use case (leaderboards) with performance characteristics. This is complementary information that fits naturally as a child example under the general data structures concept.", '
        f'"action": "add_under", "target_block_id": "4", "confidence": 0.85}}\n\n'
        f"Example 4 - add_section (no semantically related existing content):\n"
        f"Knowledge: GraphQL subscriptions enable real-time data updates using WebSockets\n"
        f"Candidate Page: APIs\n"
        f"Existing Blocks: REST endpoint design, API versioning strategies\n\n"
        f"Output:\n"
        f'{{"target_page": "APIs", '
        f'"reasoning": "The page covers API concepts but has no existing content about GraphQL or real-time patterns. This knowledge introduces a new subtopic (GraphQL subscriptions) that relates to the broader API theme but doesn\'t fit as a child of REST or versioning content.", '
        f'"action": "add_section", "confidence": 0.75}}\n\n'
        f"Example 5 - multiple valid insertion points:\n"
        f"Knowledge: PostgreSQL EXPLAIN ANALYZE shows actual execution time and row counts\n"
        f"Candidate Page: Databases\n"
        f"Existing Blocks:\n"
        f"  - Block 5: PostgreSQL performance optimization\n"
        f"  - Block 6: Query analysis tools\n\n"
        f"Output:\n"
        f'{{"target_page": "Databases", '
        f'"reasoning": "EXPLAIN ANALYZE is a PostgreSQL-specific performance tool. It fits naturally under the PostgreSQL performance optimization section as a concrete technique.", '
        f'"action": "add_under", "target_block_id": "5", "confidence": 0.92}}\n'
        f'{{"target_page": "Databases", '
        f'"reasoning": "EXPLAIN ANALYZE is also a query analysis tool that could fit in the broader tools section. However, confidence is slightly lower since this is less specific than the PostgreSQL section.", '
        f'"action": "add_under", "target_block_id": "6", "confidence": 0.78}}\n\n'
        f"Example 6 - best match even with weak semantic fit:\n"
        f"Knowledge: Git rebase rewrites commit history by replaying commits on a new base\n"
        f"Candidate Pages: Version Control, Software Architecture\n"
        f"Existing: Version Control has basic git commands; Software Architecture has general design principles\n\n"
        f"Output:\n"
        f'{{"target_page": "Version Control", '
        f'"reasoning": "While the knowledge is about a specific git feature, Version Control is the most semantically related page. The confidence is moderate because existing content only covers basic commands, but this is still the best available match.", '
        f'"action": "add_section", "confidence": 0.55}}\n\n'
        f"Example 7 - finding best placement among multiple decent options:\n"
        f"Knowledge: Async/await syntax makes asynchronous code more readable\n"
        f"Candidate Pages: JavaScript, Python, Programming Patterns\n\n"
        f"Output:\n"
        f'{{"target_page": "JavaScript", '
        f'"reasoning": "JavaScript async/await is a language feature. This fits well as a section about modern JavaScript asynchronous patterns.", '
        f'"action": "add_section", "confidence": 0.88}}\n'
        f'{{"target_page": "Python", '
        f'"reasoning": "Python also has async/await syntax. Could fit here as well, though the knowledge doesn\'t specify the language.", '
        f'"action": "add_section", "confidence": 0.85}}\n'
        f'{{"target_page": "Programming Patterns", '
        f'"reasoning": "Async/await is a general pattern across languages. Lower confidence as it\'s more abstract than language-specific pages.", '
        f'"action": "add_section", "confidence": 0.62}}\n\n'
        f"</examples>\n\n"
        f"CRITICAL REMINDERS:\n"
        f"1. Reasoning first, then action and confidence\n"
        f"2. Output at least ONE decision (required)\n"
        f"3. Output 2-3 decisions when multiple good placements exist (preferred)\n"
        f"4. Output NDJSON only - start with '{{'\n\n"
        f"<task>\n"
        f"{xml_formatted_content}\n"
        f"</task>\n\n"
        f"Output first JSON object now (start with '{{'):"
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
        # Set knowledge_block_id from context (not from LLM output)
        chunk.knowledge_block_id = edited_content.block_id

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

        # Set translated target_block_id
        chunk.target_block_id = target_hybrid_id

        # Derive target_block_title from RAG context (if target_block_id exists)
        if target_hybrid_id and target_hybrid_id in context_by_block_id:
            chunk.target_block_title = _extract_block_title_from_context(
                context_by_block_id[target_hybrid_id]
            )

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
