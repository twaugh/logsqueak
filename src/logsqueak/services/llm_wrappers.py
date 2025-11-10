"""LLM wrapper functions that wrap LLMClient.stream_ndjson with specific prompts."""

from typing import AsyncIterator
from logseq_outline.parser import LogseqOutline, LogseqBlock
from logseq_outline.context import generate_full_context
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
        journal_outline: Parsed journal entry outline

    Yields:
        KnowledgeClassificationChunk for each block classified as knowledge

    Example:
        >>> async for chunk in classify_blocks(client, outline):
        ...     print(f"Block {chunk.block_id}: {chunk.confidence}")
    """
    indent_style = _detect_indent_style(journal_outline)
    journal_content = journal_outline.render()

    system_prompt = (
        f"You are an AI assistant that identifies lasting knowledge in Logseq journal entries. "
        f"Logseq uses markdown with indented bullets ({indent_style}). "
        f"Blocks can have continuation lines (indented text without bullets) and properties (key:: value). "
        f"Each block has an id:: property for identification.\n\n"
        f"Analyze the journal entry and return ONLY blocks that contain lasting insights worth preserving "
        f"(skip temporal events/tasks/activities). When a child block contains knowledge that depends on parent context, "
        f"return ONLY the specific child block ID - not all parent blocks. "
        f"Output one JSON object per line (NDJSON format), not a JSON array. "
        f"Each object must have: block_id (string), confidence (float 0-1), reason (string explaining why this is knowledge worth preserving)."
    )

    prompt = f"Analyze this Logseq journal entry:\n\n{journal_content}"

    async for chunk in llm_client.stream_ndjson(
        prompt=prompt,
        system_prompt=system_prompt,
        chunk_model=KnowledgeClassificationChunk
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
        f"You are an AI assistant that transforms journal-style content into evergreen knowledge. "
        f"You will receive blocks wrapped in XML, each with full parent context including page properties. "
        f"The blocks are properly indented following Logseq's indentation ({indent_style}).\n\n"
        f"Remove temporal context (dates, 'today', 'yesterday'), convert first-person to third-person or neutral, "
        f"and make the content timeless. Preserve technical details and insights. "
        f"Use parent context to understand meaning, but only reword the specific block "
        f"(the deepest/last block in each context hierarchy).\n\n"
        f"IMPORTANT: Resolve all pronouns and references using parent context. "
        f"If the block says 'this' or 'it', replace with the actual subject from parent context. "
        f"Example: if parent says 'Tried Textual framework' and child says 'This is Python-specific', "
        f"reword as 'The Textual framework is Python-specific'.\n\n"
        f"Output one JSON object per line (NDJSON format), not a JSON array. "
        f"Each object must have: block_id (string), reworded_content (string)."
    )

    prompt = xml_blocks

    async for chunk in llm_client.stream_ndjson(
        prompt=prompt,
        system_prompt=system_prompt,
        chunk_model=ContentRewordingChunk
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
        "You are an AI assistant that decides where to integrate knowledge into a Logseq knowledge base. "
        "You will receive knowledge blocks with their original journal context and candidate target pages with their structure.\n\n"
        "RELEVANCE FILTERING:\n"
        "- Only output decisions where confidence ≥ 0.30 (30%)\n"
        "- Only output decisions where semantic connection is clear and actionable\n"
        "- Maximum 2 decisions per (knowledge_block, target_page) pair - choose the 2 best integration points if multiple locations are relevant\n"
        "- If a candidate page is not relevant, omit it entirely (do not output a decision for it)\n"
        "- If NO candidate pages meet the threshold, output nothing (empty stream)\n\n"
        "DUPLICATE DETECTION:\n"
        "- FIRST check if similar/identical knowledge already exists in each candidate page\n"
        "- If duplicate found, use action='skip_exists' with target_block_id pointing to existing block\n"
        "- For skip_exists: provide reasoning explaining why it's a duplicate\n"
        "- For skip_exists: set target_block_title to describe location (e.g., \"Already exists at 'Section Name'\")\n\n"
        "For each relevant NEW integration, choose the best action:\n"
        "- 'add_section': Create new top-level section on the page\n"
        "- 'add_under': Add as child under an existing block (provide target_block_id)\n"
        "- 'replace': Replace an existing block's content (provide target_block_id)\n"
        "- 'skip_exists': Knowledge already exists in target page (provide target_block_id of existing block)\n\n"
        "OUTPUT ORDERING (CRITICAL):\n"
        "- Output ALL decisions for a given knowledge_block_id CONSECUTIVELY before moving to the next knowledge block\n"
        "- Example correct order: [blockA-decision1, blockA-decision2, blockB-decision1, blockC-decision1, blockC-decision2]\n"
        "- Example INCORRECT order: [blockA-decision1, blockB-decision1, blockA-decision2] ← DO NOT DO THIS\n"
        "- This ordering is required for the system to batch decisions correctly\n\n"
        "Output one JSON object per line (NDJSON format), not a JSON array. "
        "Each object must have: knowledge_block_id (string), target_page (string), action (string), "
        "target_block_id (string or null), target_block_title (string or null), confidence (float 0-1), "
        "reasoning (string explaining why this integration makes sense or why it's a duplicate)."
    )

    prompt = xml_formatted_content

    async for chunk in llm_client.stream_ndjson(
        prompt=prompt,
        system_prompt=system_prompt,
        chunk_model=IntegrationDecisionChunk
    ):
        yield chunk
