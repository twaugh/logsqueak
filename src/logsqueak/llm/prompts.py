"""Shared LLM prompt templates and builders.

This module contains the canonical prompt templates used across the system.
All prompt construction should use these functions to maintain DRY principle.
"""

from textwrap import dedent
from typing import List

from logsqueak.llm.client import PageCandidate


def build_page_selection_messages(
    knowledge_content: str,
    candidates: List[PageCandidate],
    indent_str: str = "  ",
) -> List[dict]:
    """Build Stage 2 (page selection) prompt messages.

    This is the canonical implementation of the page selection prompt.
    Used by both the LLM provider and the token counter.

    Args:
        knowledge_content: The extracted knowledge text
        candidates: Candidate pages from RAG search
        indent_str: Indentation string detected from source (e.g., "  ", "\t")

    Returns:
        List of message dicts for chat completion API
    """
    # Describe the indentation style for the LLM
    indent_desc = repr(indent_str)

    system_prompt = dedent(f"""
        You are a knowledge organization assistant for a personal knowledge base.

        Your task is to determine the best location for a piece of knowledge within existing pages.

        FORMAT: Pages are in Logseq-flavored Markdown with these conventions:
        - Bullets start with "- " and can be nested with indentation
        - Page links use [[Page Name]] syntax
        - Headings can appear as bullets (e.g., "- ## Section Name")
        - Indentation uses {indent_desc} per level to indicate hierarchy

        FRONTMATTER vs SECTIONS (IMPORTANT):
        - Page previews may include <frontmatter> tags containing page-level properties
        - Frontmatter properties (like "type::", "area::", "tags::") are NOT sections
        - Frontmatter provides CONTEXT about the page but CANNOT be used as target_section
        - Only BULLET BLOCKS (lines starting with "- ") are valid sections
        - When frontmatter appears, use it to understand the page's purpose, but select actual bullet content as target_section

        You will receive:
        1. Knowledge to organize (in <knowledge_to_organize> tags)
        2. Top candidate pages from semantic search (in <candidate_pages> tags)

        Each candidate includes:
        - page_name: The name of the page
        - similarity_score: How semantically similar it is (0.0-1.0)
        - preview: A preview of the page content (may include <frontmatter> tags)
        - rank: Ranking by similarity (1 = most similar)

        You must select:
        - Which page is most appropriate (use the page_name)
        - Which section within that page (if applicable)
        - Whether to add as child bullet or create new section

        Return a JSON object with this structure:
        {{
          "target_page": "Page Name",
          "target_section": ["Section", "Subsection"],
          "suggested_action": "add_child",
          "reasoning": "Brief explanation of your choice"
        }}

        CRITICAL - target_section format:
        - target_section contains the text content after the bullet marker ("- ")
        - Strip the bullet marker but keep everything else (including "##" if present)
        - target_section can be null if knowledge should go at page root

        Example - if the page has this content:
          - ## People
          - Projects
            - ## Active

        Then valid target_section values are:
          null - for page root (no specific section)
          ["## People"] - NOT ["- ## People"] or ["People"]
          ["Projects"] - NOT ["- Projects"]
          ["Projects", "## Active"] - NOT ["Projects", "Active"] or ["Projects", "- ## Active"]

        Notes:
        - target_section can be null if knowledge should go at page root
        - target_section MUST ONLY contain bullet block content, NEVER frontmatter properties
        - suggested_action must be either "add_child" or "create_section"
        - Use "add_child" when a suitable section exists
        - Use "create_section" when a new organizational section is needed
        - Higher similarity scores indicate better semantic matches
    """).strip()

    # Format candidates for prompt using XML structure
    candidates_xml = []
    for i, candidate in enumerate(candidates, 1):
        candidates_xml.append(
            f"<candidate rank=\"{i}\">\n"
            f"  <page_name>{candidate.page_name}</page_name>\n"
            f"  <similarity_score>{candidate.similarity_score:.2f}</similarity_score>\n"
            f"  <preview>\n{candidate.preview}\n  </preview>\n"
            f"</candidate>"
        )

    user_prompt = (
        f"<knowledge_to_organize>\n{knowledge_content}\n</knowledge_to_organize>\n\n"
        f"<candidate_pages>\n{chr(10).join(candidates_xml)}\n</candidate_pages>\n\n"
        f"Select the best page and section for this knowledge."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_decider_messages(
    knowledge_text: str,
    candidate_chunks: List[dict],
) -> List[dict]:
    """Build Phase 3 (Decider) prompt messages.

    This is the canonical implementation of the Decider prompt.
    The Decider LLM decides what action to take with the knowledge.

    Args:
        knowledge_text: The full-context knowledge text to integrate
        candidate_chunks: List of candidate chunks from Phase 2 RAG
                         Each dict has: page_name, target_id, content, similarity_score

    Returns:
        List of message dicts for chat completion API
    """
    system_prompt = dedent("""
        You are a knowledge integration assistant for a personal knowledge base.

        Your task is to decide what action to take with a piece of knowledge from a journal entry.

        You will receive:
        1. Knowledge to integrate (in <knowledge> tags)
        2. Candidate chunks from semantic search (in <candidate_chunks> tags)

        Each candidate chunk represents a specific block in the knowledge base that might be relevant.
        Candidates include:
        - page_name: The page containing this block
        - target_id: The unique ID of this block
        - content: The current content of the block
        - similarity_score: How semantically similar it is (0.0-1.0)

        You must decide ONE of these actions:

        1. IGNORE_ALREADY_PRESENT - The knowledge already exists in the target location
           Use when: A candidate block contains essentially the same information
           Returns: target_id of the block with existing knowledge

        2. IGNORE_IRRELEVANT - The knowledge doesn't relate to any candidate
           Use when: None of the candidates are appropriate targets for this knowledge
           Returns: target_id null, page_name null

        3. UPDATE - Modify an existing block's content
           Use when: The knowledge updates or refines existing information in a candidate
           Returns: target_id of block to update

        4. APPEND_CHILD - Add as a child bullet under an existing block
           Use when: The knowledge adds detail/context to an existing candidate block
           Returns: target_id of parent block

        5. APPEND_ROOT - Add at page root level
           Use when: The knowledge relates to a page but doesn't fit under any specific block
           Returns: target_id null, page_name of the target page

        Return a JSON object with this structure:
        {{
          "action": "APPEND_CHILD",
          "target_id": "65f3a8e0-1234-5678-9abc-def012345678",
          "reasoning": "Brief explanation of your decision"
        }}

        Or for APPEND_ROOT:
        {{
          "action": "APPEND_ROOT",
          "target_id": null,
          "page_name": "Project X",
          "reasoning": "Brief explanation of your decision"
        }}

        CRITICAL RULES:
        - If no candidates are provided, use IGNORE_IRRELEVANT
        - Higher similarity scores indicate better semantic matches
        - IGNORE_ALREADY_PRESENT requires very high confidence (near-duplicate content)
        - IGNORE_IRRELEVANT means none of the candidates are appropriate (not that knowledge is irrelevant)
        - UPDATE should only be used when truly updating/correcting existing info
        - When unsure between UPDATE and APPEND_CHILD, prefer APPEND_CHILD (safer)
        - APPEND_ROOT requires page_name - choose the most relevant page from candidates
        - target_id is required for: IGNORE_ALREADY_PRESENT, UPDATE, APPEND_CHILD
        - page_name is required only for: APPEND_ROOT
        - action must be one of: IGNORE_ALREADY_PRESENT, IGNORE_IRRELEVANT, UPDATE, APPEND_CHILD, APPEND_ROOT
    """).strip()

    # Format candidates for prompt using XML structure
    if not candidate_chunks:
        chunks_xml = "<candidate_chunks>\n(No candidates found - consider APPEND_ROOT)\n</candidate_chunks>"
    else:
        chunks_xml_items = []
        for i, chunk in enumerate(candidate_chunks[:10], 1):  # Limit to top 10
            chunks_xml_items.append(
                f"<chunk rank=\"{i}\">\n"
                f"  <page_name>{chunk['page_name']}</page_name>\n"
                f"  <target_id>{chunk['target_id']}</target_id>\n"
                f"  <similarity_score>{chunk['similarity_score']:.3f}</similarity_score>\n"
                f"  <content>\n{chunk['content']}\n  </content>\n"
                f"</chunk>"
            )
        chunks_xml = f"<candidate_chunks>\n{chr(10).join(chunks_xml_items)}\n</candidate_chunks>"

    user_prompt = (
        f"<knowledge>\n{knowledge_text}\n</knowledge>\n\n"
        f"{chunks_xml}\n\n"
        f"Decide what action to take with this knowledge."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_reworder_messages(knowledge_full_text: str) -> List[dict]:
    """Build Phase 3.2 (Reworder) prompt messages.

    This is the canonical implementation of the Reworder prompt.
    The Reworder LLM transforms journal-specific knowledge into clean,
    evergreen content suitable for integration into pages.

    Args:
        knowledge_full_text: The full-context knowledge text from journal

    Returns:
        List of message dicts for chat completion API
    """
    system_prompt = dedent("""
        You are a knowledge rephrasing assistant for a personal knowledge base.

        Your task is to transform knowledge extracted from journal entries into clean,
        evergreen content suitable for integration into permanent pages.

        FORMAT: The input is Logseq-flavored Markdown from a journal entry.
        - Bullets start with "- " and can be nested with indentation
        - Page links use [[Page Name]] syntax - MUST be preserved exactly
        - Block references use (((block-id))) syntax - MUST be preserved exactly
        - The deepest nested bullet is the knowledge to extract
        - Parent bullets provide context for understanding
        - Your output will be inserted as a single bullet in the target page (no bullet marker needed)

        You will receive knowledge text that includes:
        - Hierarchical bullet structure showing parent context
        - The deepest bullet contains the specific knowledge to preserve
        - Possibly journal-specific language like "today", "this morning", etc.

        Your job is to:
        1. Remove journal-specific temporal context ("today", "this morning" -> use past tense or timeless phrasing)
        2. PRESERVE ALL page links ([[Page Name]]) and block references (((block-id))) - these are critical
        3. Preserve technical details, decisions, and rationale
        4. Create standalone, timeless content that makes sense without the journal context
        5. Keep the content concise and focused on the lasting knowledge
        6. Output plain text content suitable for a single bullet point (no bullet marker needed)

        DO NOT:
        - Add new information not present in the source
        - Remove or modify page links [[Page Name]] or block references (((block-id)))
        - Change the meaning or intent of the knowledge
        - Add flowery language or unnecessary elaboration
        - Include bullet markers (-, *) in your output - just the content

        Return ONLY the rephrased content as plain text. Do not wrap it in JSON, XML, or any other format.
        Output the clean, evergreen markdown text directly.

        EXAMPLES:

        Example 1 - Use parent context to create standalone content:
        Input:
        - Working on [[RHEL Documentation]]
          - Updated security guidelines
            - Added section on container scanning
        Output: Added section on container scanning to [[RHEL Documentation]] security guidelines

        Example 2 - Remove temporal language, preserve technical detail:
        Input:
        - Today learned that [[Python]] asyncio has a subtle bug with task cancellation - need to use shield()
        Output: [[Python]] asyncio has a subtle bug with task cancellation - use shield() to avoid it

        Example 3 - Preserve attribution from parent, remove temporal context:
        Input:
        - Met with [[Alice]] this morning
          - She suggested using [[ChromaDB]] for the vector store instead of FAISS
        Output: Suggestion from [[Alice]]: Use [[ChromaDB]] for vector store instead of FAISS

        Example 4 - Extract decision from nested context:
        Input:
        - Working on [[logsqueak]]
          - Pipeline refactoring
            - Decided to use 5-phase approach instead of 2-stage
        Output: [[logsqueak]] pipeline: Decided to use 5-phase approach instead of 2-stage
    """).strip()

    user_prompt = f"Rephrase this knowledge into clean, evergreen content:\n\n{knowledge_full_text}"

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
