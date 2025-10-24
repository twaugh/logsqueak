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
