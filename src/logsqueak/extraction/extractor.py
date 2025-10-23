"""Knowledge extraction orchestrator.

This module implements the two-stage extraction process:
1. Stage 1: Extract knowledge blocks from journal (content + confidence)
2. Stage 2: Match knowledge to target pages using RAG candidates
"""

from datetime import date
from typing import List

from logsqueak.llm.client import ExtractionResult, LLMClient
from logsqueak.models.journal import JournalEntry
from logsqueak.models.knowledge import ActionType, KnowledgeBlock


class Extractor:
    """Orchestrates knowledge extraction from journal entries.

    Uses LLM to identify lasting knowledge (vs. activity logs) and
    extract knowledge blocks with confidence scores.
    """

    def __init__(self, llm_client: LLMClient):
        """Initialize the extractor.

        Args:
            llm_client: LLM client for knowledge extraction
        """
        self.llm_client = llm_client

    def extract_knowledge(
        self, journal: JournalEntry
    ) -> List[ExtractionResult]:
        """Extract knowledge blocks from journal entry (Stage 1).

        This is the first stage of the two-stage extraction process.
        The LLM identifies pieces of information with lasting value,
        filtering out temporary activity logs.

        Stage 2 (page selection) happens separately after RAG search.

        Args:
            journal: Journal entry to extract from

        Returns:
            List of extracted knowledge blocks with confidence scores

        Raises:
            LLMError: If LLM request fails
        """
        # Call LLM to extract knowledge blocks
        results = self.llm_client.extract_knowledge(
            journal_content=journal.raw_content,
            journal_date=journal.date,
        )

        return results


def create_knowledge_block(
    extraction: ExtractionResult,
    source_date: date,
    target_page: str,
    target_section: List[str] | None,
    suggested_action: ActionType,
) -> KnowledgeBlock:
    """Create a KnowledgeBlock from extraction result and page selection.

    This is a helper function used after both Stage 1 (extraction) and
    Stage 2 (page selection) have completed.

    Args:
        extraction: Stage 1 extraction result
        source_date: Journal date
        target_page: Selected target page (from Stage 2)
        target_section: Selected section path (from Stage 2)
        suggested_action: How to integrate (from Stage 2)

    Returns:
        Complete KnowledgeBlock ready for integration
    """
    return KnowledgeBlock(
        content=extraction.content,
        source_date=source_date,
        confidence=extraction.confidence,
        target_page=target_page,
        target_section=target_section,
        suggested_action=suggested_action,
    )
