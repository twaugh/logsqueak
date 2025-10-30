"""LLM client interface for knowledge extraction.

This module defines the abstract interface for LLM providers used in the
two-stage extraction process:
1. Extract knowledge blocks from journal entries
2. Select target pages and sections from RAG candidates
"""

from abc import ABC, abstractmethod
from datetime import date
from typing import AsyncIterator, List, Optional

from logsqueak.models.knowledge import ActionType


class ExtractionResult:
    """Result from Stage 1: Knowledge extraction from journal.

    Attributes:
        content: The extracted knowledge text
        confidence: LLM confidence score (0.0-1.0)
    """

    def __init__(self, content: str, confidence: float):
        self.content = content
        self.confidence = confidence


class PageCandidate:
    """A candidate target page from RAG search.

    Attributes:
        page_name: Name of the page
        similarity_score: Semantic similarity score (0.0-1.0)
        preview: First ~200 characters of page content
    """

    def __init__(self, page_name: str, similarity_score: float, preview: str):
        self.page_name = page_name
        self.similarity_score = similarity_score
        self.preview = preview


class PageSelectionResult:
    """Result from Stage 2: Target page and section selection.

    Attributes:
        target_page: Selected page name
        target_section: Hierarchical path (e.g., ["Projects", "Timeline"]) or None
        suggested_action: How to integrate (ADD_CHILD or CREATE_SECTION)
        reasoning: LLM's explanation for the selection
    """

    def __init__(
        self,
        target_page: str,
        target_section: Optional[List[str]],
        suggested_action: ActionType,
        reasoning: str,
    ):
        self.target_page = target_page
        self.target_section = target_section
        self.suggested_action = suggested_action
        self.reasoning = reasoning


class DecisionResult:
    """Result from Phase 3: Decider LLM decision.

    The Decider LLM chooses what action to take with the knowledge
    based on candidate chunks from Phase 2.

    Attributes:
        action: What to do with the knowledge (IGNORE_*, UPDATE, APPEND_CHILD, APPEND_ROOT)
        page_name: Target page name (None only for IGNORE_IRRELEVANT)
        target_id: Hybrid ID of target block (None for IGNORE_IRRELEVANT and APPEND_ROOT)
        reasoning: LLM's explanation for the decision
    """

    def __init__(
        self,
        action: ActionType,
        page_name: Optional[str],
        target_id: Optional[str],
        reasoning: str,
    ):
        self.action = action
        self.page_name = page_name
        self.target_id = target_id
        self.reasoning = reasoning


class RephrasedContent:
    """Result from Phase 3.2: Reworder LLM rephrasing.

    The Reworder LLM transforms journal-specific knowledge into
    clean, evergreen content suitable for integration into pages.

    Attributes:
        content: Rephrased content (journal context removed, links preserved)
    """

    def __init__(self, content: str):
        self.content = content


class LLMClient(ABC):
    """Abstract interface for LLM providers.

    Implementations must provide methods for both stages of the extraction process:
    - Stage 1: Extract knowledge blocks from journal content
    - Stage 2: Select target page and section from RAG candidates
    """

    @abstractmethod
    def extract_knowledge(
        self, journal_content: str, journal_date: date, indent_str: str = "  "
    ) -> List[ExtractionResult]:
        """Extract knowledge blocks from journal entry (Stage 1).

        The LLM should identify pieces of information with lasting value,
        filtering out temporary activity logs and todos.

        Args:
            journal_content: Full text of journal entry
            journal_date: Date of the journal entry
            indent_str: Indentation string detected from source (e.g., "  ", "\t")

        Returns:
            List of extracted knowledge blocks with confidence scores

        Raises:
            LLMError: If API request fails or returns invalid response
        """
        pass

    @abstractmethod
    def select_target_page(
        self,
        knowledge_content: str,
        candidates: List[PageCandidate],
        indent_str: str = "  ",
    ) -> PageSelectionResult:
        """Select best target page and section from RAG candidates (Stage 2).

        The LLM should analyze the knowledge content and choose the most
        appropriate page and section from the top-5 RAG candidates.

        Args:
            knowledge_content: The extracted knowledge text
            candidates: Top-5 candidate pages from RAG search (ordered by similarity)
            indent_str: Indentation string detected from source (e.g., "  ", "\t")

        Returns:
            Selected target page, section path, and suggested action

        Raises:
            LLMError: If API request fails or returns invalid response
        """
        pass

    @abstractmethod
    def decide_action(
        self,
        knowledge_text: str,
        candidate_chunks: List[dict],
    ) -> DecisionResult:
        """Decide what action to take with knowledge (Phase 3: Decider).

        The LLM analyzes the knowledge and candidate chunks to decide:
        - Does this knowledge relate to any candidates?
        - If yes, which page and block should receive it?
        - Should it UPDATE existing content or APPEND as new content?

        Args:
            knowledge_text: The full-context knowledge text
            candidate_chunks: List of candidate chunks (dicts with page_name, target_id, content, score)

        Returns:
            Decision with action type, page_name, target_id, and reasoning

        Raises:
            LLMError: If API request fails or returns invalid response
        """
        pass

    @abstractmethod
    def rephrase_content(self, knowledge_full_text: str) -> RephrasedContent:
        """Rephrase knowledge into clean, evergreen content (Phase 3.2: Reworder).

        The LLM transforms journal-specific knowledge into timeless content
        suitable for integration into permanent pages. Removes journal context
        while preserving links and technical details.

        Args:
            knowledge_full_text: The full-context knowledge text from journal

        Returns:
            Rephrased content ready for integration

        Raises:
            LLMError: If API request fails or returns invalid response
        """
        pass

    @abstractmethod
    async def stream_extract_ndjson(self, blocks: List[dict]) -> AsyncIterator[dict]:
        """Stream knowledge extraction results as NDJSON.

        Phase 1 of the extraction pipeline: Classify each journal block
        as either "knowledge" (lasting information) or "activity" (temporary log).

        Args:
            blocks: List of journal blocks to classify. Each dict contains:
                - block_id: str - Hybrid ID (id:: property or content hash)
                - content: str - Block text content
                - hierarchy: str - Full hierarchical context (parent blocks)

        Yields:
            dict: One classification result per block:
                {
                    "block_id": "abc123",
                    "is_knowledge": true,
                    "confidence": 0.85
                }

        Raises:
            LLMError: Network error, API failure, or timeout

        Notes:
            - Each yielded dict is a complete, parseable JSON object
            - Objects arrive in arbitrary order (not necessarily input order)
            - Malformed lines are logged and skipped (parser handles gracefully)
            - Stream may end early on network errors (partial results preserved)
        """
        pass

    @abstractmethod
    async def stream_decisions_ndjson(
        self, knowledge_block: dict, candidate_pages: List[dict]
    ) -> AsyncIterator[dict]:
        """Stream integration decisions for one knowledge block across candidate pages.

        Phase 3.1 (Decider): For each candidate page, decide what action to take
        (skip, add_section, add_under, replace).

        Args:
            knowledge_block: Dict with:
                - block_id: str - Source block ID
                - content: str - Full-context knowledge text
                - hierarchical_text: str - Hierarchical markdown representation

            candidate_pages: List of candidate pages from Phase 2 RAG search. Each dict:
                - page_name: str - Page name
                - similarity_score: float - Semantic match score (0.0-1.0)
                - chunks: list[dict] - Blocks within page with:
                    - target_id: str - Hybrid ID of block
                    - content: str - Block text
                    - title: str - Human-readable block identifier

        Yields:
            dict: One decision per candidate page:
                {
                    "page": "Software Architecture",
                    "action": "add_section",
                    "target_id": null,
                    "target_title": null,
                    "confidence": 0.88,
                    "reasoning": "This knowledge belongs as a new root-level section"
                }

        Raises:
            LLMError: Network error, API failure, or timeout

        Notes:
            - One decision per candidate page (order may not match input)
            - LLM may decide "skip" for all pages (valid outcome)
            - target_id and target_title are null for "skip" and "add_section"
            - target_id and target_title are required for "add_under" and "replace"
        """
        pass

    @abstractmethod
    async def stream_rewording_ndjson(self, decisions: List[dict]) -> AsyncIterator[dict]:
        """Stream reworded content for accepted integration decisions.

        Phase 3.2 (Reworder): Transform journal-specific knowledge into clean,
        evergreen content suitable for integration into permanent pages.

        Args:
            decisions: List of accepted decisions from Phase 3.1 (action != "skip"). Each dict:
                - knowledge_block_id: str - Source block ID
                - page: str - Target page name
                - action: str - Action type ("add_section", "add_under", "replace")
                - full_text: str - Full-context knowledge text
                - hierarchical_text: str - Hierarchical markdown representation

        Yields:
            dict: One reworded result per decision:
                {
                    "knowledge_block_id": "abc123",
                    "page": "Software Architecture",
                    "refined_text": "Bounded contexts matter more than service size"
                }

        Raises:
            LLMError: Network error, API failure, or timeout

        Notes:
            - refined_text should be plain block content (no bullet markers or indentation)
            - Refined text removes journal-specific context (dates, "Today I learned", etc.)
            - Preserves important links: [[Page Links]] and ((block refs))
            - Results may arrive in any order
            - Each result corresponds to exactly one input decision
        """
        pass


class LLMError(Exception):
    """Base exception for LLM client errors.

    This includes network errors, API authentication failures, rate limits,
    malformed responses, and timeout errors.
    """

    pass


class LLMAPIError(LLMError):
    """API-level error (authentication, rate limit, etc.)."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class LLMResponseError(LLMError):
    """Invalid or malformed response from LLM."""

    pass


class LLMTimeoutError(LLMError):
    """Request timeout error."""

    pass
