"""LLM client interface for knowledge extraction.

This module defines the abstract interface for LLM providers used in the
two-stage extraction process:
1. Extract knowledge blocks from journal entries
2. Select target pages and sections from RAG candidates
"""

from abc import ABC, abstractmethod
from datetime import date
from typing import List, Optional

from logsqueak.models.knowledge import ActionType, KnowledgeBlock


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


class LLMClient(ABC):
    """Abstract interface for LLM providers.

    Implementations must provide methods for both stages of the extraction process:
    - Stage 1: Extract knowledge blocks from journal content
    - Stage 2: Select target page and section from RAG candidates
    """

    @abstractmethod
    def extract_knowledge(
        self, journal_content: str, journal_date: date
    ) -> List[ExtractionResult]:
        """Extract knowledge blocks from journal entry (Stage 1).

        The LLM should identify pieces of information with lasting value,
        filtering out temporary activity logs and todos.

        Args:
            journal_content: Full text of journal entry
            journal_date: Date of the journal entry

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
    ) -> PageSelectionResult:
        """Select best target page and section from RAG candidates (Stage 2).

        The LLM should analyze the knowledge content and choose the most
        appropriate page and section from the top-5 RAG candidates.

        Args:
            knowledge_content: The extracted knowledge text
            candidates: Top-5 candidate pages from RAG search (ordered by similarity)

        Returns:
            Selected target page, section path, and suggested action

        Raises:
            LLMError: If API request fails or returns invalid response
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
