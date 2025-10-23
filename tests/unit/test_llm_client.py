"""Tests for LLM client interface and error handling."""

from datetime import date
from typing import List

import pytest

from logsqueak.llm.client import (
    ExtractionResult,
    LLMAPIError,
    LLMClient,
    LLMError,
    LLMResponseError,
    LLMTimeoutError,
    PageCandidate,
    PageSelectionResult,
)
from logsqueak.models.knowledge import ActionType


# Mock LLM client for testing
class MockLLMClient(LLMClient):
    """Mock LLM client for testing."""

    def __init__(self, extraction_results=None, selection_result=None, raise_error=None):
        self.extraction_results = extraction_results or []
        self.selection_result = selection_result
        self.raise_error = raise_error
        self.last_journal_content = None
        self.last_journal_date = None
        self.last_knowledge_content = None
        self.last_candidates = None

    def extract_knowledge(
        self, journal_content: str, journal_date: date
    ) -> List[ExtractionResult]:
        """Mock implementation."""
        self.last_journal_content = journal_content
        self.last_journal_date = journal_date

        if self.raise_error:
            raise self.raise_error

        return self.extraction_results

    def select_target_page(
        self, knowledge_content: str, candidates: List[PageCandidate]
    ) -> PageSelectionResult:
        """Mock implementation."""
        self.last_knowledge_content = knowledge_content
        self.last_candidates = candidates

        if self.raise_error:
            raise self.raise_error

        return self.selection_result


# Test ExtractionResult data class
def test_extraction_result_creation():
    """Test ExtractionResult can be created with content and confidence."""
    result = ExtractionResult("Test knowledge", 0.85)
    assert result.content == "Test knowledge"
    assert result.confidence == 0.85


def test_extraction_result_low_confidence():
    """Test ExtractionResult with low confidence (activity log)."""
    result = ExtractionResult("Attended meeting", 0.2)
    assert result.content == "Attended meeting"
    assert result.confidence == 0.2


# Test PageCandidate data class
def test_page_candidate_creation():
    """Test PageCandidate can be created with all attributes."""
    candidate = PageCandidate("Project X", 0.92, "This is a preview...")
    assert candidate.page_name == "Project X"
    assert candidate.similarity_score == 0.92
    assert candidate.preview == "This is a preview..."


def test_page_candidate_zero_similarity():
    """Test PageCandidate with zero similarity score."""
    candidate = PageCandidate("Unrelated Page", 0.0, "No match")
    assert candidate.similarity_score == 0.0


# Test PageSelectionResult data class
def test_page_selection_result_with_section():
    """Test PageSelectionResult with section path."""
    result = PageSelectionResult(
        target_page="Project X",
        target_section=["Timeline", "Q1"],
        suggested_action=ActionType.ADD_CHILD,
        reasoning="Best match for timeline information",
    )
    assert result.target_page == "Project X"
    assert result.target_section == ["Timeline", "Q1"]
    assert result.suggested_action == ActionType.ADD_CHILD
    assert result.reasoning == "Best match for timeline information"


def test_page_selection_result_no_section():
    """Test PageSelectionResult with no section (add at page end)."""
    result = PageSelectionResult(
        target_page="Notes",
        target_section=None,
        suggested_action=ActionType.ADD_CHILD,
        reasoning="No clear section match",
    )
    assert result.target_page == "Notes"
    assert result.target_section is None


def test_page_selection_result_create_section():
    """Test PageSelectionResult with CREATE_SECTION action."""
    result = PageSelectionResult(
        target_page="Project X",
        target_section=["Competitors"],
        suggested_action=ActionType.CREATE_SECTION,
        reasoning="Need to create Competitors section",
    )
    assert result.suggested_action == ActionType.CREATE_SECTION


# Test MockLLMClient
def test_mock_llm_client_extract_knowledge():
    """Test MockLLMClient returns configured extraction results."""
    results = [
        ExtractionResult("Knowledge 1", 0.9),
        ExtractionResult("Knowledge 2", 0.85),
    ]
    client = MockLLMClient(extraction_results=results)

    journal_content = "Test journal content"
    journal_date = date(2025, 1, 15)

    extracted = client.extract_knowledge(journal_content, journal_date)

    assert len(extracted) == 2
    assert extracted[0].content == "Knowledge 1"
    assert extracted[1].confidence == 0.85
    assert client.last_journal_content == journal_content
    assert client.last_journal_date == journal_date


def test_mock_llm_client_select_target_page():
    """Test MockLLMClient returns configured page selection."""
    selection = PageSelectionResult(
        target_page="Database Design",
        target_section=["Architecture"],
        suggested_action=ActionType.ADD_CHILD,
        reasoning="Best match",
    )
    client = MockLLMClient(selection_result=selection)

    candidates = [PageCandidate("Database Design", 0.92, "Preview...")]
    result = client.select_target_page("Test knowledge", candidates)

    assert result.target_page == "Database Design"
    assert result.target_section == ["Architecture"]
    assert client.last_knowledge_content == "Test knowledge"
    assert client.last_candidates == candidates


# Test error handling
def test_mock_llm_client_raises_llm_error():
    """Test MockLLMClient can raise LLMError."""
    client = MockLLMClient(raise_error=LLMError("Generic error"))

    with pytest.raises(LLMError, match="Generic error"):
        client.extract_knowledge("content", date.today())


def test_mock_llm_client_raises_api_error():
    """Test MockLLMClient can raise LLMAPIError with status code."""
    client = MockLLMClient(raise_error=LLMAPIError("Unauthorized", status_code=401))

    with pytest.raises(LLMAPIError) as exc_info:
        client.extract_knowledge("content", date.today())

    assert exc_info.value.status_code == 401
    assert "Unauthorized" in str(exc_info.value)


def test_mock_llm_client_raises_response_error():
    """Test MockLLMClient can raise LLMResponseError."""
    client = MockLLMClient(raise_error=LLMResponseError("Invalid JSON"))

    with pytest.raises(LLMResponseError, match="Invalid JSON"):
        client.select_target_page("knowledge", [])


def test_mock_llm_client_raises_timeout_error():
    """Test MockLLMClient can raise LLMTimeoutError."""
    client = MockLLMClient(raise_error=LLMTimeoutError("Request timed out"))

    with pytest.raises(LLMTimeoutError, match="Request timed out"):
        client.extract_knowledge("content", date.today())


def test_llm_api_error_without_status_code():
    """Test LLMAPIError can be created without status code."""
    error = LLMAPIError("Network error")
    assert error.status_code is None
    assert "Network error" in str(error)


def test_mock_llm_client_raises_rate_limit_error():
    """Test MockLLMClient can raise rate limit error (429)."""
    client = MockLLMClient(
        raise_error=LLMAPIError("Rate limit exceeded", status_code=429)
    )

    with pytest.raises(LLMAPIError) as exc_info:
        client.select_target_page("knowledge", [])

    assert exc_info.value.status_code == 429
    assert "Rate limit exceeded" in str(exc_info.value)
