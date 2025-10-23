"""Tests for OpenAI-compatible provider with mocked httpx responses."""

from datetime import date
from unittest.mock import Mock, patch

import httpx
import pytest

from logsqueak.llm.client import (
    LLMAPIError,
    LLMResponseError,
    LLMTimeoutError,
    PageCandidate,
)
from logsqueak.llm.providers.openai_compat import OpenAICompatibleProvider
from logsqueak.models.knowledge import ActionType


@pytest.fixture
def provider():
    """Create a test provider instance."""
    return OpenAICompatibleProvider(
        endpoint="http://localhost:11434/v1",
        api_key="test-key",
        model="test-model",
        timeout=30.0,
    )


def test_provider_initialization(provider):
    """Test provider is initialized with correct parameters."""
    assert provider.endpoint == "http://localhost:11434/v1"
    assert provider.api_key == "test-key"
    assert provider.model == "test-model"
    assert provider.timeout == 30.0


def test_provider_default_timeout():
    """Test provider uses default 60s timeout."""
    provider = OpenAICompatibleProvider(
        endpoint="http://localhost:11434/v1",
        api_key="test-key",
        model="test-model",
    )
    assert provider.timeout == 60.0


@patch("logsqueak.llm.providers.openai_compat.httpx.Client")
def test_extract_knowledge_success(mock_client_class, provider):
    """Test successful knowledge extraction with mocked httpx response."""
    # Mock successful API response
    mock_response = Mock()
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": '{"knowledge_blocks": [{"content": "Test knowledge", "confidence": 0.9}]}'
                }
            }
        ]
    }
    mock_client = Mock()
    mock_client.post.return_value = mock_response
    provider.client = mock_client

    results = provider.extract_knowledge("Test journal", date(2025, 1, 15))

    assert len(results) == 1
    assert results[0].content == "Test knowledge"
    assert results[0].confidence == 0.9

    # Verify request was made correctly
    mock_client.post.assert_called_once()
    call_args = mock_client.post.call_args
    assert call_args[0][0].endswith("/chat/completions")
    assert call_args[1]["json"]["model"] == "test-model"
    assert call_args[1]["json"]["response_format"] == {"type": "json_object"}
    assert call_args[1]["headers"]["Authorization"] == "Bearer test-key"


@patch("logsqueak.llm.providers.openai_compat.httpx.Client")
def test_extract_knowledge_multiple_blocks(mock_client_class, provider):
    """Test extraction with multiple knowledge blocks."""
    mock_response = Mock()
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": '{"knowledge_blocks": [{"content": "K1", "confidence": 0.9}, {"content": "K2", "confidence": 0.7}]}'
                }
            }
        ]
    }
    mock_client = Mock()
    mock_client.post.return_value = mock_response
    provider.client = mock_client

    results = provider.extract_knowledge("Test journal", date(2025, 1, 15))

    assert len(results) == 2
    assert results[0].content == "K1"
    assert results[1].confidence == 0.7


@patch("logsqueak.llm.providers.openai_compat.httpx.Client")
def test_select_target_page_success(mock_client_class, provider):
    """Test successful page selection with mocked httpx response."""
    mock_response = Mock()
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": '{"target_page": "Project X", "target_section": ["Timeline"], "suggested_action": "add_child", "reasoning": "Best match"}'
                }
            }
        ]
    }
    mock_client = Mock()
    mock_client.post.return_value = mock_response
    provider.client = mock_client

    candidates = [PageCandidate("Project X", 0.92, "Preview...")]
    result = provider.select_target_page("Test knowledge", candidates)

    assert result.target_page == "Project X"
    assert result.target_section == ["Timeline"]
    assert result.suggested_action == ActionType.ADD_CHILD
    assert result.reasoning == "Best match"


@patch("logsqueak.llm.providers.openai_compat.httpx.Client")
def test_select_target_page_no_section(mock_client_class, provider):
    """Test page selection with no section (null target_section)."""
    mock_response = Mock()
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": '{"target_page": "Notes", "target_section": null, "suggested_action": "add_child", "reasoning": "No match"}'
                }
            }
        ]
    }
    mock_client = Mock()
    mock_client.post.return_value = mock_response
    provider.client = mock_client

    result = provider.select_target_page("Knowledge", [])

    assert result.target_section is None


@patch("logsqueak.llm.providers.openai_compat.httpx.Client")
def test_select_target_page_create_section(mock_client_class, provider):
    """Test page selection with CREATE_SECTION action."""
    mock_response = Mock()
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": '{"target_page": "Project X", "target_section": ["Competitors"], "suggested_action": "create_section", "reasoning": "Need new section"}'
                }
            }
        ]
    }
    mock_client = Mock()
    mock_client.post.return_value = mock_response
    provider.client = mock_client

    result = provider.select_target_page("Competitor info", [])

    assert result.suggested_action == ActionType.CREATE_SECTION


@patch("logsqueak.llm.providers.openai_compat.httpx.Client")
def test_extract_knowledge_invalid_json(mock_client_class, provider):
    """Test extraction fails with invalid JSON response."""
    mock_response = Mock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Not valid JSON"}}]
    }
    mock_client = Mock()
    mock_client.post.return_value = mock_response
    provider.client = mock_client

    with pytest.raises(LLMResponseError, match="Failed to parse JSON"):
        provider.extract_knowledge("Content", date.today())


@patch("logsqueak.llm.providers.openai_compat.httpx.Client")
def test_extract_knowledge_missing_field(mock_client_class, provider):
    """Test extraction fails when response missing knowledge_blocks field."""
    mock_response = Mock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": '{"wrong_field": []}'}}]
    }
    mock_client = Mock()
    mock_client.post.return_value = mock_response
    provider.client = mock_client

    with pytest.raises(LLMResponseError, match="missing 'knowledge_blocks'"):
        provider.extract_knowledge("Content", date.today())


@patch("logsqueak.llm.providers.openai_compat.httpx.Client")
def test_select_target_page_invalid_action(mock_client_class, provider):
    """Test page selection fails with invalid action type."""
    mock_response = Mock()
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": '{"target_page": "X", "suggested_action": "invalid_action", "reasoning": "Test"}'
                }
            }
        ]
    }
    mock_client = Mock()
    mock_client.post.return_value = mock_response
    provider.client = mock_client

    with pytest.raises(LLMResponseError, match="Invalid action type"):
        provider.select_target_page("Knowledge", [])


@patch("logsqueak.llm.providers.openai_compat.httpx.Client")
def test_extract_knowledge_timeout(mock_client_class, provider):
    """Test extraction raises LLMTimeoutError on timeout."""
    mock_client = Mock()
    mock_client.post.side_effect = httpx.TimeoutException("Timeout")
    provider.client = mock_client

    with pytest.raises(LLMTimeoutError, match="Request timeout"):
        provider.extract_knowledge("Content", date.today())


@patch("logsqueak.llm.providers.openai_compat.httpx.Client")
def test_extract_knowledge_api_error(mock_client_class, provider):
    """Test extraction raises LLMAPIError on HTTP error."""
    mock_response = Mock()
    mock_response.status_code = 401
    mock_response.text = "Unauthorized"

    mock_client = Mock()
    mock_client.post.side_effect = httpx.HTTPStatusError(
        "401 Unauthorized", request=Mock(), response=mock_response
    )
    provider.client = mock_client

    with pytest.raises(LLMAPIError) as exc_info:
        provider.extract_knowledge("Content", date.today())

    assert exc_info.value.status_code == 401


@patch("logsqueak.llm.providers.openai_compat.httpx.Client")
def test_endpoint_url_construction(mock_client_class, provider):
    """Test endpoint URL is correctly constructed."""
    mock_response = Mock()
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": '{"knowledge_blocks": []}'
                }
            }
        ]
    }
    mock_client = Mock()
    mock_client.post.return_value = mock_response
    provider.client = mock_client

    provider.extract_knowledge("Content", date.today())

    # Verify endpoint URL ends with /chat/completions
    call_args = mock_client.post.call_args
    assert call_args[0][0] == "http://localhost:11434/v1/chat/completions"
