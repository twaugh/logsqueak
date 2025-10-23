"""Integration tests for prompt inspection with OpenAI provider."""

from datetime import date
from io import StringIO
from unittest.mock import MagicMock, patch

import httpx
import pytest

from logsqueak.llm.client import PageCandidate
from logsqueak.llm.prompt_logger import PromptLogger
from logsqueak.llm.providers.openai_compat import OpenAICompatibleProvider


class TestPromptInspectionIntegration:
    """Test prompt logging integration with OpenAI provider."""

    def test_extract_knowledge_logs_request_and_response(self):
        """Should log both request and response during extraction."""
        output = StringIO()
        logger = PromptLogger(output=output, include_timestamps=False)

        provider = OpenAICompatibleProvider(
            endpoint="http://localhost:8000",
            api_key="test-key",
            model="test-model",
            prompt_logger=logger,
        )

        # Mock the HTTP response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"knowledge_blocks": [{"content": "Test knowledge", "confidence": 0.9}]}'
                    }
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(provider.client, "post", return_value=mock_response):
            result = provider.extract_knowledge(
                journal_content="Test journal content",
                journal_date=date(2025, 1, 15),
            )

        log_output = output.getvalue()

        # Check request logging
        assert "[1] LLM REQUEST - extraction" in log_output
        assert "Model: test-model" in log_output
        assert "journal_date" in log_output
        assert "2025-01-15" in log_output
        assert "Test journal content" in log_output
        assert "Journal entry from 2025-01-15:" in log_output

        # Check response logging
        assert "[1] LLM RESPONSE - extraction" in log_output
        assert "Status: Success" in log_output
        assert "Parsed Content:" in log_output
        assert "knowledge_blocks" in log_output
        assert "Test knowledge" in log_output
        # Should not include raw response
        assert "Raw Response:" not in log_output

        # Verify result
        assert len(result) == 1
        assert result[0].content == "Test knowledge"
        assert result[0].confidence == 0.9

    def test_select_target_page_logs_request_and_response(self):
        """Should log both request and response during page selection."""
        output = StringIO()
        logger = PromptLogger(output=output, include_timestamps=False)

        provider = OpenAICompatibleProvider(
            endpoint="http://localhost:8000",
            api_key="test-key",
            model="test-model",
            prompt_logger=logger,
        )

        candidates = [
            PageCandidate("Page 1", 0.9, "Preview of page 1"),
            PageCandidate("Page 2", 0.7, "Preview of page 2"),
        ]

        # Mock the HTTP response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"target_page": "Page 1", "target_section": ["Section A"], "suggested_action": "add_child", "reasoning": "Best match"}'
                    }
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(provider.client, "post", return_value=mock_response):
            result = provider.select_target_page(
                knowledge_content="Important insight about X",
                candidates=candidates,
            )

        log_output = output.getvalue()

        # Check request logging
        assert "[1] LLM REQUEST - page_selection" in log_output
        assert "Model: test-model" in log_output
        assert "knowledge_content" in log_output
        assert "Important insight about X" in log_output
        assert "num_candidates" in log_output
        assert "Page 1" in log_output
        # Check XML structure
        assert "<knowledge_to_organize>" in log_output
        assert "<candidate_pages>" in log_output
        assert "<page_name>Page 1</page_name>" in log_output
        assert "<similarity_score>0.90</similarity_score>" in log_output
        assert "<preview>" in log_output
        assert 'rank="1"' in log_output

        # Check response logging
        assert "[1] LLM RESPONSE - page_selection" in log_output
        assert "Status: Success" in log_output
        assert "Parsed Content:" in log_output
        assert "target_page" in log_output
        assert "Best match" in log_output
        # Should not include raw response
        assert "Raw Response:" not in log_output

        # Verify result
        assert result.target_page == "Page 1"
        assert result.target_section == ["Section A"]

    def test_error_logging_on_network_failure(self):
        """Should log error when network request fails."""
        output = StringIO()
        logger = PromptLogger(output=output, include_timestamps=False)

        provider = OpenAICompatibleProvider(
            endpoint="http://localhost:8000",
            api_key="test-key",
            model="test-model",
            prompt_logger=logger,
        )

        # Mock network error
        with patch.object(
            provider.client,
            "post",
            side_effect=httpx.RequestError("Connection failed"),
        ):
            with pytest.raises(Exception):  # LLMError
                provider.extract_knowledge(
                    journal_content="Test",
                    journal_date=date(2025, 1, 15),
                )

        log_output = output.getvalue()

        # Check that error was logged
        assert "ERROR:" in log_output
        assert "RequestError" in log_output or "Connection failed" in log_output

    def test_error_logging_on_api_error(self):
        """Should log error when API returns error status."""
        output = StringIO()
        logger = PromptLogger(output=output, include_timestamps=False)

        provider = OpenAICompatibleProvider(
            endpoint="http://localhost:8000",
            api_key="test-key",
            model="test-model",
            prompt_logger=logger,
        )

        # Mock HTTP error
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        with patch.object(
            provider.client,
            "post",
            side_effect=httpx.HTTPStatusError(
                "401 Unauthorized",
                request=MagicMock(),
                response=mock_response,
            ),
        ):
            with pytest.raises(Exception):  # LLMAPIError
                provider.extract_knowledge(
                    journal_content="Test",
                    journal_date=date(2025, 1, 15),
                )

        log_output = output.getvalue()

        # Check that error was logged
        assert "ERROR:" in log_output
        assert "HTTPStatusError" in log_output

    def test_no_logging_when_logger_not_provided(self):
        """Should work normally without logger."""
        # No logger provided
        provider = OpenAICompatibleProvider(
            endpoint="http://localhost:8000",
            api_key="test-key",
            model="test-model",
        )

        # Mock successful response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"knowledge_blocks": []}'
                    }
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(provider.client, "post", return_value=mock_response):
            result = provider.extract_knowledge(
                journal_content="Test",
                journal_date=date(2025, 1, 15),
            )

        # Should complete without errors
        assert result == []

    def test_multiple_interactions_increment_counter(self):
        """Should increment counter across multiple LLM calls."""
        output = StringIO()
        logger = PromptLogger(output=output, include_timestamps=False)

        provider = OpenAICompatibleProvider(
            endpoint="http://localhost:8000",
            api_key="test-key",
            model="test-model",
            prompt_logger=logger,
        )

        # Mock responses
        mock_extraction_response = MagicMock()
        mock_extraction_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"knowledge_blocks": [{"content": "Test", "confidence": 0.9}]}'
                    }
                }
            ]
        }
        mock_extraction_response.raise_for_status = MagicMock()

        mock_selection_response = MagicMock()
        mock_selection_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"target_page": "Page 1", "suggested_action": "add_child"}'
                    }
                }
            ]
        }
        mock_selection_response.raise_for_status = MagicMock()

        with patch.object(
            provider.client,
            "post",
            side_effect=[mock_extraction_response, mock_selection_response],
        ):
            # Call 1: extraction
            provider.extract_knowledge("Test", date(2025, 1, 15))

            # Call 2: page selection
            candidates = [PageCandidate("Page 1", 0.9, "Preview")]
            provider.select_target_page("Test", candidates)

        log_output = output.getvalue()

        # Both calls should be numbered
        assert "[1] LLM REQUEST - extraction" in log_output
        assert "[1] LLM RESPONSE - extraction" in log_output
        assert "[2] LLM REQUEST - page_selection" in log_output
        assert "[2] LLM RESPONSE - page_selection" in log_output

    def test_long_content_truncation_in_metadata(self):
        """Should truncate long knowledge content in metadata."""
        output = StringIO()
        logger = PromptLogger(output=output, include_timestamps=False)

        provider = OpenAICompatibleProvider(
            endpoint="http://localhost:8000",
            api_key="test-key",
            model="test-model",
            prompt_logger=logger,
        )

        # Create very long content
        long_content = "A" * 300

        candidates = [PageCandidate("Page 1", 0.9, "Preview")]

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"target_page": "Page 1", "suggested_action": "add_child"}'
                    }
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(provider.client, "post", return_value=mock_response):
            provider.select_target_page(long_content, candidates)

        log_output = output.getvalue()

        # Metadata should contain truncated version
        assert "knowledge_content" in log_output
        assert "..." in log_output  # Truncation indicator
