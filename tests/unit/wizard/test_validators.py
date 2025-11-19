"""Unit tests for wizard validators."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import httpx

from logsqueak.wizard.validators import (
    ValidationResult,
    validate_graph_path,
    check_disk_space,
    validate_ollama_connection,
    validate_openai_connection,
    check_embedding_model_cached,
    validate_embedding_model,
)
from logsqueak.wizard.providers import OllamaModel


class TestValidateGraphPath:
    """Tests for validate_graph_path function."""

    def test_valid_graph_path(self, tmp_path):
        """Test validation of valid Logseq graph directory."""
        # Create valid graph structure
        graph_dir = tmp_path / "graph"
        graph_dir.mkdir()
        (graph_dir / "journals").mkdir()
        (graph_dir / "logseq").mkdir()

        result = validate_graph_path(str(graph_dir))

        assert result.success is True
        assert result.error_message is None
        assert result.data["path"] == str(graph_dir)

    def test_path_does_not_exist(self, tmp_path):
        """Test validation of non-existent path."""
        nonexistent = tmp_path / "nonexistent"

        result = validate_graph_path(str(nonexistent))

        assert result.success is False
        assert "does not exist" in result.error_message
        assert str(nonexistent) in result.error_message

    def test_path_is_file_not_directory(self, tmp_path):
        """Test validation when path is a file, not directory."""
        file_path = tmp_path / "somefile.txt"
        file_path.write_text("content")

        result = validate_graph_path(str(file_path))

        assert result.success is False
        assert "not a directory" in result.error_message

    def test_missing_journals_directory(self, tmp_path):
        """Test validation when journals/ directory is missing."""
        graph_dir = tmp_path / "graph"
        graph_dir.mkdir()
        (graph_dir / "logseq").mkdir()
        # journals/ missing

        result = validate_graph_path(str(graph_dir))

        assert result.success is False
        assert "Missing journals/" in result.error_message

    def test_missing_logseq_directory(self, tmp_path):
        """Test validation when logseq/ directory is missing."""
        graph_dir = tmp_path / "graph"
        graph_dir.mkdir()
        (graph_dir / "journals").mkdir()
        # logseq/ missing

        result = validate_graph_path(str(graph_dir))

        assert result.success is False
        assert "Missing logseq/" in result.error_message

    def test_tilde_expansion(self, tmp_path, monkeypatch):
        """Test that ~ in path is expanded correctly."""
        # Create valid structure in actual home
        import os
        original_home = os.environ.get("HOME")

        try:
            # Set HOME to tmp_path
            os.environ["HOME"] = str(tmp_path)

            graph_dir = tmp_path / "graph"
            graph_dir.mkdir()
            (graph_dir / "journals").mkdir()
            (graph_dir / "logseq").mkdir()

            result = validate_graph_path("~/graph")

            assert result.success is True
            assert result.data["path"] == str(graph_dir)
        finally:
            # Restore original HOME
            if original_home:
                os.environ["HOME"] = original_home
            else:
                os.environ.pop("HOME", None)


class TestCheckDiskSpace:
    """Tests for check_disk_space function."""

    def test_sufficient_disk_space(self, monkeypatch):
        """Test when sufficient disk space is available."""
        # Mock disk_usage to return plenty of space (10GB free)
        mock_usage = Mock()
        mock_usage.free = 10 * 1024 * 1024 * 1024  # 10GB in bytes

        with patch("shutil.disk_usage", return_value=mock_usage):
            result = check_disk_space(required_mb=1024)

        assert result.success is True
        assert result.error_message is None
        assert result.data["available_mb"] == 10 * 1024

    def test_insufficient_disk_space(self, monkeypatch):
        """Test when insufficient disk space is available."""
        # Mock disk_usage to return low space (500MB free)
        mock_usage = Mock()
        mock_usage.free = 500 * 1024 * 1024  # 500MB in bytes

        with patch("shutil.disk_usage", return_value=mock_usage):
            result = check_disk_space(required_mb=1024)

        assert result.success is False
        assert "Low disk space" in result.error_message
        assert "500 MB available" in result.error_message
        assert result.data["available_mb"] == 500


@pytest.mark.asyncio
class TestOllamaConnection:
    """Tests for validate_ollama_connection function."""

    async def test_successful_connection(self):
        """Test successful connection to Ollama API."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "models": [
                {
                    "name": "mistral:7b-instruct",
                    "size": 4109865159,
                    "modified_at": "2025-01-15T10:30:00Z",
                },
                {
                    "name": "llama2:latest",
                    "size": 3826793677,
                    "modified_at": "2025-01-14T09:20:00Z",
                },
            ]
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            result = await validate_ollama_connection("http://localhost:11434")

        assert result.success is True
        assert result.error_message is None
        assert len(result.data["models"]) == 2
        assert result.data["models"][0].name == "mistral:7b-instruct"
        assert result.data["models"][1].name == "llama2:latest"

    async def test_connection_error(self):
        """Test connection error (Ollama not running)."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=httpx.ConnectError("Connection refused")
            )

            result = await validate_ollama_connection("http://localhost:11434")

        assert result.success is False
        assert "Could not connect" in result.error_message
        assert "http://localhost:11434" in result.error_message

    async def test_http_status_error(self):
        """Test HTTP error response from Ollama API."""
        mock_response = Mock()
        mock_response.status_code = 404

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=httpx.HTTPStatusError(
                    "Not found", request=Mock(), response=mock_response
                )
            )

            result = await validate_ollama_connection("http://localhost:11434")

        assert result.success is False
        assert "Ollama API error" in result.error_message
        assert "404" in result.error_message

    async def test_endpoint_normalization(self):
        """Test that /v1 suffix is removed for /api/tags call."""
        mock_response = Mock()
        mock_response.json.return_value = {"models": []}

        with patch("httpx.AsyncClient") as mock_client:
            mock_get = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.get = mock_get

            await validate_ollama_connection("http://localhost:11434/v1")

            # Verify called without /v1 suffix
            mock_get.assert_called_once()
            called_url = mock_get.call_args[0][0]
            assert called_url == "http://localhost:11434/api/tags"

    async def test_unexpected_error(self):
        """Test handling of unexpected errors."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=ValueError("Unexpected error")
            )

            result = await validate_ollama_connection("http://localhost:11434")

        assert result.success is False
        assert "Unexpected error" in result.error_message


class TestCheckEmbeddingModelCached:
    """Tests for check_embedding_model_cached function."""

    def test_model_is_cached(self):
        """Test when model is cached (loads successfully in offline mode)."""
        mock_model = Mock()

        with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
            result = check_embedding_model_cached()

        assert result is True

    def test_model_not_cached(self):
        """Test when model is not cached (fails to load in offline mode)."""
        with patch(
            "sentence_transformers.SentenceTransformer",
            side_effect=Exception("Model not found")
        ):
            result = check_embedding_model_cached()

        assert result is False

    def test_offline_mode_environment_variable(self):
        """Test that HF_HUB_OFFLINE is set during check."""
        import os

        # Ensure variable is not set initially
        original_value = os.environ.pop("HF_HUB_OFFLINE", None)

        try:
            with patch("sentence_transformers.SentenceTransformer") as mock_st:
                def check_env(*args, **kwargs):
                    # Verify offline mode is set when SentenceTransformer is called
                    assert os.environ.get("HF_HUB_OFFLINE") == "1"
                    return Mock()

                mock_st.side_effect = check_env
                check_embedding_model_cached()

            # Verify environment variable is restored after check
            assert "HF_HUB_OFFLINE" not in os.environ or os.environ["HF_HUB_OFFLINE"] != "1"
        finally:
            # Restore original value
            if original_value is not None:
                os.environ["HF_HUB_OFFLINE"] = original_value


@pytest.mark.asyncio
class TestValidateEmbeddingModel:
    """Tests for validate_embedding_model function."""

    async def test_successful_validation(self):
        """Test successful model loading."""
        mock_model = Mock()

        with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
            result = await validate_embedding_model()

        assert result.success is True
        assert result.error_message is None

    async def test_failed_validation(self):
        """Test failed model loading."""
        with patch(
            "sentence_transformers.SentenceTransformer",
            side_effect=Exception("Network error")
        ):
            result = await validate_embedding_model()

        assert result.success is False
        assert "Failed to load embedding model" in result.error_message
        assert "Network error" in result.error_message

    async def test_progress_callback(self):
        """Test that progress callback is supported (even if not implemented yet)."""
        mock_model = Mock()
        progress_calls = []

        def progress_callback(current, total):
            progress_calls.append((current, total))

        with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
            result = await validate_embedding_model(progress_callback=progress_callback)

        assert result.success is True
        # Note: Progress callback not implemented yet, but should not cause errors


@pytest.mark.asyncio
class TestValidateOpenAIConnection:
    """Tests for validate_openai_connection function."""

    async def test_successful_connection(self):
        """Test successful OpenAI API connection."""
        mock_response = Mock()
        mock_response.json.return_value = {"data": []}

        with patch("httpx.AsyncClient") as mock_client:
            mock_get = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.get = mock_get

            result = await validate_openai_connection(
                "https://api.openai.com/v1",
                "sk-test-key",
                "gpt-4o"
            )

        assert result.success is True
        assert result.error_message is None

    async def test_connection_error(self):
        """Test connection error to OpenAI API."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=httpx.ConnectError("Connection failed")
            )

            result = await validate_openai_connection(
                "https://api.openai.com/v1",
                "sk-test-key",
                "gpt-4o"
            )

        assert result.success is False
        assert "Could not connect to API" in result.error_message
        assert "https://api.openai.com/v1" in result.error_message
        assert "service is running" in result.error_message

    async def test_invalid_api_key(self):
        """Test 401 Unauthorized error (invalid API key)."""
        mock_response = Mock()
        mock_response.status_code = 401

        with patch("httpx.AsyncClient") as mock_client:
            mock_get = AsyncMock(side_effect=httpx.HTTPStatusError(
                "401 Unauthorized",
                request=Mock(),
                response=mock_response
            ))
            mock_client.return_value.__aenter__.return_value.get = mock_get

            result = await validate_openai_connection(
                "https://api.openai.com/v1",
                "sk-invalid-key",
                "gpt-4o"
            )

        assert result.success is False
        assert "Invalid API key" in result.error_message
        assert "401 Unauthorized" in result.error_message
        assert "check that your api key is correct" in result.error_message.lower()

    async def test_forbidden_error(self):
        """Test 403 Forbidden error (no permission)."""
        mock_response = Mock()
        mock_response.status_code = 403

        with patch("httpx.AsyncClient") as mock_client:
            mock_get = AsyncMock(side_effect=httpx.HTTPStatusError(
                "403 Forbidden",
                request=Mock(),
                response=mock_response
            ))
            mock_client.return_value.__aenter__.return_value.get = mock_get

            result = await validate_openai_connection(
                "https://api.openai.com/v1",
                "sk-test-key",
                "gpt-4o"
            )

        assert result.success is False
        assert "does not have permission" in result.error_message
        assert "403 Forbidden" in result.error_message
        assert "account settings" in result.error_message.lower()

    async def test_not_found_error(self):
        """Test 404 Not Found error (endpoint doesn't exist)."""
        mock_response = Mock()
        mock_response.status_code = 404

        with patch("httpx.AsyncClient") as mock_client:
            mock_get = AsyncMock(side_effect=httpx.HTTPStatusError(
                "404 Not Found",
                request=Mock(),
                response=mock_response
            ))
            mock_client.return_value.__aenter__.return_value.get = mock_get

            result = await validate_openai_connection(
                "https://api.openai.com/v1",
                "sk-test-key",
                "gpt-4o"
            )

        assert result.success is False
        assert "endpoint not found" in result.error_message.lower()
        assert "404" in result.error_message
        assert "https://api.openai.com/v1/models" in result.error_message

    async def test_other_http_error(self):
        """Test other HTTP errors (e.g., 500)."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        with patch("httpx.AsyncClient") as mock_client:
            mock_get = AsyncMock(side_effect=httpx.HTTPStatusError(
                "500 Internal Server Error",
                request=Mock(),
                response=mock_response
            ))
            mock_client.return_value.__aenter__.return_value.get = mock_get

            result = await validate_openai_connection(
                "https://api.openai.com/v1",
                "sk-test-key",
                "gpt-4o"
            )

        assert result.success is False
        assert "API error 500" in result.error_message
        assert "Internal Server Error" in result.error_message

    async def test_endpoint_normalization(self):
        """Test that endpoint URL is normalized correctly."""
        mock_response = Mock()
        mock_response.json.return_value = {"data": []}

        with patch("httpx.AsyncClient") as mock_client:
            mock_get = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.get = mock_get

            await validate_openai_connection(
                "https://api.openai.com/v1/",
                "sk-test-key",
                "gpt-4o"
            )

            # Verify called with normalized URL
            mock_get.assert_called_once()
            called_url = mock_get.call_args[0][0]
            assert called_url == "https://api.openai.com/v1/models"

    async def test_custom_timeout(self):
        """Test that custom timeout is respected."""
        mock_response = Mock()
        mock_response.json.return_value = {"data": []}

        with patch("httpx.AsyncClient") as mock_client:
            mock_get = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.get = mock_get

            await validate_openai_connection(
                "https://api.openai.com/v1",
                "sk-test-key",
                "gpt-4o",
                timeout=60
            )

            # Verify AsyncClient was created with correct timeout and verify=False
            mock_client.assert_called_once_with(timeout=60, verify=False)

    async def test_timeout_error(self):
        """Test timeout error handling."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=httpx.TimeoutException("Request timed out")
            )

            result = await validate_openai_connection(
                "https://api.openai.com/v1",
                "sk-test-key",
                "gpt-4o",
                timeout=30
            )

        assert result.success is False
        assert "timed out" in result.error_message.lower()
        assert "30 seconds" in result.error_message
        assert "slow to respond" in result.error_message.lower()

    async def test_invalid_url_error(self):
        """Test invalid URL error handling."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=httpx.InvalidURL("Invalid URL")
            )

            result = await validate_openai_connection(
                "not-a-valid-url",
                "sk-test-key",
                "gpt-4o"
            )

        assert result.success is False
        assert "Invalid endpoint URL" in result.error_message
        assert "not-a-valid-url" in result.error_message

    async def test_unexpected_error(self):
        """Test handling of unexpected errors."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=ValueError("Unexpected error")
            )

            result = await validate_openai_connection(
                "https://api.openai.com/v1",
                "sk-test-key",
                "gpt-4o"
            )

        assert result.success is False
        assert "Unexpected error" in result.error_message
        assert "ValueError" in result.error_message
        assert "https://api.openai.com/v1" in result.error_message
