"""Unit tests for Config model extension with llm_providers field."""

import pytest
import tempfile
from pathlib import Path
from pydantic import HttpUrl
from unittest.mock import AsyncMock, MagicMock, patch

from logsqueak.models.config import Config, LLMConfig, LogseqConfig, RAGConfig
from logsqueak.wizard.wizard import WizardState, configure_ollama


@pytest.fixture
def test_graph_path(tmp_path):
    """Create a temporary Logseq graph directory."""
    graph_dir = tmp_path / "test-graph"
    graph_dir.mkdir()
    (graph_dir / "journals").mkdir()
    (graph_dir / "logseq").mkdir()
    return str(graph_dir)


def test_config_with_llm_providers(test_graph_path):
    """Test that Config model accepts llm_providers field."""
    config = Config(
        llm=LLMConfig(
            endpoint=HttpUrl("http://localhost:11434/v1"),
            api_key="ollama",
            model="mistral:7b-instruct"
        ),
        logseq=LogseqConfig(graph_path=test_graph_path),
        rag=RAGConfig(),
        llm_providers={
            "ollama_local": {
                "endpoint": "http://localhost:11434/v1",
                "api_key": "ollama",
                "model": "mistral:7b-instruct"
            }
        }
    )

    assert config.llm_providers is not None
    assert "ollama_local" in config.llm_providers
    assert config.llm_providers["ollama_local"]["model"] == "mistral:7b-instruct"


def test_config_without_llm_providers(test_graph_path):
    """Test that llm_providers field is optional (backwards compatibility)."""
    config = Config(
        llm=LLMConfig(
            endpoint=HttpUrl("http://localhost:11434/v1"),
            api_key="ollama",
            model="mistral:7b-instruct"
        ),
        logseq=LogseqConfig(graph_path=test_graph_path),
        rag=RAGConfig()
    )

    assert config.llm_providers is None


def test_config_with_multiple_providers(test_graph_path):
    """Test that multiple providers can be stored in llm_providers."""
    config = Config(
        llm=LLMConfig(
            endpoint=HttpUrl("http://localhost:11434/v1"),
            api_key="ollama",
            model="mistral:7b-instruct"
        ),
        logseq=LogseqConfig(graph_path=test_graph_path),
        rag=RAGConfig(),
        llm_providers={
            "ollama_local": {
                "endpoint": "http://localhost:11434/v1",
                "api_key": "ollama",
                "model": "mistral:7b-instruct",
                "num_ctx": 32768
            },
            "openai": {
                "endpoint": "https://api.openai.com/v1",
                "api_key": "sk-test123",
                "model": "gpt-4o"
            },
            "custom_azure": {
                "endpoint": "https://myazure.openai.azure.com/v1",
                "api_key": "azure-key",
                "model": "gpt-4"
            }
        }
    )

    assert len(config.llm_providers) == 3
    assert "ollama_local" in config.llm_providers
    assert "openai" in config.llm_providers
    assert "custom_azure" in config.llm_providers


@pytest.mark.asyncio
async def test_configure_ollama_loads_paired_endpoint_and_model(test_graph_path):
    """Test that configure_ollama loads endpoint and model as a pair from llm_providers.

    This verifies the fix for the issue where switching from a custom provider
    back to Ollama would show the default localhost endpoint instead of the
    user's pre-configured remote Ollama endpoint.
    """
    # Setup: Config with custom provider active, but ollama_remote in llm_providers
    existing_config = Config(
        llm=LLMConfig(
            endpoint=HttpUrl("https://custom.api.com/v1"),
            api_key="custom-key",
            model="custom-model"
        ),
        logseq=LogseqConfig(graph_path=test_graph_path),
        rag=RAGConfig(),
        llm_providers={
            "ollama_remote": {
                "endpoint": "http://scooby:11434/v1",
                "api_key": "ollama",
                "model": "qwen2.5:14b",
                "num_ctx": 32768
            },
            "custom": {
                "endpoint": "https://custom.api.com/v1",
                "api_key": "custom-key",
                "model": "custom-model"
            }
        }
    )

    state = WizardState(existing_config=existing_config)

    # Mock the validation and prompts
    mock_models = [
        MagicMock(name="qwen2.5:14b", size=8000000000),
        MagicMock(name="mistral:7b-instruct", size=4000000000)
    ]

    with patch("logsqueak.wizard.wizard.validate_ollama_connection") as mock_validate, \
         patch("logsqueak.wizard.wizard.prompt_ollama_model") as mock_prompt_model, \
         patch("logsqueak.wizard.wizard.rprint") as mock_rprint, \
         patch("logsqueak.wizard.wizard.Status"):

        # Mock successful connection with pre-configured endpoint
        result_mock = MagicMock()
        result_mock.success = True
        result_mock.data = {"models": mock_models}
        mock_validate.return_value = result_mock

        # User selects the existing model
        mock_prompt_model.return_value = "qwen2.5:14b"

        # Execute configure_ollama
        success = await configure_ollama(state)

        # Verify success
        assert success is True

        # Verify endpoint/model pair was loaded from ollama_remote
        assert state.ollama_endpoint == "http://scooby:11434/v1"
        assert state.ollama_model == "qwen2.5:14b"

        # Verify validate_ollama_connection was called with the remote endpoint
        # (not localhost)
        validate_calls = [call[0][0] for call in mock_validate.call_args_list]
        assert "http://scooby:11434/v1" in validate_calls

        # Verify prompt_ollama_model was called with the correct default model
        assert mock_prompt_model.called
        args, kwargs = mock_prompt_model.call_args
        # prompt_ollama_model(models, default) - second positional arg
        assert args[1] == "qwen2.5:14b" or kwargs.get("default") == "qwen2.5:14b"
