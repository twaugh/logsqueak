"""Unit tests for Config model extension with llm_providers field."""

import pytest
import tempfile
from pathlib import Path
from pydantic import HttpUrl

from logsqueak.models.config import Config, LLMConfig, LogseqConfig, RAGConfig


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
