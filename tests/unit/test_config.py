"""Unit tests for configuration models."""

import pytest
from pathlib import Path
import tempfile
import os
import stat

from logsqueak.models.config import LLMConfig, LogseqConfig, RAGConfig, Config


class TestLLMConfig:
    """Test LLM configuration model."""

    def test_valid_llm_config(self):
        """Test creating valid LLM config."""
        config = LLMConfig(
            endpoint="https://api.openai.com/v1",
            api_key="sk-test-key",
            model="gpt-4-turbo-preview"
        )

        assert "api.openai.com/v1" in str(config.endpoint)
        assert config.api_key == "sk-test-key"
        assert config.model == "gpt-4-turbo-preview"
        assert config.num_ctx == 32768  # Default

    def test_llm_config_with_custom_num_ctx(self):
        """Test LLM config with custom context window."""
        config = LLMConfig(
            endpoint="http://localhost:11434/v1",
            api_key="ollama",
            model="llama2",
            num_ctx=16384
        )

        assert config.num_ctx == 16384

    def test_llm_config_immutable(self):
        """Test that LLM config is frozen (immutable)."""
        config = LLMConfig(
            endpoint="https://api.openai.com/v1",
            api_key="sk-test",
            model="gpt-4"
        )

        with pytest.raises(Exception):  # Pydantic ValidationError
            config.api_key = "new-key"


class TestLogseqConfig:
    """Test Logseq configuration model."""

    def test_valid_logseq_config(self, tmp_path):
        """Test creating valid Logseq config with existing directory."""
        graph_dir = tmp_path / "logseq-graph"
        graph_dir.mkdir()

        config = LogseqConfig(graph_path=str(graph_dir))

        assert Path(config.graph_path) == graph_dir

    def test_logseq_config_nonexistent_path(self):
        """Test Logseq config fails with nonexistent path."""
        with pytest.raises(ValueError, match="Graph path does not exist"):
            LogseqConfig(graph_path="/nonexistent/path/to/graph")

    def test_logseq_config_file_not_directory(self, tmp_path):
        """Test Logseq config fails when path is a file, not directory."""
        file_path = tmp_path / "not-a-dir.txt"
        file_path.touch()

        with pytest.raises(ValueError, match="not a directory"):
            LogseqConfig(graph_path=str(file_path))


class TestRAGConfig:
    """Test RAG configuration model."""

    def test_rag_config_defaults(self):
        """Test RAG config with default values."""
        config = RAGConfig()

        assert config.top_k == 20

    def test_rag_config_custom_top_k(self):
        """Test RAG config with custom top_k."""
        config = RAGConfig(top_k=30)

        assert config.top_k == 30

    def test_rag_config_validates_top_k_range(self):
        """Test RAG config validates top_k is in valid range."""
        # Too low
        with pytest.raises(Exception):  # Pydantic ValidationError
            RAGConfig(top_k=0)

        # Too high
        with pytest.raises(Exception):  # Pydantic ValidationError
            RAGConfig(top_k=101)


class TestConfig:
    """Test root configuration model."""

    def test_config_load_valid_yaml(self, tmp_path):
        """Test loading valid YAML config file."""
        config_file = tmp_path / "config.yaml"
        graph_dir = tmp_path / "graph"
        graph_dir.mkdir()

        # Write valid config
        config_file.write_text(f"""
llm:
  endpoint: https://api.openai.com/v1
  api_key: sk-test-key
  model: gpt-4-turbo-preview

logseq:
  graph_path: {graph_dir}

rag:
  top_k: 15
""")

        # Set correct permissions (600)
        os.chmod(config_file, stat.S_IRUSR | stat.S_IWUSR)

        config = Config.load(config_file)

        assert config.llm.model == "gpt-4-turbo-preview"
        assert config.logseq.graph_path == str(graph_dir)
        assert config.rag.top_k == 15

    def test_config_load_with_defaults(self, tmp_path):
        """Test loading config with optional RAG section omitted."""
        config_file = tmp_path / "config.yaml"
        graph_dir = tmp_path / "graph"
        graph_dir.mkdir()

        # Write config without RAG section
        config_file.write_text(f"""
llm:
  endpoint: https://api.openai.com/v1
  api_key: sk-test
  model: gpt-4

logseq:
  graph_path: {graph_dir}
""")

        os.chmod(config_file, stat.S_IRUSR | stat.S_IWUSR)

        config = Config.load(config_file)

        # Should use RAG defaults
        assert config.rag.top_k == 20

    def test_config_load_file_not_found(self, tmp_path):
        """Test loading config fails when file doesn't exist."""
        nonexistent = tmp_path / "nonexistent.yaml"

        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            Config.load(nonexistent)

    def test_config_load_wrong_permissions(self, tmp_path):
        """Test loading config fails when permissions are too open."""
        config_file = tmp_path / "config.yaml"
        graph_dir = tmp_path / "graph"
        graph_dir.mkdir()

        config_file.write_text(f"""
llm:
  endpoint: https://api.openai.com/v1
  api_key: sk-test
  model: gpt-4

logseq:
  graph_path: {graph_dir}
""")

        # Set wrong permissions (644 - group/world readable)
        os.chmod(config_file, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)

        with pytest.raises(PermissionError, match="overly permissive permissions"):
            Config.load(config_file)

    def test_config_immutable(self, tmp_path):
        """Test that root config is frozen (immutable)."""
        config_file = tmp_path / "config.yaml"
        graph_dir = tmp_path / "graph"
        graph_dir.mkdir()

        config_file.write_text(f"""
llm:
  endpoint: https://api.openai.com/v1
  api_key: sk-test
  model: gpt-4

logseq:
  graph_path: {graph_dir}
""")

        os.chmod(config_file, stat.S_IRUSR | stat.S_IWUSR)

        config = Config.load(config_file)

        with pytest.raises(Exception):  # Pydantic ValidationError
            config.rag = RAGConfig(top_k=20)
