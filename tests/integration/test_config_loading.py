"""Integration tests for configuration loading."""

import pytest
from pathlib import Path
import os
import stat

from logsqueak.config import ConfigManager
from logsqueak.models.config import Config


class TestConfigLoading:
    """Integration tests for full config loading workflow."""

    def test_load_complete_config_from_file(self, tmp_path):
        """Test loading complete configuration from YAML file."""
        config_file = tmp_path / "config.yaml"
        graph_dir = tmp_path / "logseq-graph"
        graph_dir.mkdir()

        # Write complete config file
        config_file.write_text(f"""
llm:
  endpoint: https://api.openai.com/v1
  api_key: sk-test-api-key-12345
  model: gpt-4-turbo-preview
  num_ctx: 16384

logseq:
  graph_path: {graph_dir}

rag:
  top_k: 20
""")

        # Set correct permissions
        os.chmod(config_file, stat.S_IRUSR | stat.S_IWUSR)

        # Load via ConfigManager
        config_mgr = ConfigManager.load_from_path(config_file)

        # Verify all sections loaded correctly
        assert config_mgr.llm.model == "gpt-4-turbo-preview"
        assert config_mgr.llm.api_key == "sk-test-api-key-12345"
        assert config_mgr.llm.num_ctx == 16384

        assert config_mgr.logseq.graph_path == str(graph_dir)

        assert config_mgr.rag.top_k == 20

    def test_lazy_validation_llm_config(self, tmp_path):
        """Test lazy validation - LLM config validated on first access."""
        config_file = tmp_path / "config.yaml"
        graph_dir = tmp_path / "logseq-graph"
        graph_dir.mkdir()

        # Write config with invalid LLM endpoint
        config_file.write_text(f"""
llm:
  endpoint: not-a-valid-url
  api_key: sk-test
  model: gpt-4

logseq:
  graph_path: {graph_dir}
""")

        os.chmod(config_file, stat.S_IRUSR | stat.S_IWUSR)

        # Loading should fail because Pydantic validates on load
        with pytest.raises(ValueError):
            ConfigManager.load_from_path(config_file)

    def test_lazy_validation_logseq_config(self, tmp_path):
        """Test lazy validation - Logseq config validated on first access."""
        config_file = tmp_path / "config.yaml"

        # Write config with nonexistent graph path
        config_file.write_text("""
llm:
  endpoint: https://api.openai.com/v1
  api_key: sk-test
  model: gpt-4

logseq:
  graph_path: /nonexistent/graph/path
""")

        os.chmod(config_file, stat.S_IRUSR | stat.S_IWUSR)

        # Loading should fail because validation happens on load
        with pytest.raises(ValueError):
            ConfigManager.load_from_path(config_file)

    def test_config_manager_default_path(self, tmp_path, monkeypatch):
        """Test ConfigManager loading from default path."""
        # Mock home directory
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        config_dir = fake_home / ".config" / "logsqueak"
        config_dir.mkdir(parents=True)

        config_file = config_dir / "config.yaml"
        graph_dir = fake_home / "logseq"
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

        # Mock Path.home()
        monkeypatch.setattr(Path, "home", lambda: fake_home)

        # Load from default path
        config_mgr = ConfigManager.load_default()

        assert config_mgr.llm.model == "gpt-4"
        assert config_mgr.logseq.graph_path == str(graph_dir)

    def test_config_with_defaults(self, tmp_path):
        """Test config with optional sections using defaults."""
        config_file = tmp_path / "config.yaml"
        graph_dir = tmp_path / "logseq-graph"
        graph_dir.mkdir()

        # Write minimal config (RAG section omitted)
        config_file.write_text(f"""
llm:
  endpoint: https://api.openai.com/v1
  api_key: sk-test
  model: gpt-4

logseq:
  graph_path: {graph_dir}
""")

        os.chmod(config_file, stat.S_IRUSR | stat.S_IWUSR)

        config_mgr = ConfigManager.load_from_path(config_file)

        # RAG should use defaults
        assert config_mgr.rag.top_k == 10

    def test_permission_check_integration(self, tmp_path):
        """Test that permission check works in full loading workflow."""
        config_file = tmp_path / "config.yaml"
        graph_dir = tmp_path / "logseq-graph"
        graph_dir.mkdir()

        config_file.write_text(f"""
llm:
  endpoint: https://api.openai.com/v1
  api_key: sk-test
  model: gpt-4

logseq:
  graph_path: {graph_dir}
""")

        # Set wrong permissions (world-readable)
        os.chmod(config_file, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)

        with pytest.raises(PermissionError, match="overly permissive"):
            ConfigManager.load_from_path(config_file)

    def test_missing_config_file(self, tmp_path):
        """Test helpful error when config file missing."""
        nonexistent = tmp_path / "nonexistent.yaml"

        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            ConfigManager.load_from_path(nonexistent)

    def test_ollama_config(self, tmp_path):
        """Test loading Ollama-specific config."""
        config_file = tmp_path / "config.yaml"
        graph_dir = tmp_path / "logseq-graph"
        graph_dir.mkdir()

        # Ollama config
        config_file.write_text(f"""
llm:
  endpoint: http://localhost:11434/v1
  api_key: ollama
  model: llama2
  num_ctx: 32768

logseq:
  graph_path: {graph_dir}
""")

        os.chmod(config_file, stat.S_IRUSR | stat.S_IWUSR)

        config_mgr = ConfigManager.load_from_path(config_file)

        assert "localhost" in str(config_mgr.llm.endpoint)
        assert config_mgr.llm.model == "llama2"
        assert config_mgr.llm.num_ctx == 32768
