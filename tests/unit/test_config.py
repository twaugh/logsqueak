"""Unit tests for Configuration model."""

import tempfile
from pathlib import Path

import pytest
import yaml

from logsqueak.models.config import Configuration, LLMConfig, LogseqConfig


class TestLLMConfig:
    """Tests for LLMConfig model."""

    def test_valid_config(self):
        """Test creating valid LLM config."""
        config = LLMConfig(
            endpoint="https://api.openai.com/v1",
            api_key="sk-test-key",
            model="gpt-4-turbo-preview",
        )
        assert "https://api.openai.com/v1" in str(config.endpoint)
        assert config.api_key == "sk-test-key"
        assert config.model == "gpt-4-turbo-preview"

    def test_default_model(self):
        """Test default model is set correctly."""
        config = LLMConfig(endpoint="https://api.openai.com/v1", api_key="sk-test-key")
        assert config.model == "gpt-4-turbo-preview"

    def test_empty_api_key_fails(self):
        """Test that empty API key fails validation."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            LLMConfig(endpoint="https://api.openai.com/v1", api_key="")


class TestLogseqConfig:
    """Tests for LogseqConfig model."""

    def test_valid_config(self, tmp_path):
        """Test creating valid Logseq config."""
        config = LogseqConfig(graph_path=tmp_path)
        assert config.graph_path == tmp_path

    def test_nonexistent_path_fails(self):
        """Test that nonexistent path fails validation."""
        with pytest.raises(ValueError, match="does not exist"):
            LogseqConfig(graph_path=Path("/nonexistent/path"))

    def test_file_path_fails(self, tmp_path):
        """Test that file path (not directory) fails validation."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("test")

        with pytest.raises(ValueError, match="does not exist"):
            LogseqConfig(graph_path=file_path)


class TestConfiguration:
    """Tests for complete Configuration model."""

    def test_load_valid_config(self, tmp_path):
        """Test loading valid configuration from YAML file."""
        # Create test graph directory
        graph_path = tmp_path / "graph"
        graph_path.mkdir()

        # Create config file
        config_file = tmp_path / "config.yaml"
        config_data = {
            "llm": {
                "endpoint": "https://api.openai.com/v1",
                "api_key": "sk-test-key",
                "model": "gpt-4",
            },
            "logseq": {"graph_path": str(graph_path)},
        }

        with config_file.open("w") as f:
            yaml.dump(config_data, f)

        # Load config
        config = Configuration.load(config_file)

        assert "https://api.openai.com/v1" in str(config.llm.endpoint)
        assert config.llm.api_key == "sk-test-key"
        assert config.llm.model == "gpt-4"
        assert config.logseq.graph_path == graph_path

    def test_load_missing_config_fails(self):
        """Test that missing config file raises error."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            Configuration.load(Path("/nonexistent/config.yaml"))

    def test_load_invalid_yaml_fails(self, tmp_path):
        """Test that invalid YAML fails to load."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("invalid: yaml: content:")

        with pytest.raises(Exception):  # yaml.YAMLError or similar
            Configuration.load(config_file)

    def test_load_missing_required_fields(self, tmp_path):
        """Test that missing required fields fails validation."""
        config_file = tmp_path / "config.yaml"
        config_data = {"llm": {"endpoint": "https://api.openai.com/v1"}}  # Missing api_key

        with config_file.open("w") as f:
            yaml.dump(config_data, f)

        with pytest.raises(Exception):  # Pydantic ValidationError
            Configuration.load(config_file)
