"""Integration tests for setup wizard.

These tests verify the wizard creates valid config files and handles basic flows.
More detailed testing is done in unit tests for individual functions.
"""

import os
import stat
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from logsqueak.models.config import Config
from logsqueak.wizard.providers import OllamaModel
from logsqueak.wizard.validators import ValidationResult
from logsqueak.wizard.wizard import write_config, assemble_config, WizardState


@pytest.fixture
def temp_graph_dir(tmp_path):
    """Create a temporary Logseq graph directory structure."""
    graph_dir = tmp_path / "test-graph"
    graph_dir.mkdir()
    (graph_dir / "journals").mkdir()
    (graph_dir / "logseq").mkdir()
    (graph_dir / "pages").mkdir()
    return graph_dir


@pytest.fixture
def mock_ollama_models():
    """Mock Ollama models response."""
    return [
        OllamaModel(name="mistral:7b-instruct", size=4109865159, modified_at="2025-01-15"),
        OllamaModel(name="llama2:latest", size=3826793677, modified_at="2025-01-14"),
    ]


class TestWizardConfigGeneration:
    """Tests for wizard config file generation."""

    @pytest.mark.asyncio
    async def test_assemble_config_for_ollama(self, temp_graph_dir):
        """Test assembling config from wizard state for Ollama provider."""
        state = WizardState(
            graph_path=str(temp_graph_dir),
            provider_type="ollama",
            ollama_endpoint="http://localhost:11434/v1",
            ollama_model="mistral:7b-instruct",
        )

        config = assemble_config(state)

        assert str(config.llm.endpoint) == "http://localhost:11434/v1"
        assert config.llm.model == "mistral:7b-instruct"
        assert config.logseq.graph_path == str(temp_graph_dir)

        # Verify llm_providers field preserves Ollama credentials
        assert config.llm_providers is not None
        assert "ollama_local" in config.llm_providers
        assert config.llm_providers["ollama_local"]["endpoint"] == "http://localhost:11434/v1"

    @pytest.mark.asyncio
    async def test_assemble_config_with_advanced_settings(self, temp_graph_dir):
        """Test assembling config with advanced settings."""
        state = WizardState(
            graph_path=str(temp_graph_dir),
            provider_type="ollama",
            ollama_endpoint="http://localhost:11434/v1",
            ollama_model="mistral:7b-instruct",
            num_ctx=16384,
            top_k=15,
        )

        config = assemble_config(state)

        assert config.llm.num_ctx == 16384
        assert config.rag.top_k == 15

    @pytest.mark.asyncio
    async def test_write_config_creates_file_with_correct_permissions(self, temp_graph_dir, tmp_path):
        """Test that write_config creates file with mode 600."""
        config_file = tmp_path / "config" / "config.yaml"

        config = Config(
            llm={
                "endpoint": "http://localhost:11434/v1",
                "model": "mistral:7b-instruct",
                "api_key": "not-required",
            },
            logseq={"graph_path": str(temp_graph_dir)},
        )

        await write_config(config, config_file)

        # Verify file exists
        assert config_file.exists()

        # Verify permissions are mode 600
        file_stat = os.stat(config_file)
        file_mode = stat.S_IMODE(file_stat.st_mode)
        assert file_mode == 0o600

        # Verify content is valid YAML
        with open(config_file) as f:
            config_data = yaml.safe_load(f)

        assert config_data["llm"]["endpoint"] == "http://localhost:11434/v1"
        assert config_data["llm"]["model"] == "mistral:7b-instruct"
        assert config_data["logseq"]["graph_path"] == str(temp_graph_dir)

    @pytest.mark.asyncio
    async def test_write_config_creates_directory_if_needed(self, temp_graph_dir, tmp_path):
        """Test that write_config creates parent directory if it doesn't exist."""
        config_dir = tmp_path / "nonexistent" / "config"
        config_file = config_dir / "config.yaml"

        config = Config(
            llm={
                "endpoint": "http://localhost:11434/v1",
                "model": "mistral:7b-instruct",
                "api_key": "not-required",
            },
            logseq={"graph_path": str(temp_graph_dir)},
        )

        await write_config(config, config_file)

        # Verify directory was created
        assert config_dir.exists()
        assert config_file.exists()

    @pytest.mark.asyncio
    async def test_config_can_be_loaded_after_write(self, temp_graph_dir, tmp_path):
        """Test that config written by wizard can be loaded by ConfigManager."""
        config_file = tmp_path / "config.yaml"

        original_config = Config(
            llm={
                "endpoint": "http://localhost:11434/v1",
                "model": "mistral:7b-instruct",
                "api_key": "not-required",
            },
            logseq={"graph_path": str(temp_graph_dir)},
            llm_providers={
                "ollama_local": {
                    "endpoint": "http://localhost:11434/v1",
                    "model": "mistral:7b-instruct",
                    "api_key": "not-required",
                }
            },
        )

        await write_config(original_config, config_file)

        # Load config and verify it matches
        loaded_config = Config.load(config_file)
        assert loaded_config.llm.endpoint == original_config.llm.endpoint
        assert loaded_config.llm.model == original_config.llm.model
        assert loaded_config.logseq.graph_path == original_config.logseq.graph_path
        assert loaded_config.llm_providers == original_config.llm_providers
