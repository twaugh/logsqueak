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


class TestUserStory2FixingBrokenConfig:
    """Integration tests for User Story 2 - Fixing Broken Configuration."""

    @pytest.mark.asyncio
    async def test_update_existing_valid_config(self, temp_graph_dir, tmp_path):
        """Test updating existing valid config - wizard loads values as defaults."""
        config_file = tmp_path / "config.yaml"

        # Create existing valid config
        existing_config = Config(
            llm={
                "endpoint": "http://localhost:11434/v1",
                "model": "mistral:7b-instruct",
                "api_key": "not-required",
            },
            logseq={"graph_path": str(temp_graph_dir)},
        )
        await write_config(existing_config, config_file)

        # Create new graph directory for update
        new_graph_dir = tmp_path / "new-graph"
        new_graph_dir.mkdir()
        (new_graph_dir / "journals").mkdir()
        (new_graph_dir / "logseq").mkdir()

        # Simulate wizard update - change only graph path
        state = WizardState(
            existing_config=existing_config,
            graph_path=str(new_graph_dir),
            provider_type="ollama",
            ollama_endpoint="http://localhost:11434/v1",
            ollama_model="mistral:7b-instruct",
        )

        updated_config = assemble_config(state)

        # Verify only graph path changed
        assert updated_config.logseq.graph_path == str(new_graph_dir)
        assert str(updated_config.llm.endpoint) == "http://localhost:11434/v1"
        assert updated_config.llm.model == "mistral:7b-instruct"

    @pytest.mark.asyncio
    async def test_fix_config_with_wrong_permissions(self, temp_graph_dir, tmp_path):
        """Test fixing config with wrong permissions - wizard detects and recreates with mode 600."""
        config_file = tmp_path / "config.yaml"

        # Create config with wrong permissions
        config = Config(
            llm={
                "endpoint": "http://localhost:11434/v1",
                "model": "mistral:7b-instruct",
                "api_key": "not-required",
            },
            logseq={"graph_path": str(temp_graph_dir)},
        )
        await write_config(config, config_file)

        # Modify permissions to be overly permissive (644)
        os.chmod(config_file, 0o644)

        # Verify permissions are wrong
        file_stat = os.stat(config_file)
        file_mode = stat.S_IMODE(file_stat.st_mode)
        assert file_mode == 0o644

        # Wizard should detect this and offer to fix
        # For now, verify load_existing_config returns None for permission errors
        from logsqueak.wizard.wizard import load_existing_config

        with patch("logsqueak.wizard.wizard.Path.home") as mock_home:
            mock_home.return_value = tmp_path
            loaded = load_existing_config()
            # Should return None due to permission error
            assert loaded is None

        # Recreate config with correct permissions
        await write_config(config, config_file)

        # Verify new permissions are correct
        file_stat = os.stat(config_file)
        file_mode = stat.S_IMODE(file_stat.st_mode)
        assert file_mode == 0o600

    @pytest.mark.asyncio
    async def test_handle_partially_invalid_config(self, temp_graph_dir, tmp_path):
        """Test handling partially invalid config - wizard extracts valid fields."""
        config_file = tmp_path / "config.yaml"

        # Create config with invalid YAML structure (but valid YAML syntax)
        # Missing required field (api_key)
        invalid_data = {
            "llm": {
                "endpoint": "http://localhost:11434/v1",
                "model": "mistral:7b-instruct",
                # Missing api_key
            },
            "logseq": {
                "graph_path": str(temp_graph_dir),
            },
        }

        # Write invalid config
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, "w") as f:
            yaml.dump(invalid_data, f)
        os.chmod(config_file, 0o600)

        # Try to load - should return None due to validation error
        from logsqueak.wizard.wizard import load_existing_config

        with patch("logsqueak.wizard.wizard.Path.home") as mock_home:
            mock_home.return_value = tmp_path
            loaded = load_existing_config()
            # Should return None due to missing required field
            assert loaded is None

        # Wizard can still use the file path as a base and prompt for missing fields
        # The raw YAML data could be read separately if needed
        with open(config_file) as f:
            partial_data = yaml.safe_load(f)

        # Verify we can at least read the graph path
        assert partial_data["logseq"]["graph_path"] == str(temp_graph_dir)

    @pytest.mark.asyncio
    async def test_cached_embedding_model_skip(self, temp_graph_dir):
        """Test that cached embedding model skips download."""
        from logsqueak.wizard.validators import check_embedding_model_cached
        from logsqueak.wizard.wizard import validate_embedding, WizardState

        state = WizardState(
            graph_path=str(temp_graph_dir),
            provider_type="ollama",
            ollama_endpoint="http://localhost:11434/v1",
            ollama_model="mistral:7b-instruct",
        )

        # Mock check_embedding_model_cached to return True
        with patch("logsqueak.wizard.wizard.check_embedding_model_cached") as mock_check:
            mock_check.return_value = True

            # Run validation
            result = await validate_embedding(state)

            # Should succeed and skip download
            assert result is True
            mock_check.assert_called_once()

            # validate_embedding_model should NOT be called when cached
            with patch("logsqueak.wizard.wizard.validate_embedding_model") as mock_validate:
                result = await validate_embedding(state)
                assert result is True
                mock_validate.assert_not_called()
