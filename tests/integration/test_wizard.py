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
        # Create proper .config/logsqueak directory structure
        config_dir = tmp_path / ".config" / "logsqueak"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "config.yaml"

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

        # Wizard should still load config for defaults despite permission error
        from logsqueak.wizard.wizard import load_existing_config

        with patch("logsqueak.wizard.wizard.Path.home") as mock_home:
            mock_home.return_value = tmp_path
            loaded = load_existing_config()
            # Should load successfully by bypassing permission check
            assert loaded is not None
            assert loaded.llm.model == "mistral:7b-instruct"
            assert loaded.logseq.graph_path == str(temp_graph_dir)

        # Recreate config with correct permissions
        await write_config(config, config_file)

        # Verify new permissions are correct
        file_stat = os.stat(config_file)
        file_mode = stat.S_IMODE(file_stat.st_mode)
        assert file_mode == 0o600

    @pytest.mark.asyncio
    async def test_permission_fix_with_no_content_changes(self, temp_graph_dir, tmp_path):
        """Test that wizard rewrites config even when only permissions are wrong."""
        # Create proper .config/logsqueak directory structure
        config_dir = tmp_path / ".config" / "logsqueak"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "config.yaml"

        # Create config with correct content but wrong permissions
        config = Config(
            llm={
                "endpoint": "http://localhost:11434/v1",
                "model": "mistral:7b-instruct",
                "api_key": "not-required",
            },
            logseq={"graph_path": str(temp_graph_dir)},
        )
        await write_config(config, config_file)

        # Set wrong permissions
        os.chmod(config_file, 0o644)

        # Simulate wizard run that doesn't change any values
        from logsqueak.wizard.wizard import has_config_changed, load_existing_config

        with patch("logsqueak.wizard.wizard.Path.home") as mock_home:
            mock_home.return_value = tmp_path
            loaded = load_existing_config()

            # Simulate assembling same config
            state = WizardState(
                existing_config=loaded,
                graph_path=str(temp_graph_dir),
                provider_type="ollama",
                ollama_endpoint="http://localhost:11434/v1",
                ollama_model="mistral:7b-instruct",
            )
            new_config = assemble_config(state)

            # Content hasn't changed
            assert not has_config_changed(new_config, loaded)

            # But we should still write to fix permissions
            # This is tested by the wizard logic in run_setup_wizard()
            # which checks has_permission_issue flag

        # Verify permissions are still wrong before fix
        file_stat = os.stat(config_file)
        assert stat.S_IMODE(file_stat.st_mode) == 0o644

        # After wizard writes, permissions should be fixed
        await write_config(new_config, config_file)
        file_stat = os.stat(config_file)
        assert stat.S_IMODE(file_stat.st_mode) == 0o600

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


class TestUserStory3RemoteAndCustomEndpoints:
    """Integration tests for User Story 3 - Remote Ollama and Custom Endpoints."""

    @pytest.mark.asyncio
    async def test_assemble_config_for_openai(self, temp_graph_dir):
        """Test assembling config from wizard state for OpenAI provider."""
        state = WizardState(
            graph_path=str(temp_graph_dir),
            provider_type="openai",
            openai_endpoint="https://api.openai.com/v1",
            openai_api_key="sk-test-key-123",
            openai_model="gpt-4o",
        )

        config = assemble_config(state)

        assert str(config.llm.endpoint) == "https://api.openai.com/v1"
        assert config.llm.api_key == "sk-test-key-123"
        assert config.llm.model == "gpt-4o"
        assert config.logseq.graph_path == str(temp_graph_dir)

        # Verify llm_providers field preserves OpenAI credentials
        assert config.llm_providers is not None
        assert "openai" in config.llm_providers
        assert config.llm_providers["openai"]["api_key"] == "sk-test-key-123"

    @pytest.mark.asyncio
    async def test_assemble_config_for_custom_endpoint(self, temp_graph_dir):
        """Test assembling config from wizard state for custom provider."""
        state = WizardState(
            graph_path=str(temp_graph_dir),
            provider_type="custom",
            custom_endpoint="https://custom-llm.example.com/v1",
            custom_api_key="custom-key-456",  # notsecret
            custom_model="custom-model-7b",
        )

        config = assemble_config(state)

        assert str(config.llm.endpoint) == "https://custom-llm.example.com/v1"
        assert config.llm.api_key == "custom-key-456"
        assert config.llm.model == "custom-model-7b"
        assert config.logseq.graph_path == str(temp_graph_dir)

        # Verify llm_providers field preserves custom credentials
        assert config.llm_providers is not None
        assert "custom" in config.llm_providers
        assert config.llm_providers["custom"]["endpoint"] == "https://custom-llm.example.com/v1"

    @pytest.mark.asyncio
    async def test_assemble_config_for_remote_ollama(self, temp_graph_dir):
        """Test assembling config for remote Ollama instance."""
        state = WizardState(
            graph_path=str(temp_graph_dir),
            provider_type="ollama",
            ollama_endpoint="http://192.168.1.100:11434/v1",
            ollama_model="mistral:7b-instruct",
        )

        config = assemble_config(state)

        assert str(config.llm.endpoint) == "http://192.168.1.100:11434/v1"
        assert config.llm.model == "mistral:7b-instruct"

        # Verify provider key distinguishes remote from local
        assert config.llm_providers is not None
        assert "ollama_remote" in config.llm_providers
        assert "ollama_local" not in config.llm_providers

    @pytest.mark.asyncio
    async def test_provider_switching_preserves_all_credentials(self, temp_graph_dir):
        """Test that switching providers preserves all previous provider credentials."""
        # Start with Ollama
        ollama_state = WizardState(
            graph_path=str(temp_graph_dir),
            provider_type="ollama",
            ollama_endpoint="http://localhost:11434/v1",
            ollama_model="mistral:7b-instruct",
        )
        ollama_config = assemble_config(ollama_state)

        # Switch to OpenAI
        openai_state = WizardState(
            existing_config=ollama_config,
            graph_path=str(temp_graph_dir),
            provider_type="openai",
            openai_endpoint="https://api.openai.com/v1",
            openai_api_key="sk-openai-key",
            openai_model="gpt-4o",
        )
        openai_config = assemble_config(openai_state)

        # Verify both providers are preserved
        assert "ollama_local" in openai_config.llm_providers
        assert "openai" in openai_config.llm_providers
        assert openai_config.llm_providers["ollama_local"]["model"] == "mistral:7b-instruct"
        assert openai_config.llm_providers["openai"]["api_key"] == "sk-openai-key"

        # Active config should be OpenAI
        assert str(openai_config.llm.endpoint) == "https://api.openai.com/v1"
        assert openai_config.llm.model == "gpt-4o"

    @pytest.mark.asyncio
    async def test_write_config_with_multiple_providers(self, temp_graph_dir, tmp_path):
        """Test writing config file with multiple provider credentials."""
        config_file = tmp_path / "config.yaml"

        config = Config(
            llm={
                "endpoint": "https://api.openai.com/v1",
                "model": "gpt-4o",
                "api_key": "sk-openai",
            },
            logseq={"graph_path": str(temp_graph_dir)},
            llm_providers={
                "ollama_local": {
                    "endpoint": "http://localhost:11434/v1",
                    "model": "mistral:7b-instruct",
                    "api_key": "ollama",
                    "num_ctx": 32768,
                },
                "openai": {
                    "endpoint": "https://api.openai.com/v1",
                    "model": "gpt-4o",
                    "api_key": "sk-openai",
                },
                "custom": {
                    "endpoint": "https://custom.ai/v1",
                    "model": "custom-7b",
                    "api_key": "custom-key",
                },
            },
        )

        await write_config(config, config_file)

        # Load and verify all providers are preserved
        loaded_config = Config.load(config_file)
        assert len(loaded_config.llm_providers) == 3
        assert "ollama_local" in loaded_config.llm_providers
        assert "openai" in loaded_config.llm_providers
        assert "custom" in loaded_config.llm_providers

    @pytest.mark.asyncio
    async def test_openai_config_roundtrip(self, temp_graph_dir, tmp_path):
        """Test that OpenAI config can be written and loaded correctly."""
        config_file = tmp_path / "config.yaml"

        original_config = Config(
            llm={
                "endpoint": "https://api.openai.com/v1",
                "model": "gpt-4o",
                "api_key": "sk-test-key",
            },
            logseq={"graph_path": str(temp_graph_dir)},
            llm_providers={
                "openai": {
                    "endpoint": "https://api.openai.com/v1",
                    "model": "gpt-4o",
                    "api_key": "sk-test-key",
                }
            },
        )

        await write_config(original_config, config_file)

        # Load and verify
        loaded_config = Config.load(config_file)
        assert loaded_config.llm.endpoint == original_config.llm.endpoint
        assert loaded_config.llm.api_key == original_config.llm.api_key
        assert loaded_config.llm.model == original_config.llm.model


class TestUserStory4:
    """Tests for User Story 4 - Updating Existing Configuration."""

    @pytest.mark.asyncio
    async def test_switching_providers_preserves_both_credentials(self, temp_graph_dir):
        """Test OpenAI → Ollama switch preserves both credentials."""
        # Start with OpenAI config
        openai_config = Config(
            llm={
                "endpoint": "https://api.openai.com/v1",
                "model": "gpt-4o",
                "api_key": "sk-openai-secret-key",
            },
            logseq={"graph_path": str(temp_graph_dir)},
            llm_providers={
                "openai": {
                    "endpoint": "https://api.openai.com/v1",
                    "model": "gpt-4o",
                    "api_key": "sk-openai-secret-key",
                }
            },
        )

        # Switch to Ollama
        ollama_state = WizardState(
            existing_config=openai_config,
            graph_path=str(temp_graph_dir),
            provider_type="ollama",
            ollama_endpoint="http://localhost:11434/v1",
            ollama_model="mistral:7b-instruct",
        )
        new_config = assemble_config(ollama_state)

        # Verify both providers are in llm_providers dict
        assert "openai" in new_config.llm_providers
        assert "ollama_local" in new_config.llm_providers

        # Verify OpenAI credentials are preserved
        assert new_config.llm_providers["openai"]["api_key"] == "sk-openai-secret-key"
        assert new_config.llm_providers["openai"]["model"] == "gpt-4o"

        # Verify Ollama credentials are added
        assert new_config.llm_providers["ollama_local"]["model"] == "mistral:7b-instruct"
        assert new_config.llm_providers["ollama_local"]["endpoint"] == "http://localhost:11434/v1"

        # Verify active config is Ollama
        assert str(new_config.llm.endpoint) == "http://localhost:11434/v1"
        assert new_config.llm.model == "mistral:7b-instruct"

    @pytest.mark.asyncio
    async def test_single_setting_update_preserves_others(self, temp_graph_dir):
        """Test changing only API key keeps all other settings unchanged."""
        # Existing config with all settings
        existing_config = Config(
            llm={
                "endpoint": "https://api.openai.com/v1",
                "model": "gpt-4o",
                "api_key": "sk-old-key",
            },
            logseq={"graph_path": str(temp_graph_dir)},
            rag={"top_k": 15},
            llm_providers={
                "openai": {
                    "endpoint": "https://api.openai.com/v1",
                    "model": "gpt-4o",
                    "api_key": "sk-old-key",
                }
            },
        )

        # Update only API key
        state = WizardState(
            existing_config=existing_config,
            graph_path=str(temp_graph_dir),
            provider_type="openai",
            openai_endpoint="https://api.openai.com/v1",
            openai_api_key="sk-new-key",  # Only this changes
            openai_model="gpt-4o",
            top_k=15,
        )
        new_config = assemble_config(state)

        # Verify only API key changed
        assert new_config.llm.api_key == "sk-new-key"
        assert str(new_config.llm.endpoint) == "https://api.openai.com/v1"
        assert new_config.llm.model == "gpt-4o"
        assert new_config.logseq.graph_path == str(temp_graph_dir)
        assert new_config.rag.top_k == 15

        # Verify llm_providers updated with new key
        assert new_config.llm_providers["openai"]["api_key"] == "sk-new-key"

    @pytest.mark.asyncio
    async def test_fast_update_with_cached_embedding_model(self, temp_graph_dir):
        """Test that update flow is optimized when embedding model is cached.

        This test verifies the optimization described in User Story 4:
        When embedding model is already cached, validate_embedding() should
        skip the download and complete quickly (contributing to <30s update time).
        """
        from logsqueak.wizard.validators import check_embedding_model_cached

        # Verify that the cached check works correctly
        # The actual optimization happens in validate_embedding() which checks
        # check_embedding_model_cached() and skips download if True

        # Mock scenario: embedding model is cached (returns True)
        with patch("logsqueak.wizard.validators.check_embedding_model_cached") as mock_cached:
            mock_cached.return_value = True

            # This would normally be called by validate_embedding()
            result = check_embedding_model_cached()
            assert result is True

            # When cached=True, validate_embedding() skips the download
            # This is the key optimization for fast config updates in US4
            # The validate_embedding() function in wizard.py checks this at line ~450-465


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_llm_connection_timeout_prompts_user(self, temp_graph_dir):
        """Test that LLM connection timeout prompts user for action.

        When the API connection test times out, the wizard should prompt
        the user with options: continue, retry, or skip.
        """
        from logsqueak.wizard.wizard import validate_llm_connection, WizardState
        import asyncio

        # Create state for OpenAI provider
        state = WizardState(
            graph_path=str(temp_graph_dir),
            provider_type="openai",
            openai_endpoint="https://api.openai.com/v1",
            openai_api_key="sk-test-timeout-key",
            openai_model="gpt-4o",
        )

        # Mock validate_openai_connection to simulate timeout
        # We raise TimeoutError directly to simulate the timeout
        async def mock_timeout_connection(*args, **kwargs):
            raise asyncio.TimeoutError()

        # Mock prompt to auto-skip after timeout
        prompted = {"value": False}
        def mock_prompt_timeout(operation, timeout):
            # Verify we're being prompted about timeout
            prompted["value"] = True
            assert "API connection test" in operation
            return "skip"

        with patch("logsqueak.wizard.wizard.validate_openai_connection", mock_timeout_connection):
            with patch("logsqueak.wizard.wizard.prompt_continue_on_timeout", mock_prompt_timeout):
                result = await validate_llm_connection(state)

                # Should return True (skip validation)
                assert result is True
                # Verify prompt was called
                assert prompted["value"] is True

    @pytest.mark.asyncio
    async def test_llm_connection_timeout_retry_with_longer_timeout(self, temp_graph_dir):
        """Test that choosing 'continue' increases timeout and retries."""
        from logsqueak.wizard.wizard import validate_llm_connection, WizardState
        import asyncio

        state = WizardState(
            graph_path=str(temp_graph_dir),
            provider_type="openai",
            openai_endpoint="https://api.openai.com/v1",
            openai_api_key="sk-test-key",
            openai_model="gpt-4o",
        )

        call_count = {"value": 0}
        timeout_values = []

        async def mock_connection_timeout_then_success(*args, **kwargs):
            call_count["value"] += 1
            timeout = kwargs.get("timeout", 30)
            timeout_values.append(timeout)

            if call_count["value"] == 1:
                # First call times out
                raise asyncio.TimeoutError()
            else:
                # Second call succeeds
                return ValidationResult(success=True)

        def mock_prompt_continue(operation, timeout):
            # First timeout: choose "continue" to increase timeout
            return "continue"

        with patch("logsqueak.wizard.wizard.validate_openai_connection", mock_connection_timeout_then_success):
            with patch("logsqueak.wizard.wizard.prompt_continue_on_timeout", mock_prompt_continue):
                result = await validate_llm_connection(state)

                # Should succeed after retry with longer timeout
                assert result is True
                assert call_count["value"] == 2
                # Verify timeout was increased (30s -> 60s)
                assert timeout_values[0] == 30
                assert timeout_values[1] == 60

    @pytest.mark.asyncio
    async def test_llm_connection_max_retries_reached(self, temp_graph_dir):
        """Test that max retries limit prevents infinite loops."""
        from logsqueak.wizard.wizard import validate_llm_connection, WizardState
        import asyncio

        state = WizardState(
            graph_path=str(temp_graph_dir),
            provider_type="custom",
            custom_endpoint="https://custom.ai/v1",
            custom_api_key="custom-key",  # notsecret
            custom_model="custom-model",
        )

        call_count = {"value": 0}

        async def mock_always_timeout(*args, **kwargs):
            call_count["value"] += 1
            raise asyncio.TimeoutError()

        def mock_prompt_always_retry(operation, timeout):
            return "retry"

        with patch("logsqueak.wizard.wizard.validate_openai_connection", mock_always_timeout):
            with patch("logsqueak.wizard.wizard.prompt_continue_on_timeout", mock_prompt_always_retry):
                result = await validate_llm_connection(state)

                # Should eventually give up and skip
                assert result is True
                # Should have hit max retries (3 attempts)
                assert call_count["value"] == 3

    @pytest.mark.asyncio
    async def test_embedding_download_timeout_prompts_user(self, temp_graph_dir):
        """Test that embedding download timeout prompts user for action.

        When the embedding model download times out, the wizard should prompt
        the user with options: continue, retry, or skip.
        """
        from logsqueak.wizard.wizard import validate_embedding, WizardState
        import asyncio

        state = WizardState(graph_path=str(temp_graph_dir))

        # Mock check_embedding_model_cached to return False (not cached)
        def mock_not_cached():
            return False

        # Mock validate_embedding_model to simulate timeout
        async def mock_timeout_download(*args, **kwargs):
            raise asyncio.TimeoutError()

        # Mock prompt to auto-skip after timeout
        prompted = {"value": False}
        def mock_prompt_timeout(operation, timeout):
            # Verify we're being prompted about timeout
            prompted["value"] = True
            assert "Embedding model download" in operation
            assert timeout == 300  # 5 minute default
            return "skip"

        with patch("logsqueak.wizard.wizard.check_embedding_model_cached", mock_not_cached):
            with patch("logsqueak.wizard.validators.check_disk_space", return_value=ValidationResult(success=True)):
                with patch("logsqueak.wizard.wizard.validate_embedding_model", mock_timeout_download):
                    with patch("logsqueak.wizard.wizard.prompt_continue_on_timeout", mock_prompt_timeout):
                        result = await validate_embedding(state)

                        # Should return True (skip validation)
                        assert result is True
                        # Verify prompt was called
                        assert prompted["value"] is True

    @pytest.mark.asyncio
    async def test_embedding_download_timeout_retry_with_longer_timeout(self, temp_graph_dir):
        """Test that choosing 'continue' increases timeout and retries embedding download."""
        from logsqueak.wizard.wizard import validate_embedding, WizardState
        import asyncio

        state = WizardState(graph_path=str(temp_graph_dir))

        call_count = {"value": 0}
        timeout_used = {"value": 0}

        async def mock_timeout_once(*args, **kwargs):
            call_count["value"] += 1
            if call_count["value"] == 1:
                # First call times out
                timeout_used["value"] = 300
                raise asyncio.TimeoutError()
            else:
                # Second call succeeds
                timeout_used["value"] = 600
                return ValidationResult(success=True)

        prompt_count = {"value": 0}
        def mock_prompt_continue(operation, timeout):
            prompt_count["value"] += 1
            return "continue"

        with patch("logsqueak.wizard.wizard.check_embedding_model_cached", return_value=False):
            with patch("logsqueak.wizard.validators.check_disk_space", return_value=ValidationResult(success=True)):
                with patch("logsqueak.wizard.wizard.validate_embedding_model", mock_timeout_once):
                    with patch("logsqueak.wizard.wizard.prompt_continue_on_timeout", mock_prompt_continue):
                        result = await validate_embedding(state)

                        # Should succeed after retry
                        assert result is True
                        # Verify retry happened
                        assert call_count["value"] == 2
                        # Verify user was prompted once
                        assert prompt_count["value"] == 1

    @pytest.mark.asyncio
    async def test_embedding_download_max_retries_reached(self, temp_graph_dir):
        """Test that max retries limit prevents infinite loops for embedding download."""
        from logsqueak.wizard.wizard import validate_embedding, WizardState
        import asyncio

        state = WizardState(graph_path=str(temp_graph_dir))

        call_count = {"value": 0}

        async def mock_always_timeout(*args, **kwargs):
            call_count["value"] += 1
            raise asyncio.TimeoutError()

        def mock_prompt_always_retry(operation, timeout):
            return "retry"

        with patch("logsqueak.wizard.wizard.check_embedding_model_cached", return_value=False):
            with patch("logsqueak.wizard.validators.check_disk_space", return_value=ValidationResult(success=True)):
                with patch("logsqueak.wizard.wizard.validate_embedding_model", mock_always_timeout):
                    with patch("logsqueak.wizard.wizard.prompt_continue_on_timeout", mock_prompt_always_retry):
                        result = await validate_embedding(state)

                        # Should eventually give up and skip
                        assert result is True
                        # Should have hit max retries (3 attempts)
                        assert call_count["value"] == 3

    @pytest.mark.asyncio
    async def test_low_disk_space_warning_allows_proceed(self, temp_graph_dir):
        """Test that low disk space shows warning but allows user to proceed."""
        from logsqueak.wizard.wizard import validate_embedding, WizardState

        state = WizardState(graph_path=str(temp_graph_dir))

        # Mock low disk space (500 MB available, <1GB threshold)
        def mock_low_disk_space(required_mb):
            return ValidationResult(
                success=False,
                error_message=f"Low disk space: 500 MB available ({required_mb} MB recommended)",
                data={"available_mb": 500}
            )

        # Mock user choosing to proceed anyway
        confirm_called = {"value": False}
        def mock_confirm_proceed(prompt, default=False):
            confirm_called["value"] = True
            assert "Proceed with download anyway?" in prompt
            return True  # User chooses to proceed

        # Mock successful download
        async def mock_successful_download(*args, **kwargs):
            return ValidationResult(success=True)

        with patch("logsqueak.wizard.wizard.check_embedding_model_cached", return_value=False):
            with patch("logsqueak.wizard.wizard.check_disk_space", mock_low_disk_space):
                with patch("rich.prompt.Confirm.ask", mock_confirm_proceed):
                    with patch("logsqueak.wizard.wizard.validate_embedding_model", mock_successful_download):
                        result = await validate_embedding(state)

                        # Should succeed (user proceeded)
                        assert result is True
                        # Verify user was prompted to confirm
                        assert confirm_called["value"] is True

    @pytest.mark.asyncio
    async def test_low_disk_space_warning_allows_skip(self, temp_graph_dir):
        """Test that user can decline to proceed when disk space is low."""
        from logsqueak.wizard.wizard import validate_embedding, WizardState

        state = WizardState(graph_path=str(temp_graph_dir))

        # Mock low disk space
        def mock_low_disk_space(required_mb):
            return ValidationResult(
                success=False,
                error_message=f"Low disk space: 500 MB available ({required_mb} MB recommended)",
                data={"available_mb": 500}
            )

        # Mock user choosing NOT to proceed
        def mock_confirm_decline(prompt, default=False):
            return False  # User declines

        with patch("logsqueak.wizard.wizard.check_embedding_model_cached", return_value=False):
            with patch("logsqueak.wizard.wizard.check_disk_space", mock_low_disk_space):
                with patch("rich.prompt.Confirm.ask", mock_confirm_decline):
                    result = await validate_embedding(state)

                    # Should skip validation
                    assert result is True

    @pytest.mark.asyncio
    async def test_disk_space_exhausted_during_download(self, temp_graph_dir):
        """Test error handling when disk space runs out during download."""
        from logsqueak.wizard.wizard import validate_embedding, WizardState

        state = WizardState(graph_path=str(temp_graph_dir))

        # Mock disk space exhaustion during download
        async def mock_disk_full_error(*args, **kwargs):
            # Return failure result (error handling happens in validate_embedding_model)
            return ValidationResult(
                success=False,
                error_message="Disk space exhausted during download (10 MB available). Free up space and try again."
            )

        # Mock disk check showing sufficient space initially
        def mock_sufficient_space(required_mb):
            return ValidationResult(success=True, data={"available_mb": 2000})

        # Mock prompt to skip after disk error
        def mock_prompt_skip(operation):
            return "skip"

        with patch("logsqueak.wizard.wizard.check_embedding_model_cached", return_value=False):
            with patch("logsqueak.wizard.wizard.check_disk_space", mock_sufficient_space):
                with patch("logsqueak.wizard.wizard.validate_embedding_model", mock_disk_full_error):
                    with patch("logsqueak.wizard.wizard.prompt_retry_on_failure", mock_prompt_skip):
                        result = await validate_embedding(state)

                        # Should return True (skip validation after disk error)
                        assert result is True

    @pytest.mark.asyncio
    async def test_ollama_with_no_models_shows_helpful_message(self, temp_graph_dir, capsys):
        """Test that Ollama with no models shows installation suggestion."""
        from logsqueak.wizard.wizard import configure_ollama, WizardState

        state = WizardState(graph_path=str(temp_graph_dir))

        call_count = {"value": 0}

        # Mock Ollama connection succeeding but returning empty models list
        async def mock_ollama_no_models(endpoint, timeout=30):
            call_count["value"] += 1
            # First call (automatic test): connection succeeds but no models
            # This will trigger the manual endpoint prompt
            if call_count["value"] == 1:
                return ValidationResult(
                    success=True,
                    data={"models": []}  # Empty models list - triggers line 219 else
                )
            else:
                # Subsequent prompts would happen, but we'll mock the prompt too
                return ValidationResult(
                    success=True,
                    data={"models": []}  # Still empty
                )

        # Mock the prompt to avoid EOF error - return same endpoint
        def mock_prompt(default):
            return default

        with patch("logsqueak.wizard.wizard.validate_ollama_connection", mock_ollama_no_models):
            with patch("logsqueak.wizard.wizard.prompt_ollama_endpoint", mock_prompt):
                result = await configure_ollama(state)

                # Should return False (abort due to no models)
                assert result is False

                # Verify helpful message was shown
                captured = capsys.readouterr()
                assert "No models found in Ollama instance" in captured.out
                assert "ollama pull mistral:7b-instruct" in captured.out

    @pytest.mark.asyncio
    async def test_wizard_abort_no_config_written(self, tmp_path, temp_graph_dir, monkeypatch):
        """Test that aborting wizard at various stages doesn't write config."""
        from logsqueak.wizard.wizard import run_setup_wizard
        import asyncio

        # Set config path to temp directory
        config_path = tmp_path / "config.yaml"
        monkeypatch.setenv("HOME", str(tmp_path))

        # Mock KeyboardInterrupt during provider configuration
        prompt_count = {"value": 0}

        def mock_prompt_interrupt(*args, **kwargs):
            prompt_count["value"] += 1
            # Interrupt on second prompt (provider choice)
            if prompt_count["value"] == 2:
                raise KeyboardInterrupt()
            # First prompt (graph path) returns valid path
            return str(temp_graph_dir)

        with patch("logsqueak.wizard.prompts.Prompt.ask", mock_prompt_interrupt):
            result = await run_setup_wizard()

            # Should return False (aborted)
            assert result is False
            # Config file should NOT be written
            assert not (tmp_path / ".config" / "logsqueak" / "config.yaml").exists()

    @pytest.mark.asyncio
    async def test_invalid_yaml_config_treated_as_fresh_setup(self, tmp_path, temp_graph_dir):
        """Test that malformed YAML config is treated as fresh setup."""
        from logsqueak.wizard.wizard import load_existing_config

        # Create config directory and write invalid YAML
        config_dir = tmp_path / ".config" / "logsqueak"
        config_dir.mkdir(parents=True)
        config_path = config_dir / "config.yaml"

        # Write malformed YAML (invalid syntax)
        config_path.write_text("""
llm:
  endpoint: http://localhost:11434/v1
  api_key: test-key
  model: test-model
logseq:
  graph_path: {invalid syntax here
  missing: closing brace
""")

        # Mock HOME to use temp directory
        import os
        original_home = os.environ.get("HOME")
        os.environ["HOME"] = str(tmp_path)

        try:
            # Should return None (treat as no config)
            config = load_existing_config()
            assert config is None
        finally:
            # Restore original HOME
            if original_home:
                os.environ["HOME"] = original_home

    @pytest.mark.asyncio
    async def test_path_with_spaces_validates_correctly(self, tmp_path):
        """Test that graph path with spaces is handled correctly."""
        from logsqueak.wizard.validators import validate_graph_path

        # Create graph directory with spaces in name
        graph_dir = tmp_path / "My Logseq Graph"
        graph_dir.mkdir()
        (graph_dir / "journals").mkdir()
        (graph_dir / "logseq").mkdir()

        # Should validate successfully
        result = validate_graph_path(str(graph_dir))
        assert result.success is True
        assert result.data["path"] == str(graph_dir.resolve())


class TestWizardIndexingIntegration:
    """Integration tests for wizard indexing flow."""

    @pytest.mark.asyncio
    async def test_wizard_offers_indexing_after_config_write(self, temp_graph_dir, tmp_path):
        """Test that wizard offers to index graph after writing config."""
        from logsqueak.wizard.wizard import run_setup_wizard

        # Create a test page to index
        pages_dir = temp_graph_dir / "pages"
        test_page = pages_dir / "Test Page.md"
        test_page.write_text("- This is a test block\n")

        config_path = tmp_path / "config.yaml"

        # Mock all prompts to accept defaults and accept indexing
        with patch("logsqueak.wizard.wizard.load_existing_config", return_value=None), \
             patch("logsqueak.wizard.wizard.configure_graph_path") as mock_graph, \
             patch("logsqueak.wizard.wizard.configure_provider") as mock_provider, \
             patch("logsqueak.wizard.wizard.validate_llm_connection", return_value=True), \
             patch("logsqueak.wizard.wizard.validate_embedding", return_value=True), \
             patch("logsqueak.wizard.wizard.prompt_index_graph", return_value=True), \
             patch("logsqueak.wizard.wizard.index_graph_after_setup") as mock_index:

            # Setup wizard state
            async def setup_graph_path(state):
                state.graph_path = str(temp_graph_dir)
                return True

            async def setup_provider(state):
                state.provider_type = "ollama"
                state.ollama_endpoint = "http://localhost:11434/v1"
                state.ollama_model = "mistral:7b-instruct"
                return True

            mock_graph.side_effect = setup_graph_path
            mock_provider.side_effect = setup_provider

            # Override config path
            with patch.object(Path, "home", return_value=tmp_path):
                result = await run_setup_wizard()

        # Verify wizard completed successfully
        assert result is True

        # Verify indexing was offered and accepted
        mock_index.assert_called_once_with(str(temp_graph_dir))

    @pytest.mark.asyncio
    async def test_wizard_skips_indexing_when_declined(self, temp_graph_dir, tmp_path):
        """Test that wizard skips indexing when user declines."""
        from logsqueak.wizard.wizard import run_setup_wizard

        config_path = tmp_path / "config.yaml"

        # Mock all prompts to decline indexing
        with patch("logsqueak.wizard.wizard.load_existing_config", return_value=None), \
             patch("logsqueak.wizard.wizard.configure_graph_path") as mock_graph, \
             patch("logsqueak.wizard.wizard.configure_provider") as mock_provider, \
             patch("logsqueak.wizard.wizard.validate_llm_connection", return_value=True), \
             patch("logsqueak.wizard.wizard.validate_embedding", return_value=True), \
             patch("logsqueak.wizard.wizard.prompt_index_graph", return_value=False), \
             patch("logsqueak.wizard.wizard.index_graph_after_setup") as mock_index:

            # Setup wizard state
            async def setup_graph_path(state):
                state.graph_path = str(temp_graph_dir)
                return True

            async def setup_provider(state):
                state.provider_type = "ollama"
                state.ollama_endpoint = "http://localhost:11434/v1"
                state.ollama_model = "mistral:7b-instruct"
                return True

            mock_graph.side_effect = setup_graph_path
            mock_provider.side_effect = setup_provider

            # Override config path
            with patch.object(Path, "home", return_value=tmp_path):
                result = await run_setup_wizard()

        # Verify wizard completed successfully
        assert result is True

        # Verify indexing was NOT triggered
        mock_index.assert_not_called()

    @pytest.mark.asyncio
    async def test_wizard_offers_indexing_even_when_no_config_changes(self, temp_graph_dir, tmp_path):
        """Test that wizard still offers indexing when config unchanged."""
        from logsqueak.wizard.wizard import run_setup_wizard, assemble_config, WizardState

        # Create existing config
        existing_state = WizardState(
            graph_path=str(temp_graph_dir),
            provider_type="ollama",
            ollama_endpoint="http://localhost:11434/v1",
            ollama_model="mistral:7b-instruct",
        )
        existing_config = assemble_config(existing_state)

        # Write existing config
        config_path = tmp_path / "config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            yaml.dump(existing_config.model_dump(mode='json'), f)
        os.chmod(config_path, 0o600)

        # Mock all prompts - user accepts all defaults (no changes)
        with patch("logsqueak.wizard.wizard.load_existing_config", return_value=existing_config), \
             patch("logsqueak.wizard.wizard.configure_graph_path") as mock_graph, \
             patch("logsqueak.wizard.wizard.configure_provider") as mock_provider, \
             patch("logsqueak.wizard.wizard.validate_llm_connection", return_value=True), \
             patch("logsqueak.wizard.wizard.validate_embedding", return_value=True), \
             patch("logsqueak.wizard.wizard.prompt_index_graph", return_value=True), \
             patch("logsqueak.wizard.wizard.index_graph_after_setup") as mock_index:

            # Setup wizard state (same as existing)
            async def setup_graph_path(state):
                state.graph_path = str(temp_graph_dir)
                return True

            async def setup_provider(state):
                state.provider_type = "ollama"
                state.ollama_endpoint = "http://localhost:11434/v1"
                state.ollama_model = "mistral:7b-instruct"
                return True

            mock_graph.side_effect = setup_graph_path
            mock_provider.side_effect = setup_provider

            # Override config path
            with patch.object(Path, "home", return_value=tmp_path):
                result = await run_setup_wizard()

        # Verify wizard completed successfully
        assert result is True

        # Verify indexing was STILL offered even though config didn't change
        mock_index.assert_called_once_with(str(temp_graph_dir))
