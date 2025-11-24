"""Unit tests for wizard orchestration functions."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from logsqueak.wizard.wizard import offer_graph_indexing, index_graph_after_setup


class TestOfferGraphIndexing:
    """Tests for offer_graph_indexing function."""

    @pytest.mark.asyncio
    @patch("logseq_outline.graph.GraphPaths")
    @patch("logsqueak.services.page_indexer.PageIndexer")
    @patch("logsqueak.services.rag_search.RAGSearch")
    @patch("logsqueak.wizard.wizard.prompt_index_graph")
    @patch("logsqueak.wizard.wizard.index_graph_after_setup")
    async def test_calls_index_when_user_accepts(self, mock_index, mock_prompt, mock_rag, mock_indexer, mock_graph_paths):
        """Test that indexing is triggered when user accepts."""
        # Setup: Index doesn't exist
        mock_rag_instance = MagicMock()
        mock_rag_instance.has_indexed_data.return_value = False
        mock_rag.return_value = mock_rag_instance

        mock_prompt.return_value = True
        mock_index.return_value = True

        await offer_graph_indexing("/path/to/graph")

        mock_prompt.assert_called_once()
        mock_index.assert_called_once_with("/path/to/graph")

    @pytest.mark.asyncio
    @patch("logseq_outline.graph.GraphPaths")
    @patch("logsqueak.services.page_indexer.PageIndexer")
    @patch("logsqueak.services.rag_search.RAGSearch")
    @patch("logsqueak.wizard.wizard.prompt_index_graph")
    @patch("logsqueak.wizard.wizard.index_graph_after_setup")
    async def test_skips_index_when_user_declines(self, mock_index, mock_prompt, mock_rag, mock_indexer, mock_graph_paths):
        """Test that indexing is skipped when user declines."""
        # Setup: Index doesn't exist
        mock_rag_instance = MagicMock()
        mock_rag_instance.has_indexed_data.return_value = False
        mock_rag.return_value = mock_rag_instance

        mock_prompt.return_value = False

        await offer_graph_indexing("/path/to/graph")

        mock_prompt.assert_called_once()
        mock_index.assert_not_called()

    @pytest.mark.asyncio
    @patch("logseq_outline.graph.GraphPaths")
    @patch("logsqueak.services.page_indexer.PageIndexer")
    @patch("logsqueak.services.rag_search.RAGSearch")
    @patch("logsqueak.wizard.wizard.prompt_index_graph")
    @patch("logsqueak.wizard.wizard.index_graph_after_setup")
    @patch("logsqueak.wizard.wizard.rprint")
    async def test_shows_skip_message_when_declined(self, mock_rprint, mock_index, mock_prompt, mock_rag, mock_indexer, mock_graph_paths):
        """Test that helpful message is shown when user declines."""
        # Setup: Index doesn't exist
        mock_rag_instance = MagicMock()
        mock_rag_instance.has_indexed_data.return_value = False
        mock_rag.return_value = mock_rag_instance

        mock_prompt.return_value = False

        await offer_graph_indexing("/path/to/graph")

        # Verify informative message was shown
        assert mock_rprint.call_count >= 2
        calls_text = " ".join(str(call) for call in mock_rprint.call_args_list)
        assert "later" in calls_text.lower()
        assert "search" in calls_text.lower()

    @pytest.mark.asyncio
    @patch("logseq_outline.graph.GraphPaths")
    @patch("logsqueak.services.page_indexer.PageIndexer")
    @patch("logsqueak.services.rag_search.RAGSearch")
    @patch("logsqueak.wizard.wizard.prompt_index_graph")
    @patch("logsqueak.wizard.wizard.index_graph_after_setup")
    async def test_skips_prompt_when_index_exists(self, mock_index, mock_prompt, mock_rag, mock_indexer, mock_graph_paths):
        """Test that prompt is skipped when index already exists."""
        # Setup: Index already exists
        mock_rag_instance = MagicMock()
        mock_rag_instance.has_indexed_data.return_value = True
        mock_rag.return_value = mock_rag_instance

        await offer_graph_indexing("/path/to/graph")

        # Should not prompt or index
        mock_prompt.assert_not_called()
        mock_index.assert_not_called()

    @pytest.mark.asyncio
    @patch("logseq_outline.graph.GraphPaths", side_effect=Exception("Graph error"))
    @patch("logsqueak.wizard.wizard.prompt_index_graph")
    @patch("logsqueak.wizard.wizard.index_graph_after_setup")
    async def test_skips_prompt_on_check_error(self, mock_index, mock_prompt, mock_graph_paths):
        """Test that prompt is skipped if index check fails."""
        await offer_graph_indexing("/path/to/graph")

        # Should not prompt or index (safe default)
        mock_prompt.assert_not_called()
        mock_index.assert_not_called()


class TestIndexGraphAfterSetup:
    """Tests for index_graph_after_setup function."""

    @pytest.mark.asyncio
    @patch("logseq_outline.graph.GraphPaths")
    @patch("logsqueak.services.page_indexer.PageIndexer")
    @patch("logsqueak.services.rag_search.RAGSearch")
    @patch("logsqueak.utils.index_progress.create_index_progress_callback")
    async def test_successful_initial_indexing(self, mock_progress, mock_rag, mock_indexer_class, mock_graph_paths):
        """Test successful initial graph indexing (no existing data)."""
        # Setup mocks
        mock_indexer = MagicMock()
        mock_indexer.build_index = AsyncMock(return_value=10)  # 10 pages indexed
        mock_indexer_class.return_value = mock_indexer

        mock_rag_instance = MagicMock()
        mock_rag_instance.has_indexed_data.return_value = False  # No existing data
        mock_rag.return_value = mock_rag_instance

        mock_callback = MagicMock()
        mock_cleanup = MagicMock()
        mock_progress.return_value = (mock_callback, mock_cleanup)

        # Execute
        result = await index_graph_after_setup("/path/to/graph")

        # Verify
        assert result is True
        mock_indexer.build_index.assert_called_once_with(progress_callback=mock_callback)
        mock_cleanup.assert_called_once()
        mock_rag_instance.has_indexed_data.assert_called_once()

    @pytest.mark.asyncio
    @patch("logseq_outline.graph.GraphPaths")
    @patch("logsqueak.services.page_indexer.PageIndexer")
    @patch("logsqueak.services.rag_search.RAGSearch")
    @patch("logsqueak.utils.index_progress.create_index_progress_callback")
    async def test_successful_update_indexing(self, mock_progress, mock_rag, mock_indexer_class, mock_graph_paths):
        """Test successful index update (existing data)."""
        # Setup mocks
        mock_indexer = MagicMock()
        mock_indexer.build_index = AsyncMock(return_value=3)  # 3 pages updated
        mock_indexer_class.return_value = mock_indexer

        mock_rag_instance = MagicMock()
        mock_rag_instance.has_indexed_data.return_value = True  # Existing data
        mock_rag.return_value = mock_rag_instance

        mock_callback = MagicMock()
        mock_cleanup = MagicMock()
        mock_progress.return_value = (mock_callback, mock_cleanup)

        # Execute
        result = await index_graph_after_setup("/path/to/graph")

        # Verify
        assert result is True
        mock_cleanup.assert_called_once()

    @pytest.mark.asyncio
    @patch("logseq_outline.graph.GraphPaths")
    @patch("logsqueak.services.page_indexer.PageIndexer")
    @patch("logsqueak.services.rag_search.RAGSearch")
    @patch("logsqueak.utils.index_progress.create_index_progress_callback")
    async def test_no_pages_need_indexing(self, mock_progress, mock_rag, mock_indexer_class, mock_graph_paths):
        """Test when all pages are already up-to-date."""
        # Setup mocks
        mock_indexer = MagicMock()
        mock_indexer.build_index = AsyncMock(return_value=0)  # 0 pages indexed
        mock_indexer_class.return_value = mock_indexer

        mock_rag_instance = MagicMock()
        mock_rag_instance.has_indexed_data.return_value = True
        mock_rag.return_value = mock_rag_instance

        mock_callback = MagicMock()
        mock_cleanup = MagicMock()
        mock_progress.return_value = (mock_callback, mock_cleanup)

        # Execute
        result = await index_graph_after_setup("/path/to/graph")

        # Verify - should still return True (success)
        assert result is True
        mock_cleanup.assert_called_once()

    @pytest.mark.asyncio
    @patch("logseq_outline.graph.GraphPaths")
    @patch("logsqueak.services.page_indexer.PageIndexer")
    @patch("logsqueak.services.rag_search.RAGSearch")
    @patch("logsqueak.utils.index_progress.create_index_progress_callback")
    async def test_handles_indexing_error(self, mock_progress, mock_rag, mock_indexer_class, mock_graph_paths):
        """Test error handling during indexing."""
        # Setup mocks
        mock_indexer = MagicMock()
        mock_indexer.build_index = AsyncMock(side_effect=Exception("Index error"))
        mock_indexer_class.return_value = mock_indexer

        mock_rag_instance = MagicMock()
        mock_rag_instance.has_indexed_data.return_value = False
        mock_rag.return_value = mock_rag_instance

        mock_callback = MagicMock()
        mock_cleanup = MagicMock()
        mock_progress.return_value = (mock_callback, mock_cleanup)

        # Execute
        with patch("logsqueak.wizard.wizard.rprint") as mock_rprint:
            result = await index_graph_after_setup("/path/to/graph")

        # Verify
        assert result is False
        mock_cleanup.assert_called_once()  # Cleanup should still be called
        # Verify error message was displayed
        calls_text = " ".join(str(call) for call in mock_rprint.call_args_list)
        assert "failed" in calls_text.lower() or "error" in calls_text.lower()

    @pytest.mark.asyncio
    @patch("logseq_outline.graph.GraphPaths", side_effect=Exception("Invalid graph path"))
    async def test_handles_initialization_error(self, mock_graph_paths):
        """Test error handling during service initialization."""
        # Execute
        with patch("logsqueak.wizard.wizard.rprint") as mock_rprint:
            result = await index_graph_after_setup("/invalid/path")

        # Verify
        assert result is False
        # Verify error message was displayed
        calls_text = " ".join(str(call) for call in mock_rprint.call_args_list)
        assert "failed" in calls_text.lower() or "error" in calls_text.lower()

    @pytest.mark.asyncio
    @patch("logseq_outline.graph.GraphPaths")
    @patch("logsqueak.services.page_indexer.PageIndexer")
    @patch("logsqueak.services.rag_search.RAGSearch")
    @patch("logsqueak.utils.index_progress.create_index_progress_callback")
    async def test_uses_correct_progress_callback_settings(self, mock_progress, mock_rag, mock_indexer_class, mock_graph_paths):
        """Test that progress callback is created with correct settings."""
        # Setup mocks
        mock_indexer = MagicMock()
        mock_indexer.build_index = AsyncMock(return_value=5)
        mock_indexer_class.return_value = mock_indexer

        mock_rag_instance = MagicMock()
        mock_rag_instance.has_indexed_data.return_value = False
        mock_rag.return_value = mock_rag_instance

        mock_callback = MagicMock()
        mock_cleanup = MagicMock()
        mock_progress.return_value = (mock_callback, mock_cleanup)

        # Execute
        await index_graph_after_setup("/path/to/graph")

        # Verify progress callback was created with wizard-appropriate settings
        mock_progress.assert_called_once()
        call_kwargs = mock_progress.call_args[1]
        assert call_kwargs["reindex"] is False  # Wizard never forces reindex
        assert call_kwargs["use_echo"] is False  # Wizard uses rprint style
        assert "has_data" in call_kwargs


class TestConfigureOpenAI:
    """Tests for configure_openai function."""

    @pytest.mark.asyncio
    async def test_prompts_for_api_key_and_model(self):
        """Test OpenAI configuration prompts for API key and model."""
        from logsqueak.wizard.wizard import configure_openai, WizardState

        state = WizardState()

        # Mock prompts
        with patch("logsqueak.wizard.wizard.prompt_openai_api_key") as mock_key, \
             patch("logsqueak.wizard.wizard.prompt_openai_model") as mock_model, \
             patch("logsqueak.wizard.wizard.rprint"):

            mock_key.return_value = "sk-test-key-123"
            mock_model.return_value = "gpt-4o"

            result = await configure_openai(state)

            # Verify prompts were called
            assert result is True
            mock_key.assert_called_once_with(None)
            mock_model.assert_called_once_with("gpt-4o")

            # Verify state was updated
            assert state.openai_api_key == "sk-test-key-123"
            assert state.openai_model == "gpt-4o"
            assert state.openai_endpoint == "https://api.openai.com/v1"

    @pytest.mark.asyncio
    async def test_uses_existing_credentials_as_defaults(self, tmp_path):
        """Test that existing OpenAI config is used as defaults."""
        from logsqueak.wizard.wizard import configure_openai, WizardState
        from logsqueak.models.config import Config

        # Create valid graph directory
        graph_dir = tmp_path / "graph"
        graph_dir.mkdir()
        (graph_dir / "journals").mkdir()
        (graph_dir / "logseq").mkdir()

        # Create existing config with OpenAI credentials
        existing_config = Config(
            llm={
                "endpoint": "https://api.openai.com/v1",
                "model": "gpt-4-turbo",
                "api_key": "sk-existing-key",
            },
            logseq={"graph_path": str(graph_dir)},
            llm_providers={
                "openai": {
                    "endpoint": "https://api.openai.com/v1",
                    "model": "gpt-4-turbo",
                    "api_key": "sk-existing-key",
                }
            }
        )

        state = WizardState(existing_config=existing_config)

        # Mock prompts
        with patch("logsqueak.wizard.wizard.prompt_openai_api_key") as mock_key, \
             patch("logsqueak.wizard.wizard.prompt_openai_model") as mock_model, \
             patch("logsqueak.wizard.wizard.rprint"):

            mock_key.return_value = "sk-new-key"
            mock_model.return_value = "gpt-4o"

            result = await configure_openai(state)

            # Verify existing key was passed as default
            mock_key.assert_called_once_with("sk-existing-key")
            # Verify existing model was passed as default
            mock_model.assert_called_once_with("gpt-4-turbo")

            assert result is True

    @pytest.mark.asyncio
    async def test_sets_openai_endpoint_correctly(self):
        """Test that OpenAI endpoint is always set to api.openai.com."""
        from logsqueak.wizard.wizard import configure_openai, WizardState

        state = WizardState()

        with patch("logsqueak.wizard.wizard.prompt_openai_api_key", return_value="sk-key"), \
             patch("logsqueak.wizard.wizard.prompt_openai_model", return_value="gpt-4o"), \
             patch("logsqueak.wizard.wizard.rprint"):

            await configure_openai(state)

            # Verify endpoint is hardcoded to OpenAI
            assert state.openai_endpoint == "https://api.openai.com/v1"


class TestConfigureCustom:
    """Tests for configure_custom function."""

    @pytest.mark.asyncio
    async def test_prompts_for_endpoint_key_and_model(self):
        """Test custom provider configuration prompts for all required fields."""
        from logsqueak.wizard.wizard import configure_custom, WizardState

        state = WizardState()

        # Mock prompts
        with patch("logsqueak.wizard.wizard.prompt_custom_endpoint") as mock_endpoint, \
             patch("logsqueak.wizard.wizard.prompt_custom_api_key") as mock_key, \
             patch("logsqueak.wizard.wizard.prompt_custom_model") as mock_model, \
             patch("logsqueak.wizard.wizard.rprint"):

            mock_endpoint.return_value = "https://custom.ai/v1"
            mock_key.return_value = "custom-key-456"
            mock_model.return_value = "custom-model-7b"

            result = await configure_custom(state)

            # Verify all prompts were called
            assert result is True
            mock_endpoint.assert_called_once_with(None)
            mock_key.assert_called_once_with(None)
            mock_model.assert_called_once_with(None)

            # Verify state was updated
            assert state.custom_endpoint == "https://custom.ai/v1"
            assert state.custom_api_key == "custom-key-456"  # notsecret
            assert state.custom_model == "custom-model-7b"

    @pytest.mark.asyncio
    async def test_uses_existing_custom_config_as_defaults(self, tmp_path):
        """Test that existing custom config is used as defaults."""
        from logsqueak.wizard.wizard import configure_custom, WizardState
        from logsqueak.models.config import Config

        # Create valid graph directory
        graph_dir = tmp_path / "graph"
        graph_dir.mkdir()
        (graph_dir / "journals").mkdir()
        (graph_dir / "logseq").mkdir()

        # Create existing config with custom provider
        existing_config = Config(
            llm={
                "endpoint": "https://old-custom.ai/v1",
                "model": "old-model",
                "api_key": "old-key",
            },
            logseq={"graph_path": str(graph_dir)},
            llm_providers={
                "custom": {
                    "endpoint": "https://old-custom.ai/v1",
                    "model": "old-model",
                    "api_key": "old-key",
                }
            }
        )

        state = WizardState(existing_config=existing_config)

        # Mock prompts
        with patch("logsqueak.wizard.wizard.prompt_custom_endpoint") as mock_endpoint, \
             patch("logsqueak.wizard.wizard.prompt_custom_api_key") as mock_key, \
             patch("logsqueak.wizard.wizard.prompt_custom_model") as mock_model, \
             patch("logsqueak.wizard.wizard.rprint"):

            mock_endpoint.return_value = "https://new-custom.ai/v1"
            mock_key.return_value = "new-key"
            mock_model.return_value = "new-model"

            result = await configure_custom(state)

            # Verify existing values were passed as defaults
            mock_endpoint.assert_called_once_with("https://old-custom.ai/v1")
            mock_key.assert_called_once_with("old-key")
            mock_model.assert_called_once_with("old-model")

            assert result is True


class TestDetectProviderType:
    """Tests for detect_provider_type function."""

    @pytest.mark.asyncio
    async def test_detects_ollama_provider(self):
        """Test detecting Ollama by successful connection test."""
        from logsqueak.wizard.wizard import detect_provider_type
        from logsqueak.wizard.validators import ValidationResult

        # Mock Ollama connection succeeding
        async def mock_ollama_success(endpoint, timeout=5):
            return ValidationResult(success=True, data={"models": []})

        with patch("logsqueak.wizard.wizard.validate_ollama_connection", mock_ollama_success):
            provider_type = await detect_provider_type("http://localhost:11434/v1")
            assert provider_type == "ollama"

    @pytest.mark.asyncio
    async def test_detects_openai_by_url_pattern(self):
        """Test detecting OpenAI by URL pattern when Ollama test fails."""
        from logsqueak.wizard.wizard import detect_provider_type
        from logsqueak.wizard.validators import ValidationResult

        # Mock Ollama connection failing
        async def mock_ollama_fail(endpoint, timeout=5):
            return ValidationResult(success=False, error_message="Not Ollama")

        with patch("logsqueak.wizard.wizard.validate_ollama_connection", mock_ollama_fail):
            provider_type = await detect_provider_type("https://api.openai.com/v1")
            assert provider_type == "openai"

    @pytest.mark.asyncio
    async def test_defaults_to_custom_for_unknown_endpoint(self):
        """Test defaulting to custom for unknown endpoints."""
        from logsqueak.wizard.wizard import detect_provider_type
        from logsqueak.wizard.validators import ValidationResult

        # Mock Ollama connection failing
        async def mock_ollama_fail(endpoint, timeout=5):
            return ValidationResult(success=False, error_message="Not Ollama")

        with patch("logsqueak.wizard.wizard.validate_ollama_connection", mock_ollama_fail):
            provider_type = await detect_provider_type("https://custom-llm.example.com/v1")
            assert provider_type == "custom"


class TestShowSuccessMessage:
    """Tests for show_success_message function."""

    def test_displays_next_steps(self):
        """Test that success message shows next steps."""
        from logsqueak.wizard.wizard import show_success_message
        from pathlib import Path

        config_path = Path("/home/user/.config/logsqueak/config.yaml")

        with patch("logsqueak.wizard.wizard.rprint") as mock_rprint:
            show_success_message(config_path)

            # Verify rprint was called (Panel is printed)
            mock_rprint.assert_called_once()

            # Get the Panel argument
            panel = mock_rprint.call_args[0][0]
            panel_text = str(panel.renderable)

            # Verify message contains key information
            assert "Configuration saved successfully" in panel_text
            assert "logsqueak extract" in panel_text
            assert "logsqueak search" in panel_text
            assert "logsqueak init" in panel_text
            assert str(config_path) in panel_text

    def test_includes_config_path(self):
        """Test that config path is shown in message."""
        from logsqueak.wizard.wizard import show_success_message
        from pathlib import Path

        config_path = Path("/custom/path/config.yaml")

        with patch("logsqueak.wizard.wizard.rprint") as mock_rprint:
            show_success_message(config_path)

            panel = mock_rprint.call_args[0][0]
            panel_text = str(panel.renderable)

            assert "/custom/path/config.yaml" in panel_text


class TestConfigureGraphPath:
    """Tests for configure_graph_path function."""

    @pytest.mark.asyncio
    async def test_retry_on_validation_failure(self, tmp_path):
        """Test retry loop when validation fails."""
        from logsqueak.wizard.wizard import configure_graph_path, WizardState
        from logsqueak.wizard.validators import ValidationResult

        state = WizardState()

        # Mock validation to fail once, then succeed
        call_count = {"value": 0}

        def mock_validate(path):
            call_count["value"] += 1
            if call_count["value"] == 1:
                return ValidationResult(success=False, error_message="Invalid path")
            else:
                return ValidationResult(success=True, data={"path": str(tmp_path)})

        # Mock prompts
        prompt_calls = []

        def mock_prompt(default):
            prompt_calls.append(default)
            return str(tmp_path)

        def mock_retry_prompt(operation):
            return "retry"

        with patch("logsqueak.wizard.wizard.validate_graph_path", mock_validate), \
             patch("logsqueak.wizard.wizard.prompt_graph_path", mock_prompt), \
             patch("logsqueak.wizard.wizard.prompt_retry_on_failure", mock_retry_prompt), \
             patch("logsqueak.wizard.wizard.rprint"):

            result = await configure_graph_path(state)

            # Should succeed after retry
            assert result is True
            assert state.graph_path == str(tmp_path)
            assert call_count["value"] == 2

    @pytest.mark.asyncio
    async def test_abort_on_user_choice(self):
        """Test abort when user chooses to abort after failure."""
        from logsqueak.wizard.wizard import configure_graph_path, WizardState
        from logsqueak.wizard.validators import ValidationResult

        state = WizardState()

        def mock_validate(path):
            return ValidationResult(success=False, error_message="Invalid path")

        def mock_prompt(default):
            return "/invalid/path"

        def mock_retry_prompt(operation):
            return "abort"

        with patch("logsqueak.wizard.wizard.validate_graph_path", mock_validate), \
             patch("logsqueak.wizard.wizard.prompt_graph_path", mock_prompt), \
             patch("logsqueak.wizard.wizard.prompt_retry_on_failure", mock_retry_prompt), \
             patch("logsqueak.wizard.wizard.rprint"):

            result = await configure_graph_path(state)

            # Should abort
            assert result is False

    @pytest.mark.asyncio
    async def test_skip_not_allowed_for_required_field(self, tmp_path):
        """Test that skip shows error for required graph path."""
        from logsqueak.wizard.wizard import configure_graph_path, WizardState
        from logsqueak.wizard.validators import ValidationResult

        state = WizardState()

        call_count = {"value": 0}

        def mock_validate(path):
            return ValidationResult(success=False, error_message="Invalid path")

        def mock_prompt(default):
            return "/invalid/path"

        def mock_retry_prompt(operation):
            call_count["value"] += 1
            if call_count["value"] == 1:
                return "skip"  # Try to skip
            else:
                return "abort"  # Then abort to exit test

        with patch("logsqueak.wizard.wizard.validate_graph_path", mock_validate), \
             patch("logsqueak.wizard.wizard.prompt_graph_path", mock_prompt), \
             patch("logsqueak.wizard.wizard.prompt_retry_on_failure", mock_retry_prompt), \
             patch("logsqueak.wizard.wizard.rprint") as mock_rprint:

            result = await configure_graph_path(state)

            # Should eventually abort after skip attempt
            assert result is False
            # Verify warning message was shown
            calls_text = " ".join(str(call) for call in mock_rprint.call_args_list)
            assert "required" in calls_text.lower() or "try again" in calls_text.lower()

    @pytest.mark.asyncio
    async def test_keyboard_interrupt_handled(self):
        """Test keyboard interrupt handling."""
        from logsqueak.wizard.wizard import configure_graph_path, WizardState

        state = WizardState()

        def mock_prompt(default):
            raise KeyboardInterrupt()

        with patch("logsqueak.wizard.wizard.prompt_graph_path", mock_prompt), \
             patch("logsqueak.wizard.wizard.rprint"):

            result = await configure_graph_path(state)

            # Should return False on keyboard interrupt
            assert result is False


class TestConfigureProvider:
    """Tests for configure_provider function."""

    @pytest.mark.asyncio
    async def test_detects_existing_provider_type(self, tmp_path):
        """Test provider detection from existing config."""
        from logsqueak.wizard.wizard import configure_provider, WizardState, detect_provider_type
        from logsqueak.models.config import Config

        # Create valid graph directory
        graph_dir = tmp_path / "graph"
        graph_dir.mkdir()
        (graph_dir / "journals").mkdir()
        (graph_dir / "logseq").mkdir()

        # Create existing config with Ollama
        existing_config = Config(
            llm={
                "endpoint": "http://localhost:11434/v1",
                "model": "mistral:7b-instruct",
                "api_key": "ollama",
            },
            logseq={"graph_path": str(graph_dir)},
        )

        state = WizardState(existing_config=existing_config)

        # Mock provider detection and configuration
        with patch("logsqueak.wizard.wizard.detect_provider_type", return_value="ollama") as mock_detect, \
             patch("logsqueak.wizard.wizard.prompt_provider_choice", return_value="ollama"), \
             patch("logsqueak.wizard.wizard.configure_ollama", return_value=True), \
             patch("logsqueak.wizard.wizard.prompt_advanced_settings", return_value=False), \
             patch("logsqueak.wizard.wizard.rprint"):

            result = await configure_provider(state)

            # Should detect existing provider
            assert result is True
            mock_detect.assert_called_once()

    @pytest.mark.asyncio
    async def test_advanced_settings_for_ollama(self, tmp_path):
        """Test advanced settings prompt for Ollama (num_ctx, top_k)."""
        from logsqueak.wizard.wizard import configure_provider, WizardState

        state = WizardState()

        # Mock provider selection and advanced settings
        with patch("logsqueak.wizard.wizard.prompt_provider_choice", return_value="ollama"), \
             patch("logsqueak.wizard.wizard.configure_ollama", return_value=True), \
             patch("logsqueak.wizard.wizard.prompt_advanced_settings", return_value=True), \
             patch("logsqueak.wizard.wizard.prompt_num_ctx", return_value=16384) as mock_num_ctx, \
             patch("logsqueak.wizard.wizard.prompt_top_k", return_value=15) as mock_top_k, \
             patch("logsqueak.wizard.wizard.rprint"):

            result = await configure_provider(state)

            # Should prompt for advanced settings
            assert result is True
            mock_num_ctx.assert_called_once()
            mock_top_k.assert_called_once()
            assert state.num_ctx == 16384
            assert state.top_k == 15

    @pytest.mark.asyncio
    async def test_advanced_settings_for_openai(self):
        """Test advanced settings prompt for OpenAI (top_k only)."""
        from logsqueak.wizard.wizard import configure_provider, WizardState

        state = WizardState()

        # Mock provider selection and advanced settings
        with patch("logsqueak.wizard.wizard.prompt_provider_choice", return_value="openai"), \
             patch("logsqueak.wizard.wizard.configure_openai", return_value=True), \
             patch("logsqueak.wizard.wizard.prompt_advanced_settings", return_value=True), \
             patch("logsqueak.wizard.wizard.prompt_top_k", return_value=25) as mock_top_k, \
             patch("logsqueak.wizard.wizard.rprint"):

            result = await configure_provider(state)

            # Should prompt for top_k only (not num_ctx for OpenAI)
            assert result is True
            mock_top_k.assert_called_once()
            assert state.top_k == 25

    @pytest.mark.asyncio
    async def test_advanced_settings_for_custom(self):
        """Test advanced settings prompt for custom (top_k only)."""
        from logsqueak.wizard.wizard import configure_provider, WizardState

        state = WizardState()

        # Mock provider selection and advanced settings
        with patch("logsqueak.wizard.wizard.prompt_provider_choice", return_value="custom"), \
             patch("logsqueak.wizard.wizard.configure_custom", return_value=True), \
             patch("logsqueak.wizard.wizard.prompt_advanced_settings", return_value=True), \
             patch("logsqueak.wizard.wizard.prompt_top_k", return_value=30) as mock_top_k, \
             patch("logsqueak.wizard.wizard.rprint"):

            result = await configure_provider(state)

            # Should prompt for top_k only
            assert result is True
            mock_top_k.assert_called_once()
            assert state.top_k == 30

    @pytest.mark.asyncio
    async def test_unknown_provider_type_error(self):
        """Test error handling for unknown provider types."""
        from logsqueak.wizard.wizard import configure_provider, WizardState

        state = WizardState()

        # Mock invalid provider type
        with patch("logsqueak.wizard.wizard.prompt_provider_choice", return_value="invalid_provider"), \
             patch("logsqueak.wizard.wizard.rprint"):

            result = await configure_provider(state)

            # Should return False for unknown provider
            assert result is False


class TestHasConfigChanged:
    """Tests for has_config_changed function."""

    def test_detects_endpoint_change(self, tmp_path):
        """Test detection of endpoint changes."""
        from logsqueak.wizard.wizard import has_config_changed
        from logsqueak.models.config import Config

        # Create valid graph directory
        graph_dir = tmp_path / "graph"
        graph_dir.mkdir()
        (graph_dir / "journals").mkdir()
        (graph_dir / "logseq").mkdir()

        old_config = Config(
            llm={"endpoint": "http://localhost:11434/v1", "model": "test", "api_key": "key"},
            logseq={"graph_path": str(graph_dir)},
        )

        new_config = Config(
            llm={"endpoint": "http://remote:11434/v1", "model": "test", "api_key": "key"},
            logseq={"graph_path": str(graph_dir)},
        )

        assert has_config_changed(new_config, old_config) is True

    def test_detects_model_change(self, tmp_path):
        """Test detection of model changes."""
        from logsqueak.wizard.wizard import has_config_changed
        from logsqueak.models.config import Config

        # Create valid graph directory
        graph_dir = tmp_path / "graph"
        graph_dir.mkdir()
        (graph_dir / "journals").mkdir()
        (graph_dir / "logseq").mkdir()

        old_config = Config(
            llm={"endpoint": "http://localhost:11434/v1", "model": "old-model", "api_key": "key"},
            logseq={"graph_path": str(graph_dir)},
        )

        new_config = Config(
            llm={"endpoint": "http://localhost:11434/v1", "model": "new-model", "api_key": "key"},
            logseq={"graph_path": str(graph_dir)},
        )

        assert has_config_changed(new_config, old_config) is True

    def test_detects_num_ctx_change(self, tmp_path):
        """Test detection of num_ctx changes."""
        from logsqueak.wizard.wizard import has_config_changed
        from logsqueak.models.config import Config

        # Create valid graph directory
        graph_dir = tmp_path / "graph"
        graph_dir.mkdir()
        (graph_dir / "journals").mkdir()
        (graph_dir / "logseq").mkdir()

        old_config = Config(
            llm={"endpoint": "http://localhost:11434/v1", "model": "test", "api_key": "key", "num_ctx": 32768},
            logseq={"graph_path": str(graph_dir)},
        )

        new_config = Config(
            llm={"endpoint": "http://localhost:11434/v1", "model": "test", "api_key": "key", "num_ctx": 16384},
            logseq={"graph_path": str(graph_dir)},
        )

        assert has_config_changed(new_config, old_config) is True

    def test_detects_graph_path_change(self, tmp_path):
        """Test detection of graph path changes."""
        from logsqueak.wizard.wizard import has_config_changed
        from logsqueak.models.config import Config

        # Create valid graph directories
        graph_dir1 = tmp_path / "graph1"
        graph_dir1.mkdir()
        (graph_dir1 / "journals").mkdir()
        (graph_dir1 / "logseq").mkdir()

        graph_dir2 = tmp_path / "graph2"
        graph_dir2.mkdir()
        (graph_dir2 / "journals").mkdir()
        (graph_dir2 / "logseq").mkdir()

        old_config = Config(
            llm={"endpoint": "http://localhost:11434/v1", "model": "test", "api_key": "key"},
            logseq={"graph_path": str(graph_dir1)},
        )

        new_config = Config(
            llm={"endpoint": "http://localhost:11434/v1", "model": "test", "api_key": "key"},
            logseq={"graph_path": str(graph_dir2)},
        )

        assert has_config_changed(new_config, old_config) is True

    def test_detects_top_k_change(self, tmp_path):
        """Test detection of top_k changes."""
        from logsqueak.wizard.wizard import has_config_changed
        from logsqueak.models.config import Config

        # Create valid graph directory
        graph_dir = tmp_path / "graph"
        graph_dir.mkdir()
        (graph_dir / "journals").mkdir()
        (graph_dir / "logseq").mkdir()

        old_config = Config(
            llm={"endpoint": "http://localhost:11434/v1", "model": "test", "api_key": "key"},
            logseq={"graph_path": str(graph_dir)},
            rag={"top_k": 10},
        )

        new_config = Config(
            llm={"endpoint": "http://localhost:11434/v1", "model": "test", "api_key": "key"},
            logseq={"graph_path": str(graph_dir)},
            rag={"top_k": 20},
        )

        assert has_config_changed(new_config, old_config) is True

    def test_returns_false_when_configs_identical(self, tmp_path):
        """Test that identical configs return False."""
        from logsqueak.wizard.wizard import has_config_changed
        from logsqueak.models.config import Config

        # Create valid graph directory
        graph_dir = tmp_path / "graph"
        graph_dir.mkdir()
        (graph_dir / "journals").mkdir()
        (graph_dir / "logseq").mkdir()

        config1 = Config(
            llm={"endpoint": "http://localhost:11434/v1", "model": "test", "api_key": "key", "num_ctx": 32768},
            logseq={"graph_path": str(graph_dir)},
            rag={"top_k": 10},
        )

        config2 = Config(
            llm={"endpoint": "http://localhost:11434/v1", "model": "test", "api_key": "key", "num_ctx": 32768},
            logseq={"graph_path": str(graph_dir)},
            rag={"top_k": 10},
        )

        assert has_config_changed(config2, config1) is False


class TestAssembleConfig:
    """Tests for assemble_config error handling."""

    def test_raises_error_for_unknown_provider_type(self, tmp_path):
        """Test error handling for unsupported provider types."""
        from logsqueak.wizard.wizard import assemble_config, WizardState

        # Create valid graph directory
        graph_dir = tmp_path / "graph"
        graph_dir.mkdir()
        (graph_dir / "journals").mkdir()
        (graph_dir / "logseq").mkdir()

        state = WizardState(
            graph_path=str(graph_dir),
            provider_type="invalid_provider",  # Invalid provider type
        )

        with pytest.raises(ValueError, match="Unsupported provider type"):
            assemble_config(state)

    def test_preserves_old_provider_when_switching(self, tmp_path):
        """Test old provider preservation in assemble_config."""
        from logsqueak.wizard.wizard import assemble_config, WizardState
        from logsqueak.models.config import Config

        # Create valid graph directory
        graph_dir = tmp_path / "graph"
        graph_dir.mkdir()
        (graph_dir / "journals").mkdir()
        (graph_dir / "logseq").mkdir()

        # Start with OpenAI config
        existing_config = Config(
            llm={
                "endpoint": "https://api.openai.com/v1",
                "model": "gpt-4o",
                "api_key": "sk-old-key",
            },
            logseq={"graph_path": str(graph_dir)},
            llm_providers={
                "openai": {
                    "endpoint": "https://api.openai.com/v1",
                    "model": "gpt-4o",
                    "api_key": "sk-old-key",
                }
            },
        )

        # Switch to Ollama
        state = WizardState(
            existing_config=existing_config,
            graph_path=str(graph_dir),
            provider_type="ollama",
            ollama_endpoint="http://localhost:11434/v1",
            ollama_model="mistral:7b-instruct",
        )

        new_config = assemble_config(state)

        # Verify both providers are preserved
        assert "openai" in new_config.llm_providers
        assert "ollama_local" in new_config.llm_providers
        assert new_config.llm_providers["openai"]["api_key"] == "sk-old-key"


class TestValidateLLMConnection:
    """Additional tests for validate_llm_connection edge cases."""

    @pytest.mark.asyncio
    async def test_shows_warning_for_missing_config(self, tmp_path):
        """Test warning message when provider config is incomplete."""
        from logsqueak.wizard.wizard import validate_llm_connection, WizardState

        # Create valid graph directory
        graph_dir = tmp_path / "graph"
        graph_dir.mkdir()
        (graph_dir / "journals").mkdir()
        (graph_dir / "logseq").mkdir()

        # Create state with OpenAI but missing fields
        state = WizardState(
            graph_path=str(graph_dir),
            provider_type="openai",
            openai_endpoint=None,  # Missing endpoint
            openai_api_key=None,
            openai_model=None,
        )

        with patch("logsqueak.wizard.wizard.rprint") as mock_rprint:
            result = await validate_llm_connection(state)

            # Should continue anyway but show warning
            assert result is True
            calls_text = " ".join(str(call) for call in mock_rprint.call_args_list)
            assert "missing" in calls_text.lower() or "warning" in calls_text.lower()

    @pytest.mark.asyncio
    async def test_retry_on_connection_failure(self, tmp_path):
        """Test retry logic when connection fails."""
        from logsqueak.wizard.wizard import validate_llm_connection, WizardState
        from logsqueak.wizard.validators import ValidationResult

        # Create valid graph directory
        graph_dir = tmp_path / "graph"
        graph_dir.mkdir()
        (graph_dir / "journals").mkdir()
        (graph_dir / "logseq").mkdir()

        state = WizardState(
            graph_path=str(graph_dir),
            provider_type="openai",
            openai_endpoint="https://api.openai.com/v1",
            openai_api_key="sk-test-key",
            openai_model="gpt-4o",
        )

        call_count = {"value": 0}

        async def mock_connection_fail_then_success(*args, **kwargs):
            call_count["value"] += 1
            if call_count["value"] == 1:
                return ValidationResult(success=False, error_message="Connection failed")
            else:
                return ValidationResult(success=True)

        def mock_retry_prompt(operation):
            return "retry"

        with patch("logsqueak.wizard.wizard.validate_openai_connection", mock_connection_fail_then_success), \
             patch("logsqueak.wizard.wizard.prompt_retry_on_failure", mock_retry_prompt), \
             patch("logsqueak.wizard.wizard.rprint"):

            result = await validate_llm_connection(state)

            # Should succeed after retry
            assert result is True
            assert call_count["value"] == 2

    @pytest.mark.asyncio
    async def test_abort_on_connection_failure(self, tmp_path):
        """Test abort when user chooses to abort after failure."""
        from logsqueak.wizard.wizard import validate_llm_connection, WizardState
        from logsqueak.wizard.validators import ValidationResult

        # Create valid graph directory
        graph_dir = tmp_path / "graph"
        graph_dir.mkdir()
        (graph_dir / "journals").mkdir()
        (graph_dir / "logseq").mkdir()

        state = WizardState(
            graph_path=str(graph_dir),
            provider_type="custom",
            custom_endpoint="https://custom.ai/v1",
            custom_api_key="custom-key",  # notsecret
            custom_model="custom-model",
        )

        async def mock_connection_fail(*args, **kwargs):
            return ValidationResult(success=False, error_message="Connection failed")

        def mock_retry_prompt(operation):
            return "abort"

        with patch("logsqueak.wizard.wizard.validate_openai_connection", mock_connection_fail), \
             patch("logsqueak.wizard.wizard.prompt_retry_on_failure", mock_retry_prompt), \
             patch("logsqueak.wizard.wizard.rprint"):

            result = await validate_llm_connection(state)

            # Should abort
            assert result is False

    @pytest.mark.asyncio
    async def test_skip_on_connection_failure(self, tmp_path):
        """Test skip when user chooses to skip after failure."""
        from logsqueak.wizard.wizard import validate_llm_connection, WizardState
        from logsqueak.wizard.validators import ValidationResult

        # Create valid graph directory
        graph_dir = tmp_path / "graph"
        graph_dir.mkdir()
        (graph_dir / "journals").mkdir()
        (graph_dir / "logseq").mkdir()

        state = WizardState(
            graph_path=str(graph_dir),
            provider_type="openai",
            openai_endpoint="https://api.openai.com/v1",
            openai_api_key="sk-test-key",
            openai_model="gpt-4o",
        )

        async def mock_connection_fail(*args, **kwargs):
            return ValidationResult(success=False, error_message="Connection failed")

        def mock_retry_prompt(operation):
            return "skip"

        with patch("logsqueak.wizard.wizard.validate_openai_connection", mock_connection_fail), \
             patch("logsqueak.wizard.wizard.prompt_retry_on_failure", mock_retry_prompt), \
             patch("logsqueak.wizard.wizard.rprint"):

            result = await validate_llm_connection(state)

            # Should skip and continue
            assert result is True


class TestValidateEmbedding:
    """Additional tests for validate_embedding edge cases."""

    @pytest.mark.asyncio
    async def test_retry_on_validation_failure(self, tmp_path):
        """Test retry logic when validation fails."""
        from logsqueak.wizard.wizard import validate_embedding, WizardState
        from logsqueak.wizard.validators import ValidationResult

        # Create valid graph directory
        graph_dir = tmp_path / "graph"
        graph_dir.mkdir()
        (graph_dir / "journals").mkdir()
        (graph_dir / "logseq").mkdir()

        state = WizardState(graph_path=str(graph_dir))

        call_count = {"value": 0}

        async def mock_validation_fail_then_success(*args, **kwargs):
            call_count["value"] += 1
            if call_count["value"] == 1:
                return ValidationResult(success=False, error_message="Validation failed")
            else:
                return ValidationResult(success=True)

        def mock_retry_prompt(operation):
            return "retry"

        with patch("logsqueak.wizard.wizard.check_embedding_model_cached", return_value=False), \
             patch("logsqueak.wizard.wizard.check_disk_space", return_value=ValidationResult(success=True)), \
             patch("logsqueak.wizard.wizard.validate_embedding_model", mock_validation_fail_then_success), \
             patch("logsqueak.wizard.wizard.prompt_retry_on_failure", mock_retry_prompt), \
             patch("logsqueak.wizard.wizard.rprint"):

            result = await validate_embedding(state)

            # Should succeed after retry
            assert result is True
            assert call_count["value"] == 2

    @pytest.mark.asyncio
    async def test_abort_on_validation_failure(self, tmp_path):
        """Test abort when user chooses to abort after failure."""
        from logsqueak.wizard.wizard import validate_embedding, WizardState
        from logsqueak.wizard.validators import ValidationResult

        # Create valid graph directory
        graph_dir = tmp_path / "graph"
        graph_dir.mkdir()
        (graph_dir / "journals").mkdir()
        (graph_dir / "logseq").mkdir()

        state = WizardState(graph_path=str(graph_dir))

        async def mock_validation_fail(*args, **kwargs):
            return ValidationResult(success=False, error_message="Validation failed")

        def mock_retry_prompt(operation):
            return "abort"

        with patch("logsqueak.wizard.wizard.check_embedding_model_cached", return_value=False), \
             patch("logsqueak.wizard.wizard.check_disk_space", return_value=ValidationResult(success=True)), \
             patch("logsqueak.wizard.wizard.validate_embedding_model", mock_validation_fail), \
             patch("logsqueak.wizard.wizard.prompt_retry_on_failure", mock_retry_prompt), \
             patch("logsqueak.wizard.wizard.rprint"):

            result = await validate_embedding(state)

            # Should abort
            assert result is False


class TestWriteConfig:
    """Additional tests for write_config error scenarios."""

    @pytest.mark.asyncio
    async def test_permission_error_on_directory_creation(self, tmp_path):
        """Test helpful error message when cannot create directory."""
        from logsqueak.wizard.wizard import write_config
        from logsqueak.models.config import Config
        from pathlib import Path

        # Create valid graph directory
        graph_dir = tmp_path / "graph"
        graph_dir.mkdir()
        (graph_dir / "journals").mkdir()
        (graph_dir / "logseq").mkdir()

        config = Config(
            llm={"endpoint": "http://localhost:11434/v1", "model": "test", "api_key": "key"},
            logseq={"graph_path": str(graph_dir)},
        )

        # Mock Path.mkdir to raise PermissionError
        config_path = tmp_path / "no-permission" / "config.yaml"

        original_mkdir = Path.mkdir

        def mock_mkdir(self, *args, **kwargs):
            if "no-permission" in str(self):
                raise PermissionError("Permission denied")
            return original_mkdir(self, *args, **kwargs)

        with patch.object(Path, "mkdir", mock_mkdir):
            with pytest.raises(PermissionError, match="PERMISSION DENIED"):
                await write_config(config, config_path)

    @pytest.mark.asyncio
    async def test_os_error_on_directory_creation(self, tmp_path):
        """Test error handling for filesystem errors."""
        from logsqueak.wizard.wizard import write_config
        from logsqueak.models.config import Config
        from pathlib import Path

        # Create valid graph directory
        graph_dir = tmp_path / "graph"
        graph_dir.mkdir()
        (graph_dir / "journals").mkdir()
        (graph_dir / "logseq").mkdir()

        config = Config(
            llm={"endpoint": "http://localhost:11434/v1", "model": "test", "api_key": "key"},
            logseq={"graph_path": str(graph_dir)},
        )

        # Mock Path.mkdir to raise OSError
        config_path = tmp_path / "fs-error" / "config.yaml"

        original_mkdir = Path.mkdir

        def mock_mkdir(self, *args, **kwargs):
            if "fs-error" in str(self):
                raise OSError("Filesystem error")
            return original_mkdir(self, *args, **kwargs)

        with patch.object(Path, "mkdir", mock_mkdir):
            with pytest.raises(OSError, match="FILESYSTEM ERROR"):
                await write_config(config, config_path)

    @pytest.mark.asyncio
    async def test_cleanup_on_write_failure(self, tmp_path):
        """Test temp file cleanup on write failure."""
        from logsqueak.wizard.wizard import write_config
        from logsqueak.models.config import Config
        import tempfile
        import os

        # Create valid graph directory
        graph_dir = tmp_path / "graph"
        graph_dir.mkdir()
        (graph_dir / "journals").mkdir()
        (graph_dir / "logseq").mkdir()

        config = Config(
            llm={"endpoint": "http://localhost:11434/v1", "model": "test", "api_key": "key"},
            logseq={"graph_path": str(graph_dir)},
        )

        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_path = config_dir / "config.yaml"

        temp_files_created = []

        # Track temp files created
        original_mkstemp = tempfile.mkstemp

        def mock_mkstemp(*args, **kwargs):
            fd, path = original_mkstemp(*args, **kwargs)
            temp_files_created.append(path)
            return fd, path

        # Mock chmod to raise exception (simulating write failure)
        with patch("tempfile.mkstemp", mock_mkstemp), \
             patch("os.chmod", side_effect=Exception("Write failed")):

            try:
                await write_config(config, config_path)
            except Exception:
                pass

            # Verify temp file was cleaned up
            for temp_file in temp_files_created:
                assert not os.path.exists(temp_file), f"Temp file not cleaned up: {temp_file}"


class TestLoadExistingConfig:
    """Tests for load_existing_config edge cases."""

    def test_returns_none_for_empty_file(self, tmp_path):
        """Test that empty config file returns None."""
        from logsqueak.wizard.wizard import load_existing_config

        # Create empty config file
        config_dir = tmp_path / ".config" / "logsqueak"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "config.yaml"
        config_file.write_text("")

        with patch("logsqueak.wizard.wizard.Path.home", return_value=tmp_path):
            config = load_existing_config()
            assert config is None

    def test_returns_none_for_whitespace_only_file(self, tmp_path):
        """Test that whitespace-only config returns None."""
        from logsqueak.wizard.wizard import load_existing_config

        # Create whitespace-only config file
        config_dir = tmp_path / ".config" / "logsqueak"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "config.yaml"
        config_file.write_text("   \n\n  \t  \n")

        with patch("logsqueak.wizard.wizard.Path.home", return_value=tmp_path):
            config = load_existing_config()
            assert config is None

    def test_returns_none_on_read_error(self, tmp_path):
        """Test that file read errors return None."""
        from logsqueak.wizard.wizard import load_existing_config

        # Create config file
        config_dir = tmp_path / ".config" / "logsqueak"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "config.yaml"
        config_file.write_text("llm:\n  endpoint: test\n")

        with patch("logsqueak.wizard.wizard.Path.home", return_value=tmp_path):
            # Mock open to raise exception
            with patch("builtins.open", side_effect=IOError("Read error")):
                config = load_existing_config()
                assert config is None

    def test_bypasses_permission_error_to_read_config(self, tmp_path):
        """Test that permission errors are bypassed to read defaults."""
        from logsqueak.wizard.wizard import load_existing_config
        from logsqueak.models.config import Config
        import yaml

        # Create valid graph directory
        graph_dir = tmp_path / "graph"
        graph_dir.mkdir()
        (graph_dir / "journals").mkdir()
        (graph_dir / "logseq").mkdir()

        # Create valid config file with wrong permissions
        config_dir = tmp_path / ".config" / "logsqueak"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "config.yaml"

        config_data = {
            "llm": {
                "endpoint": "http://localhost:11434/v1",
                "model": "mistral:7b-instruct",
                "api_key": "test-key",
            },
            "logseq": {
                "graph_path": str(graph_dir),
            }
        }

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        with patch("logsqueak.wizard.wizard.Path.home", return_value=tmp_path):
            # Mock Config.load to raise PermissionError
            with patch.object(Config, "load", side_effect=PermissionError("Permission denied")):
                config = load_existing_config()

                # Should still load by bypassing permission check
                assert config is not None
                assert config.llm.model == "mistral:7b-instruct"

    def test_returns_none_when_bypass_read_fails(self, tmp_path):
        """Test that None is returned when bypass read also fails."""
        from logsqueak.wizard.wizard import load_existing_config
        from logsqueak.models.config import Config

        # Create config file
        config_dir = tmp_path / ".config" / "logsqueak"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "config.yaml"
        config_file.write_text("invalid: yaml: content:")

        with patch("logsqueak.wizard.wizard.Path.home", return_value=tmp_path):
            # Mock Config.load to raise PermissionError, then open to fail
            with patch.object(Config, "load", side_effect=PermissionError("Permission denied")):
                with patch("builtins.open", side_effect=IOError("Cannot read")):
                    config = load_existing_config()
                    assert config is None


class TestConfigModelWithLLMProviders:
    """Tests for Config model with llm_providers field.

    These tests verify that the Config model correctly handles the llm_providers
    dictionary for storing multiple provider credentials.
    """

    def test_config_with_llm_providers(self, tmp_path):
        """Test that Config model accepts llm_providers field."""
        from logsqueak.models.config import Config

        # Create valid graph directory
        graph_dir = tmp_path / "test-graph"
        graph_dir.mkdir()
        (graph_dir / "journals").mkdir()
        (graph_dir / "logseq").mkdir()

        config = Config(
            llm={
                "endpoint": "http://localhost:11434/v1",
                "api_key": "ollama",
                "model": "mistral:7b-instruct"
            },
            logseq={"graph_path": str(graph_dir)},
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

    def test_config_without_llm_providers(self, tmp_path):
        """Test that llm_providers field is optional (backwards compatibility)."""
        from logsqueak.models.config import Config

        # Create valid graph directory
        graph_dir = tmp_path / "test-graph"
        graph_dir.mkdir()
        (graph_dir / "journals").mkdir()
        (graph_dir / "logseq").mkdir()

        config = Config(
            llm={
                "endpoint": "http://localhost:11434/v1",
                "api_key": "ollama",
                "model": "mistral:7b-instruct"
            },
            logseq={"graph_path": str(graph_dir)}
        )

        assert config.llm_providers is None

    def test_config_with_multiple_providers(self, tmp_path):
        """Test that multiple providers can be stored in llm_providers."""
        from logsqueak.models.config import Config

        # Create valid graph directory
        graph_dir = tmp_path / "test-graph"
        graph_dir.mkdir()
        (graph_dir / "journals").mkdir()
        (graph_dir / "logseq").mkdir()

        config = Config(
            llm={
                "endpoint": "http://localhost:11434/v1",
                "api_key": "ollama",
                "model": "mistral:7b-instruct"
            },
            logseq={"graph_path": str(graph_dir)},
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


class TestConfigureOllamaEndpointModelPairing:
    """Tests for configure_ollama endpoint/model pairing behavior.

    These tests verify the fix for the issue where switching from a custom provider
    back to Ollama would show the default localhost endpoint instead of the user's
    pre-configured remote Ollama endpoint.
    """

    @pytest.mark.asyncio
    async def test_configure_ollama_loads_paired_endpoint_and_model(self, tmp_path):
        """Test that configure_ollama loads endpoint and model as a pair from llm_providers.

        This verifies the fix for the issue where switching from a custom provider
        back to Ollama would show the default localhost endpoint instead of the
        user's pre-configured remote Ollama endpoint.
        """
        from logsqueak.wizard.wizard import configure_ollama, WizardState
        from logsqueak.models.config import Config

        # Create valid graph directory
        graph_dir = tmp_path / "test-graph"
        graph_dir.mkdir()
        (graph_dir / "journals").mkdir()
        (graph_dir / "logseq").mkdir()

        # Setup: Config with custom provider active, but ollama_remote in llm_providers
        existing_config = Config(
            llm={
                "endpoint": "https://custom.api.com/v1",
                "api_key": "custom-key",
                "model": "custom-model"
            },
            logseq={"graph_path": str(graph_dir)},
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
             patch("logsqueak.wizard.wizard.prompt_ollama_endpoint") as mock_prompt_endpoint, \
             patch("logsqueak.wizard.wizard.prompt_ollama_model") as mock_prompt_model, \
             patch("logsqueak.wizard.wizard.rprint") as mock_rprint, \
             patch("logsqueak.wizard.wizard.Status"):

            # Mock successful connection with pre-configured endpoint
            result_mock = MagicMock()
            result_mock.success = True
            result_mock.data = {"models": mock_models}
            mock_validate.return_value = result_mock

            # User accepts the default endpoint (remote endpoint)
            mock_prompt_endpoint.return_value = "http://scooby:11434/v1"

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
            # (should be called once to test default endpoint)
            assert mock_validate.called
            validate_calls = [call[0][0] for call in mock_validate.call_args_list]
            assert "http://scooby:11434/v1" in validate_calls

            # Verify prompt_ollama_endpoint was called with the remote endpoint as default
            assert mock_prompt_endpoint.called
            args, kwargs = mock_prompt_endpoint.call_args
            # prompt_ollama_endpoint(default) - first positional arg
            assert args[0] == "http://scooby:11434/v1" or kwargs.get("default") == "http://scooby:11434/v1"

            # Verify prompt_ollama_model was called with the correct default model
            # (from the SAME provider as the endpoint - ollama_remote)
            assert mock_prompt_model.called
            args, kwargs = mock_prompt_model.call_args
            # prompt_ollama_model(models, default) - second positional arg
            assert args[1] == "qwen2.5:14b" or kwargs.get("default") == "qwen2.5:14b"

    @pytest.mark.asyncio
    async def test_configure_ollama_clears_model_default_when_endpoint_changed(self, tmp_path):
        """Test that changing the endpoint clears the model default (no stale pairing).

        When a user changes the endpoint from the remembered one, the model default
        should be cleared because we don't want to pair a model from one endpoint
        with a different endpoint.
        """
        from logsqueak.wizard.wizard import configure_ollama, WizardState
        from logsqueak.models.config import Config

        # Create valid graph directory
        graph_dir = tmp_path / "test-graph"
        graph_dir.mkdir()
        (graph_dir / "journals").mkdir()
        (graph_dir / "logseq").mkdir()

        # Setup: Config with ollama_remote remembered
        existing_config = Config(
            llm={
                "endpoint": "https://custom.api.com/v1",
                "api_key": "custom-key",
                "model": "custom-model"
            },
            logseq={"graph_path": str(graph_dir)},
            llm_providers={
                "ollama_remote": {
                    "endpoint": "http://scooby:11434/v1",
                    "api_key": "ollama",
                    "model": "qwen2.5:14b",
                    "num_ctx": 32768
                }
            }
        )

        state = WizardState(existing_config=existing_config)

        # Mock models available at BOTH endpoints
        mock_models = [
            MagicMock(name="mistral:7b-instruct", size=4000000000),
            MagicMock(name="llama2:13b", size=7000000000)
        ]

        with patch("logsqueak.wizard.wizard.validate_ollama_connection") as mock_validate, \
             patch("logsqueak.wizard.wizard.prompt_ollama_endpoint") as mock_prompt_endpoint, \
             patch("logsqueak.wizard.wizard.prompt_ollama_model") as mock_prompt_model, \
             patch("logsqueak.wizard.wizard.rprint") as mock_rprint, \
             patch("logsqueak.wizard.wizard.Status"):

            # Mock validation - both endpoints succeed
            result_mock = MagicMock()
            result_mock.success = True
            result_mock.data = {"models": mock_models}
            mock_validate.return_value = result_mock

            # User CHANGES endpoint to localhost
            mock_prompt_endpoint.return_value = "http://localhost:11434/v1"

            # User selects a different model
            mock_prompt_model.return_value = "mistral:7b-instruct"

            # Execute configure_ollama
            success = await configure_ollama(state)

            # Verify success
            assert success is True

            # Verify new endpoint was accepted
            assert state.ollama_endpoint == "http://localhost:11434/v1"
            assert state.ollama_model == "mistral:7b-instruct"

            # Verify validate_ollama_connection was called for both endpoints
            # (once for default test, once for new endpoint)
            assert mock_validate.call_count >= 2
            validate_calls = [call[0][0] for call in mock_validate.call_args_list]
            assert "http://scooby:11434/v1" in validate_calls
            assert "http://localhost:11434/v1" in validate_calls

            # Verify prompt_ollama_model was called with NO default
            # (because endpoint changed, we don't want stale model pairing)
            assert mock_prompt_model.called
            args, kwargs = mock_prompt_model.call_args
            # prompt_ollama_model(models, default) - second positional arg should be None
            assert args[1] is None or kwargs.get("default") is None
