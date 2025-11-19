"""Unit tests for wizard prompt helpers."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from logsqueak.wizard.prompts import (
    prompt_advanced_settings,
    prompt_confirm_overwrite,
    prompt_continue_on_timeout,
    prompt_custom_api_key,
    prompt_custom_endpoint,
    prompt_custom_model,
    prompt_graph_path,
    prompt_num_ctx,
    prompt_ollama_endpoint,
    prompt_ollama_model,
    prompt_openai_api_key,
    prompt_openai_model,
    prompt_provider_choice,
    prompt_retry_on_failure,
    prompt_top_k,
)
from logsqueak.wizard.providers import OllamaModel


class TestPromptGraphPath:
    """Tests for prompt_graph_path function."""

    @patch("logsqueak.wizard.prompts.Prompt.ask")
    def test_prompts_with_default(self, mock_ask):
        """Test prompting with default value from existing config."""
        mock_ask.return_value = "/home/user/logseq"

        result = prompt_graph_path(default="/home/user/logseq")

        mock_ask.assert_called_once()
        call_args = mock_ask.call_args
        assert call_args[1]["default"] == "/home/user/logseq"
        assert result == str(Path("/home/user/logseq").resolve())

    @patch("logsqueak.wizard.prompts.Prompt.ask")
    def test_prompts_without_default(self, mock_ask):
        """Test prompting without default value."""
        mock_ask.return_value = "~/Documents/my-graph"

        result = prompt_graph_path()

        mock_ask.assert_called_once()
        call_args = mock_ask.call_args
        assert call_args[1]["default"] == "~/Documents/logseq"
        # Result should be expanded
        assert "~" not in result

    @patch("logsqueak.wizard.prompts.Prompt.ask")
    def test_expands_tilde_in_path(self, mock_ask):
        """Test that ~ is expanded to home directory."""
        mock_ask.return_value = "~/Documents/logseq"

        result = prompt_graph_path()

        assert "~" not in result
        assert result.startswith("/")

    @patch("logsqueak.wizard.prompts.Prompt.ask")
    def test_resolves_relative_paths(self, mock_ask):
        """Test that relative paths are resolved to absolute."""
        mock_ask.return_value = "./logseq"

        result = prompt_graph_path()

        assert result.startswith("/")
        assert result == str(Path("./logseq").resolve())


class TestPromptProviderChoice:
    """Tests for prompt_provider_choice function."""

    @patch("logsqueak.wizard.prompts.Prompt.ask")
    def test_returns_ollama_for_choice_1(self, mock_ask):
        """Test that choice 1 returns ollama."""
        mock_ask.return_value = "1"

        result = prompt_provider_choice()

        assert result == "ollama"
        mock_ask.assert_called_once()
        call_args = mock_ask.call_args
        assert call_args[1]["choices"] == ["1", "2", "3"]
        assert call_args[1]["default"] == "1"

    @patch("logsqueak.wizard.prompts.Prompt.ask")
    def test_returns_openai_for_choice_2(self, mock_ask):
        """Test that choice 2 returns openai."""
        mock_ask.return_value = "2"

        result = prompt_provider_choice()

        assert result == "openai"

    @patch("logsqueak.wizard.prompts.Prompt.ask")
    def test_returns_custom_for_choice_3(self, mock_ask):
        """Test that choice 3 returns custom."""
        mock_ask.return_value = "3"

        result = prompt_provider_choice()

        assert result == "custom"

    @patch("logsqueak.wizard.prompts.Prompt.ask")
    def test_uses_default_provider(self, mock_ask):
        """Test that default provider is used."""
        mock_ask.return_value = "2"

        result = prompt_provider_choice(default="openai")

        mock_ask.assert_called_once()
        call_args = mock_ask.call_args
        assert call_args[1]["default"] == "2"
        assert result == "openai"


class TestPromptOllamaEndpoint:
    """Tests for prompt_ollama_endpoint function."""

    @patch("logsqueak.wizard.prompts.Prompt.ask")
    def test_uses_default_localhost_endpoint(self, mock_ask):
        """Test that default localhost endpoint is used."""
        mock_ask.return_value = "http://localhost:11434"

        result = prompt_ollama_endpoint()

        mock_ask.assert_called_once()
        call_args = mock_ask.call_args
        assert call_args[1]["default"] == "http://localhost:11434"
        assert result == "http://localhost:11434/v1"

    @patch("logsqueak.wizard.prompts.Prompt.ask")
    def test_adds_v1_suffix_if_missing(self, mock_ask):
        """Test that /v1 suffix is added if missing."""
        mock_ask.return_value = "http://remote-server:11434"

        result = prompt_ollama_endpoint()

        assert result == "http://remote-server:11434/v1"

    @patch("logsqueak.wizard.prompts.Prompt.ask")
    def test_preserves_v1_suffix_if_present(self, mock_ask):
        """Test that existing /v1 suffix is preserved."""
        mock_ask.return_value = "http://remote-server:11434/v1"

        result = prompt_ollama_endpoint()

        assert result == "http://remote-server:11434/v1"

    @patch("logsqueak.wizard.prompts.Prompt.ask")
    def test_strips_trailing_slash(self, mock_ask):
        """Test that trailing slash is stripped before adding /v1."""
        mock_ask.return_value = "http://remote-server:11434/"

        result = prompt_ollama_endpoint()

        assert result == "http://remote-server:11434/v1"


class TestPromptOllamaModel:
    """Tests for prompt_ollama_model function."""

    @patch("logsqueak.wizard.prompts.console.print")
    @patch("logsqueak.wizard.prompts.Prompt.ask")
    def test_displays_models_and_prompts_selection(self, mock_ask, mock_print):
        """Test that models are displayed in table and user is prompted."""
        models = [
            OllamaModel(name="mistral:7b-instruct", size=4109865159, modified_at="2025-01-15"),
            OllamaModel(name="llama2:latest", size=3826793677, modified_at="2025-01-14"),
        ]
        mock_ask.return_value = "1"

        result = prompt_ollama_model(models)

        assert result == "mistral:7b-instruct"
        mock_print.assert_called()
        mock_ask.assert_called_once()
        call_args = mock_ask.call_args
        assert call_args[1]["choices"] == ["1", "2"]

    @patch("logsqueak.wizard.prompts.console.print")
    @patch("logsqueak.wizard.prompts.Prompt.ask")
    def test_defaults_to_recommended_model(self, mock_ask, mock_print):
        """Test that recommended model is used as default."""
        models = [
            OllamaModel(name="llama2:latest", size=3826793677, modified_at="2025-01-14"),
            OllamaModel(name="mistral:7b-instruct", size=4109865159, modified_at="2025-01-15"),
        ]
        mock_ask.return_value = "2"

        result = prompt_ollama_model(models)

        assert result == "mistral:7b-instruct"
        call_args = mock_ask.call_args
        assert call_args[1]["default"] == "2"

    @patch("logsqueak.wizard.prompts.console.print")
    @patch("logsqueak.wizard.prompts.Prompt.ask")
    def test_uses_current_model_as_default(self, mock_ask, mock_print):
        """Test that current model from config is used as default."""
        models = [
            OllamaModel(name="llama2:latest", size=3826793677, modified_at="2025-01-14"),
            OllamaModel(name="mistral:7b-instruct", size=4109865159, modified_at="2025-01-15"),
        ]
        mock_ask.return_value = "1"

        result = prompt_ollama_model(models, default="llama2:latest")

        assert result == "llama2:latest"
        call_args = mock_ask.call_args
        assert call_args[1]["default"] == "1"

    @patch("logsqueak.wizard.prompts.console.print")
    @patch("logsqueak.wizard.prompts.Prompt.ask")
    def test_returns_selected_model_by_index(self, mock_ask, mock_print):
        """Test that correct model is returned based on user choice."""
        models = [
            OllamaModel(name="llama2:latest", size=3826793677, modified_at="2025-01-14"),
            OllamaModel(name="mistral:7b-instruct", size=4109865159, modified_at="2025-01-15"),
            OllamaModel(name="codellama:latest", size=3825819519, modified_at="2025-01-13"),
        ]
        mock_ask.return_value = "3"

        result = prompt_ollama_model(models)

        assert result == "codellama:latest"


class TestPromptOpenaiApiKey:
    """Tests for prompt_openai_api_key function."""

    @patch("logsqueak.wizard.prompts.Prompt.ask")
    def test_prompts_for_new_key(self, mock_ask):
        """Test prompting for new API key without existing key."""
        mock_ask.return_value = "sk-proj-new-key-12345"

        result = prompt_openai_api_key()

        assert result == "sk-proj-new-key-12345"
        mock_ask.assert_called_once()
        call_args = mock_ask.call_args
        assert "default" not in call_args[1]

    @patch("logsqueak.wizard.prompts.Prompt.ask")
    def test_shows_masked_existing_key(self, mock_ask):
        """Test that existing key is shown masked with option to keep."""
        existing_key = "sk-proj-existing-key-12345"
        mock_ask.return_value = existing_key

        result = prompt_openai_api_key(existing_key=existing_key)

        assert result == existing_key
        mock_ask.assert_called_once()
        call_args = mock_ask.call_args
        assert call_args[1]["default"] == existing_key
        assert call_args[1]["show_default"] is False

    @patch("logsqueak.wizard.prompts.Prompt.ask")
    def test_allows_updating_existing_key(self, mock_ask):
        """Test that user can update existing key."""
        existing_key = "sk-proj-old-key"
        new_key = "sk-proj-new-key"
        mock_ask.return_value = new_key

        result = prompt_openai_api_key(existing_key=existing_key)

        assert result == new_key


class TestPromptOpenaiModel:
    """Tests for prompt_openai_model function."""

    @patch("logsqueak.wizard.prompts.Prompt.ask")
    def test_returns_gpt4o_for_choice_1(self, mock_ask):
        """Test that choice 1 returns gpt-4o."""
        mock_ask.return_value = "1"

        result = prompt_openai_model()

        assert result == "gpt-4o"

    @patch("logsqueak.wizard.prompts.Prompt.ask")
    def test_returns_gpt4_turbo_for_choice_2(self, mock_ask):
        """Test that choice 2 returns gpt-4-turbo."""
        mock_ask.return_value = "2"

        result = prompt_openai_model()

        assert result == "gpt-4-turbo"

    @patch("logsqueak.wizard.prompts.Prompt.ask")
    def test_returns_gpt35_for_choice_3(self, mock_ask):
        """Test that choice 3 returns gpt-3.5-turbo."""
        mock_ask.return_value = "3"

        result = prompt_openai_model()

        assert result == "gpt-3.5-turbo"

    @patch("logsqueak.wizard.prompts.Prompt.ask")
    def test_prompts_for_custom_model_name(self, mock_ask):
        """Test that choice 4 prompts for custom model name."""
        mock_ask.side_effect = ["4", "my-custom-model"]

        result = prompt_openai_model()

        assert result == "my-custom-model"
        assert mock_ask.call_count == 2


class TestPromptCustomEndpoint:
    """Tests for prompt_custom_endpoint function."""

    @patch("logsqueak.wizard.prompts.Prompt.ask")
    def test_uses_default_endpoint(self, mock_ask):
        """Test that default localhost endpoint is used."""
        mock_ask.return_value = "http://localhost:8000/v1"

        result = prompt_custom_endpoint()

        assert result == "http://localhost:8000/v1"
        call_args = mock_ask.call_args
        assert call_args[1]["default"] == "http://localhost:8000/v1"

    @patch("logsqueak.wizard.prompts.Prompt.ask")
    def test_adds_v1_suffix_if_missing(self, mock_ask):
        """Test that /v1 suffix is added if missing."""
        mock_ask.return_value = "https://api.example.com"

        result = prompt_custom_endpoint()

        assert result == "https://api.example.com/v1"

    @patch("logsqueak.wizard.prompts.Prompt.ask")
    def test_preserves_v1_suffix_if_present(self, mock_ask):
        """Test that existing /v1 suffix is preserved."""
        mock_ask.return_value = "https://api.example.com/v1"

        result = prompt_custom_endpoint()

        assert result == "https://api.example.com/v1"


class TestPromptCustomApiKey:
    """Tests for prompt_custom_api_key function."""

    @patch("logsqueak.wizard.prompts.Prompt.ask")
    def test_prompts_for_api_key_with_default(self, mock_ask):
        """Test that API key is prompted with 'none' default."""
        mock_ask.return_value = "custom-api-key-12345"

        result = prompt_custom_api_key()

        assert result == "custom-api-key-12345"
        call_args = mock_ask.call_args
        assert call_args[1]["default"] == "none"

    @patch("logsqueak.wizard.prompts.Prompt.ask")
    def test_allows_none_as_api_key(self, mock_ask):
        """Test that 'none' can be used as API key."""
        mock_ask.return_value = "none"

        result = prompt_custom_api_key()

        assert result == "none"


class TestPromptCustomModel:
    """Tests for prompt_custom_model function."""

    @patch("logsqueak.wizard.prompts.Prompt.ask")
    def test_prompts_for_model_name(self, mock_ask):
        """Test that model name is prompted."""
        mock_ask.return_value = "my-custom-model"

        result = prompt_custom_model()

        assert result == "my-custom-model"
        mock_ask.assert_called_once()


class TestPromptConfirmOverwrite:
    """Tests for prompt_confirm_overwrite function."""

    @patch("logsqueak.wizard.prompts.Confirm.ask")
    def test_returns_true_when_user_confirms(self, mock_ask):
        """Test that True is returned when user confirms."""
        mock_ask.return_value = True

        result = prompt_confirm_overwrite()

        assert result is True
        mock_ask.assert_called_once()

    @patch("logsqueak.wizard.prompts.Confirm.ask")
    def test_returns_false_when_user_declines(self, mock_ask):
        """Test that False is returned when user declines."""
        mock_ask.return_value = False

        result = prompt_confirm_overwrite()

        assert result is False

    @patch("logsqueak.wizard.prompts.Confirm.ask")
    def test_defaults_to_false(self, mock_ask):
        """Test that default is False."""
        mock_ask.return_value = False

        prompt_confirm_overwrite()

        call_args = mock_ask.call_args
        assert call_args[1]["default"] is False


class TestPromptRetryOnFailure:
    """Tests for prompt_retry_on_failure function."""

    @patch("logsqueak.wizard.prompts.Prompt.ask")
    def test_returns_retry_for_choice_1(self, mock_ask):
        """Test that choice 1 returns retry."""
        mock_ask.return_value = "1"

        result = prompt_retry_on_failure("Test operation")

        assert result == "retry"

    @patch("logsqueak.wizard.prompts.Prompt.ask")
    def test_returns_skip_for_choice_2(self, mock_ask):
        """Test that choice 2 returns skip."""
        mock_ask.return_value = "2"

        result = prompt_retry_on_failure("Test operation")

        assert result == "skip"

    @patch("logsqueak.wizard.prompts.Prompt.ask")
    def test_returns_abort_for_choice_3(self, mock_ask):
        """Test that choice 3 returns abort."""
        mock_ask.return_value = "3"

        result = prompt_retry_on_failure("Test operation")

        assert result == "abort"

    @patch("logsqueak.wizard.prompts.Prompt.ask")
    def test_defaults_to_retry(self, mock_ask):
        """Test that default is retry."""
        mock_ask.return_value = "1"

        prompt_retry_on_failure("Test operation")

        call_args = mock_ask.call_args
        assert call_args[1]["default"] == "1"


class TestPromptContinueOnTimeout:
    """Tests for prompt_continue_on_timeout function."""

    @patch("logsqueak.wizard.prompts.Prompt.ask")
    def test_returns_continue_for_choice_1(self, mock_ask):
        """Test that choice 1 returns continue."""
        mock_ask.return_value = "1"

        result = prompt_continue_on_timeout("Test operation", timeout=30)

        assert result == "continue"

    @patch("logsqueak.wizard.prompts.Prompt.ask")
    def test_returns_retry_for_choice_2(self, mock_ask):
        """Test that choice 2 returns retry."""
        mock_ask.return_value = "2"

        result = prompt_continue_on_timeout("Test operation", timeout=30)

        assert result == "retry"

    @patch("logsqueak.wizard.prompts.Prompt.ask")
    def test_returns_skip_for_choice_3(self, mock_ask):
        """Test that choice 3 returns skip."""
        mock_ask.return_value = "3"

        result = prompt_continue_on_timeout("Test operation", timeout=30)

        assert result == "skip"

    @patch("logsqueak.wizard.prompts.Prompt.ask")
    def test_defaults_to_retry(self, mock_ask):
        """Test that default is retry."""
        mock_ask.return_value = "2"

        prompt_continue_on_timeout("Test operation", timeout=30)

        call_args = mock_ask.call_args
        assert call_args[1]["default"] == "2"


class TestPromptAdvancedSettings:
    """Tests for prompt_advanced_settings function."""

    @patch("logsqueak.wizard.prompts.Confirm.ask")
    def test_returns_true_when_user_confirms(self, mock_ask):
        """Test that True is returned when user wants advanced settings."""
        mock_ask.return_value = True

        result = prompt_advanced_settings()

        assert result is True

    @patch("logsqueak.wizard.prompts.Confirm.ask")
    def test_returns_false_when_user_declines(self, mock_ask):
        """Test that False is returned when user declines advanced settings."""
        mock_ask.return_value = False

        result = prompt_advanced_settings()

        assert result is False

    @patch("logsqueak.wizard.prompts.Confirm.ask")
    def test_defaults_to_false(self, mock_ask):
        """Test that default is False."""
        mock_ask.return_value = False

        prompt_advanced_settings()

        call_args = mock_ask.call_args
        assert call_args[1]["default"] is False


class TestPromptNumCtx:
    """Tests for prompt_num_ctx function."""

    @patch("logsqueak.wizard.prompts.Prompt.ask")
    def test_returns_valid_integer_value(self, mock_ask):
        """Test that valid integer is returned."""
        mock_ask.return_value = "16384"

        result = prompt_num_ctx()

        assert result == 16384

    @patch("logsqueak.wizard.prompts.Prompt.ask")
    def test_uses_default_value(self, mock_ask):
        """Test that default value is used."""
        mock_ask.return_value = "32768"

        result = prompt_num_ctx(default=32768)

        assert result == 32768
        call_args = mock_ask.call_args
        assert call_args[1]["default"] == "32768"

    @patch("logsqueak.wizard.prompts.Prompt.ask")
    def test_returns_default_on_invalid_input(self, mock_ask):
        """Test that default is returned for invalid input."""
        mock_ask.return_value = "not-a-number"

        result = prompt_num_ctx(default=32768)

        assert result == 32768

    @patch("logsqueak.wizard.prompts.Prompt.ask")
    def test_accepts_custom_value(self, mock_ask):
        """Test that custom value is accepted."""
        mock_ask.return_value = "8192"

        result = prompt_num_ctx()

        assert result == 8192


class TestPromptTopK:
    """Tests for prompt_top_k function."""

    @patch("logsqueak.wizard.prompts.Prompt.ask")
    def test_returns_valid_integer_value(self, mock_ask):
        """Test that valid integer is returned."""
        mock_ask.return_value = "15"

        result = prompt_top_k()

        assert result == 15

    @patch("logsqueak.wizard.prompts.Prompt.ask")
    def test_uses_default_value(self, mock_ask):
        """Test that default value is used."""
        mock_ask.return_value = "10"

        result = prompt_top_k(default=10)

        assert result == 10
        call_args = mock_ask.call_args
        assert call_args[1]["default"] == "10"

    @patch("logsqueak.wizard.prompts.Prompt.ask")
    def test_returns_default_on_invalid_input(self, mock_ask):
        """Test that default is returned for invalid input."""
        mock_ask.return_value = "not-a-number"

        result = prompt_top_k(default=10)

        assert result == 10

    @patch("logsqueak.wizard.prompts.Prompt.ask")
    def test_rejects_value_below_range(self, mock_ask):
        """Test that values below 1 are rejected."""
        mock_ask.return_value = "0"

        result = prompt_top_k(default=10)

        assert result == 10

    @patch("logsqueak.wizard.prompts.Prompt.ask")
    def test_rejects_value_above_range(self, mock_ask):
        """Test that values above 100 are rejected."""
        mock_ask.return_value = "101"

        result = prompt_top_k(default=10)

        assert result == 10

    @patch("logsqueak.wizard.prompts.Prompt.ask")
    def test_accepts_value_at_lower_bound(self, mock_ask):
        """Test that value 1 is accepted."""
        mock_ask.return_value = "1"

        result = prompt_top_k()

        assert result == 1

    @patch("logsqueak.wizard.prompts.Prompt.ask")
    def test_accepts_value_at_upper_bound(self, mock_ask):
        """Test that value 100 is accepted."""
        mock_ask.return_value = "100"

        result = prompt_top_k()

        assert result == 100
