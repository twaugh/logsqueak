"""Unit tests for wizard provider helpers."""

import pytest

from logsqueak.wizard.providers import (
    OllamaModel,
    get_recommended_ollama_model,
    format_model_size,
    mask_api_key,
    get_provider_key,
)


class TestGetRecommendedOllamaModel:
    """Tests for get_recommended_ollama_model function."""

    def test_recommends_mistral_7b_instruct(self):
        """Test that mistral:7b-instruct is recommended when available."""
        models = [
            OllamaModel(name="llama2:latest", size=3826793677, modified_at="2025-01-14"),
            OllamaModel(name="mistral:7b-instruct", size=4109865159, modified_at="2025-01-15"),
            OllamaModel(name="codellama:latest", size=3825819519, modified_at="2025-01-13"),
        ]

        result = get_recommended_ollama_model(models)

        assert result == "mistral:7b-instruct"

    def test_returns_none_when_no_models(self):
        """Test that None is returned when models list is empty."""
        result = get_recommended_ollama_model([])

        assert result is None

    def test_returns_none_when_mistral_not_found(self):
        """Test that None is returned when Mistral is not in the list."""
        models = [
            OllamaModel(name="llama2:latest", size=3826793677, modified_at="2025-01-14"),
            OllamaModel(name="codellama:latest", size=3825819519, modified_at="2025-01-13"),
        ]

        result = get_recommended_ollama_model(models)

        assert result is None

    def test_matches_mistral_variations(self):
        """Test that various Mistral 7B Instruct variations are recognized."""
        test_cases = [
            "mistral:7b-instruct",
            "mistral:7b-instruct-q4",
            "mistral:7b-instruct-v0.2",
            "mistral:7b-instruct-latest",
        ]

        for model_name in test_cases:
            models = [OllamaModel(name=model_name, size=4109865159, modified_at="2025-01-15")]
            result = get_recommended_ollama_model(models)
            assert result == model_name, f"Failed to match {model_name}"


class TestFormatModelSize:
    """Tests for format_model_size function."""

    def test_formats_bytes(self):
        """Test formatting of sizes in bytes."""
        assert format_model_size(512) == "512 bytes"
        assert format_model_size(1023) == "1023 bytes"

    def test_formats_kilobytes(self):
        """Test formatting of sizes in KB."""
        assert format_model_size(1024) == "1.0 KB"
        assert format_model_size(1536) == "1.5 KB"
        assert format_model_size(10240) == "10.0 KB"

    def test_formats_megabytes(self):
        """Test formatting of sizes in MB."""
        assert format_model_size(1048576) == "1.0 MB"  # 1024^2
        assert format_model_size(1572864) == "1.5 MB"
        assert format_model_size(104857600) == "100.0 MB"

    def test_formats_gigabytes(self):
        """Test formatting of sizes in GB."""
        assert format_model_size(1073741824) == "1.0 GB"  # 1024^3
        assert format_model_size(1610612736) == "1.5 GB"
        assert format_model_size(4294967296) == "4.0 GB"

    def test_real_world_model_sizes(self):
        """Test formatting of actual Ollama model sizes."""
        # Mistral 7B Instruct (~4.1GB)
        assert format_model_size(4109865159) == "3.8 GB"

        # Llama2 7B (~3.8GB)
        assert format_model_size(3826793677) == "3.6 GB"

    def test_zero_size(self):
        """Test formatting of zero size."""
        assert format_model_size(0) == "0 bytes"


class TestMaskApiKey:
    """Tests for mask_api_key function."""

    def test_masks_standard_api_key(self):
        """Test masking of standard OpenAI-style API key."""
        api_key = "sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz1234567890"

        result = mask_api_key(api_key)

        assert result.startswith("sk-proj-")
        assert result.endswith("7890")
        assert "..." in result
        assert len(result) < len(api_key)

    def test_masks_short_key(self):
        """Test masking of short API key (less than 12 characters)."""
        api_key = "short123"

        result = mask_api_key(api_key)

        # Should still mask but may show less
        assert "..." in result
        assert result != api_key

    def test_masks_long_key(self):
        """Test masking of very long API key."""
        api_key = "a" * 100

        result = mask_api_key(api_key)

        assert result.startswith("a" * 8)
        assert result.endswith("a" * 4)
        assert "..." in result
        assert len(result) < len(api_key)

    def test_preserves_beginning_and_end(self):
        """Test that first 8 and last 4 characters are preserved."""
        api_key = "12345678middle90ab"

        result = mask_api_key(api_key)

        assert result.startswith("12345678")
        assert result.endswith("90ab")
        assert "middle" not in result


class TestGetProviderKey:
    """Tests for get_provider_key function."""

    def test_ollama_local_key(self):
        """Test provider key for local Ollama instance."""
        result = get_provider_key("ollama", "http://localhost:11434")

        assert result == "ollama_local"

    def test_ollama_remote_key(self):
        """Test provider key for remote Ollama instance."""
        result = get_provider_key("ollama", "http://remote-server:11434")

        assert result == "ollama_remote"

    def test_openai_key(self):
        """Test provider key for OpenAI."""
        result = get_provider_key("openai", "https://api.openai.com/v1")

        assert result == "openai"

    def test_custom_key_with_azure(self):
        """Test provider key for custom Azure provider."""
        result = get_provider_key("custom", "https://my-azure-endpoint.openai.azure.com/v1")

        assert result == "custom_azure"

    def test_custom_key_with_other(self):
        """Test provider key for other custom provider."""
        result = get_provider_key("custom", "http://localhost:8000/v1")

        assert result == "custom"

    def test_normalizes_endpoint_url(self):
        """Test that endpoint URL is normalized (trailing slashes, etc)."""
        result1 = get_provider_key("ollama", "http://remote-server:11434/")
        result2 = get_provider_key("ollama", "http://remote-server:11434")

        assert result1 == result2 == "ollama_remote"
