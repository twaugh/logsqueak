"""Provider-specific logic for LLM configuration."""

from dataclasses import dataclass


@dataclass
class OllamaModel:
    """Represents a model available in an Ollama instance.

    Attributes:
        name: Model identifier (e.g., "mistral:7b-instruct")
        size: Model size in bytes
        modified_at: Last modified timestamp (ISO format)
    """
    name: str
    size: int
    modified_at: str


async def fetch_ollama_models(endpoint: str, timeout: int = 30) -> list[OllamaModel]:
    """Fetch list of installed models from Ollama instance.

    Args:
        endpoint: Ollama endpoint URL
        timeout: Request timeout in seconds

    Returns:
        List of OllamaModel instances

    Raises:
        httpx.HTTPError: If request fails
        asyncio.TimeoutError: If request exceeds timeout
    """
    from logsqueak.wizard.validators import test_ollama_connection

    result = await test_ollama_connection(endpoint, timeout)
    if not result.success:
        raise Exception(result.error_message)

    return result.data["models"]


def get_recommended_ollama_model(models: list[OllamaModel]) -> str | None:
    """Find recommended Ollama model (Mistral 7B Instruct) in model list.

    Args:
        models: List of available models

    Returns:
        Model name if Mistral 7B Instruct found, None otherwise
    """
    # Look for Mistral 7B Instruct variants
    for model in models:
        name_lower = model.name.lower()
        if "mistral" in name_lower and ("7b" in name_lower or "7-b" in name_lower):
            if "instruct" in name_lower:
                return model.name

    return None


def format_model_size(size_bytes: int) -> str:
    """Format model size in human-readable units.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted string (e.g., "4.1 GB", "512 MB")
    """
    # Use 1024 as base (binary)
    kb = 1024
    mb = kb * 1024
    gb = mb * 1024

    if size_bytes >= gb:
        return f"{size_bytes / gb:.1f} GB"
    elif size_bytes >= mb:
        return f"{size_bytes / mb:.1f} MB"
    elif size_bytes >= kb:
        return f"{size_bytes / kb:.1f} KB"
    else:
        return f"{size_bytes} bytes"


def mask_api_key(api_key: str) -> str:
    """Mask API key for display (show first 8 + "..." + last 4).

    Args:
        api_key: Full API key

    Returns:
        Masked string (e.g., "sk-proj-abc...xyz9")
    """
    if len(api_key) <= 12:
        # For short keys, just mask middle
        return api_key[:4] + "..." + api_key[-4:]

    return api_key[:8] + "..." + api_key[-4:]


def get_provider_key(provider_type: str, endpoint: str) -> str:
    """Generate unique key for provider in llm_providers dict.

    Args:
        provider_type: "ollama", "openai", or "custom"
        endpoint: Provider endpoint URL

    Returns:
        Key string (e.g., "ollama_local", "openai", "custom_azure")
    """
    if provider_type == "ollama":
        if "localhost" in endpoint or "127.0.0.1" in endpoint:
            return "ollama_local"
        else:
            return "ollama_remote"
    elif provider_type == "openai":
        return "openai"
    elif provider_type == "custom":
        # Extract hostname from endpoint for custom providers
        if "azure" in endpoint.lower():
            return "custom_azure"
        elif "together" in endpoint.lower():
            return "custom_together"
        else:
            return "custom"
    else:
        return provider_type
