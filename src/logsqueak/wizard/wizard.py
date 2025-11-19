"""Main wizard orchestration logic."""

from dataclasses import dataclass, field

from logsqueak.models.config import Config


@dataclass
class WizardState:
    """Tracks wizard progress and user inputs.

    Attributes:
        existing_config: Loaded from existing config file (if exists)
        graph_path: User-provided Logseq graph directory path
        provider_type: Selected LLM provider ("ollama", "openai", "custom")
        ollama_endpoint: Ollama API endpoint (if provider is ollama)
        ollama_model: Selected Ollama model name
        openai_api_key: OpenAI API key (if provider is openai)
        openai_model: OpenAI model name
        custom_endpoint: Custom OpenAI-compatible endpoint
        custom_api_key: Custom provider API key
        custom_model: Custom provider model name
        num_ctx: Ollama context window size (optional)
        top_k: RAG search top_k value (default: 20)
    """
    existing_config: Config | None = None
    graph_path: str | None = None
    provider_type: str | None = None
    ollama_endpoint: str | None = None
    ollama_model: str | None = None
    openai_api_key: str | None = None
    openai_model: str | None = None
    custom_endpoint: str | None = None
    custom_api_key: str | None = None
    custom_model: str | None = None
    num_ctx: int | None = None
    top_k: int = 20


async def run_setup_wizard() -> bool:
    """Run the interactive setup wizard.

    Returns:
        True if setup completed successfully, False if aborted
    """
    # Implementation will be added in later tasks
    return False
