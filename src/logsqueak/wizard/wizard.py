"""Main wizard orchestration logic."""

import asyncio
import os
import tempfile
import yaml
from dataclasses import dataclass
from pathlib import Path

from pydantic import HttpUrl
from rich import print as rprint
from rich.panel import Panel
from rich.status import Status

from logsqueak.models.config import Config, LLMConfig, LogseqConfig, RAGConfig
from logsqueak.wizard.prompts import (
    prompt_advanced_settings,
    prompt_confirm_overwrite,
    prompt_graph_path,
    prompt_num_ctx,
    prompt_ollama_endpoint,
    prompt_ollama_model,
    prompt_provider_choice,
    prompt_retry_on_failure,
    prompt_top_k,
)
from logsqueak.wizard.providers import fetch_ollama_models, get_provider_key
from logsqueak.wizard.validators import (
    check_disk_space,
    check_embedding_model_cached,
    test_ollama_connection,
    validate_embedding_model,
    validate_graph_path,
)


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


def load_existing_config() -> Config | None:
    """Load existing config if present, return None if not found or invalid.

    Returns:
        Config instance if loaded successfully, None otherwise
    """
    config_path = Path.home() / ".config" / "logsqueak" / "config.yaml"

    if not config_path.exists():
        return None

    try:
        return Config.load(config_path)
    except Exception:
        # Any error loading config - treat as no config
        return None


async def detect_provider_type(endpoint: str) -> str:
    """Detect provider type by testing the endpoint at runtime.

    Args:
        endpoint: Endpoint URL to test

    Returns:
        Provider type: "ollama", "openai", or "custom"
    """
    endpoint_str = str(endpoint).lower()

    # Try Ollama API first (it's fast and unique)
    result = await test_ollama_connection(endpoint, timeout=5)
    if result.success:
        return "ollama"

    # Check for obvious OpenAI patterns
    if "openai.com" in endpoint_str or "api.openai.com" in endpoint_str:
        return "openai"

    # Default to custom for anything else
    return "custom"


async def configure_graph_path(state: WizardState) -> bool:
    """Prompt and validate graph path.

    Args:
        state: Wizard state to update

    Returns:
        False if user aborts, True otherwise
    """
    # Section header
    rprint("[bold cyan]═══ Step 1: Logseq Graph Location ═══[/bold cyan]")
    rprint("[dim]This is where your Logseq notes are stored on disk.[/dim]\n")

    # Get default from existing config
    default = None
    if state.existing_config:
        default = state.existing_config.logseq.graph_path

    while True:
        try:
            path = prompt_graph_path(default)
            result = validate_graph_path(path)

            if result.success:
                state.graph_path = result.data["path"]
                rprint(f"[green]✓[/green] Graph path validated: {state.graph_path}\n")
                return True
            else:
                rprint(f"[red]✗[/red] {result.error_message}")
                choice = prompt_retry_on_failure("Graph path validation")
                if choice == "abort":
                    return False
                elif choice == "skip":
                    # Can't skip graph path - it's required
                    rprint("[yellow]Graph path is required, please try again[/yellow]")
                # retry continues loop

        except KeyboardInterrupt:
            rprint("\n[yellow]Setup cancelled[/yellow]")
            return False


async def configure_ollama(state: WizardState) -> bool:
    """Ollama-specific configuration flow.

    Args:
        state: Wizard state to update

    Returns:
        False if user aborts, True otherwise
    """
    rprint("[dim]Ollama is a local LLM runtime. Logsqueak uses it to analyze your notes.[/dim]\n")

    # Get default endpoint from existing config
    default_endpoint = "http://localhost:11434"
    if state.existing_config and state.existing_config.llm.endpoint:
        endpoint_str = str(state.existing_config.llm.endpoint)
        if "ollama" in endpoint_str or "11434" in endpoint_str:
            default_endpoint = endpoint_str

    # Try existing endpoint first, then localhost, then prompt
    endpoints_to_try = [default_endpoint]
    if default_endpoint != "http://localhost:11434":
        endpoints_to_try.append("http://localhost:11434")

    models = None
    successful_endpoint = None

    with Status("[cyan]Testing Ollama connection...[/cyan]") as status:
        for endpoint in endpoints_to_try:
            result = await test_ollama_connection(endpoint)
            if result.success:
                models = result.data["models"]
                successful_endpoint = endpoint
                break

    if models and successful_endpoint:
        state.ollama_endpoint = successful_endpoint
        rprint(f"[green]✓[/green] Connected to Ollama at {successful_endpoint}")
    else:
        # Prompt for custom endpoint
        while True:
            endpoint = prompt_ollama_endpoint(default_endpoint)
            with Status(f"[cyan]Testing connection to {endpoint}...[/cyan]"):
                result = await test_ollama_connection(endpoint)

            if result.success:
                models = result.data["models"]
                state.ollama_endpoint = endpoint
                rprint(f"[green]✓[/green] Connected to Ollama")
                break
            else:
                rprint(f"[red]✗[/red] {result.error_message}")
                choice = prompt_retry_on_failure("Ollama connection")
                if choice == "abort":
                    return False
                elif choice == "skip":
                    rprint("[yellow]Ollama connection is required, please try again[/yellow]")

    # Check if we have models
    if not models:
        rprint("[red]✗[/red] No models found in Ollama instance")
        rprint("\n[yellow]You need to install an LLM model first.[/yellow]")
        rprint("[dim]We recommend Mistral 7B Instruct for this application:[/dim]")
        rprint("  [cyan]ollama pull mistral:7b-instruct[/cyan]\n")
        return False

    # Get default model from existing config
    default_model = None
    if state.existing_config:
        default_model = state.existing_config.llm.model

    # Show model selection help
    from logsqueak.wizard.providers import get_recommended_ollama_model
    recommended = get_recommended_ollama_model(models)
    if recommended:
        rprint(f"\n[dim]We recommend [cyan]{recommended}[/cyan] - it provides a good balance")
        rprint(f"[dim]of quality and speed for analyzing journal entries.[/dim]")
    else:
        rprint(f"\n[dim]Tip: Mistral 7B Instruct works well for this application.[/dim]")
        rprint(f"[dim]If you don't see it below, install it with:[/dim]")
        rprint(f"  [cyan]ollama pull mistral:7b-instruct[/cyan]")

    # Prompt for model selection
    state.ollama_model = prompt_ollama_model(models, default_model)
    rprint(f"[green]✓[/green] Selected model: {state.ollama_model}\n")

    return True


async def configure_provider(state: WizardState) -> bool:
    """Prompt for provider and configure provider-specific settings.

    Args:
        state: Wizard state to update

    Returns:
        False if user aborts, True otherwise
    """
    # Section header
    rprint("[bold cyan]═══ Step 2: AI Assistant Configuration ═══[/bold cyan]")
    rprint("[dim]Choose the LLM provider that will analyze your journal entries.[/dim]\n")

    # Get default from existing config by runtime detection
    default_provider = None
    if state.existing_config:
        endpoint = str(state.existing_config.llm.endpoint)
        rprint(f"[dim]Detecting existing provider type...[/dim]")
        default_provider = await detect_provider_type(endpoint)
        if default_provider:
            provider_names = {"ollama": "Ollama", "openai": "OpenAI", "custom": "OpenAI-compatible"}
            rprint(f"[dim]Found: {provider_names[default_provider]}[/dim]\n")

    state.provider_type = prompt_provider_choice(default_provider)

    if state.provider_type == "ollama":
        success = await configure_ollama(state)
        if not success:
            return False

        # Prompt for advanced settings
        if prompt_advanced_settings():
            state.num_ctx = prompt_num_ctx(state.num_ctx or 32768)
            state.top_k = prompt_top_k(state.top_k)
    else:
        # OpenAI and Custom not implemented yet (will be in User Story 3)
        rprint("[yellow]OpenAI and Custom providers will be implemented in next phase[/yellow]")
        return False

    return True


async def validate_llm_connection(state: WizardState) -> bool:
    """Test LLM connection with timeout handling.

    Args:
        state: Wizard state with provider settings

    Returns:
        False if user skips, True otherwise
    """
    # For Ollama, we already tested connection in configure_ollama
    if state.provider_type == "ollama":
        rprint("[green]✓[/green] LLM connection validated\n")
        return True

    # OpenAI and Custom will be implemented in User Story 3
    return True


async def validate_embedding(state: WizardState) -> bool:
    """Validate embedding model with disk check and progress.

    Args:
        state: Wizard state

    Returns:
        False if user skips, True otherwise
    """
    # Section header
    rprint("[bold cyan]═══ Step 3: Semantic Search Setup ═══[/bold cyan]")
    rprint("[dim]The embedding model enables searching your knowledge base by meaning,[/dim]")
    rprint("[dim]not just keywords. It converts text into numerical representations.[/dim]\n")

    # Check if already cached
    if check_embedding_model_cached():
        rprint("[dim]Checking embedding model cache...[/dim]")
        rprint("[green]✓[/green] Embedding model already cached (skipping download)\n")
        return True

    # Model needs to be downloaded
    rprint("[dim]Embedding model not found in cache. Download required (~420MB).[/dim]")

    # Check disk space
    rprint("[dim]Checking available disk space...[/dim]")
    disk_result = check_disk_space(512)  # Need ~420MB for model
    if not disk_result.success:
        rprint(f"[yellow]⚠[/yellow] {disk_result.error_message}")
        choice = prompt_retry_on_failure("Disk space check")
        if choice == "abort":
            return False
        elif choice == "skip":
            rprint("[yellow]Skipping embedding model validation[/yellow]")
            return True

    # Download and validate model
    rprint("\n[cyan]Downloading embedding model (all-mpnet-base-v2, ~420MB)...[/cyan]")
    rprint("[dim]This will be cached for future use. May take a few minutes...[/dim]\n")

    with Status("[cyan]Loading embedding model...[/cyan]"):
        result = await validate_embedding_model()

    if result.success:
        rprint("[green]✓[/green] Embedding model loaded successfully\n")
        return True
    else:
        rprint(f"[red]✗[/red] {result.error_message}")
        choice = prompt_retry_on_failure("Embedding model validation")
        if choice == "abort":
            return False
        elif choice == "skip":
            rprint("[yellow]Skipping embedding model validation[/yellow]")
            return True
        # retry would need to loop - for now just continue
        return True


def has_config_changed(new_config: Config, old_config: Config | None) -> bool:
    """Check if the new config differs from the old config.

    Args:
        new_config: Newly assembled config
        old_config: Existing config (or None)

    Returns:
        True if configs differ, False if identical
    """
    if old_config is None:
        return True  # No old config, so this is new

    # Compare key fields
    if str(new_config.llm.endpoint) != str(old_config.llm.endpoint):
        return True
    if new_config.llm.model != old_config.llm.model:
        return True
    if new_config.llm.num_ctx != old_config.llm.num_ctx:
        return True
    if new_config.logseq.graph_path != old_config.logseq.graph_path:
        return True
    if new_config.rag.top_k != old_config.rag.top_k:
        return True

    return False


def assemble_config(state: WizardState) -> Config:
    """Assemble final Config from WizardState, preserving existing providers.

    Args:
        state: Wizard state with all settings

    Returns:
        Complete Config instance
    """
    # Create LLMConfig from current provider
    if state.provider_type == "ollama":
        llm_config = LLMConfig(
            endpoint=HttpUrl(state.ollama_endpoint),
            api_key="ollama",
            model=state.ollama_model,
            num_ctx=state.num_ctx or 32768
        )
    else:
        # OpenAI and Custom will be implemented in User Story 3
        raise ValueError(f"Unsupported provider type: {state.provider_type}")

    # Create LogseqConfig
    logseq_config = LogseqConfig(graph_path=state.graph_path)

    # Create RAGConfig
    rag_config = RAGConfig(top_k=state.top_k)

    # Preserve existing providers and add current one
    llm_providers = {}
    if state.existing_config and state.existing_config.llm_providers:
        llm_providers = dict(state.existing_config.llm_providers)

    # Add current provider
    provider_key = get_provider_key(state.provider_type, state.ollama_endpoint)
    llm_providers[provider_key] = {
        "endpoint": state.ollama_endpoint,
        "api_key": llm_config.api_key,
        "model": llm_config.model,
        "num_ctx": llm_config.num_ctx
    }

    return Config(
        llm=llm_config,
        logseq=logseq_config,
        rag=rag_config,
        llm_providers=llm_providers
    )


async def write_config(config: Config, config_path: Path) -> None:
    """Write config to YAML file with mode 600 permissions.

    Args:
        config: Config instance to write
        config_path: Path to write config file
    """
    # Ensure directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert config to dict for YAML serialization
    config_dict = config.model_dump(mode='json')

    # Write to temp file first (atomic write)
    temp_fd, temp_path = tempfile.mkstemp(
        dir=config_path.parent,
        prefix=".config_",
        suffix=".yaml"
    )

    try:
        with os.fdopen(temp_fd, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        # Set permissions to 600
        os.chmod(temp_path, 0o600)

        # Atomic rename
        os.rename(temp_path, config_path)

    except Exception:
        # Clean up temp file on error
        try:
            os.unlink(temp_path)
        except Exception:
            pass
        raise


def show_success_message(config_path: Path) -> None:
    """Display success message with next steps.

    Args:
        config_path: Path where config was written
    """
    message = (
        "[bold green]✓ Configuration saved successfully![/bold green]\n\n"
        "[bold]Next steps:[/bold]\n"
        "  1. Extract knowledge from your journal:\n"
        "     [cyan]logsqueak extract[/cyan]\n\n"
        "  2. Search your knowledge base:\n"
        "     [cyan]logsqueak search \"your query\"[/cyan]\n\n"
        "  3. Update configuration anytime:\n"
        "     [cyan]logsqueak init[/cyan]\n\n"
        f"[dim]Configuration file: {config_path}[/dim]"
    )

    panel = Panel(message, border_style="green", padding=(1, 2))
    rprint(panel)


async def run_setup_wizard() -> bool:
    """Run the interactive setup wizard.

    Returns:
        True if setup completed successfully, False if aborted
    """
    rprint("\n[bold cyan]Logsqueak Setup Wizard[/bold cyan]\n")

    state = WizardState()

    try:
        # Load existing config if present
        state.existing_config = load_existing_config()
        if state.existing_config:
            rprint("[dim]Found existing configuration, using as defaults[/dim]\n")

        # Configure graph path
        if not await configure_graph_path(state):
            return False

        # Configure provider
        if not await configure_provider(state):
            return False

        # Validate LLM connection
        if not await validate_llm_connection(state):
            return False

        # Validate embedding model
        if not await validate_embedding(state):
            return False

        # Assemble final config
        config = assemble_config(state)

        # Check if anything actually changed
        if not has_config_changed(config, state.existing_config):
            rprint("[bold cyan]═══ No Changes Detected ═══[/bold cyan]")
            rprint("[dim]Your configuration is already up to date. Nothing to save.[/dim]\n")
            return True

        # Check if config file exists and prompt for overwrite
        config_path = Path.home() / ".config" / "logsqueak" / "config.yaml"
        if config_path.exists():
            rprint("[bold cyan]═══ Save Configuration ═══[/bold cyan]")
            if not prompt_confirm_overwrite():
                rprint("[yellow]Setup cancelled - existing config preserved[/yellow]")
                return False

        # Write config
        await write_config(config, config_path)

        # Show success message
        show_success_message(config_path)

        return True

    except KeyboardInterrupt:
        rprint("\n[yellow]Setup cancelled[/yellow]")
        return False
    except Exception as e:
        rprint(f"\n[red]✗ Setup failed: {str(e)}[/red]")
        return False
