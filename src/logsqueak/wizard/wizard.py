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
from logsqueak.utils.logging import get_logger

logger = get_logger(__name__)
from logsqueak.wizard.prompts import (
    prompt_advanced_settings,
    prompt_confirm_overwrite,
    prompt_continue_on_timeout,
    prompt_custom_api_key,
    prompt_custom_endpoint,
    prompt_custom_model,
    prompt_graph_path,
    prompt_index_graph,
    prompt_num_ctx,
    prompt_ollama_endpoint,
    prompt_ollama_model,
    prompt_openai_api_key,
    prompt_openai_model,
    prompt_provider_choice,
    prompt_retry_on_failure,
    prompt_top_k,
)
from logsqueak.wizard.providers import fetch_ollama_models, get_provider_key
from logsqueak.wizard.validators import (
    check_disk_space,
    check_embedding_model_cached,
    validate_ollama_connection,
    validate_openai_connection,
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
        openai_endpoint: OpenAI API endpoint (if provider is openai)
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
    openai_endpoint: str | None = None
    openai_api_key: str | None = None
    openai_model: str | None = None
    custom_endpoint: str | None = None
    custom_api_key: str | None = None
    custom_model: str | None = None
    num_ctx: int | None = None
    top_k: int = 20


def load_existing_config() -> Config | None:
    """Load existing config if present, return None if not found or invalid.

    Handles partial config extraction for broken configs by logging errors
    but attempting to extract valid fields where possible. For permission
    errors, bypasses permission check to read values as defaults.

    Returns:
        Config instance if loaded successfully, None otherwise
    """
    import stat

    config_path = Path.home() / ".config" / "logsqueak" / "config.yaml"

    logger.debug("load_config_started", config_path=str(config_path))

    if not config_path.exists():
        logger.debug("config_not_found", config_path=str(config_path))
        return None

    # Check for empty or whitespace-only files
    try:
        with open(config_path) as f:
            content = f.read()
            if not content.strip():
                logger.warning("config_file_empty", config_path=str(config_path))
                return None
    except Exception as e:
        logger.warning("config_read_failed", config_path=str(config_path), error=str(e))
        return None

    try:
        config = Config.load(config_path)
        logger.info("config_loaded_successfully", config_path=str(config_path))
        return config
    except PermissionError as e:
        # Log permission error but try to read anyway for defaults
        logger.warning("config_permission_error", config_path=str(config_path), error=str(e))

        # Bypass permission check - read YAML directly for defaults
        try:
            with open(config_path) as f:
                data = yaml.safe_load(f)
            config = Config(**data)
            logger.info("config_loaded_bypassing_permissions", config_path=str(config_path))
            return config
        except Exception as read_error:
            logger.error("config_bypass_read_failed", config_path=str(config_path), error=str(read_error))
            return None
    except Exception as e:
        # Log other errors (invalid YAML, validation failures, etc.)
        logger.error("config_load_failed", config_path=str(config_path), error=str(e), error_type=type(e).__name__)
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
    result = await validate_ollama_connection(endpoint, timeout=5)
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
    logger.info("configure_graph_path_started")

    # Section header
    rprint("[bold cyan]═══ Step 1: Logseq Graph Location ═══[/bold cyan]")
    rprint("[dim]This is where your Logseq notes are stored on disk.[/dim]\n")

    # Get default from existing config
    default = None
    if state.existing_config:
        default = state.existing_config.logseq.graph_path
        logger.debug("graph_path_default_from_config", default_path=default)

    while True:
        try:
            path = prompt_graph_path(default)
            logger.debug("graph_path_prompted", path=path)
            result = validate_graph_path(path)

            if result.success:
                state.graph_path = result.data["path"]
                logger.info("graph_path_validated", path=state.graph_path)
                rprint(f"[green]✓[/green] Graph path validated: {state.graph_path}\n")
                return True
            else:
                logger.warning("graph_path_validation_failed", path=path, error=result.error_message)
                rprint(f"[red]✗[/red] {result.error_message}")
                choice = prompt_retry_on_failure("Graph path validation")
                logger.debug("graph_path_retry_choice", choice=choice)
                if choice == "abort":
                    logger.info("graph_path_aborted_by_user")
                    return False
                elif choice == "skip":
                    # Can't skip graph path - it's required
                    rprint("[yellow]Graph path is required, please try again[/yellow]")
                # retry continues loop

        except KeyboardInterrupt:
            logger.info("graph_path_cancelled_keyboard_interrupt")
            rprint("\n[yellow]Setup cancelled[/yellow]")
            return False


async def configure_ollama(state: WizardState) -> bool:
    """Ollama-specific configuration flow.

    Args:
        state: Wizard state to update

    Returns:
        False if user aborts, True otherwise
    """
    logger.info("configure_ollama_started")
    rprint("[dim]Ollama is a local LLM runtime. Logsqueak uses it to analyze your notes.[/dim]\n")

    # Get default endpoint from existing config
    default_endpoint = "http://localhost:11434"

    # First, check llm_providers for any Ollama endpoint (handles provider switching)
    if state.existing_config and state.existing_config.llm_providers:
        # Check for ollama_local or ollama_remote in llm_providers
        for provider_key in ["ollama_remote", "ollama_local"]:
            if provider_key in state.existing_config.llm_providers:
                provider_config = state.existing_config.llm_providers[provider_key]
                if "endpoint" in provider_config:
                    default_endpoint = provider_config["endpoint"]
                    logger.debug("ollama_endpoint_from_providers", endpoint=default_endpoint, provider_key=provider_key)
                    break

    # Fallback: check if currently active endpoint is Ollama
    if default_endpoint == "http://localhost:11434" and state.existing_config and state.existing_config.llm.endpoint:
        endpoint_str = str(state.existing_config.llm.endpoint)
        if "ollama" in endpoint_str or "11434" in endpoint_str:
            default_endpoint = endpoint_str
            logger.debug("ollama_endpoint_from_active_config", endpoint=default_endpoint)

    # Try existing endpoint first, then localhost, then prompt
    endpoints_to_try = [default_endpoint]
    if default_endpoint != "http://localhost:11434":
        endpoints_to_try.append("http://localhost:11434")

    models = None
    successful_endpoint = None

    logger.debug("ollama_testing_endpoints", endpoints=endpoints_to_try)
    with Status("[cyan]Testing Ollama connection...[/cyan]") as status:
        for endpoint in endpoints_to_try:
            result = await validate_ollama_connection(endpoint)
            if result.success:
                models = result.data["models"]
                successful_endpoint = endpoint
                logger.info("ollama_connection_success", endpoint=endpoint, model_count=len(models))
                break
            else:
                logger.debug("ollama_connection_failed", endpoint=endpoint, error=result.error_message)

    if models and successful_endpoint:
        state.ollama_endpoint = successful_endpoint
        rprint(f"[green]✓[/green] Connected to Ollama at {successful_endpoint}")
    else:
        # Prompt for custom endpoint
        logger.debug("ollama_prompting_custom_endpoint")
        while True:
            endpoint = prompt_ollama_endpoint(default_endpoint)
            logger.debug("ollama_custom_endpoint_prompted", endpoint=endpoint)
            with Status(f"[cyan]Testing connection to {endpoint}...[/cyan]"):
                result = await validate_ollama_connection(endpoint)

            if result.success:
                models = result.data["models"]
                state.ollama_endpoint = endpoint
                logger.info("ollama_custom_connection_success", endpoint=endpoint, model_count=len(models))
                rprint(f"[green]✓[/green] Connected to Ollama")
                break
            else:
                logger.warning("ollama_custom_connection_failed", endpoint=endpoint, error=result.error_message)
                rprint(f"[red]✗[/red] {result.error_message}")
                choice = prompt_retry_on_failure("Ollama connection")
                logger.debug("ollama_retry_choice", choice=choice)
                if choice == "abort":
                    logger.info("ollama_connection_aborted_by_user")
                    return False
                elif choice == "skip":
                    rprint("[yellow]Ollama connection is required, please try again[/yellow]")

    # Check if we have models
    if not models:
        logger.error("ollama_no_models_found", endpoint=state.ollama_endpoint)
        rprint("[red]✗[/red] No models found in Ollama instance")
        rprint("\n[yellow]You need to install an LLM model first.[/yellow]")
        rprint("[dim]We recommend Mistral 7B Instruct for this application:[/dim]")
        rprint("  [cyan]ollama pull mistral:7b-instruct[/cyan]\n")
        return False

    # Get default model from existing config
    default_model = None

    # First, check llm_providers for Ollama model (handles provider switching)
    if state.existing_config and state.existing_config.llm_providers:
        for provider_key in ["ollama_remote", "ollama_local"]:
            if provider_key in state.existing_config.llm_providers:
                provider_config = state.existing_config.llm_providers[provider_key]
                if "model" in provider_config:
                    default_model = provider_config["model"]
                    logger.debug("ollama_model_from_providers", model=default_model, provider_key=provider_key)
                    break

    # Fallback: check if currently active model is suitable
    if default_model is None and state.existing_config:
        # Only use active model if active provider is Ollama
        endpoint_str = str(state.existing_config.llm.endpoint).lower()
        if "ollama" in endpoint_str or "11434" in endpoint_str:
            default_model = state.existing_config.llm.model
            logger.debug("ollama_model_from_active_config", model=default_model)

    # Show model selection help
    from logsqueak.wizard.providers import get_recommended_ollama_model
    recommended = get_recommended_ollama_model(models)
    if recommended:
        logger.debug("ollama_recommended_model_found", model=recommended)
        rprint(f"\n[dim]We recommend [cyan]{recommended}[/cyan] - it provides a good balance")
        rprint(f"[dim]of quality and speed for analyzing journal entries.[/dim]")
    else:
        logger.debug("ollama_no_recommended_model")
        rprint(f"\n[dim]Tip: Mistral 7B Instruct works well for this application.[/dim]")
        rprint(f"[dim]If you don't see it below, install it with:[/dim]")
        rprint(f"  [cyan]ollama pull mistral:7b-instruct[/cyan]")

    # Prompt for model selection
    state.ollama_model = prompt_ollama_model(models, default_model)
    logger.info("ollama_model_selected", model=state.ollama_model)
    rprint(f"[green]✓[/green] Selected model: {state.ollama_model}\n")

    return True


async def configure_openai(state: WizardState) -> bool:
    """OpenAI-specific configuration flow.

    Args:
        state: Wizard state to update

    Returns:
        False if user aborts, True otherwise
    """
    logger.info("configure_openai_started")
    rprint("[dim]OpenAI provides GPT-4 and other models via API.[/dim]\n")

    # Get existing API key from config
    existing_key = None
    if state.existing_config and state.existing_config.llm_providers:
        openai_config = state.existing_config.llm_providers.get("openai")
        if openai_config:
            existing_key = openai_config.get("api_key")
            logger.debug("openai_existing_key_found")

    # Prompt for API key
    state.openai_api_key = prompt_openai_api_key(existing_key)
    logger.debug("openai_api_key_entered")

    # Get default model from existing config
    default_model = "gpt-4o"
    if state.existing_config and state.existing_config.llm_providers:
        openai_config = state.existing_config.llm_providers.get("openai")
        if openai_config:
            default_model = openai_config.get("model", "gpt-4o")
            logger.debug("openai_model_default_from_config", default_model=default_model)

    # Prompt for model selection
    state.openai_model = prompt_openai_model(default_model)
    state.openai_endpoint = "https://api.openai.com/v1"

    logger.info("openai_configured", model=state.openai_model, endpoint=state.openai_endpoint)
    rprint(f"[green]✓[/green] Selected model: {state.openai_model}\n")

    return True


async def configure_custom(state: WizardState) -> bool:
    """Custom OpenAI-compatible endpoint configuration flow.

    Args:
        state: Wizard state to update

    Returns:
        False if user aborts, True otherwise
    """
    logger.info("configure_custom_started")
    rprint("[dim]Configure a custom OpenAI-compatible endpoint.[/dim]\n")

    # Get existing values from config
    existing_endpoint = None
    existing_key = None
    existing_model = None
    if state.existing_config and state.existing_config.llm_providers:
        custom_config = state.existing_config.llm_providers.get("custom")
        if custom_config:
            existing_endpoint = custom_config.get("endpoint")
            existing_key = custom_config.get("api_key")
            existing_model = custom_config.get("model")
            logger.debug("custom_existing_config_found", endpoint=existing_endpoint, model=existing_model)

    # Prompt for endpoint
    state.custom_endpoint = prompt_custom_endpoint(existing_endpoint)
    logger.debug("custom_endpoint_entered", endpoint=state.custom_endpoint)

    # Prompt for API key
    state.custom_api_key = prompt_custom_api_key(existing_key)
    logger.debug("custom_api_key_entered")

    # Prompt for model name
    state.custom_model = prompt_custom_model(existing_model)
    logger.info("custom_configured", endpoint=state.custom_endpoint, model=state.custom_model)

    rprint(f"[green]✓[/green] Configured custom endpoint: {state.custom_endpoint}\n")

    return True


async def configure_provider(state: WizardState) -> bool:
    """Prompt for provider and configure provider-specific settings.

    Args:
        state: Wizard state to update

    Returns:
        False if user aborts, True otherwise
    """
    logger.info("configure_provider_started")

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
            logger.info("provider_detected_from_config", provider_type=default_provider, endpoint=endpoint)
            rprint(f"[dim]Found: {provider_names[default_provider]}[/dim]\n")

    state.provider_type = prompt_provider_choice(default_provider)
    logger.info("provider_selected", provider_type=state.provider_type)

    if state.provider_type == "ollama":
        success = await configure_ollama(state)
        if not success:
            return False

        # Prompt for advanced settings
        if prompt_advanced_settings():
            logger.debug("ollama_advanced_settings_requested")
            state.num_ctx = prompt_num_ctx(state.num_ctx or 32768)
            state.top_k = prompt_top_k(state.top_k)
            logger.info("ollama_advanced_settings_configured", num_ctx=state.num_ctx, top_k=state.top_k)
    elif state.provider_type == "openai":
        success = await configure_openai(state)
        if not success:
            return False

        # Prompt for advanced settings (top_k only for OpenAI)
        if prompt_advanced_settings():
            logger.debug("openai_advanced_settings_requested")
            state.top_k = prompt_top_k(state.top_k)
            logger.info("openai_advanced_settings_configured", top_k=state.top_k)
    elif state.provider_type == "custom":
        success = await configure_custom(state)
        if not success:
            return False

        # Prompt for advanced settings (top_k only for custom)
        if prompt_advanced_settings():
            logger.debug("custom_advanced_settings_requested")
            state.top_k = prompt_top_k(state.top_k)
            logger.info("custom_advanced_settings_configured", top_k=state.top_k)
    else:
        logger.error("unknown_provider_type", provider_type=state.provider_type)
        rprint(f"[red]✗[/red] Unknown provider type: {state.provider_type}")
        return False

    return True


async def validate_llm_connection(state: WizardState) -> bool:
    """Test LLM connection with timeout handling.

    Wraps the connection test with a 30-second timeout. If timeout occurs,
    prompts user to continue waiting, retry, or skip validation.

    Args:
        state: Wizard state with provider settings

    Returns:
        False if user aborts, True otherwise
    """
    logger.info("validate_llm_connection_started", provider_type=state.provider_type)

    # For Ollama, we already tested connection in configure_ollama
    if state.provider_type == "ollama":
        logger.debug("llm_validation_skipped_ollama_already_tested")
        rprint("[green]✓[/green] LLM connection validated\n")
        return True

    # Test OpenAI or custom connection
    endpoint = None
    api_key = None
    model = None

    if state.provider_type == "openai":
        endpoint = state.openai_endpoint
        api_key = state.openai_api_key
        model = state.openai_model
    elif state.provider_type == "custom":
        endpoint = state.custom_endpoint
        api_key = state.custom_api_key
        model = state.custom_model

    if not endpoint or not api_key or not model:
        logger.warning("llm_validation_missing_config", endpoint=endpoint, has_key=bool(api_key), model=model)
        rprint("[yellow]⚠[/yellow] Missing provider configuration")
        return True  # Continue anyway

    # Test connection with timeout handling
    timeout_seconds = 30
    max_retries = 3
    retry_count = 0

    logger.debug("llm_validation_started", endpoint=endpoint, model=model, timeout=timeout_seconds)

    while retry_count < max_retries:
        try:
            with Status(f"[cyan]Testing API connection...[/cyan]"):
                # Wrap in timeout
                async with asyncio.timeout(timeout_seconds):
                    result = await validate_openai_connection(endpoint, api_key, model, timeout=timeout_seconds)

            if result.success:
                logger.info("llm_validation_success", endpoint=endpoint, model=model)
                rprint(f"[green]✓[/green] API connection validated\n")
                return True
            else:
                logger.warning("llm_validation_failed", endpoint=endpoint, error=result.error_message, retry_count=retry_count)
                rprint(f"[red]✗[/red] {result.error_message}")
                choice = prompt_retry_on_failure("API connection")
                logger.debug("llm_validation_retry_choice", choice=choice)
                if choice == "abort":
                    logger.info("llm_validation_aborted_by_user")
                    return False
                elif choice == "skip":
                    logger.info("llm_validation_skipped_by_user")
                    rprint("[yellow]Skipping API connection validation[/yellow]\n")
                    return True
                elif choice == "retry":
                    retry_count += 1
                    continue
                return True

        except asyncio.TimeoutError:
            # Connection attempt timed out
            logger.warning("llm_validation_timeout", timeout_seconds=timeout_seconds, retry_count=retry_count)
            choice = prompt_continue_on_timeout("API connection test", timeout_seconds)
            logger.debug("llm_validation_timeout_choice", choice=choice)

            if choice == "continue":
                # User wants to keep waiting - increase timeout and retry
                timeout_seconds = timeout_seconds * 2
                retry_count += 1
                logger.debug("llm_validation_timeout_continuing", new_timeout=timeout_seconds)
                rprint(f"[cyan]Retrying with {timeout_seconds}s timeout...[/cyan]\n")
                continue
            elif choice == "retry":
                # User wants to retry with same timeout
                retry_count += 1
                continue
            elif choice == "skip":
                # User wants to skip validation
                logger.info("llm_validation_skipped_after_timeout")
                rprint("[yellow]Skipping API connection validation[/yellow]\n")
                return True
            else:
                # Unknown choice, treat as skip
                return True

    # Max retries reached
    logger.warning("llm_validation_max_retries_reached", max_retries=max_retries)
    rprint("[yellow]⚠[/yellow] Maximum retry attempts reached")
    rprint("[yellow]Skipping API connection validation[/yellow]\n")
    return True


async def validate_embedding(state: WizardState) -> bool:
    """Validate embedding model with disk check and progress.

    Args:
        state: Wizard state

    Returns:
        False if user skips, True otherwise
    """
    logger.info("validate_embedding_started")

    # Section header
    rprint("[bold cyan]═══ Step 3: Semantic Search Setup ═══[/bold cyan]")
    rprint("[dim]The embedding model enables searching your knowledge base by meaning,[/dim]")
    rprint("[dim]not just keywords. It converts text into numerical representations.[/dim]\n")

    # Check if already cached
    rprint("[dim]Checking embedding model cache...[/dim]")
    if check_embedding_model_cached():
        logger.info("embedding_model_cached_found")
        rprint("[green]✓[/green] Embedding model already cached (skipping download)\n")
        return True

    # Model needs to be downloaded
    logger.info("embedding_model_download_required")
    rprint("[dim]Embedding model not found in cache. Download required (~420MB).[/dim]")

    # Check disk space
    rprint("[dim]Checking available disk space...[/dim]")
    disk_result = check_disk_space(1024)  # Warn if <1GB available
    if not disk_result.success:
        # Low disk space warning - allow user to proceed or abort
        available_mb = disk_result.data.get("available_mb", 0) if disk_result.data else 0
        logger.warning("low_disk_space_detected", available_mb=available_mb, required_mb=1024)
        rprint(f"[yellow]⚠[/yellow] Low disk space detected:")
        rprint(f"[yellow]  Available: {available_mb} MB[/yellow]")
        rprint(f"[yellow]  Recommended: 1024 MB (1 GB)[/yellow]")
        rprint(f"[yellow]  Model size: ~420 MB[/yellow]\n")

        # Prompt user whether to proceed
        from logsqueak.wizard.prompts import Confirm
        proceed = Confirm.ask(
            "[bold yellow]Proceed with download anyway?[/bold yellow]",
            default=False
        )

        if not proceed:
            logger.info("embedding_validation_cancelled_low_disk_space")
            rprint("[yellow]Skipping embedding model validation[/yellow]")
            return True
        else:
            logger.info("embedding_validation_proceeding_despite_low_disk_space")

    # Download and validate model with timeout handling
    rprint("\n[cyan]Downloading embedding model (all-mpnet-base-v2, ~420MB)...[/cyan]")
    rprint("[dim]This will be cached for future use. May take a few minutes...[/dim]\n")

    timeout_seconds = 300  # 5 minutes default timeout
    retry_count = 0
    max_retries = 3

    logger.debug("embedding_download_started", timeout_seconds=timeout_seconds, max_retries=max_retries)

    while retry_count < max_retries:
        try:
            with Status("[cyan]Loading embedding model...[/cyan]"):
                result = await asyncio.wait_for(
                    validate_embedding_model(),
                    timeout=timeout_seconds
                )

            if result.success:
                logger.info("embedding_model_loaded_successfully")
                rprint("[green]✓[/green] Embedding model loaded successfully\n")
                return True
            else:
                logger.warning("embedding_validation_failed", error=result.error_message, retry_count=retry_count)
                rprint(f"[red]✗[/red] {result.error_message}")
                choice = prompt_retry_on_failure("Embedding model validation")
                logger.debug("embedding_validation_retry_choice", choice=choice)
                if choice == "abort":
                    logger.info("embedding_validation_aborted_by_user")
                    return False
                elif choice == "skip":
                    logger.info("embedding_validation_skipped_by_user")
                    rprint("[yellow]Skipping embedding model validation[/yellow]")
                    return True
                elif choice == "retry":
                    retry_count += 1
                    continue
                return True

        except asyncio.TimeoutError:
            # Download timed out
            logger.warning("embedding_download_timeout", timeout_seconds=timeout_seconds, retry_count=retry_count)
            choice = prompt_continue_on_timeout("Embedding model download", timeout_seconds)
            logger.debug("embedding_timeout_choice", choice=choice)

            if choice == "continue":
                # User wants to keep waiting - increase timeout and retry
                timeout_seconds = timeout_seconds * 2
                retry_count += 1
                logger.debug("embedding_timeout_continuing", new_timeout=timeout_seconds)
                rprint(f"[cyan]Retrying with {timeout_seconds}s timeout...[/cyan]\n")
                continue
            elif choice == "retry":
                # User wants to retry with same timeout
                retry_count += 1
                continue
            elif choice == "skip":
                # User wants to skip validation
                logger.info("embedding_validation_skipped_after_timeout")
                rprint("[yellow]Skipping embedding model validation[/yellow]\n")
                return True
            else:
                # Unknown choice, treat as skip
                return True

    # Max retries reached
    logger.warning("embedding_validation_max_retries_reached", max_retries=max_retries)
    rprint("[yellow]⚠[/yellow] Maximum retry attempts reached")
    rprint("[yellow]Skipping embedding model validation[/yellow]\n")
    return True


async def index_graph_after_setup(graph_path: str) -> bool:
    """Build initial search index for the graph with progress feedback.

    Args:
        graph_path: Path to Logseq graph directory

    Returns:
        True if indexing succeeded, False if user skipped or error occurred
    """
    from logseq_outline.graph import GraphPaths
    from logsqueak.services.page_indexer import PageIndexer
    from logsqueak.services.rag_search import RAGSearch
    from logsqueak.utils.index_progress import create_index_progress_callback
    from rich.console import Console

    logger.info("index_graph_started", graph_path=graph_path)

    try:
        # Initialize services
        graph_paths = GraphPaths(Path(graph_path))
        page_indexer = PageIndexer(graph_paths=graph_paths)
        rag_search = RAGSearch(db_path=page_indexer.db_path)

        # Check if already indexed
        has_data = rag_search.has_indexed_data()

        # Create console for Rich output
        console = Console()

        # Create progress callback with wizard-appropriate settings
        progress_callback, cleanup = create_index_progress_callback(
            console=console,
            reindex=False,  # Wizard never forces reindex
            has_data=has_data,
            use_echo=False  # Use Rich rprint style
        )

        try:
            pages_indexed = await page_indexer.build_index(progress_callback=progress_callback)

            # Only show completion message if we did work
            if pages_indexed > 0:
                rprint()  # Newline after progress
                if not has_data:
                    rprint("[green]✓[/green] Index built successfully")
                else:
                    rprint("[green]✓[/green] Index updated successfully")
                logger.info("index_graph_completed", pages_indexed=pages_indexed)
            else:
                # No pages needed indexing (all up-to-date)
                logger.info("index_graph_skipped", reason="all_up_to_date")

            return True

        except Exception as e:
            rprint()  # Newline after progress
            logger.error("index_graph_failed", error=str(e))
            rprint(f"[red]✗[/red] Failed to build search index: {e}")
            rprint("[yellow]You can run 'logsqueak search' later to build the index[/yellow]\n")
            return False
        finally:
            cleanup()

    except Exception as e:
        logger.error("index_graph_initialization_failed", error=str(e))
        rprint(f"[red]✗[/red] Failed to initialize indexing: {e}")
        rprint("[yellow]You can run 'logsqueak search' later to build the index[/yellow]\n")
        return False


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

    When switching providers, this function ensures that credentials for ALL
    providers (both old and new) are preserved in the llm_providers dict.
    This allows users to switch between providers without re-entering credentials.

    Args:
        state: Wizard state with all settings

    Returns:
        Complete Config instance with all provider credentials preserved
    """
    # Create LLMConfig from current provider
    if state.provider_type == "ollama":
        llm_config = LLMConfig(
            endpoint=HttpUrl(state.ollama_endpoint),
            api_key="ollama",
            model=state.ollama_model,
            num_ctx=state.num_ctx or 32768
        )
        provider_endpoint = state.ollama_endpoint
        provider_dict = {
            "endpoint": state.ollama_endpoint,
            "api_key": "ollama",
            "model": state.ollama_model,
            "num_ctx": llm_config.num_ctx
        }
    elif state.provider_type == "openai":
        llm_config = LLMConfig(
            endpoint=HttpUrl(state.openai_endpoint),
            api_key=state.openai_api_key,
            model=state.openai_model
        )
        provider_endpoint = state.openai_endpoint
        provider_dict = {
            "endpoint": state.openai_endpoint,
            "api_key": state.openai_api_key,
            "model": state.openai_model
        }
    elif state.provider_type == "custom":
        llm_config = LLMConfig(
            endpoint=HttpUrl(state.custom_endpoint),
            api_key=state.custom_api_key,
            model=state.custom_model
        )
        provider_endpoint = state.custom_endpoint
        provider_dict = {
            "endpoint": state.custom_endpoint,
            "api_key": state.custom_api_key,
            "model": state.custom_model
        }
    else:
        raise ValueError(f"Unsupported provider type: {state.provider_type}")

    # Create LogseqConfig
    logseq_config = LogseqConfig(graph_path=state.graph_path)

    # Create RAGConfig
    rag_config = RAGConfig(top_k=state.top_k)

    # Preserve ALL existing providers from config (critical for provider switching)
    llm_providers = {}
    if state.existing_config and state.existing_config.llm_providers:
        # Copy all existing provider credentials
        llm_providers = dict(state.existing_config.llm_providers)

    # Add/update current provider credentials
    provider_key = get_provider_key(state.provider_type, provider_endpoint)
    llm_providers[provider_key] = provider_dict

    # Validation: Ensure both old and new provider credentials are present
    # when switching providers (US4 requirement)
    if state.existing_config and state.existing_config.llm:
        old_provider_type = _detect_provider_type(state.existing_config.llm.endpoint)
        old_provider_key = get_provider_key(
            old_provider_type,
            str(state.existing_config.llm.endpoint)
        )
        # Verify old provider is preserved when switching
        if old_provider_type != state.provider_type:
            if old_provider_key not in llm_providers:
                # This should not happen if existing_config.llm_providers was set correctly
                # But preserve it anyway from existing_config.llm
                llm_providers[old_provider_key] = {
                    "endpoint": str(state.existing_config.llm.endpoint),
                    "api_key": state.existing_config.llm.api_key,
                    "model": state.existing_config.llm.model,
                }
                if state.existing_config.llm.num_ctx:
                    llm_providers[old_provider_key]["num_ctx"] = state.existing_config.llm.num_ctx

    return Config(
        llm=llm_config,
        logseq=logseq_config,
        rag=rag_config,
        llm_providers=llm_providers
    )


def _detect_provider_type(endpoint: HttpUrl) -> str:
    """Detect provider type from endpoint URL.

    Args:
        endpoint: LLM endpoint URL

    Returns:
        Provider type ("ollama", "openai", or "custom")
    """
    endpoint_str = str(endpoint).lower()
    if "11434" in endpoint_str or "ollama" in endpoint_str:
        return "ollama"
    elif "api.openai.com" in endpoint_str:
        return "openai"
    else:
        return "custom"


async def write_config(config: Config, config_path: Path) -> None:
    """Write config to YAML file with mode 600 permissions.

    Args:
        config: Config instance to write
        config_path: Path to write config file

    Raises:
        PermissionError: If cannot create config directory or write file
        OSError: If filesystem error occurs during write
    """
    logger.info("write_config_started", config_path=str(config_path))

    # Ensure directory exists
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug("config_directory_created", directory=str(config_path.parent))
    except PermissionError as e:
        logger.error("config_directory_permission_denied", directory=str(config_path.parent), error=str(e))
        raise PermissionError(
            f"[PERMISSION DENIED] Cannot create config directory\n\n"
            f"Unable to create directory: {config_path.parent}\n\n"
            f"→ Ensure you have write permissions to ~/.config/\n"
            f"→ Run: chmod u+w ~/.config\n"
            f"→ Or choose a different config location"
        ) from e
    except OSError as e:
        logger.error("config_directory_creation_failed", directory=str(config_path.parent), error=str(e))
        raise OSError(
            f"[FILESYSTEM ERROR] Failed to create config directory\n\n"
            f"Unable to create directory: {config_path.parent}\n"
            f"Error: {e}\n\n"
            f"→ Check available disk space\n"
            f"→ Verify filesystem is healthy\n"
            f"→ Check filesystem permissions"
        ) from e

    # Convert config to dict for YAML serialization
    config_dict = config.model_dump(mode='json')

    # Write to temp file first (atomic write)
    temp_fd, temp_path = tempfile.mkstemp(
        dir=config_path.parent,
        prefix=".config_",
        suffix=".yaml"
    )
    logger.debug("config_temp_file_created", temp_path=temp_path)

    try:
        with os.fdopen(temp_fd, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        # Set permissions to 600
        os.chmod(temp_path, 0o600)
        logger.debug("config_permissions_set", permissions="600")

        # Atomic rename
        os.rename(temp_path, config_path)
        logger.info("config_written_successfully", config_path=str(config_path))

    except Exception as e:
        # Clean up temp file on error
        logger.error("config_write_failed", error=str(e), error_type=type(e).__name__)
        try:
            os.unlink(temp_path)
            logger.debug("config_temp_file_cleaned_up", temp_path=temp_path)
        except Exception:
            pass
        raise


async def offer_graph_indexing(graph_path: str) -> None:
    """Offer to index the graph and execute if user accepts.

    Only prompts if the index doesn't already exist.

    Args:
        graph_path: Path to Logseq graph directory
    """
    from logseq_outline.graph import GraphPaths
    from logsqueak.services.page_indexer import PageIndexer
    from logsqueak.services.rag_search import RAGSearch

    # Check if index already exists
    try:
        graph_paths = GraphPaths(Path(graph_path))
        page_indexer = PageIndexer(graph_paths=graph_paths)
        rag_search = RAGSearch(db_path=page_indexer.db_path)

        if rag_search.has_indexed_data():
            logger.info("graph_index_already_exists", skipping_prompt=True)
            return  # Index exists, no need to prompt
    except Exception as e:
        # If we can't check, log and skip prompting (safe default)
        logger.warning("index_check_failed", error=str(e))
        return

    # Index doesn't exist, prompt user
    if prompt_index_graph():
        logger.info("user_accepted_graph_indexing")
        await index_graph_after_setup(graph_path)
    else:
        logger.info("user_declined_graph_indexing")
        rprint("[dim]You can index your knowledge base later with:[/dim]")
        rprint("[dim]  logsqueak search \"test query\"[/dim]\n")


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

    Optimized for fast updates when modifying existing configuration:
    - Loads existing config values as defaults for all prompts
    - Skips embedding model download if already cached (validate_embedding handles this)
    - Detects when no changes were made and skips unnecessary writes
    - Preserves all provider credentials when switching providers

    Returns:
        True if setup completed successfully, False if aborted
    """
    logger.info("setup_wizard_started")
    rprint("\n[bold cyan]Logsqueak Setup Wizard[/bold cyan]\n")

    state = WizardState()

    try:
        # Check for existing config with permission issues
        config_path = Path.home() / ".config" / "logsqueak" / "config.yaml"
        has_permission_issue = False
        if config_path.exists():
            import stat
            mode = os.stat(config_path).st_mode
            if mode & (stat.S_IRWXG | stat.S_IRWXO):
                has_permission_issue = True
                logger.warning("config_permission_issue_detected", mode=oct(mode)[-3:])
                rprint("[yellow]⚠[/yellow] [bold]Config file has incorrect permissions[/bold]")
                rprint(f"[dim]Current permissions: {oct(mode)[-3:]}[/dim]")
                rprint("[dim]Config file should be readable only by owner (mode 600)[/dim]\n")
                rprint("[yellow]The wizard will create a new config with correct permissions.[/yellow]\n")

        # Load existing config if present (enables fast updates)
        state.existing_config = load_existing_config()
        if state.existing_config:
            logger.info("existing_config_loaded_as_defaults")
            rprint("[dim]Found existing configuration, using as defaults[/dim]\n")

            # Pre-populate advanced settings from existing config
            if state.existing_config.llm.num_ctx:
                state.num_ctx = state.existing_config.llm.num_ctx
            if state.existing_config.rag.top_k:
                state.top_k = state.existing_config.rag.top_k

        # Configure graph path (uses existing value as default)
        if not await configure_graph_path(state):
            logger.info("setup_wizard_aborted_at_graph_path")
            return False

        # Configure provider (uses existing provider as default)
        if not await configure_provider(state):
            logger.info("setup_wizard_aborted_at_provider")
            return False

        # Validate LLM connection (quick test, typically <30s)
        if not await validate_llm_connection(state):
            logger.info("setup_wizard_aborted_at_llm_validation")
            return False

        # Validate embedding model (optimized: skips download if cached)
        # This is where the speed improvement for updates comes from
        if not await validate_embedding(state):
            logger.info("setup_wizard_aborted_at_embedding_validation")
            return False

        # Assemble final config (preserves all provider credentials)
        logger.info("assembling_config", provider_type=state.provider_type, graph_path=state.graph_path)
        config = assemble_config(state)

        # Check if anything actually changed (but always write if permissions are wrong)
        # This optimization prevents unnecessary writes when user accepts all defaults
        if not has_config_changed(config, state.existing_config) and not has_permission_issue:
            logger.info("no_config_changes_detected")
            rprint("[bold cyan]═══ No Changes Detected ═══[/bold cyan]")
            rprint("[dim]Your configuration is already up to date. Nothing to save.[/dim]\n")

            # Still offer to index the graph even when config unchanged
            await offer_graph_indexing(state.graph_path)
            return True

        logger.info("config_changes_detected", has_permission_issue=has_permission_issue)

        # Check if config file exists and prompt for overwrite
        config_path = Path.home() / ".config" / "logsqueak" / "config.yaml"
        if config_path.exists() and not has_permission_issue:
            # Only prompt if no permission issue (permission fix is automatic)
            rprint("[bold cyan]═══ Save Configuration ═══[/bold cyan]")
            if not prompt_confirm_overwrite():
                logger.info("setup_wizard_cancelled_overwrite_declined")
                rprint("[yellow]Setup cancelled - existing config preserved[/yellow]")
                return False

        # Write config
        await write_config(config, config_path)

        # Show success message
        show_success_message(config_path)
        logger.info("setup_wizard_completed_successfully")

        # Offer to index the graph now
        await offer_graph_indexing(state.graph_path)

        return True

    except KeyboardInterrupt:
        logger.info("setup_wizard_cancelled_keyboard_interrupt")
        rprint("\n[yellow]✗ Setup cancelled by user (Ctrl+C)[/yellow]")
        rprint("[dim]No configuration was saved.[/dim]")
        return False
    except Exception as e:
        logger.error("setup_wizard_failed", error=str(e), error_type=type(e).__name__)
        rprint(f"\n[red]✗ Setup failed: {str(e)}[/red]")
        return False
