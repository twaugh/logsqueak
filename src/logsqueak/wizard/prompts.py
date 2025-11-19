"""Interactive prompt helpers using Rich library."""

from pathlib import Path

from rich import print as rprint
from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.table import Table

from logsqueak.wizard.providers import OllamaModel, format_model_size

console = Console()


def prompt_graph_path(default: str | None = None) -> str:
    """Prompt user for Logseq graph directory path.

    Args:
        default: Default value to show (from existing config)

    Returns:
        Expanded absolute path string
    """
    if default:
        path_str = Prompt.ask(
            "[bold cyan]Logseq graph directory[/bold cyan]",
            default=default
        )
    else:
        path_str = Prompt.ask(
            "[bold cyan]Logseq graph directory[/bold cyan]",
            default="~/Documents/logseq"
        )

    # Expand and resolve path
    expanded = Path(path_str).expanduser().resolve()
    return str(expanded)


def prompt_provider_choice(default: str | None = None) -> str:
    """Prompt user to select LLM provider type.

    Args:
        default: Default provider type ("ollama", "openai", "custom")

    Returns:
        Selected provider type: "ollama" | "openai" | "custom"
    """
    rprint("[bold]Select LLM Provider:[/bold]")
    rprint("  [cyan]1[/cyan]. Ollama (local or remote)")
    rprint("  [cyan]2[/cyan]. OpenAI")
    rprint("  [cyan]3[/cyan]. Other OpenAI-compatible endpoint")

    if default:
        default_num = {"ollama": "1", "openai": "2", "custom": "3"}.get(default, "1")
        choice = Prompt.ask(
            "\n[bold cyan]Choice[/bold cyan]",
            choices=["1", "2", "3"],
            default=default_num
        )
    else:
        choice = Prompt.ask(
            "\n[bold cyan]Choice[/bold cyan]",
            choices=["1", "2", "3"],
            default="1"
        )

    return {"1": "ollama", "2": "openai", "3": "custom"}[choice]


def prompt_ollama_endpoint(default: str = "http://localhost:11434") -> str:
    """Prompt user for Ollama endpoint URL.

    Args:
        default: Default endpoint URL

    Returns:
        Ollama endpoint URL
    """
    endpoint = Prompt.ask(
        "[bold cyan]Ollama endpoint URL[/bold cyan]",
        default=default
    )

    # Ensure /v1 suffix
    endpoint = endpoint.rstrip("/")
    if not endpoint.endswith("/v1"):
        endpoint += "/v1"

    return endpoint


def prompt_ollama_model(models: list[OllamaModel], default: str | None = None) -> str:
    """Display Ollama models in table and prompt user to select one.

    Args:
        models: List of available models from /api/tags
        default: Default model name to pre-select

    Returns:
        Selected model name
    """
    from logsqueak.wizard.providers import get_recommended_ollama_model

    table = Table(title="Available Ollama Models")
    table.add_column("#", style="cyan", width=3)
    table.add_column("Model", style="green")
    table.add_column("Size", justify="right")
    table.add_column("", width=4)

    recommended_model = get_recommended_ollama_model(models)
    model_names = [m.name for m in models]

    for i, model in enumerate(models, 1):
        is_recommended = model.name == recommended_model
        is_current = default and model.name == default

        # Build marker: show both if applicable
        markers = []
        if is_recommended:
            markers.append("⭐")
        if is_current:
            markers.append("✓")
        marker = " ".join(markers)

        table.add_row(
            str(i),
            model.name,
            format_model_size(model.size),
            marker
        )

    console.print(table)

    # Show legend for markers
    legend_parts = []
    if recommended_model:
        legend_parts.append("⭐ = Recommended")
    if default:
        legend_parts.append("✓ = Current")
    if legend_parts:
        rprint(f"\n[dim]{', '.join(legend_parts)}[/dim]")

    # Determine default selection
    if default and default in model_names:
        default_idx = model_names.index(default) + 1
        default_str = str(default_idx)
    elif recommended_model:
        default_idx = model_names.index(recommended_model) + 1
        default_str = str(default_idx)
    else:
        default_str = "1"

    choice = Prompt.ask(
        "\n[bold cyan]Select model[/bold cyan]",
        choices=[str(i) for i in range(1, len(models) + 1)],
        default=default_str
    )

    return model_names[int(choice) - 1]


def prompt_openai_api_key(existing_key: str | None = None) -> str:
    """Prompt user for OpenAI API key.

    Args:
        existing_key: Existing API key to display masked

    Returns:
        API key string (visible during input for verification)
    """
    from logsqueak.wizard.providers import mask_api_key

    if existing_key:
        masked = mask_api_key(existing_key)
        rprint(f"\n[dim]Current API key: {masked}[/dim]")
        rprint("[dim]Press Enter to keep current key, or enter new key:[/dim]")

        key = Prompt.ask(
            "[bold cyan]OpenAI API key[/bold cyan]",
            default=existing_key,
            show_default=False
        )
    else:
        key = Prompt.ask("[bold cyan]OpenAI API key[/bold cyan]")

    return key


def prompt_openai_model(default: str = "gpt-4o") -> str:
    """Prompt user to select OpenAI model.

    Args:
        default: Default model name

    Returns:
        Selected model name or custom model string
    """
    rprint("\n[bold]Select OpenAI Model:[/bold]")
    rprint("  [cyan]1[/cyan]. gpt-4o (recommended)")
    rprint("  [cyan]2[/cyan]. gpt-4-turbo")
    rprint("  [cyan]3[/cyan]. gpt-3.5-turbo")
    rprint("  [cyan]4[/cyan]. Custom model name")

    choice = Prompt.ask(
        "\n[bold cyan]Choice[/bold cyan]",
        choices=["1", "2", "3", "4"],
        default="1"
    )

    if choice == "4":
        return Prompt.ask("[bold cyan]Model name[/bold cyan]")

    return {"1": "gpt-4o", "2": "gpt-4-turbo", "3": "gpt-3.5-turbo"}[choice]


def prompt_custom_endpoint(existing: str | None = None) -> str:
    """Prompt user for custom OpenAI-compatible endpoint URL.

    Args:
        existing: Existing endpoint URL from config, if any

    Returns:
        Endpoint URL
    """
    default = existing if existing else "http://localhost:8000/v1"
    endpoint = Prompt.ask(
        "[bold cyan]Custom endpoint URL[/bold cyan]",
        default=default
    )

    # Ensure /v1 suffix
    endpoint = endpoint.rstrip("/")
    if not endpoint.endswith("/v1"):
        endpoint += "/v1"

    return endpoint


def prompt_custom_api_key(existing: str | None = None) -> str:
    """Prompt user for custom provider API key.

    Args:
        existing: Existing API key from config, if any

    Returns:
        API key string (visible during input for verification)
    """
    from logsqueak.wizard.providers import mask_api_key

    if existing:
        masked = mask_api_key(existing)
        rprint(f"\n[dim]Current API key: {masked}[/dim]")
        rprint("[dim]Press Enter to keep current key, or enter new key:[/dim]")

        key = Prompt.ask(
            "[bold cyan]API key[/bold cyan]",
            default=existing,
            show_default=False
        )
    else:
        key = Prompt.ask(
            "[bold cyan]API key[/bold cyan]",
            default="none"
        )

    return key


def prompt_custom_model(existing: str | None = None) -> str:
    """Prompt user for custom provider model name.

    Args:
        existing: Existing model name from config, if any

    Returns:
        Model name string
    """
    if existing:
        model = Prompt.ask("[bold cyan]Model name[/bold cyan]", default=existing)
    else:
        model = Prompt.ask("[bold cyan]Model name[/bold cyan]")
    return model


def prompt_confirm_overwrite() -> bool:
    """Prompt user for confirmation to overwrite existing config.

    Returns:
        True if user confirms overwrite, False if user declines
    """
    return Confirm.ask(
        "\n[bold yellow]Configuration file already exists. Overwrite?[/bold yellow]",
        default=False
    )


def prompt_retry_on_failure(operation: str) -> str:
    """Prompt user what to do after operation failure.

    Args:
        operation: Name of failed operation (for message)

    Returns:
        User choice: "retry" | "skip" | "abort"
    """
    rprint(f"\n[bold yellow]{operation} failed[/bold yellow]")
    rprint("  [cyan]1[/cyan]. Retry")
    rprint("  [cyan]2[/cyan]. Skip (continue setup)")
    rprint("  [cyan]3[/cyan]. Abort setup")

    choice = Prompt.ask(
        "\n[bold cyan]Choice[/bold cyan]",
        choices=["1", "2", "3"],
        default="1"
    )

    return {"1": "retry", "2": "skip", "3": "abort"}[choice]


def prompt_continue_on_timeout(operation: str, timeout: int) -> str:
    """Prompt user what to do after operation timeout.

    Args:
        operation: Name of timed-out operation
        timeout: Timeout value in seconds

    Returns:
        User choice: "continue" | "retry" | "skip"
    """
    rprint(f"\n[bold yellow]{operation} timed out after {timeout} seconds[/bold yellow]")
    rprint("  [cyan]1[/cyan]. Continue waiting")
    rprint("  [cyan]2[/cyan]. Retry")
    rprint("  [cyan]3[/cyan]. Skip")

    choice = Prompt.ask(
        "\n[bold cyan]Choice[/bold cyan]",
        choices=["1", "2", "3"],
        default="2"
    )

    return {"1": "continue", "2": "retry", "3": "skip"}[choice]


def prompt_advanced_settings() -> bool:
    """Ask if user wants to configure advanced settings.

    Returns:
        True if user wants advanced settings, False otherwise
    """
    return Confirm.ask(
        "[bold cyan]Configure advanced settings?[/bold cyan]",
        default=False
    )


def prompt_num_ctx(default: int = 32768) -> int:
    """Prompt user for Ollama context window size.

    Args:
        default: Default context window size

    Returns:
        Context window size
    """
    rprint("\n[dim]Context window size controls VRAM usage (higher = more memory)[/dim]")
    rprint("[dim]Common values: 4096, 8192, 16384, 32768[/dim]")

    num_ctx_str = Prompt.ask(
        "[bold cyan]Context window size (num_ctx)[/bold cyan]",
        default=str(default)
    )

    try:
        return int(num_ctx_str)
    except ValueError:
        rprint("[yellow]Invalid number, using default[/yellow]")
        return default


def prompt_top_k(default: int = 10) -> int:
    """Prompt user for RAG search top_k value.

    Args:
        default: Default top_k value

    Returns:
        top_k value
    """
    rprint("\n[dim]Number of similar blocks to retrieve for RAG search[/dim]")
    rprint("[dim]Higher values provide more context but may slow down processing[/dim]")

    top_k_str = Prompt.ask(
        "[bold cyan]RAG top_k[/bold cyan]",
        default=str(default)
    )

    try:
        value = int(top_k_str)
        if value < 1 or value > 100:
            rprint("[yellow]Value out of range (1-100), using default[/yellow]")
            return default
        return value
    except ValueError:
        rprint("[yellow]Invalid number, using default[/yellow]")
        return default


def prompt_index_graph() -> bool:
    """Prompt user whether to index the graph now.

    Returns:
        True if user wants to index, False otherwise
    """
    rprint("\n[bold]Would you like to index your knowledge base now?[/bold]")
    rprint("[dim]Indexing enables semantic search with the 'logsqueak search' command.[/dim]")
    rprint("[dim]This may take a few minutes depending on the size of your graph.[/dim]")

    return Confirm.ask(
        "[cyan]Index knowledge base?[/cyan]",
        default=True
    )
