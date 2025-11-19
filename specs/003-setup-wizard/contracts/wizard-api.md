# Wizard API Contract

**Module**: `src/logsqueak/wizard/`
**Purpose**: Internal API contracts for wizard components

## Public API

### Main Entry Point

```python
async def run_setup_wizard() -> bool:
    """
    Run the interactive setup wizard.

    Returns:
        True if wizard completed successfully, False if aborted by user

    Raises:
        Exception: For system-level errors (permissions, disk I/O)
    """
```

**CLI Integration**:
```python
@cli.command()
def init():
    """Initialize Logsqueak configuration interactively."""
    import asyncio
    from logsqueak.wizard.wizard import run_setup_wizard

    result = asyncio.run(run_setup_wizard())
    if result:
        click.echo("✓ Setup complete! Run 'logsqueak extract' to get started.")
    else:
        click.echo("Setup cancelled.")
        raise click.Abort()
```

---

## Internal Contracts

### Prompts Module (`prompts.py`)

```python
def prompt_graph_path(default: str | None = None) -> str:
    """
    Prompt user for Logseq graph directory path.

    Args:
        default: Default value to show (from existing config)

    Returns:
        Expanded absolute path string
    """

def prompt_provider_choice(default: str | None = None) -> str:
    """
    Prompt user to select LLM provider type.

    Args:
        default: Default provider type ("ollama", "openai", "custom")

    Returns:
        Selected provider type: "ollama" | "openai" | "custom"
    """

def prompt_ollama_endpoint(default: str = "http://localhost:11434") -> str:
    """
    Prompt user for Ollama endpoint URL.

    Args:
        default: Default endpoint URL

    Returns:
        Ollama endpoint URL
    """

def prompt_ollama_model(models: list[OllamaModel], default: str | None = None) -> str:
    """
    Display Ollama models in table and prompt user to select one.

    Args:
        models: List of available models from /api/tags
        default: Default model name to pre-select

    Returns:
        Selected model name
    """

def prompt_openai_api_key(existing_key: str | None = None) -> str:
    """
    Prompt user for OpenAI API key.

    Args:
        existing_key: Existing API key to display masked (first 8 + "..." + last 4)

    Returns:
        API key string (visible during input for verification)
    """

def prompt_openai_model(default: str = "gpt-4o") -> str:
    """
    Prompt user to select OpenAI model.

    Args:
        default: Default model name

    Returns:
        Selected model name or custom model string
    """

def prompt_custom_endpoint() -> str:
    """
    Prompt user for custom OpenAI-compatible endpoint URL.

    Returns:
        Endpoint URL
    """

def prompt_custom_api_key() -> str:
    """
    Prompt user for custom provider API key.

    Returns:
        API key string (visible during input)
    """

def prompt_custom_model() -> str:
    """
    Prompt user for custom provider model name.

    Returns:
        Model name string
    """

def prompt_confirm_overwrite() -> bool:
    """
    Prompt user for confirmation to overwrite existing config.

    Called after all settings are validated and config is assembled,
    just before writing to disk.

    Returns:
        True if user confirms overwrite, False if user declines
    """

def prompt_retry_on_failure(operation: str) -> str:
    """
    Prompt user what to do after operation failure.

    Args:
        operation: Name of failed operation (for message)

    Returns:
        User choice: "retry" | "skip" | "abort"
    """

def prompt_continue_on_timeout(operation: str, timeout: int) -> str:
    """
    Prompt user what to do after operation timeout.

    Args:
        operation: Name of timed-out operation
        timeout: Timeout value in seconds

    Returns:
        User choice: "continue" | "retry" | "skip"
    """
```

---

### Validators Module (`validators.py`)

```python
@dataclass
class ValidationResult:
    """Result of a validation operation."""
    success: bool
    error_message: str | None = None
    data: Any | None = None


def validate_graph_path(path: str) -> ValidationResult:
    """
    Validate Logseq graph directory structure.

    Args:
        path: Path to validate

    Returns:
        ValidationResult with:
        - success=True if path exists and has journals/ and logseq/
        - success=False with error_message if validation fails
    """

def check_disk_space(required_mb: int = 1024) -> ValidationResult:
    """
    Check available disk space in cache directory.

    Args:
        required_mb: Minimum required space in megabytes

    Returns:
        ValidationResult with:
        - success=True if sufficient space
        - success=False with warning message if insufficient
        - data={"available_mb": int}
    """

async def test_ollama_connection(endpoint: str, timeout: int = 30) -> ValidationResult:
    """
    Test Ollama API connectivity and retrieve models.

    Args:
        endpoint: Ollama endpoint URL
        timeout: Request timeout in seconds

    Returns:
        ValidationResult with:
        - success=True, data={"models": [OllamaModel, ...]} if connection succeeds
        - success=False, error_message=str if connection fails

    Raises:
        asyncio.TimeoutError: If request exceeds timeout
    """

async def test_openai_connection(
    endpoint: str,
    api_key: str,
    model: str,
    timeout: int = 30
) -> ValidationResult:
    """
    Test OpenAI API connectivity and model access.

    Args:
        endpoint: OpenAI endpoint URL
        api_key: API key for authentication
        model: Model name to test
        timeout: Request timeout in seconds

    Returns:
        ValidationResult with:
        - success=True if API key and model are valid
        - success=False, error_message=str if validation fails

    Raises:
        asyncio.TimeoutError: If request exceeds timeout
    """

def check_embedding_model_cached() -> bool:
    """
    Check if sentence-transformers embedding model is cached locally.

    Returns:
        True if model cache exists and is loadable, False otherwise
    """

async def validate_embedding_model(
    progress_callback: Callable[[int, int], None] | None = None
) -> ValidationResult:
    """
    Validate embedding model loads successfully (downloads if needed).

    Args:
        progress_callback: Optional callback(current, total) for download progress

    Returns:
        ValidationResult with:
        - success=True if model loads successfully
        - success=False, error_message=str if download/load fails

    Note:
        - If model is cached: Quick validation (~2-5 seconds)
        - If not cached: Downloads ~420MB with progress updates
    """
```

---

### Providers Module (`providers.py`)

```python
@dataclass
class OllamaModel:
    """Model available in Ollama instance."""
    name: str
    size: int
    modified_at: str


async def fetch_ollama_models(endpoint: str, timeout: int = 30) -> list[OllamaModel]:
    """
    Fetch list of installed models from Ollama instance.

    Args:
        endpoint: Ollama endpoint URL
        timeout: Request timeout in seconds

    Returns:
        List of OllamaModel instances

    Raises:
        httpx.HTTPError: If request fails
        asyncio.TimeoutError: If request exceeds timeout
    """

def get_recommended_ollama_model(models: list[OllamaModel]) -> str | None:
    """
    Find recommended Ollama model (Mistral 7B Instruct) in model list.

    Args:
        models: List of available models

    Returns:
        Model name if Mistral 7B Instruct found, None otherwise
    """

def format_model_size(size_bytes: int) -> str:
    """
    Format model size in human-readable units.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted string (e.g., "4.1 GB", "512 MB")
    """

def mask_api_key(api_key: str) -> str:
    """
    Mask API key for display (show first 8 + "..." + last 4).

    Args:
        api_key: Full API key

    Returns:
        Masked string (e.g., "sk-proj-abc...xyz9")
    """

def get_provider_key(provider_type: str, endpoint: str) -> str:
    """
    Generate unique key for provider in llm_providers dict.

    Args:
        provider_type: "ollama", "openai", or "custom"
        endpoint: Provider endpoint URL

    Returns:
        Key string (e.g., "ollama_local", "openai", "custom_azure")
    """
```

---

### Wizard Module (`wizard.py`)

```python
@dataclass
class WizardState:
    """Tracks wizard state across all prompts."""
    existing_config: Config | None = None
    graph_path: str | None = None
    provider_type: str | None = None
    ollama_endpoint: str | None = None
    ollama_model: str | None = None
    ollama_num_ctx: int = 32768
    openai_api_key: str | None = None
    openai_model: str | None = None
    custom_endpoint: str | None = None
    custom_api_key: str | None = None
    custom_model: str | None = None
    top_k: int = 20


async def run_setup_wizard() -> bool:
    """Main wizard orchestration (public entry point)."""


async def load_existing_config() -> Config | None:
    """Load existing config if present, return None if not found or invalid."""


async def configure_graph_path(state: WizardState) -> bool:
    """Prompt and validate graph path. Returns False if user aborts."""


async def configure_provider(state: WizardState) -> bool:
    """Prompt for provider and configure provider-specific settings.

    Returns False if user aborts.
    """


async def configure_ollama(state: WizardState) -> bool:
    """Ollama-specific configuration flow."""


async def configure_openai(state: WizardState) -> bool:
    """OpenAI-specific configuration flow."""


async def configure_custom(state: WizardState) -> bool:
    """Custom provider configuration flow."""


async def validate_llm_connection(state: WizardState) -> bool:
    """Test LLM connection with timeout handling. Returns False if user skips."""


async def validate_embedding(state: WizardState) -> bool:
    """Validate embedding model with disk check and progress. Returns False if user skips."""


def assemble_config(state: WizardState) -> Config:
    """Assemble final Config from WizardState, preserving existing providers."""


async def prompt_overwrite_if_exists(config_path: Path) -> bool:
    """Prompt user to confirm overwrite if config file exists.

    Returns True if should proceed (file doesn't exist or user confirms),
    False if user declines to overwrite.
    """


async def write_config(config: Config, config_path: Path) -> None:
    """Write config to YAML file with mode 600 permissions."""


def show_success_message() -> None:
    """Display success message with next steps."""
```

---

## Configuration File Contract

### YAML Structure

```yaml
# Required fields (runtime)
llm:
  endpoint: string  # HTTP(S) URL
  api_key: string   # Authentication key
  model: string     # Model identifier
  num_ctx: integer  # Optional, Ollama only (default: 32768)

logseq:
  graph_path: string  # Absolute path to Logseq graph

rag:
  top_k: integer  # Default: 20

# Optional field (wizard only)
llm_providers:
  <provider_key>:
    endpoint: string
    api_key: string
    model: string
    num_ctx: integer  # Optional
```

### File Permissions

- **Path**: `~/.config/logsqueak/config.yaml`
- **Permissions**: `600` (owner read/write only)
- **Ownership**: Current user

### Validation Contract

When loading config:
- File must exist and be readable
- File must have mode 600 (enforced by Config.load())
- All required fields must be present and valid
- Optional fields use defaults if missing
- `llm_providers` field is optional (backwards compatible)

When writing config:
- Directory `~/.config/logsqueak/` created if missing
- File written with mode 600 atomically
- Invalid provider settings rejected during assembly
- All providers preserved in `llm_providers` section

---

## External API Dependencies

### Ollama API

**Endpoint**: `GET {endpoint}/api/tags`

**Response**:
```json
{
  "models": [
    {
      "name": "mistral:7b-instruct",
      "size": 4109733376,
      "modified_at": "2024-11-01T12:00:00Z"
    }
  ]
}
```

**Error Handling**:
- Connection refused → "Ollama not running or unreachable"
- 404 Not Found → "Ollama endpoint invalid"
- Timeout → Prompt user (continue/retry/skip)

---

### OpenAI API

**Endpoint**: `POST {endpoint}/chat/completions`

**Request**:
```json
{
  "model": "gpt-4o",
  "messages": [{"role": "user", "content": "test"}],
  "max_tokens": 5
}
```

**Headers**:
```
Authorization: Bearer {api_key}
```

**Success Response**: `200 OK` with completion

**Error Handling**:
- 401 Unauthorized → "Invalid API key"
- 404 Not Found → "Model not found or no access"
- 429 Rate Limited → "API rate limit exceeded"
- Timeout → Prompt user (continue/retry/skip)

---

### Sentence-Transformers (HuggingFace)

**Model**: `all-mpnet-base-v2`

**Cache Location**: `~/.cache/torch/sentence_transformers/`

**Download Size**: ~420 MB

**Load Method**:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-mpnet-base-v2")
```

**Error Handling**:
- Network error during download → Show error, prompt retry/skip
- Disk full during download → Show disk space error
- Corrupt download → Detect via load failure, prompt retry
- Timeout (5 min) → Prompt user (continue/retry/skip)

---

## Error Messages Contract

All error messages must follow this pattern:

```
[ERROR TYPE] <brief description>

<detailed explanation>

<actionable guidance>
```

**Examples**:

```
[Path Error] Logseq graph directory not found

The path '/home/user/Documents/logseq' does not exist.

→ Create the directory: mkdir -p /home/user/Documents/logseq
→ Or provide a different path
```

```
[Connection Error] Ollama API unreachable

Could not connect to http://localhost:11434

→ Start Ollama: ollama serve
→ Or provide remote Ollama URL
```

```
[Authentication Error] OpenAI API key invalid

API returned 401 Unauthorized for key: sk-proj-abc...xyz9

→ Check your API key at https://platform.openai.com/api-keys
→ Generate a new key if needed
```

```
[Disk Space Warning] Low disk space for model download

Available: 800 MB | Required: 1024 MB

Embedding model download may fail if disk fills during download.
Continue anyway?
```

---

## Success Messages Contract

```
✓ Configuration saved successfully!

Next steps:
  1. Extract knowledge from your journal:
     $ logsqueak extract

  2. Search your knowledge base:
     $ logsqueak search "your query"

  3. Update configuration anytime:
     $ logsqueak init

Configuration file: ~/.config/logsqueak/config.yaml
```

---

## Testing Contract

### Unit Test Requirements

All functions must have unit tests covering:
- ✅ Happy path (valid inputs → expected outputs)
- ✅ Invalid inputs (validation errors)
- ✅ Edge cases (empty strings, None values, boundary conditions)
- ✅ Network errors (connection refused, timeouts for async functions)

### Integration Test Requirements

Full wizard flow must be tested:
- ✅ First-time setup (no existing config)
- ✅ Update existing config (preserves providers)
- ✅ Switch providers (Ollama → OpenAI, vice versa)
- ✅ Abort at various stages (no file written)
- ✅ Network failures with retry logic
- ✅ Timeout handling (LLM and embedding)
- ✅ Disk space warning flow

### Mock Requirements

Tests must mock:
- `rich.prompt.Prompt.ask()` - User input
- `rich.prompt.Confirm.ask()` - Yes/no confirmations
- `httpx.AsyncClient` - Network requests
- `SentenceTransformer.__init__()` - Model downloads
- `shutil.disk_usage()` - Disk space checks
- `Path.chmod()` - File permissions

---

## CLI Help Text Contract

```
$ logsqueak init --help
Usage: logsqueak init [OPTIONS]

  Initialize Logsqueak configuration interactively.

  Guides you through:
  - Logseq graph location
  - LLM provider selection (Ollama, OpenAI, Custom)
  - Connection validation
  - Embedding model setup

  Run this command to create or update your configuration file.

Options:
  --help  Show this message and exit.
```
