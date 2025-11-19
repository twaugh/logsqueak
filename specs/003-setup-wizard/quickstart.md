# Quickstart: Implementing the Setup Wizard

**Target Audience**: Developers implementing the setup wizard feature
**Estimated Time**: 4-6 hours for full implementation
**Prerequisites**: Python 3.11+, familiarity with Rich library, async/await

## Implementation Checklist

### Phase 0: Foundation (30 minutes)

- [ ] Create `src/logsqueak/wizard/` package
- [ ] Add `__init__.py` with public API export
- [ ] Create empty modules:
  - `wizard.py` - Main orchestration
  - `prompts.py` - Rich-based prompts
  - `validators.py` - Validation functions
  - `providers.py` - Provider-specific helpers

### Phase 1: Validators Module (45 minutes)

- [ ] Implement `ValidationResult` dataclass
- [ ] Implement `validate_graph_path()`
  - Check path exists
  - Check `journals/` subdirectory exists
  - Check `logseq/` subdirectory exists
- [ ] Implement `check_disk_space()`
  - Use `shutil.disk_usage()`
  - Calculate available MB
  - Compare to threshold (1024 MB)
- [ ] Implement `validate_ollama_connection()`
  - HTTP GET to `/api/tags`
  - Parse JSON response
  - Convert to `OllamaModel` instances
- [ ] Implement `validate_openai_connection()`
  - HTTP POST to `/chat/completions`
  - Minimal test request
  - Handle 401, 404, 429 errors
- [ ] Implement `check_embedding_model_cached()`
  - Check `~/.cache/torch/sentence_transformers/` path
- [ ] Implement `validate_embedding_model()`
  - Attempt `SentenceTransformer("all-MiniLM-L6-v2")`
  - Handle download with progress callback
  - Test encode() to ensure model works

**Testing**: Write unit tests for all validators (mock httpx, SentenceTransformer)

### Phase 2: Providers Module (30 minutes)

- [ ] Implement `OllamaModel` dataclass
- [ ] Implement `fetch_ollama_models()`
  - Call `validate_ollama_connection()`
  - Extract models from ValidationResult
- [ ] Implement `get_recommended_ollama_model()`
  - Search for "mistral" and "7b" and "instruct" in model name
  - Return first match or None
- [ ] Implement `format_model_size()`
  - Convert bytes to GB/MB/KB
  - Use proper units (1024 base)
- [ ] Implement `mask_api_key()`
  - Take first 8 chars + "..." + last 4 chars
  - Handle keys shorter than 12 chars
- [ ] Implement `get_provider_key()`
  - Parse endpoint to determine instance type (local vs remote)
  - Generate unique key: `{provider}_{instance}`

**Testing**: Write unit tests for all helper functions

### Phase 3: Prompts Module (1 hour)

- [ ] Implement `prompt_graph_path()`
  - Use `Prompt.ask()` with default
  - Expand `~` to home directory
  - Show current value if exists
- [ ] Implement `prompt_provider_choice()`
  - Use `Prompt.ask()` with choices
  - Display table with provider descriptions
  - Default to "ollama"
- [ ] Implement `prompt_ollama_endpoint()`
  - Default: "http://localhost:11434"
  - Validate URL format
- [ ] Implement `prompt_ollama_model()`
  - Display models in Rich table
  - Show name and size
  - Highlight recommended model
  - Use `Prompt.ask()` with choices
- [ ] Implement `prompt_openai_api_key()`
  - Show masked version if existing
  - Use visible input for new keys
  - Validate format (starts with "sk-")
- [ ] Implement `prompt_openai_model()`
  - Offer common models + custom
  - Use `Prompt.ask()` with choices
- [ ] Implement `prompt_custom_endpoint()`
  - Validate URL format
- [ ] Implement `prompt_custom_api_key()`
  - Visible input
- [ ] Implement `prompt_custom_model()`
  - Freeform text input
- [ ] Implement `prompt_confirm_overwrite()`
  - Use `Confirm.ask()`
  - Called just before writing config file
- [ ] Implement `prompt_retry_on_failure()`
  - Choices: retry, skip, abort
- [ ] Implement `prompt_continue_on_timeout()`
  - Choices: continue, retry, skip
  - Show timeout value in message

**Testing**: Write unit tests with mocked `Prompt.ask()` and `Confirm.ask()`

### Phase 4: Wizard Orchestration (1.5 hours)

- [ ] Implement `WizardState` dataclass
- [ ] Implement `load_existing_config()`
  - Try `Config.load()`
  - Catch all exceptions, return None
  - Log errors but don't fail
- [ ] Implement `configure_graph_path()`
  - Prompt for path
  - Validate with `validate_graph_path()`
  - Retry on failure
  - Update state
- [ ] Implement `configure_ollama()`
  - Try existing endpoint first
  - Try localhost if no existing
  - Prompt for custom if both fail
  - Fetch models
  - Prompt to select model
  - Optionally prompt for num_ctx
- [ ] Implement `configure_openai()`
  - Prompt for API key (mask existing)
  - Prompt for model
  - Update state
- [ ] Implement `configure_custom()`
  - Prompt for endpoint
  - Prompt for API key
  - Prompt for model
  - Update state
- [ ] Implement `configure_provider()`
  - Prompt for provider type
  - Branch to provider-specific function
  - Return False if user aborts
- [ ] Implement `validate_llm_connection()`
  - Call appropriate test function based on provider
  - Handle timeout with user prompt
  - Retry on failure
  - Allow skip
- [ ] Implement `validate_embedding()`
  - Check disk space, warn if low
  - Check if cached
  - If cached: quick validation
  - If not: download with progress
  - Handle timeout
  - Allow skip with warning
- [ ] Implement `assemble_config()`
  - Create LLMConfig from active provider
  - Create LogseqConfig from graph path
  - Create RAGConfig with top_k
  - Merge with existing llm_providers
  - Add current provider to llm_providers
  - Return Config instance
- [ ] Implement `write_config()`
  - Create config directory if needed
  - Serialize to YAML
  - Write atomically (temp file + rename)
  - Set permissions to 600
- [ ] Implement `show_success_message()`
  - Rich panel with success message
  - List next steps
  - Show config file path
- [ ] Implement `run_setup_wizard()`
  - Load existing config (if any, for defaults)
  - Configure graph path → abort on False
  - Configure provider → abort on False
  - Validate LLM → abort on False (unless skip)
  - Validate embedding → abort on False (unless skip)
  - Assemble config
  - Prompt to overwrite if config exists → abort if declined
  - Write config
  - Show success message
  - Return True

**Testing**: Write integration tests for full wizard flow

### Phase 5: CLI Integration (15 minutes)

- [ ] Add `init` command to `src/logsqueak/cli.py`
  - Import `run_setup_wizard`
  - Call with `asyncio.run()`
  - Handle return value (success/abort)
  - Show appropriate message
- [ ] Update CLI help text
- [ ] Test manually: `logsqueak init`

### Phase 6: Config Model Extension (30 minutes)

- [ ] Add `llm_providers` field to `Config` model
  - Type: `dict[str, dict] | None`
  - Default: `None`
  - Add to YAML serialization
- [ ] Test backwards compatibility
  - Load old config without `llm_providers`
  - Verify no errors
- [ ] Test new config structure
  - Write config with `llm_providers`
  - Load and verify all fields present

**Testing**: Update config tests to cover new field

### Phase 7: Edge Cases & Polish (1 hour)

- [ ] Handle all edge cases from spec:
  - Missing `journals/` or `logseq/` subdirectory
  - Network timeouts (LLM and embedding)
  - Invalid API key format
  - Ollama with no models installed
  - Empty or whitespace-only config file
  - Config directory creation fails (permissions)
  - Abort mid-wizard (Ctrl+C)
  - Invalid YAML syntax in existing config
  - Graph path with spaces/special characters
  - Disk space runs out during download
  - Provider switching preserves all credentials
- [ ] Add logging for all wizard actions
  - Use existing `structlog` infrastructure
  - Log: config loaded, prompts shown, validations, errors
- [ ] Add progress indicators
  - Rich Status for network operations
  - Rich Progress for embedding download
  - Spinners for validation steps
- [ ] Test all error messages
  - Verify format: [ERROR TYPE] + description + guidance
  - Ensure actionable guidance provided

### Phase 8: Testing (1 hour)

- [ ] Write unit tests (target: 20-30 tests)
  - All validators
  - All prompts
  - All provider helpers
  - Config assembly logic
- [ ] Write integration tests (target: 8-10 tests)
  - First-time setup
  - Update existing config
  - Switch providers
  - Abort scenarios
  - Network failures
  - Timeout handling
  - Disk space warnings
- [ ] Manual testing scenarios:
  - Run with no existing config
  - Run with existing valid config
  - Run with broken config (wrong permissions, invalid YAML)
  - Test Ollama flow (local and remote)
  - Test OpenAI flow
  - Test Custom flow
  - Test abort at each step
  - Test network timeout handling
  - Test embedding download with slow connection
- [ ] Verify all success criteria from spec
  - New users complete setup in <3 minutes
  - LLM validation catches invalid configs
  - Config files have mode 600
  - Provider switching preserves credentials
  - Cached embedding models skip download

---

## Key Code Patterns

### Using Rich for Prompts

```python
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.panel import Panel

console = Console()

# Simple prompt with default
path = Prompt.ask("Logseq graph path", default="/home/user/logseq")

# Confirmation prompt
overwrite = Confirm.ask("Overwrite existing config?", default=False)

# Choice prompt
provider = Prompt.ask(
    "Select LLM provider",
    choices=["ollama", "openai", "custom"],
    default="ollama"
)

# Display table
table = Table(title="Available Models")
table.add_column("Model Name")
table.add_column("Size")
table.add_row("mistral:7b-instruct", "4.1 GB")
console.print(table)

# Display panel
console.print(Panel(
    "✓ Configuration saved successfully!",
    title="Success",
    border_style="green"
))
```

### Async HTTP with httpx

```python
import httpx

async def test_connection(endpoint: str, timeout: int = 30) -> ValidationResult:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{endpoint}/api/tags",
                timeout=timeout
            )
            response.raise_for_status()
            data = response.json()
            return ValidationResult(success=True, data=data)
    except httpx.HTTPStatusError as e:
        return ValidationResult(
            success=False,
            error_message=f"HTTP {e.response.status_code}: {e.response.text}"
        )
    except httpx.ConnectError:
        return ValidationResult(
            success=False,
            error_message=f"Could not connect to {endpoint}"
        )
    except asyncio.TimeoutError:
        raise  # Re-raise for timeout handling
```

### Timeout Handling with Retry

```python
async def operation_with_timeout(operation, name: str, timeout: int):
    """Execute operation with timeout and user-controlled retry."""
    while True:
        try:
            return await asyncio.wait_for(operation(), timeout=timeout)
        except asyncio.TimeoutError:
            console.print(f"[yellow]⚠ {name} timed out ({timeout}s)[/yellow]")
            choice = Prompt.ask(
                "What would you like to do?",
                choices=["continue", "retry", "skip"],
                default="retry"
            )
            if choice == "continue":
                # Remove timeout, wait indefinitely
                return await operation()
            elif choice == "retry":
                # Loop continues, retry with same timeout
                continue
            else:  # skip
                return None
```

### Config Assembly with Provider Preservation

```python
def assemble_config(state: WizardState) -> Config:
    """Assemble Config from wizard state, preserving existing providers."""
    # Create active LLM config based on selected provider
    if state.provider_type == "ollama":
        llm_config = LLMConfig(
            endpoint=state.ollama_endpoint,
            api_key="ollama",
            model=state.ollama_model,
            num_ctx=state.ollama_num_ctx
        )
    elif state.provider_type == "openai":
        llm_config = LLMConfig(
            endpoint="https://api.openai.com/v1",
            api_key=state.openai_api_key,
            model=state.openai_model
        )
    else:  # custom
        llm_config = LLMConfig(
            endpoint=state.custom_endpoint,
            api_key=state.custom_api_key,
            model=state.custom_model
        )

    # Preserve existing providers and add current one
    llm_providers = {}
    if state.existing_config and state.existing_config.llm_providers:
        llm_providers = state.existing_config.llm_providers.copy()

    # Add current provider to preserved providers
    provider_key = get_provider_key(state.provider_type, llm_config.endpoint)
    llm_providers[provider_key] = {
        "endpoint": str(llm_config.endpoint),
        "api_key": llm_config.api_key,
        "model": llm_config.model,
    }
    if state.provider_type == "ollama":
        llm_providers[provider_key]["num_ctx"] = state.ollama_num_ctx

    return Config(
        llm=llm_config,
        logseq=LogseqConfig(graph_path=state.graph_path),
        rag=RAGConfig(top_k=state.top_k),
        llm_providers=llm_providers
    )
```

### Atomic Config Write with Permissions

```python
import tempfile
import os

async def write_config(config: Config, config_path: Path) -> None:
    """Write config to YAML atomically with mode 600."""
    # Ensure directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Serialize to YAML
    config_dict = config.model_dump()
    yaml_content = yaml.dump(config_dict, default_flow_style=False, sort_keys=False)

    # Write to temp file in same directory (atomic rename requirement)
    with tempfile.NamedTemporaryFile(
        mode='w',
        dir=config_path.parent,
        delete=False,
        prefix='.config_',
        suffix='.tmp'
    ) as tmp_file:
        tmp_file.write(yaml_content)
        tmp_path = Path(tmp_file.name)

    try:
        # Set permissions before rename (more secure)
        tmp_path.chmod(0o600)

        # Atomic rename
        tmp_path.rename(config_path)
    except Exception:
        # Clean up temp file on error
        tmp_path.unlink(missing_ok=True)
        raise
```

---

## Testing Strategy

### Unit Test Example

```python
import pytest
from unittest.mock import patch, AsyncMock
from logsqueak.wizard.validators import validate_graph_path, ValidationResult

def test_validate_graph_path_success(tmp_path):
    """Test valid graph path with journals/ and logseq/ subdirectories."""
    # Create mock graph structure
    (tmp_path / "journals").mkdir()
    (tmp_path / "logseq").mkdir()

    result = validate_graph_path(str(tmp_path))

    assert result.success is True
    assert result.error_message is None

def test_validate_graph_path_missing_journals(tmp_path):
    """Test graph path missing journals/ subdirectory."""
    (tmp_path / "logseq").mkdir()  # Only logseq/, no journals/

    result = validate_graph_path(str(tmp_path))

    assert result.success is False
    assert "journals/" in result.error_message

@pytest.mark.asyncio
async def validate_ollama_connection_success():
    """Test successful Ollama connection returns models."""
    mock_response = AsyncMock()
    mock_response.json.return_value = {
        "models": [
            {"name": "mistral:7b-instruct", "size": 4109733376, "modified_at": "2024-11-01T12:00:00Z"}
        ]
    }
    mock_response.raise_for_status = AsyncMock()

    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)

        from logsqueak.wizard.validators import validate_ollama_connection
        result = await validate_ollama_connection("http://localhost:11434")

        assert result.success is True
        assert len(result.data["models"]) == 1
        assert result.data["models"][0]["name"] == "mistral:7b-instruct"
```

### Integration Test Example

```python
@pytest.mark.asyncio
async def test_wizard_first_time_setup(tmp_path, monkeypatch):
    """Test complete wizard flow for first-time setup."""
    # Mock all prompts
    mock_prompts = [
        str(tmp_path),  # graph_path
        "ollama",       # provider_type
        "mistral:7b-instruct",  # model
    ]
    prompt_iter = iter(mock_prompts)

    def mock_prompt_ask(*args, **kwargs):
        return next(prompt_iter)

    monkeypatch.setattr("logsqueak.wizard.prompts.Prompt.ask", mock_prompt_ask)
    monkeypatch.setattr("logsqueak.wizard.prompts.Confirm.ask", lambda *args, **kwargs: True)

    # Mock validations
    monkeypatch.setattr(
        "logsqueak.wizard.validators.validate_ollama_connection",
        AsyncMock(return_value=ValidationResult(
            success=True,
            data={"models": [{"name": "mistral:7b-instruct", "size": 4109733376}]}
        ))
    )
    monkeypatch.setattr(
        "logsqueak.wizard.validators.validate_embedding_model",
        AsyncMock(return_value=ValidationResult(success=True))
    )

    # Setup graph structure
    (tmp_path / "journals").mkdir()
    (tmp_path / "logseq").mkdir()

    # Run wizard
    config_path = tmp_path / "config.yaml"
    monkeypatch.setenv("HOME", str(tmp_path.parent))  # Override config location

    from logsqueak.wizard.wizard import run_setup_wizard
    result = await run_setup_wizard()

    assert result is True
    assert config_path.exists()
    assert config_path.stat().st_mode & 0o777 == 0o600  # Check permissions

    # Verify config content
    from logsqueak.models.config import Config
    config = Config.load(config_path)
    assert config.llm.model == "mistral:7b-instruct"
    assert config.logseq.graph_path == str(tmp_path)
```

---

## Development Tips

1. **Start with validators**: They're the foundation and easiest to test in isolation
2. **Use Rich's `Console.print()` liberally**: Makes debugging interactive flows much easier
3. **Test timeout handling manually**: Automated tests can't fully capture the UX
4. **Mock SentenceTransformer early**: Prevents accidental 420MB downloads during development
5. **Use `pytest -k test_name -v`**: Run specific tests quickly during iteration
6. **Test on fresh system**: Cached models and existing configs hide bugs
7. **Verify file permissions**: `ls -la ~/.config/logsqueak/config.yaml` should show `-rw-------`
8. **Check logs**: `~/.cache/logsqueak/logs/logsqueak.log` for debugging wizard issues

---

## Common Pitfalls

- **Forgetting to expand `~` in paths**: Always use `Path.expanduser()`
- **Not handling keyboard interrupts**: Wrap wizard in try/except for `KeyboardInterrupt`
- **Mixing sync and async code**: All network operations must be async
- **Not preserving provider settings**: Easy to overwrite existing `llm_providers` dict
- **Hardcoding timeout values**: Use constants or config for timeouts
- **Not setting file permissions atomically**: Set chmod before rename, not after
- **Displaying full API keys**: Always mask existing keys in prompts
- **Not validating before writing**: Run `Config.load()` on assembled config before write

---

## Next Steps After Implementation

1. **Manual testing**: Run through all user scenarios from spec
2. **Documentation**: Update main README with setup wizard instructions
3. **User testing**: Have 2-3 people try the wizard cold (no guidance)
4. **Iteration**: Collect feedback on confusing prompts or error messages
5. **Edge case hunting**: Try to break the wizard in creative ways
6. **Performance**: Time the full wizard flow (should be <3 minutes)

---

## Reference Implementation Order

```
validators.py  →  providers.py  →  prompts.py  →  wizard.py  →  cli.py
     ↓                 ↓               ↓             ↓            ↓
Unit tests    Unit tests     Unit tests    Integration  Manual
(30 min)      (20 min)       (30 min)      tests        testing
                                           (1 hour)     (30 min)
```

**Total estimated time**: 4-6 hours (experienced Python developer)

**Checkpoint**: After each module, run tests and verify contract compliance before moving on.
