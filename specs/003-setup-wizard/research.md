# Research: Interactive Setup Wizard

**Phase**: 0 - Research & Discovery
**Date**: 2025-11-19
**Purpose**: Resolve technical unknowns and establish implementation patterns for the setup wizard feature

## Research Questions & Findings

### 1. Rich Library CLI Patterns for Interactive Wizards

**Question**: What Rich library components are best for building interactive CLI wizards with prompts, validation, and progress indicators?

**Decision**: Use Rich Console, Prompt, Table, Progress, and Status for wizard UI

**Rationale**:
- Rich is already in project dependencies (used by existing TUI and search command)
- Provides comprehensive CLI formatting: colored panels, tables, spinners, progress bars
- `rich.console.Console` for formatted output (errors, success messages)
- `rich.prompt.Prompt`, `rich.prompt.Confirm` for user input with validation
- `rich.table.Table` for displaying Ollama models
- `rich.status.Status` for spinners during network operations
- `rich.progress.Progress` for embedding model download progress
- All components work well together and have consistent API

**Alternatives Considered**:
- `click.prompt()` - Rejected: No built-in validation, no colored output, less user-friendly
- `PyInquirer` - Rejected: Additional dependency, TUI-style interface is overkill for simple wizard
- `questionary` - Rejected: Another dependency, Rich already provides what we need

**References**:
- Rich documentation: https://rich.readthedocs.io/en/stable/
- Existing usage in `src/logsqueak/cli.py` (search command uses Console and Status)

---

### 2. Config File Preservation When Switching Providers

**Question**: How should the wizard preserve existing provider settings when switching between Ollama and OpenAI?

**Decision**: Store all provider configurations in config file using provider-specific keys, preserve all when switching

**Rationale**:
- Users frequently switch providers (cost optimization, testing, availability)
- Re-entering API keys is frustrating and error-prone
- YAML structure supports multiple provider sections naturally
- Wizard can merge new provider settings with existing config without deleting other providers

**Implementation Pattern**:
```yaml
llm:
  # Active provider settings (what Logsqueak uses)
  endpoint: http://localhost:11434/v1
  api_key: ollama
  model: mistral:7b-instruct
  num_ctx: 32768

# Preserved provider settings (for future switches)
llm_providers:
  openai:
    endpoint: https://api.openai.com/v1
    api_key: sk-...
    model: gpt-4o
  ollama_local:
    endpoint: http://localhost:11434/v1
    api_key: ollama
    model: mistral:7b-instruct
    num_ctx: 32768
  ollama_remote:
    endpoint: http://192.168.1.100:11434/v1
    api_key: ollama
    model: llama2
    num_ctx: 16384
```

**Alternatives Considered**:
- Single provider only - Rejected: Users lose credentials when switching
- Multiple config files - Rejected: Violates simplicity principle, harder to manage
- Encrypted credential store - Rejected: Out of scope, mode 600 is sufficient security for POC

---

### 3. Embedding Model Validation Approach

**Question**: How should the wizard test that sentence-transformers downloads and loads successfully without blocking the UI?

**Decision**: Use PageIndexer's lazy loading mechanism with progress callback, check if model is cached first

**Rationale**:
- PageIndexer already has lazy SentenceTransformer loading logic
- Can detect if model is cached by checking `~/.cache/torch/sentence_transformers/`
- If cached: Quick validation by attempting to load (few seconds)
- If not cached: Display progress bar for ~420MB download (1-5 minutes)
- Reuses existing infrastructure, no duplication of download logic

**Implementation Pattern**:
```python
from sentence_transformers import SentenceTransformer
from pathlib import Path
import os

def check_embedding_model_cached() -> bool:
    """Check if embedding model is already cached locally."""
    model_name = "all-mpnet-base-v2"
    cache_dir = Path.home() / ".cache" / "torch" / "sentence_transformers"
    model_path = cache_dir / f"sentence-transformers_{model_name.replace('/', '_')}"
    return model_path.exists()

def validate_embedding_model(progress_callback=None) -> bool:
    """Validate embedding model loads successfully (downloads if needed)."""
    try:
        model = SentenceTransformer("all-mpnet-base-v2")
        # Test encode to ensure model works
        model.encode(["test"])
        return True
    except Exception as e:
        return False
```

**Alternatives Considered**:
- Skip embedding validation - Rejected: Users hit errors later during RAG search
- Manual download management - Rejected: Reinvents wheel, sentence-transformers handles this well
- Pre-download in background - Rejected: Adds complexity, wizard should be sequential and transparent

**References**:
- Existing PageIndexer usage in `src/logsqueak/services/page_indexer.py`
- Sentence-transformers caching: https://www.sbert.net/docs/pretrained_models.html

---

### 4. Disk Space Check Before Large Downloads

**Question**: How to check available disk space before downloading ~420MB embedding model?

**Decision**: Use `shutil.disk_usage()` to check available space, warn if less than 1GB but allow proceeding

**Rationale**:
- `shutil.disk_usage()` is cross-platform (Linux, macOS, Windows)
- 1GB threshold provides buffer for model (420MB) + overhead
- Warning approach balances safety with user control (don't block if user knows best)
- Mirrors timeout approach: inform user, let them decide

**Implementation Pattern**:
```python
import shutil
from pathlib import Path

def check_disk_space(required_mb: int = 1024) -> tuple[bool, int]:
    """Check if sufficient disk space is available.

    Args:
        required_mb: Minimum required space in megabytes

    Returns:
        (has_space, available_mb) tuple
    """
    cache_dir = Path.home() / ".cache"
    usage = shutil.disk_usage(cache_dir)
    available_mb = usage.free // (1024 * 1024)
    has_space = available_mb >= required_mb
    return has_space, available_mb
```

**Alternatives Considered**:
- `os.statvfs()` - Rejected: Not available on Windows
- `psutil.disk_usage()` - Rejected: Additional dependency
- No disk check - Rejected: Downloads fail mid-way with cryptic errors

---

### 5. LLM Connection Testing Without Full LLMClient

**Question**: Should wizard instantiate full LLMClient for connection testing, or use simpler HTTP test?

**Decision**: Use simple HTTP request to test endpoint availability, model validation is optional

**Rationale**:
- LLMClient is designed for streaming NDJSON responses (complex for simple connection test)
- Simple HTTP GET/POST to health endpoint is faster and simpler
- For Ollama: GET `/api/tags` tests connection AND retrieves models (two birds, one stone)
- For OpenAI: POST `/v1/chat/completions` with minimal prompt tests API key validity
- Avoid importing heavy LLMClient machinery into wizard module

**Implementation Pattern**:
```python
import httpx

async def validate_ollama_connection(endpoint: str, timeout: int = 30) -> tuple[bool, list[dict] | str]:
    """Validate Ollama connection and retrieve models.

    Returns:
        (success, models_or_error)
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{endpoint}/api/tags",
                timeout=timeout
            )
            response.raise_for_status()
            data = response.json()
            return True, data.get("models", [])
    except Exception as e:
        return False, str(e)

async def validate_openai_connection(endpoint: str, api_key: str, model: str, timeout: int = 30) -> tuple[bool, str]:
    """Test OpenAI API connection.

    Returns:
        (success, error_message)
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{endpoint}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "test"}],
                    "max_tokens": 5
                },
                timeout=timeout
            )
            response.raise_for_status()
            return True, ""
    except Exception as e:
        return False, str(e)
```

**Alternatives Considered**:
- Use full LLMClient - Rejected: Overkill, tight coupling to streaming logic
- No connection test - Rejected: Users discover issues only when running extract command
- Ping endpoint only - Rejected: Doesn't validate API key or model access

---

### 6. Handling Provider-Specific Settings in Config Model

**Question**: How to extend Config model to support multiple provider configurations?

**Decision**: Add optional `llm_providers` field to Config, keep existing `llm` field for active provider

**Rationale**:
- Preserves existing Config structure (backwards compatible with current code)
- `llm` field remains the source of truth for active configuration
- `llm_providers` is a write-only store managed by wizard (not used by runtime code)
- Pydantic allows optional fields with `default=None`
- Wizard writes both sections, runtime only reads `llm`

**Implementation Pattern**:
```python
class Config(BaseModel):
    """Root configuration for Logsqueak application."""

    llm: LLMConfig = Field(..., description="LLM API settings")
    logseq: LogseqConfig = Field(..., description="Logseq graph settings")
    rag: RAGConfig = Field(default_factory=RAGConfig, description="RAG search settings")

    # NEW: Provider preservation for wizard
    llm_providers: dict[str, dict] | None = Field(
        default=None,
        description="Preserved settings for all configured LLM providers"
    )
```

**Alternatives Considered**:
- Separate config file for providers - Rejected: Multiple files violate simplicity principle
- Array of provider configs - Rejected: Need to identify active provider, dict is clearer
- No preservation - Rejected: Poor UX when switching providers

---

### 7. Timeout Handling for Network Operations

**Question**: Should wizard auto-abort on timeout, or prompt user for action?

**Decision**: Prompt user with options (continue waiting, retry, skip) instead of auto-abort

**Rationale**:
- Users on slow networks need more time, auto-abort is hostile
- Users on fast networks with transient issues want retry
- Power users may want to skip validation and fix later
- Prompting respects user agency (aligns with constitution principle III)
- Different timeout values for different operations:
  - LLM connection: 30 seconds (quick API call)
  - Embedding download: 5 minutes (large download)

**Implementation Pattern**:
```python
from rich.prompt import Prompt

async def operation_with_timeout_handling(operation_func, operation_name: str, timeout: int):
    """Execute operation with timeout and user prompt on timeout."""
    try:
        return await asyncio.wait_for(operation_func(), timeout=timeout)
    except asyncio.TimeoutError:
        console.print(f"[yellow]⚠ {operation_name} timed out after {timeout}s[/yellow]")
        choice = Prompt.ask(
            "What would you like to do?",
            choices=["continue", "retry", "skip"],
            default="retry"
        )
        if choice == "continue":
            # Remove timeout, wait indefinitely
            return await operation_func()
        elif choice == "retry":
            # Retry with same timeout
            return await operation_with_timeout_handling(operation_func, operation_name, timeout)
        else:  # skip
            return None
```

**Alternatives Considered**:
- Auto-abort on timeout - Rejected: Hostile to users on slow networks
- No timeout - Rejected: Wizard could hang indefinitely
- Fixed timeout with no retry - Rejected: Transient network issues would fail permanently

---

## Summary of Decisions

| Area | Decision | Key Rationale |
|------|----------|---------------|
| UI Framework | Rich library (Console, Prompt, Table, Progress, Status) | Already in deps, comprehensive CLI formatting |
| Config Preservation | Add `llm_providers` dict to Config, preserve all provider settings | Better UX when switching providers, no re-entering credentials |
| Embedding Validation | Use PageIndexer's lazy loading, check cache first | Reuses existing logic, fast for cached models |
| Disk Space Check | `shutil.disk_usage()` with 1GB threshold, warn but allow proceeding | Cross-platform, balances safety with user control |
| Connection Testing | Simple HTTP requests (GET `/api/tags` for Ollama, POST `/v1/chat/completions` for OpenAI) | Faster and simpler than full LLMClient |
| Provider Settings | Optional `llm_providers` field in Config model | Backwards compatible, clear separation of active vs. preserved |
| Timeout Handling | Prompt user (continue/retry/skip) instead of auto-abort | Respects user agency, handles slow networks gracefully |

## Implementation Notes

1. **Module Structure**: Create `src/logsqueak/wizard/` package with:
   - `wizard.py` - Main orchestration (prompts sequence, config assembly, file write)
   - `prompts.py` - Rich-based prompt helpers (with validation and defaults)
   - `validators.py` - Validation functions (path, disk space, network tests)
   - `providers.py` - Provider-specific logic (Ollama model fetching, OpenAI validation)

2. **Testing Strategy**:
   - Mock Rich prompts using `monkeypatch` in pytest
   - Mock httpx for network tests (no real API calls in tests)
   - Integration test: Full wizard flow with mocked user input and API responses
   - Edge case tests: Timeout handling, disk space warnings, permission errors

3. **Error Handling**: All error scenarios should:
   - Use Rich panels for clear error messages
   - Provide actionable guidance (what to fix, how to fix it)
   - Allow user to retry or abort gracefully
   - Log errors using existing structlog infrastructure

4. **Configuration Flow**:
   ```
   1. Check if config exists
   2. Load existing config (if any) → extract defaults
   3. Prompt for Logseq graph path → validate
   4. Prompt for LLM provider → branch to provider flow
   5. Test LLM connection → retry on failure
   6. Check embedding model cache → download if needed
   7. Validate embedding model → retry on failure
   8. Assemble config (merge with preserved providers)
   9. Prompt to overwrite if config exists → abort if declined
   10. Write config file with mode 600
   11. Display success message with next steps
   ```

## Open Questions for Implementation

None - all clarifications resolved via spec clarification session.
