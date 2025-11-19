# Data Model: Interactive Setup Wizard

**Phase**: 1 - Design
**Date**: 2025-11-19
**Source**: Extracted from feature spec and research decisions

## Entities

### 1. WizardState

**Purpose**: Tracks wizard progress and collects user inputs across all prompts

**Fields**:
- `existing_config: Config | None` - Loaded from existing config file (if exists)
- `graph_path: str` - User-provided Logseq graph directory path
- `provider_type: str` - Selected LLM provider ("ollama", "openai", "custom")
- `ollama_endpoint: str | None` - Ollama API endpoint (if provider is ollama)
- `ollama_model: str | None` - Selected Ollama model name
- `openai_api_key: str | None` - OpenAI API key (if provider is openai)
- `openai_model: str | None` - OpenAI model name
- `custom_endpoint: str | None` - Custom OpenAI-compatible endpoint
- `custom_api_key: str | None` - Custom provider API key
- `custom_model: str | None` - Custom provider model name
- `num_ctx: int | None` - Ollama context window size (optional)
- `top_k: int` - RAG search top_k value (default: 20)

**Relationships**: None (ephemeral state object, not persisted)

**Validation Rules**:
- `graph_path` must exist and contain both `journals/` and `logseq/` subdirectories
- Exactly one of (ollama_*, openai_*, custom_*) fields must be populated based on `provider_type`
- `num_ctx` is only valid when `provider_type == "ollama"` (ignored for others)

**State Transitions**:
1. Initial: All fields None except `top_k` (default 20)
2. Config loaded: `existing_config` populated if file exists
3. Graph configured: `graph_path` validated and set
4. Provider selected: `provider_type` set, branches to provider-specific flow
5. Provider configured: Provider-specific fields populated
6. Validated: LLM and embedding tests passed
7. Final: Config assembled from state and written to disk

---

### 2. ProviderConfig (Abstract)

**Purpose**: Common interface for provider-specific configuration data

**Fields** (common to all providers):
- `endpoint: str` - API endpoint URL
- `api_key: str` - Authentication key
- `model: str` - Model identifier

**Subclasses**:

#### OllamaProviderConfig
- `endpoint: str` (default: "http://localhost:11434/v1")
- `api_key: str` (default: "ollama")
- `model: str` - Selected from available models
- `num_ctx: int` (default: 32768)

#### OpenAIProviderConfig
- `endpoint: str` (default: "https://api.openai.com/v1")
- `api_key: str` - User-provided API key
- `model: str` - One of (gpt-4o, gpt-4-turbo, gpt-3.5-turbo, custom)

#### CustomProviderConfig
- `endpoint: str` - User-provided OpenAI-compatible URL
- `api_key: str` - User-provided API key
- `model: str` - User-provided model name

**Usage**: Internal wizard representation, converted to LLMConfig before writing

---

### 3. ValidationResult

**Purpose**: Captures outcome of validation operations (path, network, model tests)

**Fields**:
- `success: bool` - Whether validation passed
- `error_message: str | None` - Error details if failed
- `data: Any | None` - Additional data (e.g., list of Ollama models, disk space info)

**Examples**:
```python
# Successful Ollama connection with models
ValidationResult(
    success=True,
    error_message=None,
    data={"models": [{"name": "mistral:7b-instruct", "size": 4109733376}, ...]}
)

# Failed OpenAI API key
ValidationResult(
    success=False,
    error_message="Invalid API key: 401 Unauthorized",
    data=None
)

# Disk space warning
ValidationResult(
    success=True,
    error_message="Low disk space: 800 MB available (1024 MB recommended)",
    data={"available_mb": 800}
)
```

**Validation Rules**: None (pure data container)

---

### 4. OllamaModel

**Purpose**: Represents a model available in Ollama instance

**Fields**:
- `name: str` - Model identifier (e.g., "mistral:7b-instruct")
- `size: int` - Model size in bytes
- `modified_at: str` - Last modified timestamp (ISO format)

**Source**: Retrieved from Ollama `/api/tags` endpoint

**Example JSON Response**:
```json
{
  "models": [
    {
      "name": "mistral:7b-instruct",
      "size": 4109733376,
      "modified_at": "2024-11-01T12:00:00Z"
    },
    {
      "name": "llama2",
      "size": 3826793728,
      "modified_at": "2024-10-15T08:30:00Z"
    }
  ]
}
```

---

### 5. ConfigFile (Extended)

**Purpose**: YAML configuration file structure (extends existing Config model)

**Existing Fields** (from `src/logsqueak/models/config.py`):
- `llm: LLMConfig` - Active LLM configuration
- `logseq: LogseqConfig` - Logseq graph path
- `rag: RAGConfig` - RAG search settings

**New Fields** (added for wizard):
- `llm_providers: dict[str, dict] | None` - Preserved provider configurations

**YAML Structure**:
```yaml
# Active configuration (used by Logsqueak runtime)
llm:
  endpoint: http://localhost:11434/v1
  api_key: ollama
  model: mistral:7b-instruct
  num_ctx: 32768

logseq:
  graph_path: /home/user/Documents/logseq-graph

rag:
  top_k: 20

# Preserved configurations (managed by wizard only)
llm_providers:
  ollama_local:
    endpoint: http://localhost:11434/v1
    api_key: ollama
    model: mistral:7b-instruct
    num_ctx: 32768
  openai:
    endpoint: https://api.openai.com/v1
    api_key: sk-proj-abc123...
    model: gpt-4o
  custom_azure:
    endpoint: https://myazure.openai.azure.com/v1
    api_key: azure-key-xyz...
    model: gpt-4
```

**File Permissions**: Must be mode 600 (read/write for owner only)

**Validation Rules**:
- `llm_providers` keys are freeform strings (wizard generates them as "{provider}_{instance}")
- Each provider dict must have at minimum: `endpoint`, `api_key`, `model`
- When loading config, `llm_providers` is optional (backwards compatible)
- When writing config, wizard includes `llm_providers` even if empty dict

---

## State Machine: Wizard Flow

```
┌─────────────────┐
│   START         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Check if config │
│ file exists     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Load existing   │
│ config (if any) │
│ for defaults    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────────┐
│ Prompt graph    │─────▶│ Validate path    │
│ path            │      │ (journals/,      │
└────────┬────────┘      │  logseq/)        │
         │               └────────┬─────────┘
         │                        │ Invalid
         │                        ▼
         │                  ┌──────────────┐
         │                  │ Show error   │
         │                  │ Retry prompt │
         │                  └──────┬───────┘
         │                         │
         │                         └──────┐
         │ Valid                          │
         ▼                                │
┌─────────────────┐                       │
│ Prompt provider │◀──────────────────────┘
│ type            │
└────────┬────────┘
         │
    ┌────┴─────┬──────────┐
    │          │          │
    ▼          ▼          ▼
┌───────┐  ┌───────┐  ┌───────┐
│Ollama │  │OpenAI │  │Custom │
│ Flow  │  │ Flow  │  │ Flow  │
└───┬───┘  └───┬───┘  └───┬───┘
    │          │          │
    └────┬─────┴──────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────────┐
│ Test LLM        │─────▶│ Connection OK?   │
│ connection      │      │ Timeout handler  │
└────────┬────────┘      └────────┬─────────┘
         │                        │ Failed
         │                        ▼
         │                  ┌──────────────┐
         │                  │ Retry/Skip   │
         │                  │ prompt       │
         │                  └──────┬───────┘
         │                         │
         │                         └──────┐
         │ Success                        │
         ▼                                │
┌─────────────────┐                       │
│ Check embedding │◀──────────────────────┘
│ model cache     │
└────────┬────────┘
         │
    ┌────┴────────┐
    │             │
    ▼             ▼
┌─────────┐  ┌──────────┐
│ Cached  │  │Download  │
│ Quick   │  │with      │
│ validate│  │progress  │
└────┬────┘  └─────┬────┘
     │             │
     └─────┬───────┘
           │
           ▼
    ┌──────────────┐
    │ Model loads? │
    └──────┬───────┘
           │ Success
           ▼
    ┌──────────────┐
    │ Assemble     │
    │ config       │
    │ (merge       │
    │  providers)  │
    └──────┬───────┘
           │
           ▼
    ┌──────────────────┐      ┌──────────────────┐
    │ Config file      │─────▶│ Prompt to        │
    │ exists?          │ Yes  │ overwrite        │
    └────────┬─────────┘      └─────────┬────────┘
             │ No                       │ No/Abort
             │                          ▼
             │                    ┌──────────────┐
             │                    │   EXIT       │
             │                    │  (no write)  │
             │                    └──────────────┘
             │                          │ Yes
             │                          │
             └────────┬─────────────────┘
                      │
                      ▼
               ┌──────────────┐
               │ Write config │
               │ (mode 600)   │
               └──────┬───────┘
                      │
                      ▼
               ┌──────────────┐
               │ Show success │
               │ & next steps │
               └──────┬───────┘
                      │
                      ▼
                   ┌──────┐
                   │ EXIT │
                   └──────┘
```

## Provider-Specific Flows

### Ollama Flow
1. Attempt connection to existing config URL (if any), else localhost:11434
2. If connection succeeds → retrieve models via `/api/tags`
3. If connection fails → prompt for custom Ollama URL → retry
4. Display models in table, highlight Mistral 7B Instruct if available
5. Prompt user to select model from list
6. Optionally prompt for num_ctx (default: 32768)

### OpenAI Flow
1. Display existing masked API key if available (first 8 + "..." + last 4)
2. Prompt for API key (visible input for new keys, masked for existing)
3. Display common models: gpt-4o, gpt-4-turbo, gpt-3.5-turbo, custom
4. If custom → prompt for model name

### Custom Flow
1. Prompt for endpoint URL (OpenAI-compatible)
2. Prompt for API key (visible input)
3. Prompt for model name

## Data Flow: Config Assembly

```
WizardState
    │
    ├─ graph_path ───────── ───┐
    │                          │
    ├─ provider_type ──── ─┐   │
    │                      │   │
    ├─ ollama_* fields ── ─┤   │
    ├─ openai_* fields ── ─┤   │
    ├─ custom_* fields ── ─┤   │
    │                      │   │
    └─ existing_config     │   │
         │                 │   │
         └─ llm_providers ─┤   │
                           │   │
                           ▼   ▼
                    ┌──────────────────┐
                    │ Config Assembly  │
                    │                  │
                    │ 1. Create new    │
                    │    LLMConfig from│
                    │    selected      │
                    │    provider      │
                    │                  │
                    │ 2. Create        │
                    │    LogseqConfig  │
                    │    from graph    │
                    │    path          │
                    │                  │
                    │ 3. Preserve      │
                    │    existing      │
                    │    llm_providers │
                    │                  │
                    │ 4. Add current   │
                    │    provider to   │
                    │    llm_providers │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Config           │
                    │ ├─ llm           │
                    │ ├─ logseq        │
                    │ ├─ rag           │
                    │ └─ llm_providers │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ YAML             │
                    │ serialization    │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ ~/.config/       │
                    │ logsqueak/       │
                    │ config.yaml      │
                    │ (mode 600)       │
                    └──────────────────┘
```

## Error Handling

All validation functions return `ValidationResult` to enable consistent error handling:

```python
# Path validation
def validate_graph_path(path: str) -> ValidationResult:
    """Validate Logseq graph path structure."""
    expanded = Path(path).expanduser()
    if not expanded.exists():
        return ValidationResult(
            success=False,
            error_message=f"Path does not exist: {expanded}",
            data=None
        )
    if not (expanded / "journals").exists():
        return ValidationResult(
            success=False,
            error_message=f"Missing journals/ directory in {expanded}",
            data=None
        )
    if not (expanded / "logseq").exists():
        return ValidationResult(
            success=False,
            error_message=f"Missing logseq/ directory in {expanded}",
            data=None
        )
    return ValidationResult(success=True, error_message=None, data=None)

# Network validation
async def test_llm_connection(config: ProviderConfig) -> ValidationResult:
    """Test LLM endpoint connectivity."""
    # Implementation from research.md
    # Returns ValidationResult with success status and error details

# Model validation
async def validate_embedding_model() -> ValidationResult:
    """Validate embedding model loads successfully."""
    # Implementation from research.md
    # Returns ValidationResult with success status
```

## Summary

This data model supports:
- ✅ Multi-provider configuration with preservation
- ✅ Validation at each step (path, network, model)
- ✅ State tracking through wizard flow
- ✅ Error handling with actionable feedback
- ✅ Backwards compatibility with existing Config model
- ✅ Clean separation of concerns (wizard state vs. persistent config)
