# Research: Interactive TUI for Knowledge Extraction

**Date**: 2025-11-04

**Feature**: 002-logsqueak-spec

## Research Questions

This document captures research findings for technical decisions needed to implement the interactive TUI feature.

---

## 1. Textual Framework Best Practices

**Question**: What are the best practices for building complex, multi-phase TUI applications with Textual?

**Research Findings**:

### Architecture Patterns

- **Screen-based navigation**: Use `Screen` classes for each phase (Phase1Screen, Phase2Screen, Phase3Screen)
- **Message passing**: Use Textual's message system for communication between widgets and screens
- **Reactive programming**: Use reactive attributes for state management (e.g., `selected_blocks = reactive(set())`)
- **Worker threads**: Use `@work` decorator for background tasks that don't block UI
- **Composition over inheritance**: Build complex UIs from smaller, reusable widgets

### Tree View Implementation

- Use `Tree` widget from Textual for hierarchical block display
- Add data to tree nodes using `.data` attribute for block references
- Use `.add()` and `.add_leaf()` methods to build hierarchy
- Style tree nodes dynamically using rich Text objects with markup
- Handle navigation with built-in arrow key support

### Markdown Rendering

- Use `markdown-it-py` for parsing markdown to tokens
- Render to Rich Text objects for display in Textual widgets
- Use `Static` widget with rich markup for rendered content
- Consider `textual-markdown` package for ready-made markdown widgets (experimental)

### Background Task Patterns

- Use `run_worker()` to launch background coroutines
- Update UI state from workers using reactive attributes or post_message()
- Use `asyncio.Queue` for streaming results from workers to UI
- Cancel workers on screen exit using `.cancel()` method

**Decision**: Use Screen-based architecture with reactive state management and worker threads for background tasks.

**Rationale**: This aligns with Textual's design philosophy and provides clean separation between phases while enabling responsive UI during LLM streaming.

**Alternatives Considered**:

- Single screen with conditional rendering: Rejected because phase transitions would be harder to manage
- Custom event loop: Rejected because Textual's built-in async support is sufficient

---

## 2. NDJSON Streaming for LLM Responses

**Question**: How should we handle streaming LLM responses with incremental updates to UI?

**Research Findings**:

### NDJSON Format (Newline-Delimited JSON)

- Each line is a complete, valid JSON object
- Lines separated by newline characters (`\n`)
- Simple to parse: read line-by-line, parse each as JSON
- No special markers needed (unlike SSE with `data:` prefix)
- Robust to errors: malformed JSON on one line doesn't corrupt remaining stream

Example NDJSON stream:

```
{"type":"classification","block_id":"abc123","is_knowledge":true,"confidence":0.92}
{"type":"classification","block_id":"def456","is_knowledge":false,"confidence":0.15}
{"type":"classification","block_id":"ghi789","is_knowledge":true,"confidence":0.87}

```

### httpx Streaming Pattern with NDJSON

```python
import httpx
import json

async def stream_ndjson_response(url, payload):
    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream("POST", url, json=payload) as response:
            async for line in response.aiter_lines():
                if line.strip():  # Skip empty lines
                    try:
                        data = json.loads(line)
                        yield data
                    except json.JSONDecodeError as e:
                        logger.error("Malformed JSON line", line=line, error=str(e))
                        continue  # Skip bad line, process remaining stream

```

### LLM API Configuration

OpenAI-compatible APIs (including Ollama) support NDJSON streaming:

- Request: Set `stream: true` in JSON payload
- Response: Each line is complete JSON object with incremental results
- Connection closes when stream complete

### Error Handling

- Try/except around JSON parsing per line (isolated failures)
- Preserve all successfully parsed results before network error
- Timeout applies to each chunk read (60s per FR-078)
- Retry logic preserves partial stream results

**Decision**: Use NDJSON format for all LLM streaming responses with httpx line-by-line async iteration.

**Rationale**:

- NDJSON is simpler than SSE (no `data:` prefix parsing)
- Each line is complete JSON object (easier to log and debug)
- Error isolation: bad JSON on one line doesn't corrupt stream
- Perfect fit with httpx.aiter_lines()
- Compatible with both OpenAI and Ollama APIs

**Alternatives Considered**:

- Server-Sent Events (SSE): Rejected because NDJSON is simpler and sufficient
- JSON streaming libraries (ijson): Rejected because NDJSON line boundaries make complex parsers unnecessary
- requests library: Rejected because it's synchronous and would block UI
- aiohttp: Rejected because httpx has better type hints and simpler API

---

## 3. ChromaDB Integration for RAG Search

**Question**: How should we structure the vector database for efficient semantic search across Logseq pages?

**Research Findings**:

### Collection Schema

- One collection per Logseq graph
- Documents: Individual blocks (not full pages)
- Metadata: page name, block ID, parent hierarchy, last modified timestamp
- Embeddings: Generate from full hierarchical context (same as content hash input)

### Indexing Strategy

- Build index on first run (background task during Phase 1)
- Persist to disk (~/.cache/logsqueak/chromadb/)
- Incremental updates: Check file modification timestamps, re-index changed pages only
- Batch embed blocks (100-500 at a time) for performance

### Query Strategy

- Embed knowledge block's full hierarchical context
- Retrieve top-k blocks (default k=10 per config)
- Post-process: Group by page, rank pages by total similarity
- Explicit link hints: Boost pages mentioned in block content (e.g., [[Page Name]])

### sentence-transformers Model

- Use `all-MiniLM-L6-v2` (384 dimensions, fast, good quality)
- ~90MB download on first run
- Produces embeddings in ~50ms per block on CPU

**Decision**: Use ChromaDB with block-level embeddings from full hierarchical context, indexed incrementally with file modification tracking.

**Rationale**: Block-level granularity matches Logseq's structure and provides better precision than page-level search. Incremental indexing reduces startup time.

**Alternatives Considered**:

- Page-level embeddings: Rejected because we need to place blocks within specific sections of pages
- Chroma ephemeral in-memory: Rejected because re-indexing on every run would be too slow
- FAISS: Rejected because ChromaDB provides simpler API and persistence out-of-box

---

## 4. YAML Configuration Management

**Question**: How should we validate and parse YAML configuration files with lazy validation and clear error messages?

**Research Findings**:

### Pydantic with YAML

```python
from pydantic import BaseModel, Field, validator
import yaml

class LLMConfig(BaseModel):
    endpoint: str
    api_key: str
    model: str
    num_ctx: int = 32768  # Optional with default

class LogseqConfig(BaseModel):
    graph_path: str

    @validator('graph_path')
    def validate_graph_path(cls, v):
        path = Path(v).expanduser()
        if not path.exists():
            raise ValueError(f"Graph path does not exist: {path}")
        if not path.is_dir():
            raise ValueError(f"Graph path is not a directory: {path}")
        return str(path)

class Config(BaseModel):
    llm: LLMConfig
    logseq: LogseqConfig
    rag: RAGConfig

    @classmethod
    def load(cls, path: Path):
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

```

### Lazy Validation Pattern

- Load YAML into raw dict on startup (no validation)
- Parse into Pydantic models only when config section first accessed
- Cache validated config sections for reuse
- Clear error messages from Pydantic ValidationError

### File Permissions Check

```python
import stat
mode = os.stat(config_path).st_mode
if mode & (stat.S_IRWXG | stat.S_IRWXO):
    raise PermissionError(f"Config file has overly permissive permissions: {oct(mode)}")

```

**Decision**: Use Pydantic models with YAML parsing, lazy validation via cached properties, and explicit file permission checks.

**Rationale**: Pydantic provides excellent validation error messages and type safety. Lazy validation means startup is fast and users only see errors for config they actually use.

**Alternatives Considered**:

- Manual dict validation: Rejected because error messages would be harder to standardize
- TOML format: Rejected because spec explicitly requires YAML
- Eager validation at startup: Rejected because spec requires lazy validation

---

## 5. Concurrent File Modification Detection

**Question**: How should we detect when Logseq files are modified externally during a TUI session and safely reload them?

**Research Findings**:

### File Modification Tracking

```python
from pathlib import Path
from typing import Dict

class FileTracker:
    def __init__(self):
        self._mtimes: Dict[Path, float] = {}

    def record(self, path: Path) -> None:
        """Record current modification time."""
        self._mtimes[path] = path.stat().st_mtime

    def is_modified(self, path: Path) -> bool:
        """Check if file modified since last record."""
        if path not in self._mtimes:
            return True
        return path.stat().st_mtime > self._mtimes[path]

    def refresh(self, path: Path) -> None:
        """Update recorded mtime after reload."""
        self._mtimes[path] = path.stat().st_mtime

```

### Reload and Re-validation Flow

1. Before any write operation: Check if target file modified
2. If modified: Reload file using parser
3. Re-validate operation (e.g., target block still exists)
4. If validation passes: Proceed with write and update mtime
5. If validation fails: Show error, mark operation as failed

### Race Condition Handling

- Use try/except around file operations to catch concurrent writes
- If write fails due to modification during write: Treat as validation failure
- Show clear error: "File was modified during operation, skipping"

**Decision**: Use modification time tracking with reload-and-revalidate pattern before all writes.

**Rationale**: mtime checking is fast and reliable across platforms. Reload-and-revalidate ensures we never write to stale file state.

**Alternatives Considered**:

- File locking: Rejected because Logseq doesn't use file locks, would cause conflicts
- Hash-based change detection: Rejected because mtime is faster and sufficient
- No detection: Rejected because spec explicitly requires concurrent modification handling

---

## 6. Network Retry and Timeout Strategy

**Question**: How should we implement automatic retry with exponential backoff for transient LLM API failures?

**Research Findings**:

### httpx Timeout Configuration

```python
timeout = httpx.Timeout(
    connect=10.0,  # Initial connection
    read=60.0,     # Per-read (important for streaming)
    write=10.0,
    pool=10.0
)
client = httpx.AsyncClient(timeout=timeout)

```

### Retry Logic

```python
from asyncio import sleep

async def retry_request(func, max_retries=1, delay=2.0):
    """Retry async function once on transient errors."""
    try:
        return await func()
    except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout) as e:
        if max_retries > 0:
            await sleep(delay)
            return await func()
        raise

```

### User Prompt After Retry Failure

- Show error modal with options: "Retry", "Skip", "Cancel"
- Preserve partial state (e.g., classifications received before failure)
- Allow continuation with partial data where applicable

**Decision**: Use httpx timeout configuration with 60s read timeout and single automatic retry after 2s delay, then prompt user.

**Rationale**: Matches spec requirements (FR-075 to FR-079). Single retry is sufficient for transient issues without annoying users.

**Alternatives Considered**:

- Exponential backoff: Rejected because spec only requires one retry
- Tenacity library: Rejected because our retry logic is simple enough to implement directly

---

## 7. Structured Logging for Debugging

**Question**: How should we structure logs to capture LLM requests/responses and user actions for debugging?

**Research Findings**:

### Python structlog Pattern

```python
import structlog

logger = structlog.get_logger()

# Log LLM request
logger.info(
    "llm_request_started",
    phase="classification",
    model="gpt-4-turbo-preview",
    num_blocks=15,
    request_id="uuid-here"
)

# Log streaming chunk
logger.debug(
    "llm_response_chunk",
    request_id="uuid-here",
    chunk_num=5,
    content_preview="..."
)

# Log user action
logger.info(
    "user_action",
    screen="phase1_selection",
    action="toggle_selection",
    block_id="abc123"
)

```

### Log File Location

- `~/.cache/logsqueak/logs/logsqueak.log`
- JSON format for machine parsing
- No automatic rotation (per spec FR-067a)

### Log Levels

- DEBUG: LLM response chunks, detailed state changes
- INFO: User actions, phase transitions, LLM requests
- WARNING: Partial failures, retry attempts
- ERROR: Operation failures, validation errors

**Decision**: Use structlog with JSON output to cache directory, no rotation.

**Rationale**: structlog provides excellent structured logging with minimal boilerplate. JSON format makes logs easy to parse for debugging sessions.

**Alternatives Considered**:

- Standard logging module: Rejected because structured logging is clearer for debugging LLM interactions
- Human-readable format: Rejected because JSON is more parseable and still readable

---

## Summary of Key Decisions

1. **TUI Framework**: Textual with Screen-based architecture, reactive state, worker threads
2. **LLM Streaming**: NDJSON format with httpx async streaming (line-by-line parsing)
3. **Vector Search**: ChromaDB with block-level embeddings, incremental indexing
4. **Configuration**: Pydantic + YAML with lazy validation and permission checks
5. **File Safety**: Modification time tracking with reload-and-revalidate before writes
6. **Network Resilience**: httpx timeouts (60s read) with single auto-retry (2s delay)
7. **Logging**: structlog with JSON output to ~/.cache/logsqueak/logs/

All decisions align with constitution principles: POC-first (pragmatic choices), non-destructive (file safety), simplicity (file-based, transparent logging, NDJSON format).
