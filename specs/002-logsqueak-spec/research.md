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

## 2. Textual Testing Strategy with Pilot API

**Question**: How should we test Textual TUI components to ensure keyboard navigation, streaming updates, and user interactions work correctly?

**Research Findings**:

### Why Test TUI Components

- **Regression prevention**: Catch UI breakage early without manual testing every change
- **Keyboard navigation complexity**: Many key combinations (j/k, Shift+j/k, Space, Tab, etc.) need verification
- **Streaming behavior**: LLM results arriving incrementally must update UI correctly
- **State management**: Selection state, background tasks, screen transitions need validation
- **Refactoring confidence**: Tests allow safe refactoring of widgets and screens

### Textual Pilot API

Textual provides a built-in testing framework (`pilot`) that runs apps headlessly and simulates user interactions:

```python
import pytest
from textual.pilot import Pilot
from logsqueak.tui.screens.block_selection import Phase1Screen

@pytest.mark.asyncio
async def test_block_navigation():
    """Test j/k keyboard navigation through blocks."""
    app = Phase1Screen(blocks=sample_blocks())

    async with app.run_test() as pilot:
        # Wait for app to be ready
        await pilot.pause()

        # Simulate keypresses
        await pilot.press("j", "j", "k")

        # Query widgets
        tree = app.query_one("#block-tree")
        assert tree.cursor_line == 1
```

**Key Pilot Methods**:

- `app.run_test()`: Run app in headless test mode
- `pilot.pause()`: Wait for async messages to process
- `pilot.press(*keys)`: Simulate keyboard input
- `pilot.click(selector)`: Simulate mouse clicks
- `app.query_one(selector)`: Find widget by CSS selector
- `app.query(selector)`: Find multiple widgets

### Test-Driven Development Workflow

**Critical**: Write tests FIRST, verify they FAIL, then implement features.

**Step 1: Write Failing Tests**

```python
# tests/ui/test_phase1_navigation.py
@pytest.mark.asyncio
async def test_navigate_with_j_k_keys():
    """Test j/k keys navigate through block tree."""
    app = Phase1Screen(blocks=[
        create_block("Block 1"),
        create_block("Block 2"),
        create_block("Block 3"),
    ])

    async with app.run_test() as pilot:
        await pilot.pause()

        tree = app.query_one("#block-tree")

        # Initially at line 0
        assert tree.cursor_line == 0

        # Press j twice
        await pilot.press("j", "j")
        assert tree.cursor_line == 2

        # Press k once
        await pilot.press("k")
        assert tree.cursor_line == 1
```

Run: `pytest tests/ui/test_phase1_navigation.py -v`

Expected: **Test FAILS** (Phase1Screen not implemented yet)

**Step 2: Implement Feature**

```python
# src/logsqueak/tui/screens/block_selection.py
class Phase1Screen(Screen):
    def on_key(self, event: Key) -> None:
        tree = self.query_one("#block-tree")

        if event.key == "j":
            tree.cursor_line += 1
        elif event.key == "k":
            tree.cursor_line = max(0, tree.cursor_line - 1)
```

**Step 3: Verify Tests Pass**

Run: `pytest tests/ui/test_phase1_navigation.py -v`

Expected: **Test PASSES** (implementation correct)

**Step 4: Manual Verification**

Run actual TUI: `logsqueak extract` and manually test j/k navigation

### Snapshot Testing for Visual Regression

Use `pytest-textual-snapshot` to capture and compare rendered UI:

```python
def test_phase1_initial_render(snap_compare):
    """Test Phase 1 screen initial appearance."""
    assert snap_compare(
        "logsqueak/tui/screens/block_selection.py",
        terminal_size=(120, 40)
    )
```

**How it works**:

1. First run: Generates SVG snapshot of rendered UI
2. Subsequent runs: Compares current render to saved snapshot
3. Detects visual regressions (layout changes, missing elements, styling issues)

**When to use**:

- Initial screen render for each phase
- After major UI changes (verify visual appearance)
- Before/after keyboard interactions (verify state changes are visible)

### Testing Patterns for Each Phase

**Phase 1 (Block Selection) Tests**:

- Navigation: `j`/`k` moves cursor, `Shift+j`/`Shift+k` jumps to knowledge blocks
- Selection: `Space` toggles selection, `a` accepts all, `c` clears all
- Streaming: LLM classifications appear incrementally, visual indicators update
- Status: Background task progress updates in status widget
- Transition: `n` key enabled only when blocks selected

**Phase 2 (Content Editing) Tests**:

- Navigation: `j`/`k` moves between blocks, auto-saves on navigation
- Focus: `Tab` focuses/unfocuses text editor
- Actions: `a` accepts LLM version, `r` reverts to original
- Blocking: `n` key disabled until RAG search completes
- Status: Multi-task progress (rewording, indexing, RAG search)

**Phase 3 (Integration Review) Tests**:

- Navigation: `j`/`k` navigates decisions, preview updates
- Actions: `y` accepts decision, `n` skips to next block, `a` accepts all
- Write status: Decisions marked ⊙ pending → ✓ completed → ⚠ failed
- Preview: Target page preview scrolls to insertion point
- Error handling: Write failures show errors, allow continuation

### Mocking LLM Responses in Tests

```python
@pytest.fixture
def mock_llm_stream():
    """Mock LLM streaming responses for testing."""
    async def stream_classifications():
        yield {"block_id": "abc123", "confidence": 0.92, "reason": "..."}
        yield {"block_id": "def456", "confidence": 0.85, "reason": "..."}

    return stream_classifications

@pytest.mark.asyncio
async def test_llm_streaming_updates(mock_llm_stream):
    """Test UI updates as LLM results stream in."""
    app = Phase1Screen(blocks=sample_blocks())
    app.llm_stream = mock_llm_stream  # Inject mock

    async with app.run_test() as pilot:
        await pilot.pause()

        # Trigger LLM classification
        app.start_classification()
        await pilot.pause()  # Let stream process

        # Verify blocks updated with LLM suggestions
        tree = app.query_one("#block-tree")
        first_node = tree.get_node_at_line(0)
        assert first_node.data.llm_classification == "knowledge"
        assert "🤖" in first_node.label
```

### Test Organization

```
tests/
├── ui/
│   ├── test_phase1_navigation.py      # j/k, Shift+j/k navigation
│   ├── test_phase1_selection.py       # Space, a, c, r actions
│   ├── test_phase1_llm_streaming.py   # LLM results appearing
│   ├── test_phase1_status.py          # Status widget updates
│   ├── test_phase1_snapshots.py       # Visual regression
│   ├── test_phase2_navigation.py      # Navigation with auto-save
│   ├── test_phase2_editing.py         # Tab focus, text editing
│   ├── test_phase2_llm_accept.py      # Accept LLM version
│   ├── test_phase2_rag_blocking.py    # RAG search blocks 'n'
│   ├── test_phase3_navigation.py      # Decision navigation
│   ├── test_phase3_accept.py          # Accept decisions, writes
│   └── test_phase3_errors.py          # Write failures
├── unit/
│   ├── test_models.py                 # Pydantic models
│   ├── test_llm_client.py             # LLM streaming
│   └── test_file_monitor.py           # File tracking
└── integration/
    ├── test_workflow.py               # Full Phase 1→2→3 flow
    └── test_rag_pipeline.py           # Indexing + search
```

### Running Tests

```bash
# Run all UI tests for Phase 1
pytest tests/ui/test_phase1_*.py -v

# Run all tests with coverage
pytest --cov=logsqueak --cov-report=html -v

# Update snapshots after intentional UI changes
pytest tests/ui/test_phase1_snapshots.py --snapshot-update

# Run specific test
pytest tests/ui/test_phase1_navigation.py::test_navigate_with_j_k_keys -v
```

**Decision**: Use Textual pilot for all TUI component testing with Test-Driven Development approach (write tests first, verify failures, implement, verify passes).

**Rationale**:

- **Automation**: Pilot enables automated testing of keyboard interactions and UI state
- **Fast feedback**: Tests run in milliseconds vs manual testing in minutes
- **Regression prevention**: Catches UI breakage immediately
- **TDD workflow**: Writing tests first clarifies requirements and edge cases
- **Refactoring safety**: Can refactor widgets confidently with test coverage
- **Snapshot testing**: Visual regression detection catches layout/styling issues
- **Mock-friendly**: Easy to mock LLM streams and background tasks

**Alternatives Considered**:

- Manual testing only: Rejected because too slow and error-prone for complex keyboard interactions
- Selenium/browser testing: Not applicable (Textual is terminal-based, not web)
- Custom test harness: Rejected because Textual pilot is built-in and well-designed

---

## 3. NDJSON Streaming for LLM Responses

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
