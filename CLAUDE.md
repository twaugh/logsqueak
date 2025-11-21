# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Logsqueak** is a TUI (Text User Interface) application for extracting lasting knowledge from Logseq journal entries using LLM-powered analysis. Users interactively review, refine, and integrate knowledge blocks into their Logseq knowledge base.

**Current Status**: ✅ **COMPLETE** - Full interactive workflow with all features implemented

- ✅ **Core Parser**: `logseq-outline-parser` library (robust Logseq markdown parsing with property order preservation)
- ✅ **Infrastructure**: Models, services, CLI, config with lazy validation
- ✅ **Phase 1 - Block Selection TUI**: Tree navigation, LLM streaming classification, manual selection, multi-journal support
- ✅ **Phase 2 - Content Editing TUI**: Three-panel layout, LLM rewording, manual editing, RAG search
- ✅ **Phase 3 - Integration Review TUI**: Decision batching, target preview, atomic writes, completion summary
- ✅ **Application Integration**: End-to-end workflow, background workers with dependency coordination
- ✅ **LLM Optimization**: Hierarchical chunks keep prompts compact (~6-12KB per block)
- ✅ **RAG Search**: PageIndexer and RAGSearch with hierarchical chunks and explicit link boosting
- ✅ **File Operations**: Atomic two-phase writes with provenance markers and concurrent modification detection
- ✅ **LLM Queue**: Priority-based request serialization with cancellation support
- ✅ **CLI Commands**: `extract` (with date/range parsing), `search` (with clickable logseq:// links), `init` (interactive setup wizard)
- ✅ **Per-Graph Indexing**: ChromaDB with deterministic directory naming
- ✅ **Logging**: Structured logging for all LLM requests/responses and user actions
- ✅ **Edge Cases**: Config errors, network errors, file modifications, malformed JSON handling
- ✅ **UX Polish**: Skip_exists decisions with clickable links, quiet indexing when no updates needed
- ✅ **Performance**: EditedContent references eliminate sync loops, pre-cleaned contexts in RAG index
- ✅ **Test Coverage**: Comprehensive test suite with unit, integration, and UI tests

## Project Structure

```
logsqueak/
├── src/
│   ├── logsqueak/                 # Main application
│   │   ├── models/                # Pydantic data models (config, block state, LLM chunks)
│   │   ├── services/              # LLMClient, FileMonitor, PageIndexer, RAGSearch
│   │   ├── tui/                   # TUI screens & widgets (Phase 1, Phase 2 complete)
│   │   ├── wizard/                # Interactive setup wizard (validators, prompts, orchestration)
│   │   ├── utils/                 # Logging, UUID generation, LLM ID mapping
│   │   ├── cli.py                 # Click-based CLI entry point
│   │   └── config.py              # ConfigManager with lazy validation
│   └── logseq-outline-parser/     # Logseq markdown parser library (COMPLETE)
│       └── src/logseq_outline/
│           ├── parser.py          # Core parsing/rendering logic
│           ├── context.py         # Full-context generation & content hashing
│           └── graph.py           # Graph path utilities
├── tests/                         # Test suite
│   ├── unit/                      # Unit tests for models, services, utils
│   ├── integration/               # Integration tests for config, LLM, RAG pipeline
│   └── ui/                        # UI tests for Phase 1 and Phase 2 (Textual pilot)
├── specs/
│   ├── 001-logsqueak/             # Original knowledge extraction spec
│   ├── 002-logsqueak-spec/        # Interactive TUI feature spec (COMPLETE)
│   │   ├── spec.md                # Complete feature specification
│   │   ├── tasks.md               # Phase 1-8 complete (T001-T147 ✅)
│   │   └── contracts/             # Service interfaces, data models
│   └── 003-setup-wizard/          # Interactive setup wizard spec (CURRENT)
│       ├── spec.md                # Setup wizard specification
│       ├── tasks.md               # Phase 1-3 complete (T001-T037 ✅)
│       ├── plan.md                # Implementation planning
│       └── contracts/             # Validation interfaces, data models
├── pyproject.toml                 # Project dependencies and config
└── CLAUDE.md                      # This file
```

## Development Commands

### Environment Setup

**IMPORTANT**: Always activate the virtual environment before running any commands:

```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Running Tests

```bash
# Run all tests
pytest -v

# Run specific test suite
pytest tests/unit/ -v           # Unit tests only
pytest tests/integration/ -v    # Integration tests only
pytest tests/ui/ -v             # UI tests only

# Run parser tests only
pytest src/logseq-outline-parser/tests/ -v

# Run single test file
pytest tests/unit/test_config.py -v

# Run single test function
pytest tests/unit/test_config.py::test_load_config_success -v

# Run with coverage
pytest --cov=logsqueak --cov=logseq_outline --cov-report=html -v
# Open htmlcov/index.html to view coverage report
```

### Code Quality

```bash
# Format code (auto-fixes)
black src/ tests/

# Lint code (shows issues)
ruff check src/ tests/

# Type checking (strict mode)
mypy src/
```

### Running the CLI

```bash
# First-time setup: Interactive configuration wizard
logsqueak init

# Launch end-to-end knowledge extraction workflow
logsqueak extract                         # Today's journal entry
logsqueak extract 2025-01-15              # Specific date
logsqueak extract 2025-01-10..2025-01-15  # Date range

# Search your knowledge base
logsqueak search "machine learning best practices"
logsqueak search "python debugging tips" --reindex  # Force rebuild index
```

## Key Architecture Components

### 1. Logseq Outline Parser - Hybrid ID System

The parser implements a **dual-mode parsing strategy** with a sophisticated hybrid ID system.

#### Parsing Modes

- **Default mode** (`strict_indent_preservation=False`): Normalizes continuation line indentation, no internal markers. **Use this for all read-only operations** including LLM prompts, UI display, semantic search, and analysis.
- **Strict mode** (`strict_indent_preservation=True`): Preserves exact indentation using internal `\x00N\x00` outdent markers for write operations. **Only use this when re-parsing content for modification and writing back to files.**

```python
from logseq_outline.parser import LogseqOutline

# Parse for reading (recommended for application code)
outline = LogseqOutline.parse(markdown_text)

# Parse for writing back to files (strict mode)
outline = LogseqOutline.parse(markdown_text, strict_indent_preservation=True)
```

**Design rationale:** Outdent markers are an implementation detail confined to the write path, preventing them from leaking into LLM prompts and UI rendering.

#### Hybrid ID System

Every block has a stable identifier for targeting and tracking:

**Two-Tier Identification:**

1. **Explicit IDs** - `id::` property (UUID format) if present in source

   ```python
   block.block_id = block.get_property("id")  # "65f3a8e0-1234-5678-9abc-def012345678"
   ```

2. **Implicit IDs** - MD5 content hash when no `id::` exists

   ```python
   from logseq_outline.context import generate_content_hash

   # Generate content-based hash (includes hierarchical context)
   content_hash = generate_content_hash(block, parent_blocks, indent_str, frontmatter)
   ```

**Why Hierarchical Context?**

- Same content at different hierarchy levels = different hashes (intentional)
- Blocks with truly identical content + hierarchy = same hash (acceptable for RAG)
- Deterministic: same input → same hash across runs
- Page-scoped: includes `page_name` prefix for global uniqueness in RAG index

**Usage Example:**

```python
# Find block by ID (explicit or hash)
block, parents = outline.find_block_by_id(block_id)

# Access block's stable ID
block_id = block.block_id  # From id:: property if present, else None
```

### 2. Background Task Mechanism - Worker Coordination

The app uses a **shared dictionary pattern** for worker coordination across all screens.

#### Dependency Chain

```
Phase 1 (App-Level):
  model_preload (app.py) ──► SentenceTransformer in thread pool
    └─ [BLOCKS] → page_indexing (Phase1Screen) ──► PageIndexer.build_index()

Phase 2:
  llm_rewording (Phase2Screen) ──► Independent, starts immediately

  rag_search (Phase2Screen)
    └─ [WAITS FOR] page_indexing completion
      └─ [TRIGGERS] llm_decisions (opportunistic)

         - If RAG completes while in Phase 2: Worker starts there
         - If user presses 'n' quickly: Worker starts in Phase 3

Phase 3:
  llm_decisions (Phase3Screen or Phase2Screen)
    └─ Polls for new decisions OR starts worker if none exist
```

#### Task Lifecycle

**Critical Pattern:** Tasks are **never deleted**, only marked as "completed" or "failed"

```python
# Worker creates task
app.background_tasks["task_name"] = BackgroundTask(
    task_type="page_indexing",
    status="running",
    progress_current=0,
    progress_total=100
)

# Worker updates progress
app.background_tasks["task_name"].progress_current = 50

# Worker marks complete (stays in dict)
app.background_tasks["task_name"].status = "completed"

# Dependent workers check completion
dependency = app.background_tasks.get("task_name")
if dependency is None or dependency.status == "completed":
    # Ready to proceed
```

**Exception:** Screen-level tasks like `llm_classification` and `llm_rewording` are deleted on screen transitions for cleanup.

#### LLM Request Serialization

**Problem:** Multiple workers trying to stream LLM responses concurrently leads to resource contention with high-latency reasoning models.

**Solution:** Priority-based request queue serializes LLM calls.

```python
from logsqueak.models.llm_priority import LLMRequestPriority

# Priority levels (lower number = higher priority)
# CLASSIFICATION = 1 (Phase 1)
# REWORDING = 2 (Phase 2)
# INTEGRATION = 3 (Phase 3)

# Workers acquire slot before LLM calls
await app.acquire_llm_slot("classification_worker", LLMRequestPriority.CLASSIFICATION)
try:
    async for chunk in llm_client.stream_ndjson(...):
        # Process chunks
finally:
    app.release_llm_slot("classification_worker")
```

#### Opportunistic Worker Execution

**Pattern:** The Integration Decision Worker can start in either Phase 2 or Phase 3, depending on when its dependencies complete and user navigation speed.

**How it works:**

- Phase 2 starts the worker opportunistically when RAG search completes
- If user waits in Phase 2: Worker runs there, decisions ready for Phase 3
- If user presses 'n' quickly: Worker starts in Phase 3 instead
- Phase 3 checks if decisions are already streaming and polls for new ones

**Why it's useful:**

- Maximizes background processing time (starts as soon as dependencies ready)
- Doesn't block user navigation (worker can start after phase transition)
- Shared `app.integration_decisions` list works across phase boundaries
- Reduces perceived latency if user reviews content slowly in Phase 2

```python
# Phase 2 (content_editing.py): Start worker opportunistically
async def _check_rag_search_complete(self):
    rag_task = self.app.background_tasks.get("rag_search")
    if rag_task and rag_task.status == "completed":
        # RAG done - start integration worker if not already started
        if "llm_decisions" not in self.app.background_tasks:
            self.run_worker(self._integration_decision_worker())

# Phase 3 (integration_review.py): Handle both scenarios
async def on_mount(self):
    if self.app.integration_decisions:
        # Worker already running in Phase 2 - poll for new decisions
        self.run_worker(self._poll_for_decisions())
    else:
        # Worker not started yet - start it now
        self.run_worker(self._integration_decision_worker())
```

### 3. RAG Chunking Strategy - Hierarchical Context

The RAG (Retrieval-Augmented Generation) pipeline uses semantic search to find relevant existing knowledge.

#### Deterministic Graph Database Naming

**Pattern:** Each Logseq graph gets a unique ChromaDB collection named `{basename}-{16-char-hash}` where the hash is derived from the absolute path.

**How it works:**

```python
def generate_graph_db_name(graph_path: Path) -> str:
    basename = graph_path.name
    abs_path = str(graph_path.resolve())
    hash_hex = hashlib.sha256(abs_path.encode('utf-8')).hexdigest()[:16]
    return f"{basename}-{hash_hex}"

# Example: /home/user/Documents/my-graph → my-graph-a1b2c3d4e5f6a7b8
```

**Why it's useful:**

- Multiple graphs with same name can coexist (different paths = different hashes)
- Same graph at same path always gets same DB (deterministic hashing)
- Readable naming includes basename for debugging
- Collision-resistant (16-char SHA-256 prefix is ~2^64 space)
- Automatically handles graph renames/moves (new path = new DB, old DB preserved)

#### Schema Version Tracking with Automatic Rebuild

**Pattern:** ChromaDB collections store an `INDEX_SCHEMA_VERSION` in metadata. On init, PageIndexer checks the version and auto-deletes outdated collections.

**How it works:**

```python
# Constants
INDEX_SCHEMA_VERSION = 3  # Current schema version

# On collection creation
collection.modify(metadata={"schema_version": str(INDEX_SCHEMA_VERSION)})

# On PageIndexer initialization
stored_version = int(collection.metadata.get("schema_version", "1"))
if stored_version != INDEX_SCHEMA_VERSION:
    logger.warning("schema_version_mismatch",
                   stored=stored_version,
                   current=INDEX_SCHEMA_VERSION)
    self.client.delete_collection(name=db_name)
    # Next build_index() call will recreate with current schema
```

**Version history:** See `INDEX_SCHEMA_VERSION` constant and comments in `src/logsqueak/services/page_indexer.py` for schema evolution details.

**Why it's useful:**

- No user intervention required for schema upgrades
- Old indexes automatically invalidated when code changes
- Next `logsqueak extract` or `logsqueak search` rebuilds transparently
- Prevents subtle bugs from schema mismatches
- Clean upgrade path as features evolve

#### Indexing Phase (PageIndexer)

**Chunk Generation:**

```python
from logseq_outline.context import generate_chunks

# Generate semantic chunks with hierarchical context
chunks = generate_chunks(outline, page_name)

for block, full_context, hybrid_id, parents in chunks:
    # full_context includes:
    # - Page frontmatter (title::, tags::, etc.)
    # - All parent blocks with indentation
    # - Current block with indentation

    # Pre-clean context during indexing
    cleaned_context = _clean_context_for_llm(full_context, outline.frontmatter)

    # Store in ChromaDB
    collection.upsert(
        documents=[cleaned_context],  # Hierarchical context
        embeddings=[embedding],
        ids=[f"{page_name}::{hybrid_id}"],
        metadatas=[{
            "page_name": page_name,
            "block_id": hybrid_id,
            "page_frontmatter": json.dumps(frontmatter)
        }]
    )
```

**Pre-Cleaning During Indexing:**

```python
def _clean_context_for_llm(context: str, page_properties: list[str]) -> str:
    """Strip id:: properties and page-level properties during indexing.

    Why clean during indexing instead of during search?

    - One-time cost vs. repeated cost per LLM call
    - Prevents duplicate frontmatter in prompts
    - Keeps prompts compact by removing internal metadata
    """
```

**Page-Level Semantic Chunks:**

Every page gets a page-level chunk containing the page name, title, and frontmatter, even if it has no blocks.

```python
# ALWAYS create page-level chunk (even for empty pages)
page_title = self._extract_page_title(outline)
page_context = f"Page: {page_name}\nTitle: {page_title}\n{frontmatter}"

page_chunk_id = f"{page_name}::__PAGE__"
chunks_by_id[page_chunk_id] = {
    "document": page_context,
    "metadata": {"block_id": "__PAGE__", "page_name": page_name, ...}
}
```

**Why it's useful:**

- Enables searching by page name alone (finds empty pages)
- Page title can differ from filename (stored in `title::` property)
- Frontmatter provides additional search context (tags, properties)
- Special `__PAGE__` block ID distinguishes from real blocks
- RAG search can suggest adding content to empty but relevant pages

#### Search Phase (RAGSearch)

**Semantic Search with Explicit Link Boosting:**

```python
# Extract [[Page Name]] links from query
explicit_links = re.findall(r'\[\[([^\]]+)\]\]', context)

# Boost similarity scores for mentioned pages
if page_name in explicit_links:
    similarity *= 1.5  # 50% boost
```

Returns hierarchical chunks directly from ChromaDB: `(page_name, block_id, hierarchical_context)` tuples.

### 4. LLM Response Streaming - NDJSON Protocol

**Multi-Format Support:**

- Direct NDJSON (one JSON object per line)
- SSE (Server-Sent Events) with `data:` prefix
- Ollama native format (`/api/chat` endpoint)
- OpenAI-compatible format (`/v1/chat/completions`)

**Provider Detection:**

```python
async def _detect_ollama() -> bool:
    """Probe /api/version endpoint to classify provider.

    Cached after first call to avoid repeated network requests.
    """
    version_url = f"{base_url}/api/version"
    response = await client.get(version_url)
    return response.status_code == 200  # Ollama responds, OpenAI doesn't
```

**HTTP 429 Retry with Exponential Backoff:**

The LLM client automatically retries rate-limited requests with intelligent backoff.

```python
# Error handling in stream_ndjson()
except httpx.HTTPStatusError as e:
    if e.response.status_code == 429:
        # Respect Retry-After header if present
        retry_after = e.response.headers.get("Retry-After")
        if retry_after:
            delay = int(retry_after)
        else:
            # Exponential backoff: 2s, 4s, 8s, 16s...
            delay = 2 ** attempt

        logger.warning("llm_rate_limit",
                      attempt=attempt,
                      max_retries=max_retries,
                      delay=delay)
        await asyncio.sleep(delay)
        # Retry request
    else:
        raise  # Other 4xx/5xx errors fail immediately
```

**Why it's useful:**

- Transparent retry without failing requests
- Respects API rate limit guidance (`Retry-After` header)
- Exponential backoff prevents thundering herd
- Particularly important during Phase 3 (multiple sequential LLM calls)
- Configurable max_retries (default: 1 retry = 2 total attempts)

**Streaming Pipeline:**

```python
async def stream_ndjson(prompt, system_prompt, chunk_model):
    # Detect format (SSE vs NDJSON, Ollama vs OpenAI)
    needs_content_extraction = is_ollama or "text/event-stream" in content_type

    accumulated_content = ""

    async for line in response.aiter_lines():
        if line.startswith("data: "):
            json_line = line[6:]  # Strip "data: " prefix

        data = json.loads(json_line)

        if needs_content_extraction:
            # Extract fragment: data["message"]["content"] (Ollama)
            #               or data["choices"][0]["delta"]["content"] (OpenAI SSE)
            content_fragment = _extract_content(data)
            accumulated_content += content_fragment

            # Try parsing complete NDJSON lines
            for complete_line in accumulated_content.split('\n')[:-1]:
                chunk_data = json.loads(complete_line)
                chunk = chunk_model(**chunk_data)
                yield chunk
        else:
            # Direct NDJSON - parse immediately
            chunk = chunk_model(**data)
            yield chunk
```

**Chunk Models** (from `models/llm_chunks.py`):

```python
class KnowledgeClassificationChunk(BaseModel):
    type: Literal["classification"] = "classification"
    block_id: str
    reasoning: str  # Chain-of-thought explanation of why this block is valuable
    confidence: float  # Based on temporal durability (0.8-1.0: fundamental, 0.5-0.7: contextual, 0.0-0.4: speculative)

class ContentRewordingChunk(BaseModel):
    type: Literal["rewording"] = "rewording"
    block_id: str
    reworded_content: str

class IntegrationDecisionChunk(BaseModel):
    type: Literal["decision"] = "decision"
    knowledge_block_id: str
    target_page: str
    action: Literal["add_section", "add_under", "replace", "skip_exists"]
    target_block_id: Optional[str]
    confidence: float
    reasoning: str  # Exposed via tooltip in Phase 3 UI
```

**Separated Classification and Rewording Workflow:**

Phase 1 classifies blocks with reasoning, Phase 2 rewords all selected blocks independently.

```python
# Phase 1 prompt identifies valuable blocks with chain-of-thought reasoning
# Each KnowledgeClassificationChunk contains:
# - block_id: Which journal block this came from
# - reasoning: Why the LLM considers this block valuable
# - confidence: Based on temporal durability

# Phase 1 worker stores classification in BlockState
block_state.llm_classification = "knowledge"
block_state.llm_reworded_content = None  # No rewording yet
block_state.llm_confidence = chunk.confidence

# Phase 1→2 transition creates empty EditedContent
app.edited_content[block_id] = EditedContent(
    block_id=block_id,
    current_content="",  # Empty - will be reworded in Phase 2
    ...
)

# Phase 2 rewording worker processes ALL selected blocks
for block_id, edited_content in app.edited_content.items():
    # Reword content, preserving [[Page]] wiki links
    async for chunk in llm_client.stream_classification_and_rewording(...):
        edited_content.current_content = chunk.reworded_content
```

**Why it's useful:**

- Separation of concerns: Phase 1 identifies what's valuable, Phase 2 handles rewording
- Chain-of-thought reasoning improves classification accuracy
- Consistent rewording quality for all blocks (LLM-classified and user-selected)
- Simplified prompts with clearer responsibilities per phase

**Short ID Optimization:**

All three phases use compact sequential IDs (1, 2, 3...) in LLM prompts instead of long hybrid IDs (UUIDs/content hashes), reducing token usage by ~600 tokens per extraction run.

**Implementation:**

```python
from logsqueak.utils.llm_id_mapper import LLMIDMapper

# Create mapper scoped to this LLM request
id_mapper = LLMIDMapper()

# Register all block IDs that will appear in the prompt
for block in blocks:
    id_mapper.add(block.block_id)  # Returns short ID ("1", "2", etc.)

# Replace hybrid IDs with short IDs in prompt
short_id = id_mapper.to_short(block.block_id)

# Translate LLM response back to hybrid IDs
async for chunk in llm_client.stream_ndjson(...):
    hybrid_id = id_mapper.try_to_hybrid(chunk.block_id)
    if hybrid_id is None:
        logger.warning("llm_invalid_block_id", short_id=chunk.block_id)
        continue  # Skip invalid chunks
    chunk.block_id = hybrid_id
    yield chunk
```

**Phase-Specific Behavior:**

- **Phase 1 (Classification)**: Maps all journal blocks, deep copies outline before replacing IDs to avoid mutation
- **Phase 2 (Rewording)**: Maps edited content blocks
- **Phase 3 (Integration)**: Creates per-call mapper (knowledge block always ID "1", RAG targets get "2", "3", etc.), translates both `knowledge_block_id` and `target_block_id`

**Why it's useful:**

- Reduces prompt size by ~2,400 characters (36-char UUIDs → 1-2 char numbers)
- Scoped per-request ensures no ID conflicts across phases
- Graceful degradation (invalid IDs logged and skipped, not crashed)
- Transparent to application logic (translation happens at LLM boundary)

### 5. Interactive UX Features

#### Clickable Logseq Links

The application generates clickable `logseq://` links in multiple places for seamless navigation to your Logseq graph.

**Link Format:**

```python
from logsqueak.utils.logseq_urls import create_logseq_url

# Page link
url = create_logseq_url(graph_name, page_name)
# → logseq://graph/my-graph?page=Python%20Debugging

# Block reference link
url = create_logseq_url(graph_name, page_name, block_id)
# → logseq://graph/my-graph?block-id=65f3a8e0-1234-5678-9abc-def012345678
```

**Where links appear:**

1. **Phase 1 - Previously Integrated Content:**
   - Shows "Previously integrated:" section for blocks with `extracted-to::` property
   - Displays clickable links to each integrated location
   - Example: `[Python Debugging]((65f3a8e0-...))` → clickable link in terminal

2. **Phase 3 - Integration Decisions List:**
   - Shows target page links for all decision types (add_section, add_under, replace, skip_exists)
   - Clickable links displayed in the decision list (left panel)

3. **Search Command Output:**
   - `logsqueak search "query"` returns results with clickable page links
   - Each result shows: page name (clickable), similarity score, hierarchical context
   - Works in modern terminals that support hyperlinks

**Terminal Compatibility:**

- Modern terminals (iTerm2, GNOME Terminal, Windows Terminal) render as clickable links
- Legacy terminals display as plain text URLs (still usable)

#### Skip_Exists Decision UI

Phase 3 displays all integration decisions transparently, including `skip_exists` for already-integrated content.

**How it works:**

```python
# Phase 3 decision list shows all decisions
for decision in decisions:
    if decision.action == "skip_exists":
        # Display with "✓ Already Exists" indicator
        # Preview shows existing block location
        # y/a keys skip over these (non-actionable)
    else:
        # Normal decision (add_section, add_under, replace)
        # y/a keys accept and write
```

**Why it's useful:**

- Full transparency - users see all LLM decisions including skips
- Audit trail - no hidden filtering of decisions
- Non-disruptive - skip_exists can't be accidentally "accepted"
- Navigation - clickable links to existing content
- Completion summary dynamically calculates skip count

#### Integration Reasoning Tooltips

Phase 3 exposes the LLM's reasoning for each integration decision via tooltips.

**Implementation:**

- `IntegrationDecisionChunk.reasoning` field contains LLM explanation
- Tooltip appears when hovering over decision in the decision list
- Shows why LLM chose specific target page and action

**Example reasoning:**
> "This debugging insight is related to the existing 'Python Debugging' page which covers pdb and debugging tools. Adding under the 'Tools' section provides thematic organization."

**Why it's useful:**

- Transparency into LLM decision-making
- Helps users understand suggested placements
- Educational - users learn to organize better over time
- Debugging - reveals when LLM misunderstands content

### 6. Structured Logging - What and Where

**Location:** `~/.cache/logsqueak/logs/logsqueak.log`

**Format:** JSON (one object per line, parseable with `jq`)

**Log Levels:**

```bash
export LOGSQUEAK_LOG_LEVEL=DEBUG  # DEBUG, INFO, WARNING, ERROR
```

**What's Logged:**

- **LLM Requests (INFO):** Request ID, model, endpoint, provider, prompt length, timestamp
- **LLM Request Payloads (DEBUG):** Full JSON payload sent to LLM API
- **LLM Response Lines (INFO):** Each complete NDJSON line received
- **LLM Raw Responses (INFO):** Full raw response content
- **User Actions (INFO):** Block selection, navigation, acceptance decisions
- **Worker Dependencies (INFO/DEBUG):** Waiting for dependencies, status changes
- **File Operations (INFO/ERROR):** Page writes, journal updates, errors

**Usage Examples:**

```bash
# View all logs with pretty-printing
tail -f ~/.cache/logsqueak/logs/logsqueak.log | jq .

# Filter LLM requests only
jq 'select(.event | startswith("llm_"))' ~/.cache/logsqueak/logs/logsqueak.log

# Filter errors
jq 'select(.level == "error")' ~/.cache/logsqueak/logs/logsqueak.log

# Track specific block through workflow
jq 'select(.block_id == "abc123")' ~/.cache/logsqueak/logs/logsqueak.log
```

### 7. Provenance Recording - Atomic Two-Phase Writes

**Goal:** Ensure both page write and journal marker succeed atomically, or neither.

**Pattern:** Two-phase commit with idempotency

#### Write Sequence

```python
async def write_integration_atomic(decision, journal_date, graph_paths, file_monitor):
    # PHASE 1: Page Write
    # 1. Check and reload page if modified
    # 2. Idempotency check - has this integration already happened?
    # 3. Apply integration (add_section, add_under, replace, skip_exists)
    # 4. Atomic write to page

    # PHASE 2: Journal Update (only if Phase 1 succeeded)
    # 5. Reload journal if modified
    # 6. Add provenance marker
    # 7. Atomic write to journal
```

**Atomic Write Implementation:**

```python
def atomic_write(path: Path, content: str, file_monitor: FileMonitor):
    """Temp-file-rename pattern with double modification check."""

    # Early check
    if file_monitor and file_monitor.is_modified(path):
        raise FileModifiedError("File modified before write")

    # Create temp file in same directory (ensures same filesystem)
    temp_path = path.parent / f".{path.name}.tmp.{os.getpid()}"

    # Write to temp
    temp_path.write_text(content)

    # fsync to ensure on disk
    with open(temp_path, 'r+') as f:
        f.flush()
        os.fsync(f.fileno())

    # Late check (before rename)
    if file_monitor and file_monitor.is_modified(path):
        temp_path.unlink()
        raise FileModifiedError("File modified during write")

    # Atomic rename (POSIX guarantee)
    temp_path.replace(path)

    # Update mtime tracking
    if file_monitor:
        file_monitor.refresh(path)
```

**Provenance Format:**

```markdown

- Today I learned about pdb debugging
  extracted-to:: [Python Debugging](((65f3a8e0-1234-5678-9abc-def012345678)))
```

The format is a Markdown link `[text](target)` where:

- Link text: target page name
- Link target: `((uuid))` - Logseq block reference syntax (two parentheses for the block reference)
- Total: three sets of parentheses (one from Markdown link syntax + two from block reference)

**Deterministic UUIDs:**

```python
from logsqueak.utils.ids import generate_integration_id

# Generate UUID v5 from integration parameters
integration_id = generate_integration_id(
    knowledge_block_id="abc123",
    target_page="Python Debugging",
    action="add_under",
    target_block_id="def456"
)
# Same parameters → same UUID (enables idempotent retries)
```

## Project Constitution

**See `.specify/memory/constitution.md` for complete project constitution.**

Key principles:

### I. Proof-of-Concept First

- Prioritize working software over perfection
- Ship iteratively, demonstrate feasibility
- No backwards compatibility guarantees (POC stage)

### II. Non-Destructive Operations (NON-NEGOTIABLE)

- All operations traceable via `extracted-to::` markers in journal entries
- APPEND operations add new blocks without modifying existing content
- Every integrated block generates unique `id::` property (UUID)
- Journal entries atomically marked with block references to integrated knowledge
- **Property order preservation is sacred** - NEVER reorder block properties

### III. Simplicity and Transparency

- Prefer file-based I/O over databases (except ChromaDB for RAG)
- Use JSON for structured LLM outputs
- Show the user what the LLM is doing
- Avoid premature abstraction

### Commit Message Requirements

All AI-assisted commits MUST include:

```
Assisted-by: Claude Code
```

(instead of 'Generated with' or 'Co-Authored-By')

## Python Version & Dependencies

- **Python**: 3.11+ (required)
- **Key runtime dependencies:**
  - `textual>=0.47.0` - TUI framework
  - `httpx>=0.27.0` - Async HTTP client for LLM API
  - `pydantic>=2.0.0` - Data validation and models
  - `click>=8.1.0` - CLI framework
  - `chromadb>=0.4.0` - Vector store for RAG
  - `sentence-transformers>=2.2.0` - Embeddings (lazy-loaded)
  - `structlog>=23.0.0` - Structured logging

- **Dev dependencies:**
  - `pytest>=7.4.0`, `pytest-asyncio>=0.21.0`, `pytest-textual-snapshot>=0.4.0`
  - `black>=23.0.0`, `ruff>=0.1.0`, `mypy>=1.7.0`

## Common Pitfalls

### Parser Property Order Preservation

**NEVER reorder block properties.** Insertion order is sacred in Logseq.

```python
# ✅ CORRECT: Preserves existing property order
block.set_property("extracted-to", value)  # Updates in place if exists, appends if new

# ❌ WRONG: Don't manually rebuild properties list
block.content = [first_line] + new_properties + rest  # Breaks order preservation
```

### Parser Indent Preservation Modes

```python
# ✅ CORRECT: Default mode for reading
outline = LogseqOutline.parse(text)  # No outdent markers

# ✅ CORRECT: Strict mode only for writes
outline = LogseqOutline.parse(text, strict_indent_preservation=True)

# ❌ WRONG: Don't use strict mode for LLM prompts
outline = LogseqOutline.parse(text, strict_indent_preservation=True)
prompt = block.get_full_content()  # May leak \x00N\x00 markers
```

### Background Task Completion Checking

```python
# ✅ CORRECT: Check for None OR "completed"
dependency = app.background_tasks.get("task_name")
if dependency is None or dependency.status == "completed":
    # Ready to proceed

# ❌ WRONG: Only checking for None
if app.background_tasks.get("task_name") is None:
    # Misses completed tasks still in dict
```

### FileMonitor Modification Detection

```python
# ✅ CORRECT: Uses != comparison (git-friendly)
if file_monitor.is_modified(path):
    # File changed (handles git reverts correctly)

# ❌ WRONG: Don't use > comparison
if new_mtime > old_mtime:
    # Breaks on git revert (older mtime)
```

### LLM ID Mapping

**ALWAYS use LLMIDMapper for LLM prompts.** Never send raw hybrid IDs to the LLM.

```python
# ✅ CORRECT: Create mapper, register IDs, translate responses
id_mapper = LLMIDMapper()
for block in blocks:
    id_mapper.add(block.block_id)

# Replace IDs in prompt copy (don't mutate original)
outline_copy = copy.deepcopy(outline)
replace_ids_in_blocks(outline_copy.blocks, id_mapper)

# Translate responses back
async for chunk in llm_client.stream_ndjson(...):
    hybrid_id = id_mapper.try_to_hybrid(chunk.block_id)
    if hybrid_id is None:
        logger.warning("llm_invalid_block_id", ...)
        continue
    chunk.block_id = hybrid_id
    yield chunk

# ❌ WRONG: Sending raw hybrid IDs to LLM
prompt = f"Block {block.block_id}: {block.content}"  # Wastes ~35 chars per ID

# ❌ WRONG: Mutating original outline
block.set_property("id", short_id)  # Application state corrupted
```

## License

GPLv3 - All code is licensed under GPLv3 regardless of authorship method (including AI-assisted development).

