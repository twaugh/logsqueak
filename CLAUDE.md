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
- ✅ **LLM Optimization**: Hierarchical chunks reduce prompts from 62KB → 2-4KB (90% reduction)
- ✅ **RAG Search**: PageIndexer and RAGSearch with hierarchical chunks and explicit link boosting
- ✅ **File Operations**: Atomic two-phase writes with provenance markers and concurrent modification detection
- ✅ **LLM Queue**: Priority-based request serialization with cancellation support
- ✅ **CLI Commands**: `extract` (with date/range parsing), `search` (with clickable logseq:// links)
- ✅ **Per-Graph Indexing**: ChromaDB with deterministic directory naming
- ✅ **Logging**: Structured logging for all LLM requests/responses and user actions
- ✅ **Edge Cases**: Config errors, network errors, file modifications, malformed JSON handling
- ✅ **UX Polish**: Skip_exists decisions with clickable links, quiet indexing when no updates needed
- ✅ **Performance**: EditedContent references eliminate sync loops, pre-cleaned contexts in RAG index
- ✅ **Test Coverage**: 280+ tests passing (173 unit, 69 integration, 38 UI)

## Project Structure

```
logsqueak/
├── src/
│   ├── logsqueak/                 # Main application
│   │   ├── models/                # Pydantic data models (config, block state, LLM chunks)
│   │   ├── services/              # LLMClient, FileMonitor, PageIndexer, RAGSearch
│   │   ├── tui/                   # TUI screens & widgets (Phase 1, Phase 2 complete)
│   │   ├── utils/                 # Logging, UUID generation
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
│   └── 002-logsqueak-spec/        # Interactive TUI feature spec (CURRENT)
│       ├── spec.md                # Complete feature specification
│       ├── tasks.md               # Phase 1-6.5 complete (T001-T108s ✅)
│       └── contracts/             # Service interfaces, data models
├── pyproject.toml                 # Project dependencies and config
└── CLAUDE.md                      # This file
```

## Currently Implemented: Logseq Outline Parser

The `logseq-outline-parser` library is a **production-ready** Logseq markdown parser with precise structure preservation.

### Key Features

#### 1. Non-Destructive Parsing & Rendering

The parser preserves **exact structure and order** from source files:

- **Property order preservation** (NON-NEGOTIABLE): Property insertion order is sacred in Logseq
- **Indentation detection**: Auto-detects 2-space indentation
- **Content structure**: All block lines (first line, continuation lines, properties) stored in unified list
- **Round-trip guarantee**: Parse → Modify → Render produces identical output (except intentional changes)

```python
from logseq_outline.parser import LogseqOutline

# Parse Logseq markdown (default mode - for read operations)
outline = LogseqOutline.parse(markdown_text)

# Parse with strict indent preservation (for write operations only)
outline = LogseqOutline.parse(markdown_text, strict_indent_preservation=True)

# Access blocks
for block in outline.blocks:
    print(block.get_full_content())
    print(f"ID: {block.block_id}")  # From id:: property or None

# Render back to markdown
output = outline.render()  # Preserves exact formatting
```

**Indent Preservation Modes:**

The parser supports two modes for handling continuation line indentation:

- **Default mode** (`strict_indent_preservation=False`): Continuation line indentation is normalized during parsing. No internal outdent markers are created. **Use this for all read-only operations** including LLM prompts, UI display, semantic search, and analysis. This is the recommended mode for application-layer code.

- **Strict mode** (`strict_indent_preservation=True`): Exact indentation is preserved using internal outdent markers (`\x00N\x00`) for continuation lines with non-standard indentation. **Only use this when re-parsing content for modification and writing back to files.** This mode is necessary for file operations that require perfect round-trip fidelity.

**Why this matters:** Outdent markers are an implementation detail for preserving formatting during write operations. By defaulting to normalized mode, the parser prevents these markers from leaking into application code, simplifying LLM prompts and UI rendering.

#### 2. Hybrid ID System

Every block has a stable identifier for targeting and tracking:

- **Explicit IDs**: `id::` property (UUID format) if present in source
- **Implicit IDs**: Content hash (MD5 of full context) for blocks without explicit IDs
- **Reproducible hashing**: Same content + hierarchy = same hash (deterministic)

```python
from logseq_outline.context import generate_content_hash

# Get block's stable ID
block_id = block.block_id  # From id:: property if present

# Or generate content-based hash
content_hash = generate_content_hash(block, parent_blocks)
```

**Design rationale**: Blocks with truly identical content and context get the same hash. This is intentional - such blocks are semantically equivalent for RAG purposes. Use explicit `id::` properties if you need to distinguish identical blocks.

#### 3. Full-Context Generation

The context module generates hierarchical context strings for blocks:

```python
from logseq_outline.context import generate_full_context

# Generate full context (includes all parent blocks)
context = generate_full_context(block, parent_list)
# Example output:
# "- Parent content\n  - Child content\n    - Current block content"
```

**Important:** When using the default parsing mode (`strict_indent_preservation=False`), the generated context strings contain clean, normalized indentation without internal markers. This makes them safe to use directly in LLM prompts, UI display, and semantic search.

This is used for:
- Content-based hashing (hybrid ID system)
- Semantic search embeddings (via RAG pipeline)
- LLM context windows (hierarchical_context in EditedContent)

### Parser API Reference

#### LogseqBlock

```python
@dataclass
class LogseqBlock:
    content: list[str]           # All block lines (first line, continuation, properties)
    indent_level: int            # Indentation level (0 = root)
    block_id: Optional[str]      # From id:: property or None
    children: list[LogseqBlock]  # Child blocks

    def get_full_content(self, normalize_whitespace: bool = False) -> str:
        """Get full block content as single string."""

    def add_child(self, content: str, position: Optional[int] = None) -> LogseqBlock:
        """Add child bullet with proper indentation."""

    def get_property(self, key: str) -> Optional[str]:
        """Get property value by key."""

    def set_property(self, key: str, value: str) -> None:
        """Set property value (preserves order if exists, appends if new)."""
```

#### LogseqOutline

```python
@dataclass
class LogseqOutline:
    blocks: list[LogseqBlock]    # Root-level blocks
    frontmatter: str             # Text before first bullet

    @classmethod
    def parse(cls, text: str, strict_indent_preservation: bool = False) -> LogseqOutline:
        """Parse Logseq markdown into outline tree.

        Args:
            text: Logseq markdown content
            strict_indent_preservation: If True, preserve exact indentation with
                outdent markers. If False (default), normalize indentation.
                Only use True for write operations.
        """

    def render(self, indent_str: str = "  ") -> str:
        """Render outline back to markdown."""

    def find_block_by_id(self, block_id: str) -> Optional[tuple[LogseqBlock, list[LogseqBlock]]]:
        """Find block by ID (explicit or hash), return (block, parents)."""
```

### Testing the Parser

```bash
# Run parser tests
pytest src/logseq-outline-parser/tests/ -v

# Key test areas
pytest src/logseq-outline-parser/tests/test_parser.py      # Parsing/rendering
pytest src/logseq-outline-parser/tests/test_context.py     # Context generation & hashing
```

## Foundational Infrastructure (IMPLEMENTED)

The foundational infrastructure is **complete and tested** (Phase 1-2 in tasks.md):

### Data Models (Pydantic)

All models in `src/logsqueak/models/`:
- **config.py**: LLMConfig, LogseqConfig, RAGConfig, Config (with YAML loading and permission validation)
- **block_state.py**: BlockState (Phase 1 selection tracking)
- **edited_content.py**: EditedContent (Phase 2 editing state)
- **integration_decision.py**: IntegrationDecision (Phase 3 write decisions)
- **background_task.py**: BackgroundTaskState enum (async task status)
- **llm_chunks.py**: NDJSON streaming chunk models (KnowledgeClassificationChunk, ContentRewordingChunk, IntegrationDecisionChunk)

### Services Layer

Implemented services in `src/logsqueak/services/`:
- **llm_client.py**: LLMClient with async NDJSON streaming, retry logic, request queue with priority
- **llm_wrappers.py**: LLM-specific prompt wrappers for knowledge classification, content rewording, and integration decisions
- **llm_helpers.py**: Decision batching, filtering, and hierarchical chunk formatting for RAG results
- **file_monitor.py**: FileMonitor with mtime tracking using `!=` comparison (git-friendly)
- **page_indexer.py**: ChromaDB vector indexing with lazy-loaded SentenceTransformer, uses `generate_chunks()`
- **rag_search.py**: Semantic search returning hierarchical chunks with explicit link boosting
- **file_operations.py**: Atomic two-phase writes with provenance markers (uses strict mode for write operations)

### RAG Pipeline & Hierarchical Chunks

The RAG (Retrieval-Augmented Generation) pipeline uses semantic search to find relevant existing knowledge:

**Indexing (Phase 1):**
- PageIndexer loads all non-journal pages from the Logseq graph
- Uses `generate_chunks()` to create semantic chunks (blocks with hierarchical context)
- Embeds chunks using lazy-loaded SentenceTransformer model
- Stores embeddings in ChromaDB with full hierarchical context as document text

**Search (Phase 2):**
- RAGSearch finds similar blocks for each knowledge block being integrated
- Returns hierarchical chunks: `(page_name, block_id, hierarchical_context)` tuples
- Hierarchical context includes parent blocks for full understanding
- Applies explicit link boosting (blocks that link to target pages rank higher)

**LLM Integration (Phase 2 → Phase 3):**
- `format_chunks_for_llm()` formats chunks as XML with page grouping
- Only sends relevant block hierarchies (not full pages) to LLM
- Strips duplicate page properties and id:: properties from content
- **Result**: 90%+ prompt reduction (62KB → 2-4KB per knowledge block)

**Why this matters:**
- Enables smaller/faster models (Mistral-7B works without context window errors)
- Reduces LLM latency and cost
- Focuses LLM attention on semantically relevant content only

### Configuration & CLI

- **cli.py**: Click-based CLI with commands:
  - `extract` - Interactive knowledge extraction workflow (date/range parsing, journal loading, TUI launcher)
  - `search` - Semantic search of knowledge base (uses RAG infrastructure, clickable logseq:// links)
- **config.py**: ConfigManager with lazy validation, helpful error messages, permission checks (mode 600)
- **app.py**: Main TUI application with screen management, phase transitions, worker coordination
- Config file: `~/.config/logsqueak/config.yaml` (user must create before first run)

### Utilities

- **logging.py**: Structured logging (structlog) to `~/.cache/logsqueak/logs/logsqueak.log`
- **ids.py**: Deterministic UUID v5 generation with Logsqueak-specific namespace (`32e497fc-abf0-4d71-8cff-e302eb3e2bb0`)

### TUI Components

Screens in `src/logsqueak/tui/screens/`:
- **block_selection.py**: Phase 1 - Tree navigation, LLM classification worker, block selection
- **content_editing.py**: Phase 2 - Three-panel layout, LLM rewording, RAG search, manual editing
- **integration_review.py**: Phase 3 - Decision review, target preview, acceptance workflow

Widgets in `src/logsqueak/tui/widgets/`:
- **block_tree.py**: Hierarchical tree view for journal blocks with selection state
- **status_panel.py**: Real-time background task status display
- **block_detail_panel.py**: Markdown rendering of selected block with context
- **content_editor.py**: Three-panel editor (original, reworded, edited) with auto-save
- **decision_list.py**: Batched integration decisions with filtering and navigation
- **target_page_preview.py**: Live preview of target page with insertion point indicator

### Test Coverage

Comprehensive test suite with **280 tests passing**:
- **173 unit tests** (`tests/unit/`): Models, config, LLM client/helpers/wrappers, file monitor, utilities, RAG services, request queue
  - Includes per-block integration planning validation (T108o)
- **69 integration tests** (`tests/integration/`): Config loading, LLM NDJSON streaming, RAG pipeline, phase transitions, end-to-end workflow
  - Includes multi-block workflow and decision batching tests (T108p)
- **38 UI tests** (`tests/ui/`): All three phases (block selection, content editing, integration review) with snapshot testing

All async fixtures use `@pytest_asyncio.fixture`. FileMonitor uses `!=` for mtime comparison to handle git reverts.

**Phase 6 Validation Complete** (all success criteria met):
- ✅ All unit tests pass (173 passed in 25.40s)
- ✅ All integration tests pass (69 passed in 18.65s)
- ✅ Manual end-to-end tests complete without errors
- ✅ Files written have correct structure and provenance
- ✅ Worker dependencies execute in correct order

## Interactive TUI Application (IMPLEMENTED)

See `specs/002-logsqueak-spec/spec.md` for the complete feature specification.

### High-Level Architecture

The TUI application has **3 interactive phases**:

**Phase 1: Block Selection**
- Display journal blocks in hierarchical tree view
- LLM classifies blocks as "knowledge" vs "activity logs" (streaming results)
- User manually selects/deselects blocks for extraction
- Background workers:
  - LLM Classification Worker (immediate, independent)
  - SentenceTransformer Loading (immediate, app-level)
  - Page Indexing Worker (waits for SentenceTransformer, then indexes pages)

**Phase 2: Content Editing**
- Display selected knowledge blocks with full context
- LLM generates reworded versions (removes temporal context)
- User can accept LLM version, edit manually, or revert to original
- Background workers:
  - LLM Rewording Worker (immediate, independent)
  - RAG Search Worker (waits for PageIndexer from Phase 1)
  - Integration Decision Worker (opportunistic - starts when RAG completes)

**Phase 3: Integration Decisions**
- LLM suggests where to integrate each knowledge block (streaming decisions)
- User reviews preview showing new content in target page context
- Accept/skip each decision (writes immediately on accept)
- Atomic provenance: Add `extracted-to::` marker to journal only after successful write
- Workers:
  - Polls for new decisions if Phase 2 worker still streaming
  - Starts Integration Decision Worker if no decisions exist yet

### Configuration

Configuration file: `~/.config/logsqueak/config.yaml`

**Required fields:**
```yaml
llm:
  endpoint: https://api.openai.com/v1  # Or Ollama: http://localhost:11434/v1
  api_key: sk-your-api-key-here
  model: gpt-4-turbo-preview

logseq:
  graph_path: ~/Documents/logseq-graph

rag:
  top_k: 10  # Optional: Number of similar blocks to retrieve per search (default: 10)
```

**Optional fields:**
```yaml
llm:
  num_ctx: 32768  # Ollama context window size (controls VRAM usage)
                  # Automatically sent via native Ollama API when Ollama is detected
                  # Ignored for OpenAI endpoints (use model's default context window)
```

**Configuration behavior** (from spec clarifications):
- User must create config file before first run (not auto-created)
- Missing config shows helpful error with example YAML
- Lazy validation: Settings validated only when first accessed
- Validation failures: Show error and exit immediately (user must edit and restart)
- File permissions: Must be mode 600 (checked on load)

### Key Design Principles

From `specs/002-logsqueak-spec/spec.md` and project constitution:

#### 1. Non-Destructive Operations (NON-NEGOTIABLE)

- All integrations are traceable via `extracted-to::` markers in journal entries
- APPEND operations add new blocks without modifying existing content
- Every integrated block gets a unique `id::` property (UUID)
- Atomic consistency: Journal marked only when page write succeeds

#### 2. Property Order Preservation (NON-NEGOTIABLE)

**NEVER reorder block properties**. Insertion order is sacred in Logseq.

- Parser preserves exact property order from source
- `set_property()` preserves location if property exists, appends if new
- Round-trip tests verify no reordering occurs

#### 3. Keyboard-Driven Interface

- All actions via keyboard (no mouse support)
- Vim-style navigation (`j`/`k`) + arrow keys
- Consistent key bindings across phases
- Context-sensitive footer shows available actions

#### 4. Streaming LLM Results

- LLM results arrive incrementally (not batch)
- UI updates in real-time as results stream in
- Background tasks don't block UI interaction
- Partial results preserved on errors

#### 5. Explicit User Control

- Users must explicitly approve all integrations (no auto-write)
- Can override any LLM suggestion at any phase
- Can skip/cancel operations without side effects

### Worker Dependencies & Lifecycle

Background workers execute in a specific dependency order to ensure correct operation:

#### Dependency Chain

```
Phase 1:
  LLM Classification (immediate) ─┐
                                   ├─ Independent workers
  SentenceTransformer Loading     ─┤
    └─ [BLOCKS] → PageIndexer     ─┘

Phase 2:
  LLM Rewording (immediate) ───────┐
                                    ├─ Independent workers
  RAG Search                        │
    └─ [WAITS FOR] PageIndexer ────┤
      └─ [TRIGGERS] Integration    │
         Decision Worker            ─┘
         (opportunistic)

Phase 3:
  Integration Decision Worker (if not started in Phase 2)
  OR polls for decisions from Phase 2 worker
```

#### Worker Details

**Phase 1 Workers:**
1. **LLM Classification** - Streams block classifications immediately
2. **SentenceTransformer Loading** (app-level) - Lazy-loads embedding model in thread pool
3. **PageIndexer** - Waits for SentenceTransformer, then builds ChromaDB vector index

**Phase 2 Workers:**
1. **LLM Rewording** - Streams reworded content immediately (independent)
2. **RAG Search** - Waits for PageIndexer completion, then searches for candidate pages
   - **Blocks Phase 2 → 3 transition** until complete
3. **Integration Decision** (opportunistic) - Starts when RAG completes
   - If user waits in Phase 2: Worker starts there, decisions ready for Phase 3
   - If user presses 'n' quickly: Worker starts in Phase 3 instead

**Phase 3 Workers:**
- If decisions already streaming from Phase 2: Polls for new decisions
- If no decisions exist: Starts Integration Decision Worker
- **Placeholder task edge case**: If Phase 2 worker completed before Phase 3 mounts:
  - If `blocks_ready > 0`: Creates placeholder task to show "Processing blocks X/Y"
  - If `blocks_ready == 0`: No placeholder (worker finished with no valid decisions)

#### Key Constraints

1. **PageIndexer cannot start** until SentenceTransformer loads
2. **RAG search cannot start** until PageIndexer completes
3. **Integration decisions cannot start** until RAG search completes
4. **Phase 2 → 3 transition is blocked** until RAG search completes
5. **Integration Decision Worker is opportunistic** (starts whenever RAG completes)
6. **LLM requests are serialized** via priority queue (prevents concurrent prompts):
   - Priority order: Classification (1) > Rewording (2) > Integration (3)
   - Workers use `acquire_llm_slot()` / `release_llm_slot()` for coordination
   - Prevents resource contention with high-latency reasoning models

#### Background Task Lifecycle

Background tasks are tracked in `app.background_tasks` dictionary and displayed in the StatusPanel.

**Task Lifecycle:**
1. Worker creates task entry: `app.background_tasks["task_name"] = BackgroundTask(...)`
2. Worker updates progress: `app.background_tasks["task_name"].progress_current = X`
3. **Worker marks task as "completed"**: `app.background_tasks["task_name"].status = "completed"`

**Polling for Completion:**
- Workers check `task.status == "completed"` OR `task is None` to detect completion
- Completed tasks stay in the dictionary (same as failed tasks), not deleted
- Exception: Some screen-level tasks (e.g., `llm_classification`, `llm_rewording`) are deleted on screen transition for cleanup

**Example:**
```python
# Worker polls for dependency completion
while True:
    dependency_task = self.app.background_tasks.get("dependency_name")
    if dependency_task is None or dependency_task.status == "completed":
        # Task deleted or completed - dependency ready
        break
    elif dependency_task.status == "failed":
        # Dependency failed
        raise RuntimeError(f"Dependency failed: {dependency_task.error_message}")
    await asyncio.sleep(0.1)
```

## Development Commands

### Environment Setup

**IMPORTANT**: Always activate the virtual environment before running any commands:

```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Running Tests

```bash
# Run all tests (148 passed)
pytest -v

# Run parser tests only
pytest src/logseq-outline-parser/tests/ -v

# Run main app tests only
pytest tests/unit/ tests/integration/ -v

# Run with coverage
pytest --cov=logsqueak --cov=logseq_outline --cov-report=html -v

# Verify foundational tests (Phase 2 checkpoint)
pytest tests/unit/ tests/integration/test_config*.py tests/integration/test_llm*.py -v
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type checking
mypy src/
```

### Running the CLI

```bash
# Launch end-to-end knowledge extraction workflow
logsqueak extract                         # Today's journal entry
logsqueak extract 2025-01-15              # Specific date
logsqueak extract 2025-01-10..2025-01-15  # Date range

# Phase 1: Select knowledge blocks (j/k navigation, Space to select, 'a' accept all)
# Phase 2: Edit/refine content (Tab to focus editor, 'a' accept LLM version, 'r' revert)
# Phase 3: Review integrations (j/k through decisions, 'y' accept, 'a' batch accept)
# All phases complete - writes to pages with provenance markers in journal

# Search your knowledge base
logsqueak search "machine learning best practices"
logsqueak search "python debugging tips" --reindex  # Force rebuild index
```

## Python Version & Dependencies

- **Python**: 3.11+ (required)
- **Runtime dependencies** (installed):
  - `textual>=0.47.0` - TUI framework
  - `httpx>=0.27.0` - Async HTTP client for LLM API
  - `pydantic>=2.0.0` - Data validation and models
  - `click>=8.1.0` - CLI framework
  - `pyyaml>=6.0.0` - YAML config parsing
  - `structlog>=23.0.0` - Structured logging
  - `chromadb>=0.4.0` - Vector store for RAG
  - `sentence-transformers>=2.2.0` - Embeddings (lazy-loaded to prevent UI blocking)
  - `markdown-it-py>=3.0.0` - Markdown rendering (Textual dependency for Markdown widget)
- **Dev dependencies** (installed):
  - `pytest>=7.4.0` - Test framework
  - `pytest-asyncio>=0.21.0` - Async test support
  - `pytest-textual-snapshot>=0.4.0` - TUI snapshot testing
  - `pytest-cov>=4.1.0` - Coverage reporting
  - `black>=23.0.0`, `ruff>=0.1.0`, `mypy>=1.7.0` - Code quality

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
- Property order preservation is sacred

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

### Commit Workflow with User Approval

When the user says **"Add and stage"** or **"Add it and stage for review"**, this means:

1. Make the requested changes to the codebase
2. Stage the changes with `git add`
3. Prepare a commit message following the project's commit message format
4. Show the user the proposed commit message for review
5. **Wait for user approval** before running `git commit`

The user will either say "Proceed" to commit, or request changes to the commit message.

**Never commit without explicit user approval of the commit message.**

## Working with Logseq Outline Format

Logseq uses indented bullets (2 spaces per level) with special features:

```markdown
- Top-level bullet
  - Child bullet (2 spaces indent)
    - Grandchild (4 spaces indent)
- Bullet with properties
  property:: value
  another-property:: value
  - Child comes after properties
- Multi-line content
  Continuation line (indented, no bullet)
  Another continuation
```

### Parser Behavior

- `LogseqOutline.parse()`: Returns tree of LogseqBlock objects
  - Default mode: Normalizes continuation line indentation (recommended for app code)
  - Strict mode (`strict_indent_preservation=True`): Preserves exact indentation for writes
- Properties detected by `key:: value` pattern
- Continuation lines: Lines at same/deeper indent without bullet
- Frontmatter: Lines before first bullet (page-level content)

### Renderer Behavior

- `LogseqOutline.render()`: Converts tree back to markdown string
- Preserves exact property order (NON-NEGOTIABLE)
- Preserves continuation line formatting
- Uses detected indentation (default: 2 spaces)

## Feature Implementation Status

**Current branch**: `002-logsqueak-spec`

**Spec location**: `specs/002-logsqueak-spec/spec.md`

**Implementation status**: ✅ **COMPLETE** - All phases implemented and tested

- ✅ **Phase 1: Setup** (T001-T009)
- ✅ **Phase 2: Foundational** (T010-T031)
- ✅ **Phase 3: User Story 1** (T032-T049) - Block Selection TUI
- ✅ **Phase 4: User Story 2** (T050-T070) - Content Editing TUI
- ✅ **Phase 5: User Story 3** (T071-T096) - Integration Review TUI
- ✅ **Phase 6: Application Integration** (T097-T108g) - End-to-end workflow
- ✅ **Phase 6.5: Prompt Optimization** (T108i-T108s) - 90% prompt reduction, LLM request queue
- ✅ **Phase 7: Edge Cases** (T110-T120) - Error handling and robustness
- ✅ **Phase 8: Polish** (T121-T147) - UX improvements, CLI search, performance optimizations

**Development guidelines**:
- Follow TDD: Write failing tests → Implement → Tests pass → Manual verify
- Use Textual pilot for all UI tests (automated keyboard/mouse simulation)
- Property order preservation is NON-NEGOTIABLE
- All file writes must be atomic with provenance markers

## License

GPLv3 - All code is licensed under GPLv3 regardless of authorship method (including AI-assisted development).

## Recent Changes
- 003-setup-wizard: Added Python 3.11+ + Click (CLI framework), Rich (formatted CLI output), httpx (HTTP client for API testing), PyYAML (config serialization), sentence-transformers (embedding model), existing Pydantic models (Config validation)

- 2025-11-17: **Specification Complete** - All phases implemented, 280+ tests passing
  - Completed all 8 phases of the interactive TUI feature spec (002-logsqueak-spec)
  - Key achievements:
    - Three-phase interactive workflow (Block Selection → Content Editing → Integration Review)
    - LLM prompt optimization (90% reduction: 62KB → 2-4KB per block)
    - RAG semantic search with hierarchical chunks
    - CLI commands: `extract` and `search` with clickable logseq:// links
    - Edge case handling (config errors, network failures, file modifications)
    - Performance optimizations (EditedContent references, pre-cleaned RAG contexts, quiet indexing)
    - Comprehensive test coverage (280+ passing tests)
- 2025-11-12: **Integration Decisions Prompt Optimization (COMPLETE)** - Reduced prompt size from 62KB to 2-4KB per block (T108i-m)
  - **Phase 1**: Created `format_chunks_for_llm()` helper in `src/logsqueak/services/llm_helpers.py`
    - Formats RAG search results as XML with hierarchical block context
    - Groups chunks by page, includes page properties, strips id:: from content
  - **Phase 2**: Implemented per-block integration decision planning (T108l, T108m, T108n)
    - Refactored `plan_integrations()` to process one knowledge block at a time
    - Added `plan_integration_for_block()` to handle single block with its candidate pages
    - Reduced initial prompt size from 62KB to ~6-12KB per block
  - **Phase 3**: Wired up hierarchical chunks in RAG pipeline
    - Updated `RAGSearch.find_candidates()` to return hierarchical chunks from ChromaDB
    - Used `generate_chunks()` to find blocks and build parent paths
    - Further reduced prompt size to ~2-4KB per block
  - **Phase 4**: Simplified to use ChromaDB documents directly (final refactor)
    - RAG now returns `(page_name, block_id, hierarchical_context)` tuples
    - Uses pre-computed context from ChromaDB instead of reconstructing from pages
    - Removed redundant page loading and tree traversal
    - Added `strip_page_properties()` to prevent duplicate frontmatter
  - **Impact**: Fixes 400 Bad Request errors from Mistral-7B context window limits
  - **Result**: 90%+ prompt size reduction enables faster responses and more reliable LLM calls
  - Added priority queue to serialize LLM requests (prevents concurrent prompts)
  - Priority order: Classification (1) > Rewording (2) > Integration (3)
  - Workers use `acquire_llm_slot()` / `release_llm_slot()` for coordination
  - Cancellation support: workers cancelled during screen transitions
    - Phase 1→2: Cancel classification worker
    - Phase 2→3: Cancel rewording worker
  - Graceful handling of `asyncio.CancelledError` in all workers
  - Test suite with 5 tests for sequential execution, priority ordering, error handling
  - Rationale: Prevents resource contention with high-latency reasoning models
  - Parser now defaults to normalized indentation (no outdent markers)
  - Added strict_indent_preservation parameter for write operations only
  - Normalized content hashing for stable IDs across parsing modes
  - Removed redundant outdent marker cleaning from llm_wrappers, TUI modules
  - Simplifies application code by handling markers at parser level
  - All 87 parser tests passing, 7 new tests for indent preservation and hash stability
  - Prevents implementation details from leaking into LLM prompts and UI
  - TargetPagePreview and DecisionList widgets complete
  - Phase3Screen with decision navigation, batch acceptance, target preview
  - File operations service with atomic two-phase writes and provenance markers
  - Integration actions: add_section, add_under, replace, skip_exists
  - Idempotent retry detection and concurrent modification handling
  - 38 UI tests passing, 37 file operations tests passing
  - All three core user stories (US1, US2, US3) now complete
  - RAG services (PageIndexer, RAGSearch) with lazy SentenceTransformer loading
  - Fixed PageIndexer to use `generate_chunks()` for proper semantic chunking
  - ContentEditor widget and Phase2Screen with vertical three-panel layout
  - Keyboard controls: j/k navigation, Tab focus, 'a' accept, 'r' revert
  - Auto-save on navigation, RAG search blocking implemented
  - Comprehensive UI test suite with snapshot testing
  - All async fixtures properly use `@pytest_asyncio.fixture`
  - BlockTree, StatusPanel, MarkdownViewer widgets complete
  - Phase1Screen with LLM streaming, keyboard navigation, manual selection
  - Journal loader service with date/range parsing
  - CLI integration launching Phase 1 TUI
  - All data models, services, CLI, config, and utilities complete
  - Fixed FileMonitor to use `!=` comparison (git-friendly)
  - Generated Logsqueak-specific UUID namespace

## Active Technologies
- Python 3.11+ + Click (CLI framework), Rich (formatted CLI output), httpx (HTTP client for API testing), PyYAML (config serialization), sentence-transformers (embedding model), existing Pydantic models (Config validation) (003-setup-wizard)
- File-based YAML config at `~/.config/logsqueak/config.yaml` with mode 600 permissions (003-setup-wizard)
