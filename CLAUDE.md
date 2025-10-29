# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Logsqueak is a CLI tool that extracts lasting knowledge from Logseq journal entries and integrates it into relevant pages using LLM-powered analysis with RAG-based semantic search. It distinguishes between temporary activity logs and knowledge blocks, presents changes in dry-run mode, and adds extracted knowledge as new child bullets with provenance links.

**Current Status**: Near-production 5-phase pipeline (better-pipeline branch, 85% complete)
- **Previous**: Basic 2-stage pipeline (extract → RAG match) on main branch (69% complete)
- **Current**: Comprehensive 5-phase pipeline with persistent ChromaDB vector store and hybrid-ID system
- **Progress**: 4/5 milestones complete, M5 (Testing & Refinement) in progress
- See PLAN.md for detailed status and FUTURE-STATE.md for architecture

## Development Commands

### Environment Setup

**IMPORTANT**: Always activate the virtual environment before running any commands:

```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Running Tests

```bash
# Run all tests (that don't require heavy dependencies)
pytest tests/unit/test_config.py tests/unit/test_journal.py tests/unit/test_knowledge.py tests/unit/test_parser.py tests/unit/test_graph.py tests/unit/test_preview.py tests/integration/

# Run specific test file
pytest tests/unit/test_parser.py -v

# Run with coverage
pytest tests/unit/ tests/integration/ --cov=logsqueak --cov-report=html

# Skip slow tests (tests that require sentence-transformers)
pytest -m "not slow"
```

**Note**: Some tests require `sentence-transformers` which is a large dependency. It's included in pyproject.toml but may take time to install.

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
# Build/rebuild vector store index (required before first extraction)
logsqueak index rebuild

# Check index status
logsqueak index status

# Extract from specific date (writes directly with processed:: markers)
logsqueak extract 2025-01-15

# Extract from date range
logsqueak extract 2025-01-10..2025-01-15

# Extract from today's journal
logsqueak extract

# Enable verbose logging to see detailed pipeline steps
logsqueak extract --verbose 2025-01-15
```

**Note**: The 5-phase pipeline writes directly to pages and adds `processed::` markers to journal entries for traceability. Dry-run and interactive approval modes have been removed in favor of this atomic approach.

## Architecture

### Data Flow

The extraction and integration process follows a **5-phase pipeline**:

**Phase 0 - Indexing** (run once, then incrementally):
- Parse all pages in Logseq graph into AST (LogseqOutline/LogseqBlock tree)
- Generate full-context chunks for each block (includes parent context)
- Extract or generate hybrid IDs (`id::` property OR content hash)
- Store chunks in persistent ChromaDB vector store with embeddings
- Maintain incremental cache manifest (detects additions/updates/deletions)

**Phase 1 - Knowledge Extraction**:
- Load journal entry from file (JournalEntry model enforces 2000-line limit)
- LLM identifies lasting knowledge blocks (vs temporary activity logs)
- Returns exact block text with original IDs (hybrid ID system)
- Post-processing adds parent context to each block (creates "Knowledge Packages")

**Phase 2 - Candidate Retrieval** (Enhanced RAG):
- **Semantic search**: Query ChromaDB for Top-K similar chunks (block-level)
- **Hinted search**: Parse `[[Page Links]]` from knowledge text for explicit references
- Aggregate and deduplicate results from both searches
- Group chunks by page, return list of candidate pages with target blocks

**Phase 3 - Decision & Rewording**:
- **Phase 3.1 - Decider**: For each (knowledge, candidate page) pair, LLM chooses action:
  - `IGNORE_*`: Skip (duplicate, low relevance, etc.)
  - `UPDATE`: Replace existing block content
  - `APPEND_CHILD`: Add as child to specific block
  - `APPEND_ROOT`: Add as root-level block
- **Phase 3.2 - Reworder**: For accepted knowledge, LLM rephrases content:
  - Remove journal-specific context
  - Preserve important links
  - Create evergreen, standalone block text

**Phase 4 - Execution & Cleanup**:
- Group all write operations by page (minimize file I/O)
- For each page: parse AST, apply all operations, generate UUIDs, serialize
- Use `find_block_by_id(target_id)` for precise block targeting
- **Atomic update**: For each write, immediately add `processed::` marker to journal
  - Format: `processed:: [page1](((uuid1))), [page2](((uuid2)))`
  - Links user back to integrated knowledge locations
- Ensures consistency: if page write fails, journal isn't marked as processed

### Five-Phase LLM Pattern

The system uses a **multi-phase approach** to balance accuracy, cost, and semantic quality:

- **Phase 1** (Extractor): Identifies knowledge blocks from journal context
- **Phase 3.1** (Decider): Makes structural decisions (where to integrate, what action)
- **Phase 3.2** (Reworder): Refines content for evergreen knowledge base

This pattern enables:
- Block-level semantic search (more precise than page-level)
- Hybrid search (semantic + explicit page references)
- Separate decision-making from content generation (configurable models)
- Atomic consistency between page writes and journal markers

### Key Modules

- **cli/**: Command-line interface with Click
  - `main.py`: Entry point, `extract` and `index` commands
  - `progress.py`: Progress feedback messages

- **config/**: Configuration management
  - Uses Pydantic for validation
  - Config file: `~/.config/logsqueak/config.yaml`
  - Supports configurable models for extractor, decider, and reworder

- **extraction/**: Knowledge extraction logic (Phase 1-3)
  - `extractor.py`: Main pipeline orchestrator (coordinates all phases)
  - Implements Phase 1 (extraction), Phase 2 (RAG), Phase 3 (decider + reworder)

- **integration/**: Page integration logic (Phase 4)
  - `executor.py`: Write operation execution with atomic journal updates
  - `journal_cleanup.py`: Adds `processed::` markers to journal entries
  - `writer.py`: Low-level block manipulation (UPDATE, APPEND)

- **llm/**: LLM client abstraction
  - `client.py`: Generic LLM interface with three operations:
    - `extract_knowledge()` - Phase 1
    - `decide_action()` - Phase 3.1
    - `rephrase_content()` - Phase 3.2
  - `providers/openai_compat.py`: OpenAI-compatible provider (OpenAI, Anthropic, Ollama)
  - `prompts.py`: All LLM prompt templates
  - `prompt_logger.py`: Prompt inspection system (logs to `~/.cache/logsqueak/prompts/`)

- **logseq/**: Logseq file format handling
  - `parser.py`: Parse outline markdown into LogseqOutline/LogseqBlock tree
    - Extracts `id::` properties for hybrid ID system
    - `find_block_by_id()` for precise block targeting
  - `renderer.py`: Render outline back to markdown (preserves property order)
  - `context.py`: Full-context generation and content hashing utilities
  - `graph.py`: Graph path operations (journals/, pages/)

- **models/**: Data models (all use dataclasses)
  - `journal.py`: JournalEntry (2000-line limit enforcement)
  - `knowledge.py`: KnowledgePackage (extracted knowledge with hybrid IDs)
  - `decisions.py`: DecisionResult, RephrasedContent (Phase 3 outputs)
  - `operations.py`: WriteOperation (Phase 4 operations)

- **rag/**: Semantic search infrastructure (Phase 0 & 2)
  - `vector_store.py`: VectorStore abstraction with ChromaDBStore implementation
  - `indexer.py`: Incremental index builder (Phase 0)
  - `chunker.py`: Block-level chunk generation with full-context text
  - `manifest.py`: Cache manifest for incremental updates
  - `embedder.py`: Embedding utilities with sentence-transformers

## Project Constitution

**See `.specify/memory/constitution.md` for the complete project constitution.**

Key principles from the constitution:

### I. Proof-of-Concept First
- Prioritize working software over perfection
- Ship iteratively, demonstrate feasibility
- No backwards compatibility guarantees (POC stage)

### II. Non-Destructive Operations (NON-NEGOTIABLE)
- All operations are traceable via `processed::` markers in journal entries
- UPDATE operations replace block content but preserve structure and IDs
- APPEND operations add new blocks without modifying existing content
- Every write generates a unique `id::` property for future tracking
- Journal entries are atomically marked with links to integrated knowledge

### III. Simplicity and Transparency
- Prefer file-based I/O over databases
- Use JSON for structured LLM outputs
- Show the user what the LLM is doing
- Avoid premature abstraction

### Commit Message Requirements
**IMPORTANT**: All AI-assisted commits MUST include:
```
Assisted-by: Claude Code
```

## Critical Design Principles

### 1. Property Order Preservation (FR-008) - NON-NEGOTIABLE

**NEVER reorder block properties**. Insertion order is sacred in Logseq. When parsing and rendering:
- Preserve exact property order from source
- Use Python 3.7+ dict ordering (insertion order guaranteed)
- Round-trip tests verify no reordering occurs

### 2. Hybrid ID System

Every block has a stable identifier for targeting and tracking:
- **Explicit IDs**: `id::` property (UUID format) if present
- **Implicit IDs**: Content hash (MD5) if no `id::` property
- Writer generates new UUIDs for all integrated blocks
- Parser extracts existing `id::` properties during round-trip
- Enables precise block targeting via `find_block_by_id()`

### 3. Journal Provenance via processed:: Markers

Every integration creates bidirectional links:
- **Forward link**: Journal block gets `processed::` marker child block
- **Format**: `processed:: [page1](((uuid1))), [page2](((uuid2)))`
- **Backward link**: Each integrated block has `id::` property
- Links use markdown syntax with block references: `[page](((uuid)))`
- Atomic consistency: journal marked only when page write succeeds

### 4. Persistent Vector Store with Incremental Indexing

ChromaDB-based semantic search with smart caching:
- Store location: `~/.cache/logsqueak/chroma/`
- Manifest tracks mtime for each page
- Detects additions, updates (mtime changed), and deletions
- Block-level embeddings with full-context text
- Performance: Initial index ~20-30s for 500 pages, incremental updates <2s

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
- Properties detected by `key:: value` pattern
- Continuation lines are lines at same/deeper indent without bullet
- Frontmatter: lines before first bullet (page-level content)

### Renderer Behavior

- `LogseqOutline.render()`: Converts tree back to markdown string
- Preserves exact property order
- Preserves continuation line formatting
- Uses detected indentation (default: 2 spaces)

## Prompt Inspection System

All LLM prompts are **automatically logged** to timestamped files in `~/.cache/logsqueak/prompts/`:

```bash
# Logs automatically created during extraction
logsqueak extract 2025-01-15  # Creates YYYYMMDD_HHMMSS.log

# Custom log file
logsqueak extract --prompt-log-file /path/to/file.log 2025-01-15
```

Log format includes:
- Sequential interaction numbers
- Stage identifiers ("extraction" or "page_selection")
- Full message history (system + user prompts)
- LLM responses with parsed content
- Error details and timestamps

## Testing Strategy

### Test Organization

- `tests/unit/`: Unit tests for all data models, parser/renderer, graph operations
- `tests/integration/`: Integration tests for round-trip parsing, full extraction workflow
- `tests/fixtures/`: Test data (sample journal entries, Logseq pages)

### Key Test Areas

- **Parser/Renderer Round-trip**: Ensures no data loss or reordering
- **Property Order Preservation**: Critical for FR-008 compliance
- **2000-line Limit**: JournalEntry model enforcement
- **Duplicate Detection**: Content hashing tests
- **Provenance Links**: 100% coverage verification
- **LLM Integration**: Contract tests for OpenAI-compatible provider

## Configuration

Configuration file: `~/.config/logsqueak/config.yaml`

```yaml
llm:
  endpoint: https://api.openai.com/v1
  api_key: sk-your-api-key-here
  model: gpt-4-turbo-preview  # Default model for all phases

  # Optional: Configure separate models for different phases
  # extractor_model: gpt-4-turbo-preview  # Phase 1 (defaults to model)
  # decider_model: gpt-3.5-turbo         # Phase 3.1 - faster/cheaper for decisions
  # reworder_model: gpt-4-turbo-preview  # Phase 3.2 - high quality for content

logseq:
  graph_path: ~/Documents/logseq-graph

rag:
  top_k: 10  # Number of similar chunks to retrieve (default: 10)
```

For Ollama (local LLM):
```yaml
llm:
  endpoint: http://localhost:11434/v1
  api_key: ollama  # Any non-empty value
  model: llama3.2
  decider_model: llama3.2  # Use same model for all phases
  reworder_model: llama3.2
  num_ctx: 32768  # Context window size (controls VRAM usage)
                  # Typical values: 8192, 16384, 32768, 65536

logseq:
  graph_path: ~/Documents/logseq-graph

rag:
  top_k: 5  # Fewer candidates for local models
```

**Ollama-specific configuration:**

- **`num_ctx`**: Context window size for Ollama models (optional)
  - **Automatically uses Ollama's native `/api/chat` endpoint** when `num_ctx` is configured
  - Controls how much GPU VRAM is used
  - Larger values allow longer prompts but use more VRAM
  - Typical values:
    - `8192`: ~4-6 GB VRAM (conservative)
    - `16384`: ~8-10 GB VRAM (balanced)
    - `32768`: ~14-18 GB VRAM (recommended for 24GB+ GPUs)
    - `65536`: ~28-32 GB VRAM (for 40GB+ GPUs)
  - If not specified, uses Ollama's model default (usually 2048-4096)
  - Set this to maximize your GPU utilization
  - **Note**: The model must be unloaded/reloaded for `num_ctx` changes to take effect
    - On Ollama server: `curl http://localhost:11434/api/generate -d '{"model": "your-model", "keep_alive": 0}'`

### RAG Configuration

**`rag.top_k`**: Number of similar chunks to retrieve from vector store (default: 10)

The Phase 2 candidate retrieval combines:
- **Semantic search**: Query ChromaDB for top_k most similar chunks
- **Hinted search**: Parse `[[Page Links]]` from knowledge text

Results are aggregated, deduplicated, and grouped by page. The Decider (Phase 3.1) then evaluates each (knowledge, candidate page) pair to determine the best action.

**Tuning recommendations:**
- **5-10 candidates**: Good balance for most use cases
- **10-20 candidates**: More thorough search, higher LLM cost (Decider evaluates all pairs)
- **3-5 candidates**: Faster, cheaper, may miss some relevant pages

Example verbose output with `--verbose`:
```bash
logsqueak extract --verbose 2025-01-15
```

Shows:
- Number of candidates retrieved (semantic + hinted)
- Decider decisions for each (knowledge, page) pair
- Final write operations

## Project Structure Conventions

- All source code in `src/logsqueak/`
- All tests in `tests/` (pytest discovers via `pyproject.toml` pythonpath)
- Feature specs in `specs/001-knowledge-extraction/`
- Entry point: `logsqueak.cli.main:main` (defined in pyproject.toml)

## Implementation Status

**Main branch**: Original 2-stage pipeline (69% complete, archived)
- Basic extract → RAG match workflow
- Session-based embedding cache
- See `specs/001-knowledge-extraction/tasks.md`

**Current branch: `better-pipeline`** - Advanced 5-phase pipeline (85% complete)
- ✅ **Milestone 1**: Hybrid-ID Foundation (5/5 tasks complete)
  - `id::` property extraction and generation
  - Full-context chunk generation with content hashing
  - Block lookup via `find_block_by_id()`
- ✅ **Milestone 2**: Persistent Vector Store (7/7 tasks complete)
  - ChromaDB integration with block-level indexing
  - Incremental index builder with manifest tracking
  - CLI commands: `logsqueak index rebuild`, `logsqueak index status`
- ✅ **Milestone 3**: Block-Level Targeting (4/4 tasks complete)
  - Precise block targeting via hybrid IDs
  - UPDATE, APPEND_CHILD, APPEND_ROOT operations
- ✅ **Milestone 4**: Multi-Stage LLM Pipeline (12/12 tasks complete)
  - Phase 1: Knowledge extraction with hybrid IDs
  - Phase 2: Enhanced RAG (semantic + hinted search)
  - Phase 3.1: Decider (action selection)
  - Phase 3.2: Reworder (content refinement)
  - Phase 4: Atomic execution with journal cleanup
  - Configurable models for each phase
- ⏳ **Milestone 5**: Testing & Refinement (3/7 tasks complete)
  - ✅ Dry-run mode removed (incompatible with new architecture)
  - ✅ Integration test for full pipeline (5 tests)
  - ✅ Error recovery tests (17 comprehensive tests)
  - ⏳ Performance benchmarks
  - ⏳ Documentation updates
  - ⏳ CLI polish

See `PLAN.md` for detailed status and `FUTURE-STATE.md` for architecture.

## Python Version & Dependencies

- **Python**: 3.11+ (required)
- **Key dependencies**:
  - httpx (LLM client)
  - markdown-it-py (parsing)
  - pydantic (validation)
  - click (CLI)
  - chromadb (persistent vector store - NEW in better-pipeline)
  - sentence-transformers (RAG embeddings - large dependency)
  - numpy (vector operations)
  - tiktoken (token counting for LLM prompts)

## Key Features of the 5-Phase Pipeline

### 1. Hybrid ID System for Precise Targeting
Every block in your Logseq graph gets a stable identifier:
- Existing blocks with `id::` properties use those UUIDs
- Blocks without IDs get content-based hashes (deterministic)
- New integrated blocks automatically receive unique UUIDs
- Enables surgical UPDATE operations without side effects

### 2. Persistent Vector Store with Incremental Updates
ChromaDB-based block-level semantic search:
- Initial index: Parse entire graph, generate embeddings (~20-30s for 500 pages)
- Incremental updates: Only process changed/new pages (~2s)
- Manifest-based tracking: Detects additions, updates (mtime), and deletions
- Survives across sessions: No re-indexing on every extraction

### 3. Enhanced RAG with Hybrid Search
Phase 2 combines two search strategies:
- **Semantic**: Vector similarity search across all blocks
- **Hinted**: Explicit `[[Page Name]]` references in journal
- Aggregates both results for comprehensive candidate list

### 4. Three-Model LLM Architecture
Separates concerns for cost/quality optimization:
- **Extractor** (Phase 1): Identifies knowledge vs activity logs
- **Decider** (Phase 3.1): Makes structural decisions (can use cheaper model)
- **Reworder** (Phase 3.2): Refines content quality (use best model)

### 5. Atomic Journal Updates with Bidirectional Links
Every integration creates traceable connections:
- Page gets new block with `id::` UUID
- Journal gets `processed::` marker with block reference
- If write fails, journal is NOT marked (consistency)
- Click block refs in journal to jump to integrated knowledge

### 6. Full CRUD Operations
Beyond simple append-only:
- **UPDATE**: Replace existing block content (preserves ID and structure)
- **APPEND_CHILD**: Add child to specific target block
- **APPEND_ROOT**: Add root-level block to page
- All operations preserve property order and formatting

## License

GPLv3 - All code is licensed under GPLv3 regardless of authorship method (including AI-assisted development).
