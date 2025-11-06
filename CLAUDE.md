# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Logsqueak** is a TUI (Text User Interface) application for extracting lasting knowledge from Logseq journal entries using LLM-powered analysis. Users interactively review, refine, and integrate knowledge blocks into their Logseq knowledge base.

**Current Status**: Foundation complete, ready for TUI implementation
- ✅ **Implemented**: `logseq-outline-parser` library (robust Logseq markdown parsing)
- ✅ **Implemented**: Foundational infrastructure (models, services, CLI, config, tests)
- 🚧 **In Progress**: Interactive TUI screens (Phase 3+ in tasks.md)
- ⏳ **Planned**: RAG semantic search (PageIndexer, RAGSearch services)

## Project Structure

```
logsqueak/
├── src/
│   ├── logsqueak/                 # Main application (FOUNDATION COMPLETE)
│   │   ├── models/                # Pydantic data models (config, block state, LLM chunks)
│   │   ├── services/              # LLMClient, FileMonitor (RAG services planned)
│   │   ├── tui/                   # TUI structure (screens/widgets to be implemented)
│   │   ├── utils/                 # Logging, UUID generation
│   │   ├── cli.py                 # Click-based CLI entry point
│   │   └── config.py              # ConfigManager with lazy validation
│   └── logseq-outline-parser/     # Logseq markdown parser library (COMPLETE)
│       └── src/logseq_outline/
│           ├── parser.py          # Core parsing/rendering logic
│           ├── context.py         # Full-context generation & content hashing
│           └── graph.py           # Graph path utilities
├── tests/                         # Test suite (125 passed, 1 skipped)
│   ├── unit/                      # Unit tests for models, services, utils
│   ├── integration/               # Integration tests for config, LLM streaming
│   └── ui/                        # UI tests (to be implemented with Textual pilot)
├── specs/
│   ├── 001-logsqueak/             # Original knowledge extraction spec
│   └── 002-logsqueak-spec/        # Interactive TUI feature spec (CURRENT)
│       ├── spec.md                # Complete feature specification
│       ├── tasks.md               # Phase 1-2 complete (T001-T031 ✅)
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

# Parse Logseq markdown
outline = LogseqOutline.parse(markdown_text)

# Access blocks
for block in outline.blocks:
    print(block.get_full_content())
    print(f"ID: {block.block_id}")  # From id:: property or None

# Render back to markdown
output = outline.render()  # Preserves exact formatting
```

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

This is used for:
- Content-based hashing (hybrid ID system)
- Semantic search embeddings (planned)
- LLM context windows (planned)

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
    def parse(cls, text: str) -> LogseqOutline:
        """Parse Logseq markdown into outline tree."""

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
- **llm_client.py**: LLMClient with async NDJSON streaming, retry logic, structured logging
- **file_monitor.py**: FileMonitor with mtime tracking using `!=` comparison (git-friendly)

Planned services (not yet implemented):
- **page_indexer.py**: ChromaDB vector indexing (required for US2)
- **rag_search.py**: Semantic search with explicit link boosting (required for US2)
- **file_operations.py**: Atomic two-phase writes with provenance markers (required for US3)

### Configuration & CLI

- **cli.py**: Click-based CLI with `extract` command (placeholder implementation)
- **config.py**: ConfigManager with lazy validation, helpful error messages, permission checks (mode 600)
- Config file: `~/.config/logsqueak/config.yaml` (user must create before first run)

### Utilities

- **logging.py**: Structured logging (structlog) to `~/.cache/logsqueak/logs/logsqueak.log`
- **ids.py**: Deterministic UUID v5 generation with Logsqueak-specific namespace (`32e497fc-abf0-4d71-8cff-e302eb3e2bb0`)

### Test Coverage

**125 tests passing, 1 skipped** across:
- **Unit tests** (`tests/unit/`): Models, config, LLM client, file monitor, utilities
- **Integration tests** (`tests/integration/`): Config loading, LLM NDJSON streaming workflows

All async mocking properly implements context managers. FileMonitor uses `!=` for mtime comparison to handle git reverts.

## Interactive TUI Application (IN PROGRESS)

See `specs/002-logsqueak-spec/spec.md` for the complete feature specification.

### High-Level Architecture

The TUI application will have **3 interactive phases**:

**Phase 1: Block Selection**
- Display journal blocks in hierarchical tree view
- LLM classifies blocks as "knowledge" vs "activity logs" (streaming results)
- User manually selects/deselects blocks for extraction
- Background: Page indexing for semantic search

**Phase 2: Content Editing**
- Display selected knowledge blocks with full context
- LLM generates reworded versions (removes temporal context)
- User can accept LLM version, edit manually, or revert to original
- Background: RAG search for candidate pages

**Phase 3: Integration Decisions**
- LLM suggests where to integrate each knowledge block (streaming decisions)
- User reviews preview showing new content in target page context
- Accept/skip each decision (writes immediately on accept)
- Atomic provenance: Add `processed::` marker to journal only after successful write

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

- All integrations are traceable via `processed::` markers in journal entries
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

## Development Commands

### Environment Setup

**IMPORTANT**: Always activate the virtual environment before running any commands:

```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Running Tests

```bash
# Run all tests (125 passed, 1 skipped)
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
# CLI entry point exists (placeholder implementation)
logsqueak extract                     # Placeholder - shows message
logsqueak extract 2025-01-15          # Placeholder - shows message
logsqueak extract 2025-01-10..2025-01-15  # Placeholder - shows message

# Future: Full TUI workflow (Phase 6 - not yet implemented)
# Will launch interactive TUI with Phase 1 → Phase 2 → Phase 3
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
  - `chromadb>=0.4.0` - Vector store for RAG (to be used)
  - `sentence-transformers>=2.2.0` - Embeddings (to be used, large dependency)
  - `markdown-it-py>=3.0.0` - Markdown rendering (to be used)
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

- All operations traceable via `processed::` markers in journal entries
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
- Properties detected by `key:: value` pattern
- Continuation lines: Lines at same/deeper indent without bullet
- Frontmatter: Lines before first bullet (page-level content)

### Renderer Behavior

- `LogseqOutline.render()`: Converts tree back to markdown string
- Preserves exact property order (NON-NEGOTIABLE)
- Preserves continuation line formatting
- Uses detected indentation (default: 2 spaces)

## Active Feature Development

**Current feature**: Interactive TUI for knowledge extraction (002-logsqueak-spec)

**Branch**: `002-logsqueak-spec`

**Spec location**: `specs/002-logsqueak-spec/spec.md`

**Implementation status** (see tasks.md for details):
- ✅ **Phase 1: Setup** (T001-T009) - Complete
- ✅ **Phase 2: Foundational** (T010-T031) - Complete and tested
- 🚧 **Phase 3: User Story 1** (T032-T049) - Block Selection TUI (next to implement)
- ⏳ **Phase 4: User Story 2** (T050-T070) - Content Editing TUI
- ⏳ **Phase 5: User Story 3** (T071-T096) - Integration Review TUI
- ⏳ **Phase 6: Application Integration** (T097-T106) - Wire up all phases
- ⏳ **Phase 7: Edge Cases** (T107-T117) - Error handling polish
- ⏳ **Phase 8: Polish & Documentation** (T118-T130) - Final validation

**Next steps** (Phase 3 - User Story 1):
1. Write UI tests FIRST using Textual pilot (T032-T037) - tests should FAIL
2. Verify tests fail with `pytest tests/ui/test_phase1_*.py -v`
3. Implement BlockTree, StatusPanel, MarkdownViewer widgets (T038-T040)
4. Implement Phase1Screen with keyboard controls and LLM streaming (T041-T048)
5. Run tests again - should NOW PASS (T049)
6. Manually test TUI: navigate blocks, see LLM streaming, select blocks

**When implementing**:
- Follow TDD: Write failing tests → Implement → Tests pass → Manual verify
- Use Textual pilot for all UI tests (automated keyboard/mouse simulation)
- Test each phase independently before proceeding
- Property order preservation is NON-NEGOTIABLE
- Get user verification at each checkpoint before continuing

## License

GPLv3 - All code is licensed under GPLv3 regardless of authorship method (including AI-assisted development).

## Recent Changes

- 2025-11-06: **Phase 1-2 Complete** - Foundational infrastructure implemented and tested (T001-T031)
  - All data models, services, CLI, config, and utilities complete
  - 125 tests passing (unit + integration)
  - Fixed FileMonitor to use `!=` comparison (git-friendly)
  - Generated Logsqueak-specific UUID namespace
  - All async tests properly mock context managers
- 2025-11-04: Created hybrid CLAUDE.md reflecting current implementation (parser only) + planned TUI feature
- 2025-11-04: Completed clarification session for 002-logsqueak-spec (configuration management, validation strategy)

## Active Technologies
- Python 3.11+ (002-logsqueak-spec)
- File-based (Logseq markdown files) + ChromaDB (vector embeddings) (002-logsqueak-spec)
