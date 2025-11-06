# Logsqueak

Turn your Logseq journal chaos into organized knowledge. Interactive TUI (Text User Interface) for extracting lasting insights from journal entries using LLM-powered analysis.

**Status**: 🚧 **Phase 4 Complete** - Content Editing TUI working, Integration Review next (Phase 5)

## Overview

Logsqueak provides an **interactive 3-phase workflow** for knowledge extraction:

**Phase 1 - Block Selection**:
- View journal blocks in hierarchical tree
- LLM streams classification results (knowledge vs. activity)
- Manually select/deselect blocks with keyboard controls
- Background: Page indexing for semantic search

**Phase 2 - Content Editing**:
- Review selected blocks with full context
- LLM generates reworded versions (removes temporal context)
- Accept LLM suggestions or edit manually
- Background: RAG search finds candidate target pages

**Phase 3 - Integration Review**:
- LLM suggests where to integrate each knowledge block
- Preview target pages with insertion points
- Accept/skip decisions (writes immediately on accept)
- Atomic consistency: Journal marked only when page write succeeds

All operations are **keyboard-driven** with vim-style navigation and **streaming LLM results** for responsive feedback.

## Features

### ✅ Implemented
- **Logseq Parser**: Production-ready markdown parser with property order preservation
- **Data Models**: Pydantic models for config, block state, LLM chunks, integration decisions
- **LLM Client**: Async NDJSON streaming with retry logic and structured logging
- **File Monitor**: Git-friendly mtime tracking for concurrent modification detection
- **Configuration**: Lazy validation with helpful error messages (mode 600 permission check)
- **Utilities**: Structured logging (structlog), deterministic UUID generation
- **Phase 1 TUI**: Block selection with tree navigation, LLM streaming, manual selection
- **Phase 2 TUI**: Content editing with three-panel layout, LLM rewording, manual editing
- **RAG Services**: PageIndexer and RAGSearch with lazy SentenceTransformer loading
- **Journal Loader**: Load and parse journal entries with date/range support
- **CLI Integration**: Working `logsqueak extract` command

### 🚧 In Progress
- **Background Workers**: LLM rewording and RAG search workers for Phase 2

### ⏳ Planned
- **Phase 3 Screen**: Integration review with target page preview
- **File Operations**: Atomic two-phase writes with provenance markers

### ⏳ Planned (After Phase 2-3 Complete)
- **Application Integration**: Wire up all 3 phases in TUI app
- **Edge Cases**: Error handling, concurrent modification detection
- **Polish**: Comprehensive logging, documentation, manual validation

### Key Design Principles
- **Non-Destructive**: All integrations traceable via `processed::` markers
- **Property Order Preservation**: NEVER reorder (insertion order sacred)
- **Keyboard-Driven**: Vim-style navigation (j/k), no mouse required
- **Streaming LLM**: Real-time updates as results arrive
- **Explicit Control**: Users approve all integrations (no auto-write)

## Installation

### Requirements

- Python 3.11 or later
- A Logseq graph with journal entries
- Access to an LLM API (OpenAI, Anthropic, or local Ollama)

### Setup

**IMPORTANT**: The following steps install all runtime and development dependencies, including large packages like sentence-transformers (~500MB) and ChromaDB.

**Option 1: Automated Setup (Recommended)**

```bash
# Clone repository
git clone https://github.com/twaugh/logsqueak.git
cd logsqueak

# Run setup script (creates venv, installs dependencies)
./setup-dev.sh
```

**Option 2: Manual Setup**

```bash
# Clone repository
git clone https://github.com/twaugh/logsqueak.git
cd logsqueak

# Create virtual environment (REQUIRED - do not skip this step!)
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip in the venv
pip install --upgrade pip

# Install logsqueak in editable mode with all dependencies
# This installs: textual, httpx, pydantic, click, structlog, chromadb,
# sentence-transformers (~500MB), and all dev dependencies
pip install -e .

# Also install the parser library in editable mode
pip install -e src/logseq-outline-parser/

# Verify installation
pytest -v
```

**What gets installed:**
- Runtime dependencies: textual, httpx, pydantic, click, pyyaml, structlog, chromadb, sentence-transformers, markdown-it-py
- Dev dependencies: pytest, pytest-asyncio, pytest-textual-snapshot, pytest-cov, black, ruff, mypy

### Configuration

Create `~/.config/logsqueak/config.yaml` with mode 600 permissions:

```yaml
llm:
  endpoint: https://api.openai.com/v1  # Or http://localhost:11434/v1 for Ollama
  api_key: sk-your-api-key-here
  model: gpt-4-turbo-preview

logseq:
  graph_path: ~/Documents/logseq-graph

rag:
  top_k: 10  # Number of similar blocks to retrieve per search (default: 10)
```

**Optional settings:**
```yaml
llm:
  num_ctx: 32768  # Ollama context window size (controls VRAM usage)
```

**Configuration behavior:**
- File must exist before first run (not auto-created)
- File permissions must be mode 600 (checked on load)
- Lazy validation: Settings validated only when first accessed
- Helpful error messages with example YAML for missing/invalid config

Set correct permissions:
```bash
chmod 600 ~/.config/logsqueak/config.yaml
```

## Usage

### Current Status: Phase 1 TUI Working

Launch the interactive block selection TUI:

```bash
# Extract from today's journal
logsqueak extract

# Extract from specific date
logsqueak extract 2025-01-15

# Extract from date range
logsqueak extract 2025-01-10..2025-01-15
```

**Phase 1 - Block Selection** (Currently Working):
- Navigate blocks with `j`/`k` (vim-style) or arrow keys
- Watch LLM classification stream in real-time (knowledge vs. activity)
- Select/deselect blocks with `Space`
- Accept all LLM suggestions with `a`
- Clear all selections with `c`, reset to LLM suggestions with `r`
- Jump to next/prev knowledge block with `Shift+j`/`Shift+k`
- View block content with full context in bottom panel
- Press `n` to proceed (Phase 2-3 not yet implemented)
- Press `q` to quit

**Future workflow** (when Phase 2-3 complete):
1. **Phase 1**: Select knowledge blocks (WORKING NOW)
2. Press `n` to proceed to Phase 2
3. **Phase 2**: Review/edit content, accept LLM rewording with `a` (not yet implemented)
4. Press `n` to proceed to Phase 3
5. **Phase 3**: Review integration decisions, accept with `y` (not yet implemented)
6. Writes complete - journal marked with `processed::` markers

All keyboard-driven, no mouse required.

## Project Structure

```
logsqueak/
├── src/
│   ├── logsqueak/                 # Main application
│   │   ├── models/                # Pydantic data models ✅
│   │   ├── services/              # LLMClient, FileMonitor, PageIndexer, RAGSearch ✅
│   │   ├── tui/                   # TUI screens & widgets (Phase 1-2 ✅, Phase 3 pending)
│   │   ├── utils/                 # Logging, UUID generation ✅
│   │   ├── cli.py                 # CLI entry point ✅
│   │   └── config.py              # ConfigManager ✅
│   └── logseq-outline-parser/     # Parser library ✅
├── tests/                         # Test suite
│   ├── unit/                      # Unit tests ✅
│   ├── integration/               # Integration tests ✅
│   └── ui/                        # UI tests for Phase 1 and Phase 2 ✅
├── specs/
│   ├── 001-logsqueak/             # Original 5-phase pipeline spec
│   └── 002-logsqueak-spec/        # Interactive TUI spec (CURRENT)
│       ├── spec.md                # Feature specification
│       ├── tasks.md               # Phase 1-4 complete ✅ (T001-T070)
│       └── contracts/             # Service interfaces, data models
└── pyproject.toml                 # Dependencies and config
```

## Development

### Running Tests

**IMPORTANT**: Always activate the virtual environment first!

```bash
# Activate venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run all tests
pytest -v

# Run only parser tests
pytest src/logseq-outline-parser/tests/ -v

# Run only main app tests
pytest tests/unit/ tests/integration/ -v

# Run with coverage
pytest --cov=logsqueak --cov=logseq_outline --cov-report=html -v

# Verify foundational tests (Phase 2 checkpoint)
pytest tests/unit/ tests/integration/test_config*.py tests/integration/test_llm*.py -v
```

### Code Quality

**IMPORTANT**: Always activate the virtual environment first!

```bash
# Activate venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type checking
mypy src/
```

## Implementation Status

**✅ Phase 1-4 Complete (Tasks T001-T070)**

Foundation, Block Selection, and Content Editing TUI ready:

- ✅ **Logseq Outline Parser** (Production-ready library)
  - Non-destructive parsing & rendering with property order preservation
  - Hybrid ID system (explicit `id::` OR content hash)
  - Full-context generation for semantic search
  - Frontmatter support, round-trip safety tests

- ✅ **Data Models** (Pydantic validation)
  - Config models (LLMConfig, LogseqConfig, RAGConfig, Config)
  - Block state models (BlockState, EditedContent, IntegrationDecision)
  - LLM chunk models (KnowledgeClassificationChunk, ContentRewordingChunk, IntegrationDecisionChunk)
  - Background task state enum

- ✅ **Services Layer**
  - LLMClient: Async NDJSON streaming, retry logic, structured logging
  - FileMonitor: Mtime tracking with `!=` comparison (git-friendly)
  - PageIndexer: ChromaDB vector indexing with lazy-loaded SentenceTransformer
  - RAGSearch: Semantic search with explicit link boosting and page-level ranking

- ✅ **Configuration & CLI**
  - ConfigManager with lazy validation
  - Mode 600 permission checking
  - Helpful error messages with example YAML
  - Click-based CLI entry point (placeholder implementation)

- ✅ **Utilities**
  - Structured logging (structlog) to `~/.cache/logsqueak/logs/logsqueak.log`
  - Deterministic UUID v5 with Logsqueak-specific namespace

- ✅ **Test Coverage**
  - Comprehensive test suite
  - Unit tests: All models, services, utilities, RAG services
  - Integration tests: Config loading, LLM NDJSON streaming, RAG pipeline
  - UI tests: Phase 1 and Phase 2 TUI with snapshot testing
  - Proper async fixtures with `@pytest_asyncio.fixture`

- ✅ **Phase 1 Block Selection TUI**
  - BlockTree widget: Hierarchical block display with expand/collapse
  - StatusPanel widget: Background task progress tracking
  - MarkdownViewer widget: Block preview with full context
  - Keyboard navigation: j/k, Space, Shift+j/k, a/c/r keys
  - LLM streaming: Real-time classification updates
  - Journal loader: Date/range parsing and multi-file support
  - CLI integration: `logsqueak extract` command

- ✅ **Phase 2 Content Editing TUI**
  - ContentEditor widget: Multi-line text editor with focus/unfocus visual indication
  - Phase2Screen: Vertical three-panel layout (original, LLM reworded, current editable)
  - Keyboard controls: j/k navigation with auto-save, Tab focus, 'a' accept, 'r' revert
  - RAG search blocking on 'n' key (waits for completion)
  - Background worker stubs ready for LLM/RAG integration

**🚧 Next Steps**

Phase 5: User Story 3 (Integration Review TUI) using Test-Driven Development

## Architecture

### Interactive 3-Phase TUI Workflow

**Phase 1 - Block Selection Screen** (User Story 1, T032-T049):
- BlockTree widget: Hierarchical display of journal blocks
- LLM worker: Streams classification results (knowledge vs. activity)
- StatusPanel widget: Shows background task progress (LLM classification, page indexing)
- User actions: Navigate (j/k), select/deselect (Space), accept all (a), proceed (n)

**Phase 2 - Content Editing Screen** (User Story 2, T050-T070):
- ContentEditor widget: Three-panel view (original, LLM reworded, current editable)
- LLM worker: Streams rewording suggestions
- RAG worker: Finds candidate target pages in background
- User actions: Navigate (j/k with auto-save), accept LLM (a), revert (r), Tab to focus editor, proceed (n) when RAG complete

**Phase 3 - Integration Review Screen** (User Story 3, T071-T096):
- DecisionList widget: Batched decisions per knowledge block
- TargetPagePreview widget: Shows target page with green bar at insertion point
- LLM worker: Streams integration decisions
- File operations: Atomic two-phase writes with provenance markers
- User actions: Navigate decisions (j/k), accept (y), skip (s), next block (n), batch accept (a)

### Key Design Principles

1. **Property Order Preservation**: NEVER reorder (insertion order sacred)
2. **Non-Destructive Operations**: All integrations traceable via `processed::` markers
3. **Atomic Consistency**: Journal marked only when page write succeeds
4. **Keyboard-Driven**: Vim-style navigation (j/k), context-sensitive shortcuts
5. **Streaming LLM**: Real-time UI updates as results arrive
6. **Explicit Control**: Users approve all integrations (no auto-write)
7. **Test-Driven Development**: Write failing tests → Implement → Tests pass → Manual verify

## License

GPLv3 - See [LICENSE](LICENSE) file.

This project uses AI assistance (Claude Code) in development. All code is licensed under GPLv3 regardless of authorship method.

## Contributing

See [CLAUDE.md](CLAUDE.md) for developer documentation, architecture details, and implementation guidance.

Key resources:
- **specs/002-logsqueak-spec/spec.md** - Complete interactive TUI feature specification
- **specs/002-logsqueak-spec/tasks.md** - Implementation tasks (Phase 1-4 complete ✅)
- **specs/002-logsqueak-spec/contracts/** - Service interfaces and data models
- **CLAUDE.md** - Developer guide with parser API, testing, and next steps

## Development Workflow

**Test-Driven Development** (Phase 3+ approach):
1. Write UI tests FIRST using Textual pilot - tests should FAIL
2. Verify tests fail: `pytest tests/ui/test_phase1_*.py -v`
3. Implement widgets and screens
4. Run tests again - should NOW PASS
5. Manual verification in TUI before proceeding

**Current Status**: Phase 4 complete - Ready to begin Phase 5 (User Story 3 - Integration Review TUI)

## Roadmap

**Completed** (Phase 1-4):
- ✅ Project structure and dependencies
- ✅ All data models with Pydantic validation
- ✅ LLM client with NDJSON streaming
- ✅ Configuration management with lazy validation
- ✅ File monitoring for concurrent edits
- ✅ Journal loading with date/range parsing
- ✅ RAG services (PageIndexer, RAGSearch) with lazy SentenceTransformer loading
- ✅ Comprehensive test suite
- ✅ User Story 1: Block Selection TUI
- ✅ User Story 2: Content Editing TUI

**Next** (Phase 5):
- 🚧 User Story 3: Integration Review TUI

**Future** (Phase 6-8):
- ⏳ Application integration (wire up all 3 phases)
- ⏳ Background workers for LLM/RAG integration
- ⏳ Edge case handling and polish
