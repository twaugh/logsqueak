# Logsqueak

Turn your Logseq journal chaos into organized knowledge. Advanced 5-phase LLM pipeline with persistent vector store automatically extracts insights and files them where they belong.

**Status**: ✅ Production Ready - All core features complete and tested

## Overview

Logsqueak implements a **5-phase pipeline** for knowledge management:

**Phase 0 - Indexing**: Parse your entire Logseq graph, generate block-level embeddings, store in persistent ChromaDB vector store with incremental updates

**Phase 1 - Extraction**: LLM identifies lasting knowledge from journal entries (vs. temporary activity logs), returns exact block text with hybrid IDs

**Phase 2 - Enhanced RAG**: Hybrid search combining semantic similarity + explicit `[[Page Links]]`, aggregates block-level candidates

**Phase 3 - Decision & Rewording**: Decider LLM chooses action (UPDATE/APPEND/IGNORE), Reworder LLM refines content for evergreen knowledge base

**Phase 4 - Atomic Execution**: Apply operations, generate UUIDs, add `processed::` markers to journal with bidirectional block references

## Features

### Core Capabilities
- **5-Phase LLM Pipeline**: Extractor → RAG → Decider → Reworder → Executor
- **Persistent Vector Store**: ChromaDB with incremental indexing (survives across sessions)
- **Hybrid ID System**: Stable block identifiers (`id::` property OR content hash)
- **Block-Level Targeting**: Precise UPDATE/APPEND operations via hybrid IDs
- **Enhanced RAG**: Semantic search + hinted search (page links in journal)
- **Configurable Models**: Use different models for extraction, decisions, and rewording
- **Atomic Journal Updates**: Bidirectional links with `processed::` markers

### Non-Destructive Operations
- **UPDATE**: Replaces block content while preserving structure and ID
- **APPEND_CHILD**: Adds child to specific block
- **APPEND_ROOT**: Adds root-level block to page
- **Property Order Preservation**: NEVER reorders block properties (FR-008)
- **Traceability**: Every integration tracked with block references in journal

## Installation

### Requirements

- Python 3.11 or later
- A Logseq graph with journal entries
- Access to an LLM API (OpenAI, Anthropic, or local Ollama)

### Setup

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

# Install logsqueak in editable mode (installs all dependencies including ChromaDB)
pip install -e .

# Note: Tests run directly from source
# Tests use: pytest (which reads pythonpath from pyproject.toml)
```

### Configuration

Create `~/.config/logsqueak/config.yaml`:

```yaml
llm:
  endpoint: https://api.openai.com/v1
  api_key: sk-your-api-key-here
  model: gpt-4-turbo-preview  # Default model for all phases

  # Optional: Configure separate models for different phases
  # extractor_model: gpt-4-turbo-preview  # Phase 1 (defaults to model)
  # decider_model: gpt-3.5-turbo         # Phase 3.1 - faster/cheaper
  # reworder_model: gpt-4-turbo-preview  # Phase 3.2 - high quality

logseq:
  graph_path: ~/Documents/logseq-graph

rag:
  top_k: 10  # Number of similar chunks to retrieve (default: 10)
```

**For Ollama (local)**:

```yaml
llm:
  endpoint: http://localhost:11434/v1
  api_key: ollama  # Any non-empty value
  model: llama3.2
  decider_model: llama3.2  # Use same model for all phases
  reworder_model: llama3.2

logseq:
  graph_path: ~/Documents/logseq-graph

rag:
  top_k: 5  # Fewer candidates for local models
```

#### RAG Configuration

The `rag.top_k` setting controls how many similar chunks are retrieved in Phase 2 (Enhanced RAG). The Decider (Phase 3.1) then evaluates each (knowledge, candidate page) pair.

**Tuning recommendations:**
- **5-10**: Good balance for most use cases
- **10-20**: More thorough search, higher LLM cost
- **3-5**: Faster, cheaper, may miss relevant pages

**Viewing pipeline details:**
```bash
# See detailed logs for all phases
logsqueak extract --verbose 2025-01-15
```

Shows candidate retrieval, decider decisions, and write operations.

## Usage

### First Time Setup

```bash
# Build the vector store index (Phase 0)
logsqueak index rebuild

# Check index status
logsqueak index status
```

### Extracting Knowledge

```bash
# Extract from specific date (writes directly with processed:: markers)
logsqueak extract 2025-01-15

# Extract from date range
logsqueak extract 2025-01-10..2025-01-15

# Extract from today's journal
logsqueak extract

# Enable verbose logging to see pipeline details
logsqueak extract --verbose 2025-01-15
```

**Note**: The 5-phase pipeline writes directly to pages and adds `processed::` markers to journal entries. Each marker contains bidirectional links in the format:
```
processed:: [PageName](((block-uuid-1))), [OtherPage](((block-uuid-2)))
```

Click these links in Logseq to jump to the integrated knowledge blocks.

## Project Structure

```
logsqueak/
├── src/logsqueak/         # Source code
│   ├── cli/               # Command-line interface
│   ├── config/            # Configuration management
│   ├── extraction/        # Knowledge extraction logic
│   ├── integration/       # Page integration logic
│   ├── llm/               # LLM client abstraction
│   ├── logseq/            # Logseq file format handling
│   ├── models/            # Data models
│   └── rag/               # RAG infrastructure
├── tests/                 # Test suite
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── fixtures/          # Test data
└── specs/                 # Feature specifications
```

## Development

### Running Tests

**IMPORTANT**: Always activate the virtual environment first!

```bash
# Activate venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run all tests (that don't require heavy dependencies)
pytest tests/unit/test_config.py tests/unit/test_journal.py tests/unit/test_knowledge.py tests/unit/test_parser.py tests/unit/test_graph.py tests/unit/test_preview.py tests/integration/

# Run with coverage
pytest tests/unit/ tests/integration/ --cov=logsqueak --cov-report=html

# Run specific test file
pytest tests/unit/test_parser.py -v

# Skip slow tests (requires sentence-transformers)
pytest -m "not slow"
```

**Note**: Some tests require `sentence-transformers` which is a large dependency. Install it separately if needed:
```bash
pip install sentence-transformers>=2.2.0
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

Production-ready 5-phase pipeline with all core features complete:

- ✅ **Hybrid-ID Foundation**
  - Parser extracts `id::` properties
  - Full-context chunk generation with parent traversal
  - Content hashing for hybrid IDs
  - Writer generates UUIDs for new blocks
  - Round-trip safety tests (preserve IDs)
  - `find_block_by_id()` for AST lookup

- ✅ **Persistent Vector Store**
  - ChromaDB dependency and integration
  - VectorStore abstraction with ChromaDBStore
  - Block-level chunking with full-context text
  - Cache manifest system (mtime tracking)
  - Incremental index builder (detects add/update/delete)
  - CLI commands: `index rebuild`, `index status`

- ✅ **Block-Level Targeting**
  - Block ID targeting infrastructure
  - UPDATE operation (replace content, preserve ID)
  - APPEND_CHILD operation (add to specific block)
  - APPEND_ROOT operation (add to page root)

- ✅ **Multi-Stage LLM Pipeline**
  - Phase 1: Knowledge extraction with hybrid IDs
  - Phase 2: Enhanced RAG (semantic + hinted search)
  - Phase 3.1: Decider LLM (action selection)
  - Phase 3.2: Reworder LLM (content refinement)
  - Phase 4: Atomic execution with journal cleanup
  - Configurable models for each phase
  - Full 5-phase pipeline integration

- ✅ **Testing & Reliability**
  - Full pipeline integration tests
  - Comprehensive error recovery tests
  - Round-trip parsing validation

## Architecture

### 5-Phase Pipeline

```
Phase 0 - Indexing (run once, then incrementally):
  Parse all pages → Generate full-context chunks → Extract hybrid IDs
  → Store in ChromaDB → Maintain manifest (mtime tracking)

Phase 1 - Knowledge Extraction:
  Load journal → LLM identifies knowledge blocks → Add parent context
  → Create Knowledge Packages with hybrid IDs

Phase 2 - Enhanced RAG:
  Semantic search (ChromaDB top_k) + Hinted search ([[Page Links]])
  → Aggregate and deduplicate → Group by page → Return candidates

Phase 3 - Decision & Rewording:
  For each (knowledge, candidate) pair:
    Phase 3.1 - Decider: LLM chooses UPDATE/APPEND_CHILD/APPEND_ROOT/IGNORE
    Phase 3.2 - Reworder: LLM refines content (if not IGNORE)
  → Build Write List

Phase 4 - Execution & Cleanup:
  Group by page → Parse AST → Apply operations → Generate UUIDs
  → Atomically update journal with processed:: markers
  → Write all changes
```

### Key Design Principles

1. **Property Order Preservation**: NEVER reorder (insertion order sacred - FR-008)
2. **Hybrid ID System**: Persistent `id::` OR content hash for stable targeting
3. **Atomic Consistency**: Journal marked only when page write succeeds
4. **Block-Level Precision**: Target specific blocks via `find_block_by_id()`
5. **Persistent Vector Store**: ChromaDB with incremental updates (manifest-based)
6. **Multi-Model Architecture**: Separate models for extraction, decisions, and rewording
7. **Bidirectional Links**: `processed::` markers in journal reference integrated blocks

## License

GPLv3 - See [LICENSE](LICENSE) file.

This project uses AI assistance (Claude Code) in development. All code is licensed under GPLv3 regardless of authorship method.

## Contributing

See [CLAUDE.md](CLAUDE.md) for developer documentation and architecture details.

Additional resources in [specs/001-knowledge-extraction/](specs/001-knowledge-extraction/):
- `spec.md` - Feature requirements and user stories
- `data-model.md` - Entity definitions
- `quickstart.md` - Developer setup guide

## Testing

Test coverage focuses on:

- **Unit Tests**: All data models, parser/renderer, graph operations
- **Integration Tests**: Round-trip parsing, full extraction workflow
- **Contract Tests**: CLI commands, LLM interactions, file operations

Run with:

```bash
pytest --cov=logsqueak --cov-report=term-missing
```

## Performance

Tested on 500+ page Logseq graph:

**Vector Store Indexing (Phase 0):**
- **Initial build**: ~20-30 seconds (parse all pages, generate block embeddings)
- **Incremental updates**: ~2 seconds (only process changed pages)
- **Storage**: `~/.cache/logsqueak/chroma/` (persistent across sessions)

**Knowledge Extraction (Phases 1-4):**
- **Phase 1-2**: ~2-3 seconds per journal (extraction + RAG search)
- **Phase 3**: Variable (depends on number of candidates and LLM speed)
- **Phase 4**: ~1-2 seconds (AST operations + file writes)

**Total**: Typically 5-10 seconds per journal entry (excluding LLM API latency)

## Roadmap

**Future Enhancements:**
- Performance benchmarks and optimization
- Batch processing optimization
- Semantic deduplication across pages
- Historical journal summarization
- Rich terminal UI (TUI)
- Async LLM calls for parallelization
