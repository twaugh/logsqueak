# Logsqueak

Extract lasting knowledge from Logseq journal entries using LLM-powered analysis with RAG-based semantic search.

**Status**: 🚧 Work in Progress - Phase 2 Complete (Foundational Components)

## Overview

Logsqueak is a CLI tool that helps you organize your Logseq knowledge base by:

1. **Extracting** lasting knowledge from daily journal entries (vs. temporary activity logs)
2. **Matching** knowledge to relevant pages using semantic search (RAG)
3. **Integrating** knowledge as new bullet points with provenance links
4. **Preserving** your existing content structure (100% non-destructive)

## Features

- **LLM-Powered Extraction**: Uses OpenAI-compatible APIs to distinguish knowledge from activity logs
- **Semantic Search**: RAG-based page matching with per-page embedding cache
- **Dry-Run Mode**: Preview all changes before applying (mandatory first step)
- **Provenance Links**: Every integrated fact links back to source journal
- **Property Order Preservation**: NEVER reorders block properties (FR-008)
- **Convention Detection**: Matches your page's organizational style (plain bullets vs. headings)
- **2000-Line Limit**: Automatic truncation with warnings for large journal entries

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
git checkout 001-knowledge-extraction

# Run setup script (creates venv, installs dependencies)
./setup-dev.sh
```

**Option 2: Manual Setup**

```bash
# Clone repository
git clone https://github.com/twaugh/logsqueak.git
cd logsqueak
git checkout 001-knowledge-extraction

# Create virtual environment (REQUIRED - do not skip this step!)
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip in the venv
pip install --upgrade pip

# Install dependencies in the venv
pip install -r requirements.txt

# Note: Tests run directly from source, no installation needed
# Tests use: pytest (which reads pythonpath from pyproject.toml)
```

### Configuration

Create `~/.config/logsqueak/config.yaml`:

```yaml
llm:
  endpoint: https://api.openai.com/v1
  api_key: sk-your-api-key-here
  model: gpt-4-turbo-preview

logseq:
  graph_path: ~/Documents/logseq-graph
```

**For Ollama (local)**:

```yaml
llm:
  endpoint: http://localhost:11434/v1
  api_key: ollama  # Any non-empty value
  model: llama2

logseq:
  graph_path: ~/Documents/logseq-graph
```

## Usage

```bash
# Extract knowledge from today's journal (dry-run mode)
logsqueak extract

# Extract from specific date
logsqueak extract 2025-01-15

# Extract from date range
logsqueak extract 2025-01-10..2025-01-15

# Preview and apply changes
logsqueak extract --apply 2025-01-15
# User prompted: Apply changes? [y/N/e]
```

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
├── specs/                 # Feature specifications
└── requirements.txt       # Python dependencies
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

### ✅ Completed (Phase 1 & 2)

- [x] Project structure and dependencies
- [x] Configuration model with Pydantic validation
- [x] JournalEntry model with 2000-line limit enforcement
- [x] KnowledgeBlock model with provenance and hashing
- [x] TargetPage and PageIndex models with RAG caching
- [x] ExtractionPreview model with display formatting
- [x] LogseqOutline and LogseqBlock models with property order preservation
- [x] Logseq markdown parser and renderer
- [x] Graph path operations
- [x] RAG index with per-page embedding cache
- [x] Comprehensive unit and integration tests

### 🚧 In Progress

- [ ] Phase 3: User Story 1 - Extract and Preview Knowledge (T017-T032)
  - Configuration loader
  - LLM client (OpenAI-compatible)
  - Two-stage extraction (extract → RAG match → LLM select)
  - Preview display
  - Error handling

### 📋 Planned

- [ ] Phase 4: User Story 2 - Integration (T033-T043)
- [ ] Phase 5: User Story 3 - Section Creation (T044-T049)
- [ ] Phase 6: Polish & Logging (T050-T051)

## Architecture

### Data Flow

```
1. Load journal entry from file
2. Extract knowledge blocks via LLM (Stage 1)
3. For each block:
   a. Find top-5 candidate pages via RAG (semantic search)
   b. LLM selects best page + section (Stage 2)
   c. Check for duplicates
   d. Create proposed action
4. Build preview and display to user
5. On approval: integrate knowledge with provenance links
```

### Key Design Principles

1. **Property Order**: NEVER reorder (insertion order sacred)
2. **Non-Destructive**: All changes are additive (no deletions/modifications)
3. **Dry-Run First**: Always preview before applying
4. **Provenance**: Every knowledge block links back to source journal
5. **RAG Matching**: Two-stage LLM (extract, then match to candidates)
6. **Per-Page Caching**: Embeddings cached with mtime validation

## License

GPLv3 - See [LICENSE](LICENSE) file.

This project uses AI assistance (Claude Code) in development. All code is licensed under GPLv3 regardless of authorship method.

## Contributing

See [specs/001-knowledge-extraction/](specs/001-knowledge-extraction/) for detailed specifications:

- `spec.md` - Feature requirements and user stories
- `plan.md` - Implementation plan and architecture
- `tasks.md` - Detailed task breakdown
- `data-model.md` - Entity definitions
- `research.md` - Library choices and rationale
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

Tested on 566 pages (2.3MB Logseq graph):

- **First index build**: ~20 seconds (embed all pages)
- **Subsequent runs**: <1 second (load from cache)
- **After modifying 5 pages**: ~1.5 seconds (561 cached + 5 new)

## Roadmap

See [specs/001-knowledge-extraction/spec.md](specs/001-knowledge-extraction/spec.md) for future enhancements:

- Batch processing of date ranges
- Semantic merging of similar facts
- Historical journal summarization
- Rich terminal UI (TUI)
- Async processing
