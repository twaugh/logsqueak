# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Logsqueak is a CLI tool that extracts lasting knowledge from Logseq journal entries and integrates it into relevant pages using LLM-powered analysis with RAG-based semantic search. It distinguishes between temporary activity logs and knowledge blocks, presents changes in dry-run mode, and adds extracted knowledge as new child bullets with provenance links.

**Current Status**: Planning major pipeline upgrade (better-pipeline branch)
- **Existing**: Basic 2-stage pipeline (extract → RAG match) working on main branch (69% complete)
- **Goal**: Comprehensive 5-phase pipeline with persistent vector store and hybrid-ID system (see FUTURE-STATE.md)
- **Plan**: 30 tasks across 5 milestones, 20-31 days estimated (see PLAN.md)

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

**Note**: Some tests require `sentence-transformers` which is a large dependency. It's included in requirements.txt but may take time to install.

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
# Preview only (dry-run mode - no writes)
logsqueak extract 2025-01-15 --dry-run

# Extract and apply with approval (normal mode)
logsqueak extract 2025-01-15

# Extract from date range
logsqueak extract 2025-01-10..2025-01-15

# Extract from today's journal
logsqueak extract
```

## Architecture

### Data Flow

The extraction and integration process follows these stages:

1. **Load**: Load journal entry from file (JournalEntry model enforces 2000-line limit)
2. **Extract**: Extract knowledge blocks via LLM (Stage 1: activity vs knowledge classification)
3. **RAG Match**: For each block, find top-5 candidate pages via semantic search (PageIndex)
4. **Page Selection**: LLM selects best page + section (Stage 2: structured output)
5. **Duplicate Detection**: Check for duplicates using content hashing
6. **Preview**: Build ExtractionPreview and display to user with diffs
7. **Approval**: Interactive y/n/e prompt (approve/reject/edit)
8. **Integration**: Write knowledge blocks with provenance links, refresh PageIndex

### Two-Stage LLM Pattern

The system uses a two-stage LLM approach to balance context window constraints with accuracy:

**Stage 1 - Extraction**: LLM reads entire journal entry and extracts knowledge blocks
**Stage 2 - Page Selection**: For each knowledge block, LLM reads the block + top-5 RAG candidates and selects target page/section

This pattern prevents sending 500+ pages to the LLM while maintaining high-quality page matching.

### Key Modules

- **cli/**: Command-line interface with Click
  - `main.py`: Entry point, `extract` command orchestration
  - `interactive.py`: Interactive approval prompts (y/n/e)
  - `progress.py`: Progress feedback messages

- **config/**: Configuration management
  - Uses Pydantic for validation
  - Config file: `~/.config/logsqueak/config.yaml`

- **extraction/**: Knowledge extraction logic
  - `extractor.py`: Main extraction orchestrator (Stage 1 LLM)
  - `classifier.py`: Activity vs knowledge classification

- **integration/**: Page integration logic
  - `integrator.py`: Main integration orchestrator
  - `writer.py`: Atomic file writes with provenance links

- **llm/**: LLM client abstraction
  - `client.py`: Generic LLM interface
  - `providers/openai_compat.py`: OpenAI-compatible provider (OpenAI, Anthropic, Ollama)
  - `prompt_logger.py`: Prompt inspection system (logs to `~/.cache/logsqueak/prompts/`)

- **logseq/**: Logseq file format handling
  - `parser.py`: Parse outline markdown into LogseqOutline/LogseqBlock tree
  - `renderer.py`: Render outline back to markdown
  - `graph.py`: Graph path operations (journals/, pages/)

- **models/**: Data models (all use dataclasses)
  - `journal.py`: JournalEntry (2000-line limit enforcement)
  - `knowledge.py`: KnowledgeBlock (with provenance and content hashing)
  - `page.py`: TargetPage, PageIndex (RAG with per-page embedding cache)
  - `preview.py`: ExtractionPreview, ProposedAction (display formatting)

- **rag/**: Semantic search infrastructure
  - `index.py`: PageIndex implementation with sentence-transformers
  - `embedder.py`: Embedding utilities with cache (`~/.cache/logsqueak/embeddings/`)

## Project Constitution

**See `.specify/memory/constitution.md` for the complete project constitution.**

Key principles from the constitution:

### I. Proof-of-Concept First
- Prioritize working software over perfection
- Ship iteratively, demonstrate feasibility
- No backwards compatibility guarantees (POC stage)

### II. Non-Destructive Operations (NON-NEGOTIABLE)
- NEVER modify or delete existing Logseq content without explicit user approval
- All changes MUST be additive (new blocks, new sections)
- Dry-run mode required before any writes
- Provenance links to source journals are MANDATORY

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

### 2. Provenance Links (FR-003)

Every knowledge block MUST link back to source journal:
- Format: `[[journal/YYYY-MM-DD]]`
- 100% coverage enforced by tests
- Links added during integration phase

### 3. RAG Caching

Per-page embedding cache for performance:
- Cache location: `~/.cache/logsqueak/embeddings/`
- Cache validation via mtime
- Performance: 566 pages in ~20s (first run), <1s (cached), ~1.5s (5 modified)

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
  model: gpt-4-turbo-preview

logseq:
  graph_path: ~/Documents/logseq-graph

rag:
  token_budget: 3000  # Token budget for Stage 2 prompts (default: null = use top 5 candidates)
```

For Ollama (local LLM):
```yaml
llm:
  endpoint: http://localhost:11434/v1
  api_key: ollama  # Any non-empty value
  model: llama2

logseq:
  graph_path: ~/Documents/logseq-graph

rag:
  token_budget: 2000  # Smaller budget for local models
```

### RAG Token Budget Tuning

The `rag.token_budget` setting controls Stage 2 prompts (page selection). The system uses **exact token counting** (tiktoken) to fit as many candidate pages as possible within the budget.

**How it works:**
- When token_budget is `null` (default): Uses top 5 candidates (backward compatible)
- When token_budget is set (e.g., 3000): System calculates tokens for each candidate page preview and includes as many as fit

**Budget recommendations:**
- **1000-1500 tokens**: Typically fits 1-2 candidates (minimal cost, may miss best match)
- **2000-3000 tokens**: Typically fits 3-5 candidates (good balance)
- **4000-6000 tokens**: Typically fits 5-10 candidates (more thorough)
- **8000+ tokens**: Fits 10-20 candidates (comprehensive but expensive)

**Token calculation (exact, using tiktoken):**
- Base prompt (actual Stage 2 system + user prompts): Calculated exactly from real prompts
- Each candidate page preview (1000 chars): Counted exactly for each candidate
- Response overhead: 300 tokens (reserved for JSON output)

The system constructs the actual prompts and counts tokens using tiktoken before adding candidates, ensuring the budget is never exceeded.

**Example:** With a 3000 token budget:
- Base prompt: ~650 tokens (actual count, varies with knowledge content)
- Response: 300 tokens (reserved)
- Available for candidates: ~2050 tokens
- Result: ~5-7 candidates (each ~300-400 tokens depending on page size)

You can override via environment variable:
- `LOGSQUEAK_RAG_TOKEN_BUDGET`: Override token_budget

**Viewing token usage logs:**

To see token budget information during extraction, use the `--verbose` flag:

```bash
logsqueak extract --verbose 2025-01-15
```

**IMPORTANT**: Token budget logging only appears when you have `token_budget` set in your config:

```yaml
rag:
  token_budget: 3000  # Without this, logs will show "using default of 5 candidates"
```

With token budget configured and `--verbose`, you'll see:
- `INFO` level: Budget usage summary and candidate selection
- `DEBUG` level: Per-candidate token counts and decisions

Example verbose output **with token budget**:
```
INFO [logsqueak.extraction.extractor] Using token budget of 3000 tokens for candidate selection
DEBUG [logsqueak.extraction.extractor] Added candidate 'Project X' (342 tokens, total: 342/2300)
DEBUG [logsqueak.extraction.extractor] Added candidate 'Team Notes' (298 tokens, total: 640/2300)
DEBUG [logsqueak.extraction.extractor] Skipping candidate 'Archive' (387 tokens) - would exceed budget
INFO [logsqueak.extraction.extractor] Selected 6 candidates within 3000 token budget (used 2847 total)
```

Example verbose output **without token budget** (default behavior):
```
INFO [logsqueak.extraction.extractor] No token budget specified, using default of 5 candidates
```

Without `--verbose`, only warnings and errors are shown.

## Project Structure Conventions

- All source code in `src/logsqueak/`
- All tests in `tests/` (pytest discovers via `pyproject.toml` pythonpath)
- Feature specs in `specs/001-knowledge-extraction/`
- Entry point: `logsqueak.cli.main:main` (defined in pyproject.toml)

## Implementation Status

**Main branch**: Working proof-of-concept (69% complete)
- Basic 2-stage pipeline (extract → RAG match)
- Session-based embedding cache
- ADD_CHILD operations only
- See `specs/001-knowledge-extraction/tasks.md`

**Current branch: `better-pipeline`** - Major pipeline redesign in planning
- **Milestone 1**: Hybrid-ID Foundation (5 tasks) - id:: properties + full-context hashing
- **Milestone 2**: Persistent Vector Store (6 tasks) - ChromaDB with block-level indexing
- **Milestone 3**: Block-Level Targeting (4 tasks) - Precise block targeting via hybrid IDs
- **Milestone 4**: Multi-Stage LLM Pipeline (9 tasks) - Decider + Reworder + journal cleanup
- **Milestone 5**: Testing & Refinement (6 tasks) - Integration tests and documentation

See `PLAN.md` for detailed task breakdown and `FUTURE-STATE.md` for target architecture.

## Python Version & Dependencies

- **Python**: 3.11+ (required)
- **Key dependencies**:
  - httpx (LLM client)
  - markdown-it-py (parsing)
  - pydantic (validation)
  - click (CLI)
  - sentence-transformers (RAG embeddings - large dependency)
  - numpy (vector operations)

## License

GPLv3 - All code is licensed under GPLv3 regardless of authorship method (including AI-assisted development).
