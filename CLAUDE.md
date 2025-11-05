# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Logsqueak** is a TUI (Text User Interface) application for extracting lasting knowledge from Logseq journal entries using LLM-powered analysis. Users interactively review, refine, and integrate knowledge blocks into their Logseq knowledge base.

**Current Status**: Early development
- ✅ **Implemented**: `logseq-outline-parser` library (robust Logseq markdown parsing)
- 🚧 **In Progress**: Interactive TUI for knowledge extraction (spec: `specs/002-logsqueak-spec/`)
- ⏳ **Planned**: LLM integration, RAG semantic search, background task orchestration

## Project Structure

```
logsqueak/
├── src/
│   └── logseq-outline-parser/     # Logseq markdown parser library (IMPLEMENTED)
│       └── src/logseq_outline/
│           ├── parser.py          # Core parsing/rendering logic
│           ├── context.py         # Full-context generation & content hashing
│           └── graph.py           # Graph path utilities
├── specs/
│   ├── 001-logsqueak/             # Original knowledge extraction spec
│   └── 002-logsqueak-spec/        # Interactive TUI feature spec (CURRENT)
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

## Planned: Interactive TUI Application

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
  top_k: 10  # Optional: Number of candidate pages to retrieve (default: 10)
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
# Run parser tests (currently the only tests)
pytest src/logseq-outline-parser/tests/ -v

# Run with coverage
pytest src/logseq-outline-parser/tests/ --cov=logseq_outline --cov-report=html

# Future: When main app tests exist
# pytest tests/unit/ tests/integration/ -v
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

### Running the CLI (Not Yet Implemented)

```bash
# Future commands (from spec):
# logsqueak extract 2025-01-15          # Extract from specific date
# logsqueak extract 2025-01-10..2025-01-15  # Date range
# logsqueak extract                     # Today's journal
```

## Python Version & Dependencies

- **Python**: 3.11+ (required)
- **Key dependencies**:
  - `textual>=0.47.0` - TUI framework (planned)
  - `httpx>=0.27.0` - LLM client (planned)
  - `pydantic>=2.0.0` - Data validation (planned)
  - `click>=8.1.0` - CLI framework (planned)
  - `chromadb>=0.4.0` - Vector store for RAG (planned)
  - `sentence-transformers>=2.2.0` - Embeddings (planned, large dependency)
  - `markdown-it-py>=3.0.0` - Markdown rendering (planned)

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

**Implementation guidance**:
1. Use Textual framework for TUI (already in dependencies)
2. Build incrementally: Phase 1 → Phase 2 → Phase 3
3. Test each phase independently before integrating
4. Use existing parser library for all Logseq file operations
5. Follow keyboard control specifications exactly (user muscle memory)

**When implementing**:
- Start with Phase 1 block selection screen (P1 priority)
- Use mock LLM responses initially (avoid API costs during dev)
- Implement streaming incrementally (start with batch, add streaming later)
- Test round-trip parsing extensively (property order preservation)

## License

GPLv3 - All code is licensed under GPLv3 regardless of authorship method (including AI-assisted development).

## Recent Changes

- 2025-11-04: Created hybrid CLAUDE.md reflecting current implementation (parser only) + planned TUI feature
- 2025-11-04: Completed clarification session for 002-logsqueak-spec (configuration management, validation strategy)
