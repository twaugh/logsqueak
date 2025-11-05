# Implementation Plan: Interactive TUI for Knowledge Extraction

**Branch**: `002-logsqueak-spec` | **Date**: 2025-11-04 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/002-logsqueak-spec/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Build an interactive TUI application for extracting lasting knowledge from Logseq journal entries using LLM-powered analysis. The application provides three interactive phases: block selection (users select knowledge blocks with LLM assistance), content editing (users refine content with LLM-suggested rewordings), and integration decisions (users review and approve where knowledge should be integrated). All operations are keyboard-driven, streaming-enabled, and non-destructive with atomic provenance tracking.

## Technical Context

**Language/Version**: Python 3.11+
**Primary Dependencies**: Textual (TUI framework), httpx (HTTP client for LLM APIs), pydantic (data validation), click (CLI framework), chromadb (vector store for RAG), sentence-transformers (embeddings), markdown-it-py (markdown rendering), existing logseq-outline-parser library
**Storage**: File-based I/O for Logseq files, ChromaDB for vector embeddings, YAML for configuration (~/.config/logsqueak/config.yaml)
**Testing**: pytest (existing), coverage for new TUI components
**Target Platform**: Linux/macOS/Windows terminal emulators with 256 color + Unicode support
**Project Type**: Single Python application (CLI with interactive TUI)
**Performance Goals**: UI responsiveness <100ms for input, <500ms visual feedback, complete typical workflow (11 root blocks, 22 total blocks, 5 knowledge blocks, 3 candidate pages each) in <3 minutes
**Constraints**: Must handle journals approaching 2000-line parser limit, streaming LLM responses must update UI incrementally, background tasks must not block UI, file permissions mode 600 for config
**Scale/Scope**: Multiple journal entries per session (naturally supported by tree view structure with date grouping nodes), ~100 blocks max per entry, 3 interactive phases (P1: block selection, P2: content editing, P3: integration decisions), keyboard-driven only

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### I. Proof-of-Concept First ✅

**Alignment**: This feature is explicitly designed as a POC to demonstrate feasibility of LLM-driven knowledge extraction. The spec prioritizes working software over perfection, shipping iteratively across three independent phases (P1 block selection, P2 content editing, P3 integration decisions), each delivering standalone value.

**Evidence**:

- Priority levels (P1/P2) allow incremental delivery
- No backwards compatibility promises required
- Success criteria focus on user experience metrics (responsiveness, override capability) not production readiness
- Streaming LLM responses demonstrate feasibility, not optimize for scale

**Status**: PASS - Feature designed for iterative POC development

### II. Non-Destructive Operations (NON-NEGOTIABLE) ✅

**Alignment**: All operations follow non-destructive principles with atomic provenance tracking.

**Evidence**:

- FR-043: Atomically add `processed::` property on journal block only after successful page write
- FR-071: Maintain provenance links from journal to integrated blocks via property that lists target page and block references
- FR-067/FR-068: Use existing Logseq parsing capabilities, preserve exact property order (property order is significant in Logseq)
- FR-070: Generate unique identifiers for all integrated blocks to enable precise future references
- Phase 3 acceptance scenarios show APPEND operations (add new sections/blocks) without modifying existing content
- User must explicitly approve all integrations via 'y' key (FR-041)
- FR-071a-d: Concurrent modification detection and automatic reload before writes

**Status**: PASS - Feature strictly adheres to non-destructive principles

### III. Simplicity and Transparency ✅

**Alignment**: Architecture favors simplicity and shows user what LLM is doing.

**Evidence**:

- File-based I/O for Logseq files (avoid database complexity)
- ChromaDB only for RAG semantic search (unavoidable for vector embeddings)
- JSON for structured LLM outputs (FR-063, FR-064)
- FR-003: Display LLM reasoning for why a block was identified as knowledge in the bottom panel
- FR-052: Display LLM reasoning for each integration decision
- FR-005: Status widget showing which background tasks are active and their progress
- Single Python application, keyboard-driven only (no web UI complexity)
- YAML configuration file with explicit structure (FR-072d)

**Status**: PASS - Feature embraces simplicity and transparency

### Overall Gate Decision: ✅ PASS

All three constitution principles are satisfied. No violations requiring justification. Proceed to Phase 0 research.

---

### Post-Design Re-Evaluation: ✅ PASS

After completing Phase 0 (research.md) and Phase 1 (data-model.md, contracts/, quickstart.md), the design still fully aligns with constitution principles:

**I. Proof-of-Concept First**: ✅

- Research decisions prioritize pragmatic over perfect (e.g., all-MiniLM-L6-v2 embeddings, simple retry logic)
- Data models use Pydantic (no complex ORM)
- Quickstart shows 4-week incremental timeline
- Services use simple async functions, not frameworks

**II. Non-Destructive Operations**: ✅

- FileOperations service implements atomic writes (contracts/services.md)
- Modification time tracking prevents stale writes
- Provenance via `processed::` property is explicit
- All actions (add_section, add_under, replace) preserve structure

**III. Simplicity and Transparency**: ✅

- LLM contracts show explicit JSON structures (contracts/llm-api.md)
- Structured logging captures all LLM interactions (research.md #7)
- File-based I/O dominates (ChromaDB only for vector search)
- No abstraction layers added (direct LogseqOutline usage)

**Status**: Design approved - ready for implementation (/speckit.tasks)

## Project Structure

### Documentation (this feature)

```text
specs/[###-feature]/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)

```

### Source Code (repository root)

```text
logsqueak/
├── src/
│   ├── logseq-outline-parser/        # Existing library (IMPLEMENTED)
│   │   ├── src/logseq_outline/
│   │   │   ├── parser.py             # Core parsing/rendering
│   │   │   ├── context.py            # Full-context generation & hashing
│   │   │   └── graph.py              # Graph path utilities
│   │   └── tests/                    # Parser unit tests
│   │
│   └── logsqueak/                    # NEW: Main application
│       ├── __init__.py
│       ├── __main__.py               # CLI entry point
│       ├── config.py                 # Configuration management (YAML)
│       ├── models.py                 # Data models (BlockState, EditedContent, IntegrationDecision, BackgroundTask)
│       │
│       ├── tui/                      # Textual TUI components
│       │   ├── __init__.py
│       │   ├── app.py                # Main TUI application
│       │   ├── phase1_selection.py   # Block selection screen (P1)
│       │   ├── phase2_editing.py     # Content editing screen (P2)
│       │   ├── phase3_integration.py # Integration decisions screen (P1)
│       │   ├── widgets/              # Reusable widgets
│       │   │   ├── block_tree.py     # Tree view for blocks
│       │   │   ├── preview_panel.py  # Preview panel with markdown
│       │   │   ├── status_widget.py  # Background task status
│       │   │   └── footer.py         # Context-sensitive key bindings
│       │   └── markdown_renderer.py  # Markdown rendering utilities
│       │
│       ├── services/                 # Business logic
│       │   ├── __init__.py
│       │   ├── llm_client.py         # LLM API client (streaming support)
│       │   ├── knowledge_classifier.py # Phase 1: LLM knowledge classification
│       │   ├── content_rewriter.py   # Phase 2: LLM content rewording
│       │   ├── integration_planner.py # Phase 3: LLM integration decisions
│       │   ├── page_indexer.py       # Background: Page index building
│       │   ├── rag_search.py         # Background: Semantic search
│       │   └── file_operations.py    # File I/O with concurrent modification detection
│       │
│       └── utils/                    # Utilities
│           ├── __init__.py
│           ├── logging.py            # Structured logging
│           └── retry.py              # Network retry logic
│
├── tests/                            # NEW: Application tests
│   ├── unit/
│   │   ├── test_config.py
│   │   ├── test_models.py
│   │   ├── test_llm_client.py
│   │   └── test_services.py
│   ├── integration/
│   │   ├── test_phase1_workflow.py
│   │   ├── test_phase2_workflow.py
│   │   └── test_phase3_workflow.py
│   └── fixtures/                     # Test data (sample journals, pages)
│
├── pyproject.toml                    # Project dependencies
└── README.md

```

**Structure Decision**: Single Python application (Option 1). This is a CLI tool with TUI interface, not a web or mobile app. The structure follows Python best practices with clear separation:

- `src/logsqueak/` for main application code
- `src/logsqueak/tui/` for Textual UI components (phase screens, widgets)
- `src/logsqueak/services/` for business logic (LLM clients, background tasks)
- `tests/` for unit and integration tests
- Existing `src/logseq-outline-parser/` library remains unchanged and is used by the new application

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

No violations - Constitution Check passed with full compliance.
