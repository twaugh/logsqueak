# Implementation Plan: Logsqueak - Interactive TUI for Knowledge Extraction

**Branch**: `002-logsqueak-spec` | **Date**: 2025-11-05 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/002-logsqueak-spec/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Build an interactive TUI application for extracting lasting knowledge from Logseq journal entries using LLM-powered analysis. Users progress through three phases: (1) block selection with LLM knowledge classification, (2) content editing with LLM rewording suggestions, and (3) integration decisions with LLM-suggested target pages. Uses Textual framework for the TUI, existing logseq-outline-parser for file operations, and NDJSON streaming for LLM responses.

## Technical Context

**Language/Version**: Python 3.11+

**Primary Dependencies**:

- Textual >=0.47.0 (TUI framework)
- httpx >=0.27.0 (async LLM client with streaming support)
- pydantic >=2.0.0 (data validation)
- click >=8.1.0 (CLI framework)
- chromadb >=0.4.0 (vector store for RAG)
- sentence-transformers >=2.2.0 (embeddings)
- markdown-it-py >=3.0.0 (markdown rendering in TUI)
- logseq-outline-parser (existing, in src/logseq-outline-parser)

**Storage**: File-based (Logseq markdown files) + ChromaDB (vector embeddings)

**Testing**: pytest (unit/integration), Textual pilot for UI testing

**Target Platform**: Linux/macOS/Windows terminal (modern terminal emulators with 256 colors + Unicode)

**Project Type**: Single project (CLI application with TUI)

**Performance Goals**:

- UI responsiveness: <100ms input acceptance, <500ms visual feedback
- Typical workflow completion: <3 minutes for ~5 knowledge blocks
- Streaming LLM results: incremental display as responses arrive

**Constraints**:

- Non-destructive operations (constitutional requirement)
- Property order preservation in Logseq files (MANDATORY)
- Background tasks must not block UI interaction
- 60s timeout for LLM API requests
- File permissions: config file must be mode 600

**Scale/Scope**:

- POC for single-user local operation
- Typical journal: 11 root blocks, 22 total blocks, ~5 knowledge blocks
- RAG search: top-k=10 candidate pages, ~2 relevant decisions per block
- Three interactive phases with 8-12 keyboard shortcuts total

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Principle I: Proof-of-Concept First
**Status**: ✅ PASS

- Feature is scoped as POC (no backwards compatibility guarantees)
- Iterative delivery: Phase 1 (block selection) → Phase 2 (editing) → Phase 3 (integration)
- Focuses on demonstrating feasibility of LLM-driven knowledge extraction
- Uses existing parser library, avoiding premature optimization

### Principle II: Non-Destructive Operations
**Status**: ✅ PASS

- All journal modifications are APPEND-only (adding `processed::` property)
- Target page modifications require explicit user approval (Phase 3 'y' key)
- Every integrated block gets deterministic `id::` (UUID v5) for traceability and idempotent retries
- Provenance links from journal to integrated blocks via `processed::` property
- Property order preservation enforced (FR-068)
- Atomic writes: journal marked only after successful page write (FR-043)
- External file modification detection and validation (FR-071a-d)

### Principle III: Simplicity and Transparency
**Status**: ✅ PASS

- File-based I/O for Logseq content (using existing parser)
- ChromaDB for vector storage (only abstraction is for RAG, complexity justified)
- LLM outputs shown to user in all phases:
  - Phase 1: Knowledge classification reasoning visible in bottom panel
  - Phase 2: Reworded content shown alongside original for comparison
  - Phase 3: Integration decisions with reasoning, confidence scores, and previews

- NDJSON streaming for LLM responses (simple, debuggable format)
- Logging of all LLM requests/responses (FR-063, FR-064)

**GATE RESULT**: ✅ ALL GATES PASS - Proceed to Phase 0

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
src/
├── logseq-outline-parser/     # Existing parser library (DO NOT MODIFY)
│   └── src/logseq_outline/
│       ├── parser.py
│       ├── context.py
│       └── graph.py
├── logsqueak/                  # Main application (NEW)
│   ├── __init__.py
│   ├── cli.py                  # CLI entry point (click)
│   ├── config.py               # Configuration management
│   ├── models/                 # Pydantic data models
│   │   ├── __init__.py
│   │   ├── block_state.py
│   │   ├── edited_content.py
│   │   ├── integration_decision.py
│   │   └── background_task.py
│   ├── services/               # Business logic
│   │   ├── __init__.py
│   │   ├── llm_client.py       # NDJSON streaming LLM client
│   │   ├── page_indexer.py     # ChromaDB indexing
│   │   ├── rag_search.py       # Semantic search
│   │   └── file_monitor.py     # External modification detection
│   ├── tui/                    # Textual screens and widgets
│   │   ├── __init__.py
│   │   ├── app.py              # Main Textual app
│   │   ├── screens/
│   │   │   ├── __init__.py
│   │   │   ├── block_selection.py    # Phase 1
│   │   │   ├── content_editing.py    # Phase 2
│   │   │   └── integration_review.py # Phase 3
│   │   └── widgets/
│   │       ├── __init__.py
│   │       ├── block_tree.py
│   │       ├── status_panel.py
│   │       └── markdown_viewer.py
│   └── utils/
│       ├── __init__.py
│       ├── logging.py
│       └── ids.py              # UUID generation

tests/
├── unit/
│   ├── test_models.py
│   ├── test_llm_client.py
│   ├── test_page_indexer.py
│   └── test_file_monitor.py
├── integration/
│   ├── test_workflow.py        # End-to-end phase flow
│   └── test_rag_pipeline.py
└── ui/
    └── test_screens.py         # Textual pilot tests

pyproject.toml                  # Dependencies and entry points

```

**Structure Decision**: Single project structure chosen. The application is a CLI tool with TUI interface, not a web or mobile app. Source organized by layer (models, services, tui) with clear separation between Textual UI components and business logic. Existing parser library kept isolated to avoid modifications.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

**Status**: No violations - all constitution principles satisfied.

---

## Phase 0 & Phase 1 Complete

### Generated Artifacts

- ✅ **research.md**: Technical decisions for Textual, NDJSON streaming, ChromaDB, configuration
- ✅ **data-model.md**: Pydantic models for BlockState, EditedContent, IntegrationDecision, Config
- ✅ **contracts/**:
  - `llm-api.md`: NDJSON formats for knowledge classification, rewording, integration decisions
  - `service-interfaces.md`: LLMClient, PageIndexer, RAGSearch, FileMonitor interfaces
  - `file-operations.md`: logseq-outline-parser operations, atomic write guarantees

- ✅ **quickstart.md**: User-focused walkthrough for installation, configuration, and all three phases

### Post-Design Constitution Check

**Re-evaluation after Phase 1 design completion:**

#### Principle I: Proof-of-Concept First
**Status**: ✅ PASS (unchanged)

- Design maintains POC scope with pragmatic technology choices
- No over-engineering detected in service interfaces or contracts
- Incremental delivery preserved (Phase 1 → 2 → 3 independence)

#### Principle II: Non-Destructive Operations
**Status**: ✅ PASS (unchanged)

- File operations contract enforces atomic two-phase writes
- Property order preservation explicitly documented as NON-NEGOTIABLE
- Concurrent modification detection pattern validated in contracts
- Provenance tracking formalized in `processed::` property format

#### Principle III: Simplicity and Transparency
**Status**: ✅ PASS (unchanged)

- NDJSON streaming simpler than SSE (research validated)
- ChromaDB only abstraction, well-justified for vector search
- All LLM interactions logged and visible to user
- Configuration uses standard YAML + Pydantic (no custom formats)

**FINAL GATE RESULT**: ✅ ALL GATES PASS - Design approved

---

## Next Steps

**Command Complete**: `/speckit.plan` has finished Phase 0 (Research) and Phase 1 (Design).

**To proceed with implementation**:

1. Run `/speckit.tasks` to generate dependency-ordered task breakdown
2. Run `/speckit.implement` to execute tasks from tasks.md

**Branch**: `002-logsqueak-spec`

**Plan Location**: `/home/twaugh/devel/logsqueak/specs/002-logsqueak-spec/plan.md`
