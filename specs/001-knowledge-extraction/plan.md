# Implementation Plan: Knowledge Extraction from Journals

**Branch**: `001-knowledge-extraction` | **Date**: 2025-10-22 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-knowledge-extraction/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Extract lasting knowledge from Logseq journal entries and intelligently integrate into relevant pages using LLM-powered analysis with RAG-based semantic search. The system will distinguish between temporary activity logs and knowledge blocks, present changes in dry-run mode for user approval, and add extracted knowledge as new child bullets with provenance links. Focus on non-destructive operations with targeted additive integration (new child blocks at appropriate locations, new organizational bullets) while strictly preserving property order and deferring semantic merging to future versions.

## Technical Context

**Language/Version**: Python 3.11+
**Primary Dependencies**: httpx (LLM client), markdown-it-py (parsing), PyYAML + pydantic (config), click (CLI), python-dateutil (date parsing), sentence-transformers + numpy (RAG/semantic search)
**Storage**: Files (Logseq markdown in user's graph directory, config in ~/.config/logsqueak/), per-page embedding cache (~/.cache/logsqueak/embeddings/)
**Testing**: pytest
**Target Platform**: Linux/macOS CLI (cross-platform Python)
**Project Type**: single (CLI tool)
**Performance Goals**: Process typical journal entries (50-200 lines) with acceptable LLM latency; support up to 2000 lines. Page index building: ~20s first run for 566 pages, <1s with cache, ~1.5s if 5 pages modified
**Constraints**: No file corruption (100% data safety), dry-run mode mandatory, 2000-line entry limit
**Scale/Scope**: Single-user POC, one journal entry at a time, personal Logseq graphs (tested scale: 566 pages / 2.3MB, in-memory embeddings scale to ~1000 pages)

**Research Complete**: See [research.md](./research.md) for dependency rationale and best practices.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Initial Check (Pre-Research)

### Principle I: Proof-of-Concept First
✅ **PASS** - Project scope is explicitly POC. Focus on working software (extract, preview, integrate) without production concerns. No backwards compatibility requirements. Iterative delivery (P1 → P2 → P3 user stories).

### Principle II: Non-Destructive Operations (NON-NEGOTIABLE)
✅ **PASS** - Requirements enforce this:

- FR-002: Dry-run mode mandatory
- FR-006: Additive only (new child bullets)
- FR-008: Preserve existing content
- FR-012: User approval required (y/n/e prompt)
- SC-004: 100% data safety guarantee
- FR-003: Provenance links mandatory

### Principle III: Simplicity and Transparency
✅ **PASS** - Architecture embraces simplicity:

- File-based I/O (FR-014), no database
- FR-015: JSON for structured LLM outputs
- FR-012: Show preview before applying
- SC-001: Clear progress feedback
- Simple CLI with interactive prompts

**Initial Gate Status**: ✅ ALL CHECKS PASS - Proceeded to Phase 0 Research

---

### Post-Design Check (After Phase 1)

**Re-evaluation after completing research.md, data-model.md, and contracts:**

### Principle I: Proof-of-Concept First
✅ **PASS** - Design choices reinforce POC simplicity:

- 8 dependencies total (minimal, well-established libraries)
- Stdlib `dataclasses` used (not heavier alternatives)
- Simple HTTP client (httpx) instead of heavy abstractions (LiteLLM, Langchain)
- In-memory numpy for vector search (not external vector DB)
- Offline embeddings (sentence-transformers) - no additional API calls
- No premature optimization (defer async, TUI, semantic merging to roadmap)

### Principle II: Non-Destructive Operations (NON-NEGOTIABLE)
✅ **PASS** - Data model and contracts enforce safety:

- `ExtractionPreview` entity shows all changes before applying
- `ProposedAction.status` tracks READY/SKIPPED/WARNING
- Duplicate detection (FR-017) via `content_hash()`
- File operations through `LogseqOutline.render()` preserve structure
- **Property order preservation**: NEVER reorder block properties (insertion order sacred)
- Children can be inserted at appropriate locations (targeted, minimal changes)
- CLI contract specifies dry-run as default mode

### Principle III: Simplicity and Transparency
✅ **PASS** - Architecture remains simple:

- Direct file I/O (`pathlib.Path`, no ORM/database)
- Structured LLM outputs (JSON + Pydantic validation)
- Clear data flow: Build Index → Load → Extract → RAG Search → Preview → Approve → Integrate
- RAG workflow transparent: show similarity scores in preview
- CLI contracts document all user-facing behavior
- Quickstart guide enables rapid onboarding

**Post-Design Gate Status**: ✅ ALL CHECKS PASS - Ready for Phase 2 (Tasks)

## Project Structure

### Documentation (this feature)

```text
specs/001-knowledge-extraction/
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
├── logsqueak/
│   ├── __init__.py
│   ├── cli/              # Command-line interface
│   │   ├── __init__.py
│   │   ├── main.py       # Entry point, argument parsing
│   │   └── interactive.py # y/n/e prompts
│   ├── config/           # Configuration management
│   │   ├── __init__.py
│   │   └── loader.py     # Load ~/.config/logsqueak/config.yaml
│   ├── extraction/       # Knowledge extraction logic
│   │   ├── __init__.py
│   │   ├── extractor.py  # Main extraction orchestrator
│   │   └── classifier.py # Activity vs. knowledge classification
│   ├── integration/      # Page integration logic
│   │   ├── __init__.py
│   │   ├── integrator.py # Main integration orchestrator
│   │   ├── matcher.py    # RAG-based page matching
│   │   └── writer.py     # Write with provenance links
│   ├── rag/              # Semantic search / embeddings
│   │   ├── __init__.py
│   │   ├── index.py      # PageIndex implementation
│   │   └── embedder.py   # Embedding utilities
│   ├── llm/              # LLM client abstraction
│   │   ├── __init__.py
│   │   ├── client.py     # Generic LLM client interface
│   │   └── providers/    # OpenAI-compatible, Ollama
│   ├── logseq/           # Logseq file format handling
│   │   ├── __init__.py
│   │   ├── parser.py     # Parse outline markdown
│   │   ├── renderer.py   # Render outline markdown
│   │   └── graph.py      # Graph path operations
│   └── models/           # Data models
│       ├── __init__.py
│       ├── journal.py    # Journal Entry model
│       ├── knowledge.py  # Knowledge Block model
│       ├── page.py       # Target Page + PageIndex models
│       └── preview.py    # Extraction Preview model

tests/
├── unit/                 # Unit tests for individual modules
│   ├── test_config.py
│   ├── test_journal.py
│   ├── test_knowledge.py
│   ├── test_parser.py
│   ├── test_graph.py
│   ├── test_preview.py
│   ├── test_page.py
│   ├── test_llm_client.py
│   ├── test_openai_provider.py
│   ├── test_extractor.py
│   ├── test_classifier.py
│   ├── test_matcher.py
│   ├── test_writer.py
│   ├── test_integrator.py
│   ├── test_interactive.py
│   ├── test_convention_detector.py
│   └── test_section_creator.py
├── integration/          # Integration tests for workflows
│   ├── test_parsing_roundtrip.py
│   ├── test_extraction_workflow.py
│   ├── test_file_safety.py
│   ├── test_provenance.py
│   ├── test_pageindex_refresh.py
│   ├── test_error_recovery.py
│   ├── test_duplicate_prevention.py
│   ├── test_integration_workflow.py
│   ├── test_section_creation.py
│   ├── test_convention_preservation.py
│   └── test_section_edge_cases.py
├── e2e/                  # End-to-end tests
│   ├── test_full_workflow.py
│   ├── test_multi_entry.py
│   ├── test_real_world.py
│   ├── test_error_scenarios.py
│   └── test_cli_integration.py
├── performance/          # Performance and scale tests
│   └── test_scale.py
├── fixtures/             # Test data
│   ├── journals/         # Sample journal entries
│   └── pages/            # Sample Logseq pages
└── conftest.py           # Shared pytest fixtures

```

**Structure Decision**: Single CLI project structure chosen because:

- This is a standalone command-line tool (not web/mobile)
- Clear separation of concerns: CLI → extraction → integration → LLM → Logseq I/O
- Models directory for data entities from spec
- Tests organized by type (unit, integration, e2e, performance) with comprehensive coverage at each level

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

No violations detected. All constitution principles satisfied.
