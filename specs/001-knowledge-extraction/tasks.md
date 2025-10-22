# Implementation Tasks: Knowledge Extraction from Journals

**Feature**: Extract lasting knowledge from Logseq journal entries and integrate into relevant pages using LLM-powered analysis with RAG-based semantic search
**Branch**: `001-knowledge-extraction`
**Generated**: 2025-10-22
**Source**: [spec.md](./spec.md), [plan.md](./plan.md), [data-model.md](./data-model.md)

## Overview

This document breaks down the implementation into executable tasks organized by user story. Each phase delivers an independently testable increment of functionality.

**MVP Scope**: User Story 1 (P1) - Extract and Preview Knowledge

- Delivers immediate value: see what knowledge would be extracted without modifying files
- Foundation for all other stories
- Validates core LLM extraction and RAG matching

## Task Summary

- **Total Tasks**: 47
- **Phase 1 (Setup)**: 5 tasks
- **Phase 2 (Foundational)**: 11 tasks
- **Phase 3 (User Story 1)**: 12 tasks
- **Phase 4 (User Story 2)**: 11 tasks
- **Phase 5 (User Story 3)**: 6 tasks
- **Phase 6 (Polish)**: 2 tasks

**Parallel Execution Opportunities**: 28 tasks marked [P] can run in parallel within their phase

## Dependencies

### User Story Dependencies

```
Phase 1 (Setup)
  ↓
Phase 2 (Foundational - models, parsing, RAG index)
  ↓
Phase 3 (User Story 1 - P1) ← Can start immediately after Phase 2
  ↓
Phase 4 (User Story 2 - P2) ← Depends on US1 completion
  ↓
Phase 5 (User Story 3 - P3) ← Depends on US2 completion
  ↓
Phase 6 (Polish)

```

**Key Insight**: User stories are sequential (each builds on the previous), but tasks within each story can be parallelized.

### Critical Path

1. Setup project structure (T001)
2. Install dependencies (T002)
3. Create core models (T006-T012)
4. Implement Logseq parser (T013-T014)
5. Build RAG index (T015-T016)
6. Implement LLM client (T018-T020)
7. Implement extraction logic (T021-T026)
8. Build preview display (T027-T032)

## Phase 1: Project Setup

**Goal**: Initialize Python project with required dependencies and structure

**Duration**: ~30 minutes

### Tasks

- [ ] T001 Create project structure per plan.md at /home/twaugh/devel/logsqueak/src/logsqueak/ with all subdirectories (cli/, config/, extraction/, integration/, rag/, llm/, logseq/, models/)
- [ ] T002 Create requirements.txt with 8 core dependencies: httpx>=0.27.0, markdown-it-py>=3.0.0, PyYAML>=6.0.1, pydantic>=2.0.0, click>=8.1.0, python-dateutil>=2.8.0, sentence-transformers>=2.2.0, numpy>=1.24.0
- [ ] T003 Create pyproject.toml with project metadata, entry points for CLI (logsqueak = logsqueak.cli.main:main), and Python 3.11+ requirement
- [ ] T004 Create tests/ directory structure with unit/, integration/, and fixtures/ subdirectories
- [ ] T005 Create LICENSE file with GPLv3 text (already exists, verify it's present)

---

## Phase 2: Foundational Components

**Goal**: Build core data models, Logseq parsing, and RAG infrastructure that all user stories depend on

**Duration**: ~4-6 hours

**Why before user stories**: These are blocking dependencies - every user story needs models, parsing, and page indexing.

### Tasks

#### Data Models (Parallel Group 1)

- [ ] T006 [P] Implement Configuration model in src/logsqueak/models/config.py with Pydantic validation for LLM config (endpoint, api_key, model) and Logseq config (graph_path)
- [ ] T007 [P] Implement JournalEntry model in src/logsqueak/models/journal.py with attributes (date, file_path, raw_content, outline, line_count) and load() classmethod
- [ ] T008 [P] Implement KnowledgeBlock model in src/logsqueak/models/knowledge.py with attributes (content, source_date, confidence, target_page, target_section: Optional[List[str]], suggested_action) and helper methods (content_hash(), provenance_link(), with_provenance(), section_path())
- [ ] T009 [P] Implement TargetPage model in src/logsqueak/models/page.py with attributes (name, file_path, outline, organizational_convention) and methods (load(), find_section(), has_duplicate())
- [ ] T010 [P] Implement ExtractionPreview model in src/logsqueak/models/preview.py with ProposedAction and display() method for terminal output
- [ ] T011 [P] Implement LogseqOutline and LogseqBlock models in src/logsqueak/logseq/parser.py with attributes preserving property order (dict in Python 3.7+), _original_line for round-trip, and add_child(content, position) method
- [ ] T012 [P] Implement PageIndex model in src/logsqueak/models/page.py with per-page caching (cache_dir, build(), find_similar(), refresh(), _load_cached_embedding(), _save_embedding())

#### Logseq File Handling (Sequential Group)

- [ ] T013 Implement Logseq markdown parser in src/logsqueak/logseq/parser.py using markdown-it-py to parse outline structure with property order preservation
- [ ] T014 Implement Logseq markdown renderer in src/logsqueak/logseq/renderer.py with minimal changes guarantee (preserve property order, use _original_line, only append children)
- [ ] T015 Implement graph path operations in src/logsqueak/logseq/graph.py with methods to locate journals/ and pages/ directories

#### RAG Infrastructure (Depends on PageIndex model)

- [ ] T016 Implement PageIndex build logic in src/logsqueak/rag/index.py with per-page caching to ~/.cache/logsqueak/embeddings/, mtime validation, and 1000-character preview

---

## Phase 3: User Story 1 - Extract and Preview Knowledge (P1)

**Goal**: Enable users to see what knowledge would be extracted from journals without modifying any files

**Independent Test**: Run `logsqueak extract 2025-01-15` on a test journal and verify preview shows extracted knowledge with target pages identified

**Success Criteria**:

- Preview displays all extracted knowledge blocks
- Each block shows target page and section (from RAG matching)
- Each block shows confidence score
- Shows which blocks would be skipped (activity logs)
- No files are modified (dry-run mode)

**Duration**: ~6-8 hours

### Tasks

#### Configuration & CLI (Parallel Group 1)

- [ ] T017 [P] [US1] Implement config loader in src/logsqueak/config/loader.py to read ~/.config/logsqueak/config.yaml with Pydantic validation and environment variable support (LOGSQUEAK_*)

- [ ] T018 [P] [US1] Implement LLM client interface in src/logsqueak/llm/client.py with methods for extraction and page selection requests

- [ ] T019 [P] [US1] Implement OpenAI-compatible provider in src/logsqueak/llm/providers/openai_compat.py using httpx with JSON mode (response_format: {type: "json_object"})

- [ ] T020 [P] [US1] Implement CLI argument parsing in src/logsqueak/cli/main.py using click with extract command, date/range parsing, --dry-run/--apply flags, --model override

#### Extraction Logic (Sequential Group - T021 must complete before T022-T026)

- [ ] T021 [US1] Implement Stage 1 extraction in src/logsqueak/extraction/extractor.py to call LLM with journal content and extract knowledge blocks (content + confidence only, no page targeting yet)

- [ ] T022 [P] [US1] Implement Stage 2 page selection in src/logsqueak/extraction/extractor.py to use PageIndex.find_similar() for top-5 candidates, then LLM selects best page + section from candidates

- [ ] T023 [P] [US1] Implement activity vs knowledge classifier in src/logsqueak/extraction/classifier.py with confidence threshold logic

- [ ] T024 [P] [US1] Implement duplicate detection in src/logsqueak/extraction/extractor.py using content_hash() comparison against target page content

- [ ] T025 [P] [US1] Implement 2000-line limit enforcement in src/logsqueak/models/journal.py load() method with truncation warning logging

- [ ] T026 [P] [US1] Implement malformed entry handling in src/logsqueak/models/journal.py with try/except, warning log, and graceful skip

#### Preview Display (Depends on Extraction Logic)

- [ ] T027 [US1] Implement preview formatter in src/logsqueak/models/preview.py display() method showing knowledge blocks, target pages with similarity scores, actions, and warnings

- [ ] T028 [US1] Implement progress feedback in src/logsqueak/cli/main.py to show index building status, extraction progress, and matching progress per SC-001

- [ ] T029 [US1] Wire up extract command in src/logsqueak/cli/main.py to load journal, build PageIndex (with cache), run extraction, display preview, exit without modifying files (dry-run)

#### Error Handling

- [ ] T030 [P] [US1] Implement LLM API error handling in src/logsqueak/llm/client.py with graceful failures, clear error messages, and no file corruption per FR-011

- [ ] T031 [P] [US1] Implement missing target page detection in src/logsqueak/extraction/extractor.py to report when TargetPage.load() returns None per FR-009

- [ ] T032 [P] [US1] Add validation for invalid journal file paths in src/logsqueak/cli/main.py with clear error message without crashing

**Deliverable**: CLI command that shows extraction preview without modifying files

**Test Command**:

```bash
logsqueak extract 2025-01-15  # Shows preview, no changes

```

---

## Phase 4: User Story 2 - Add Knowledge to Existing Pages (P2)

**Goal**: Enable users to approve previewed extractions and integrate knowledge into pages with provenance links

**Independent Test**: Run `logsqueak extract --apply 2025-01-15`, approve with "y", verify knowledge appears in correct page locations with [[2025-01-15]] provenance links

**Depends On**: User Story 1 (needs extraction and preview working)

**Success Criteria**:

- Knowledge added as child bullets in correct outline locations
- Every block includes provenance link to source journal
- Existing page content preserved (no modifications/deletions)
- Property order never reordered
- User can approve (y), reject (n), or edit (e) changes
- Page embeddings refreshed after modification

**Duration**: ~5-7 hours

### Tasks

#### Interactive Approval (Parallel Group 1)

- [ ] T033 [P] [US2] Implement interactive prompt in src/logsqueak/cli/interactive.py with y/n/e options per FR-012

- [ ] T034 [P] [US2] Implement edit mode (e option) in src/logsqueak/cli/interactive.py to allow preview modification before applying (basic implementation - can enhance later)

#### Integration Logic (Sequential Group - T035 must complete before T036-T039)

- [ ] T035 [US2] Implement integration orchestrator in src/logsqueak/integration/integrator.py to coordinate adding knowledge to pages per ProposedAction list

- [ ] T036 [P] [US2] Implement page matcher in src/logsqueak/integration/matcher.py to use TargetPage.find_section() for locating target sections in outline hierarchy

- [ ] T037 [P] [US2] Implement provenance link writer in src/logsqueak/integration/writer.py using KnowledgeBlock.with_provenance() to append [[date]] links per FR-003

- [ ] T038 [P] [US2] Implement child bullet addition in src/logsqueak/integration/writer.py using LogseqBlock.add_child(content, position) with targeted placement logic

- [ ] T039 [P] [US2] Implement fallback logic in src/logsqueak/integration/writer.py to add knowledge at page end when no clear section match exists per acceptance scenario

#### File Writing & Safety (Depends on Integration Logic)

- [ ] T040 [US2] Implement safe file writing in src/logsqueak/integration/writer.py using LogseqOutline.render() to preserve property order and structure per FR-008

- [ ] T041 [US2] Implement PageIndex refresh in src/logsqueak/integration/integrator.py to call refresh() method after modifying pages to update embeddings

- [ ] T042 [US2] Wire up --apply flag in src/logsqueak/cli/main.py to show preview, wait for approval, then call integrator to write changes

- [ ] T043 [US2] Verify provenance links in src/logsqueak/integration/writer.py - ensure 100% of integrated blocks include valid [[date]] link per SC-003

**Deliverable**: Full extraction and integration workflow with user approval

**Test Commands**:

```bash
logsqueak extract --apply 2025-01-15  # Shows preview, prompts for approval
# User types: y
# Result: Knowledge integrated into pages

```

---

## Phase 5: User Story 3 - Create New Sections When Needed (P3)

**Goal**: Intelligently create new organizational bullets when knowledge doesn't fit existing page structure

**Independent Test**: Extract knowledge that needs a new section (e.g., competitors info when page has no Competitors bullet), verify system creates appropriate structure matching page conventions

**Depends On**: User Story 2 (needs integration working)

**Success Criteria**:

- Detects when new organizational bullet is needed
- Matches page's existing outline convention (plain bullets vs "- ## Heading" style)
- Creates structure once even when multiple blocks need it
- Shows "Will create new bullet:" in preview
- Preserves property order

**Duration**: ~3-4 hours

### Tasks

#### Convention Detection (Parallel Group 1)

- [ ] T044 [P] [US3] Implement convention detector in src/logsqueak/models/page.py to analyze existing outline and determine ConventionType (PLAIN_BULLETS, HEADING_BULLETS, MIXED)

- [ ] T045 [P] [US3] Implement organizational bullet creation in src/logsqueak/integration/writer.py to create new sections matching detected convention per FR-007

#### Section Creation Logic (Depends on Convention Detection)

- [ ] T046 [US3] Implement section grouping logic in src/logsqueak/integration/integrator.py to identify when multiple knowledge blocks need the same new structure and create it once

- [ ] T047 [US3] Update preview formatter in src/logsqueak/models/preview.py to show "Will create new bullet: [Structure]" for CREATE_SECTION actions

- [ ] T048 [US3] Wire up section creation in src/logsqueak/integration/integrator.py to handle CREATE_SECTION suggested_action from LLM

- [ ] T049 [US3] Test convention matching in src/logsqueak/integration/writer.py - verify plain bullet pages get plain bullets, heading pages get headings per acceptance scenario

**Deliverable**: Intelligent section creation with convention matching

**Test Scenario**:

```bash
# Journal has: "Main competitor is Product Y"
# Target page "Project X" has no Competitors section
# Expected: Creates "- Competitors" (or "- ## Competitors" if page uses headings)
#           Adds "- Main competitor is Product Y [[2025-01-15]]" under it

```

---

## Phase 6: Polish & Cross-Cutting Concerns

**Goal**: Final touches for production readiness

**Duration**: ~2-3 hours

### Tasks

- [ ] T050 Add comprehensive error messages in src/logsqueak/llm/client.py for common issues (API key invalid, network error, rate limit, invalid JSON response)

- [ ] T051 Add logging configuration in src/logsqueak/cli/main.py with --verbose flag support and file logging to ~/.cache/logsqueak/logs/

**Deliverable**: Production-ready CLI tool with excellent error handling and logging

---

## Implementation Strategy

### MVP-First Approach

1. **Start with User Story 1 (P1)** - Extract and preview

   - Delivers immediate value (see what would be extracted)
   - No risk to user's knowledge base (read-only)
   - Validates core extraction and RAG matching

2. **Add User Story 2 (P2)** - Integration

   - Completes the value loop (actually organize knowledge)
   - Builds on validated extraction

3. **Enhance with User Story 3 (P3)** - Section creation

   - Quality-of-life improvement
   - Can be deferred if needed

### Parallelization Strategy

**Within each phase**, maximize parallel execution:

**Phase 2 Example**:

```bash
# Run in parallel (different files):

- T006 (config.py)
- T007 (journal.py)
- T008 (knowledge.py)
- T009 (page.py)
- T010 (preview.py)
- T011 (parser.py - LogseqOutline)
- T012 (page.py - PageIndex)

# Then sequentially:

- T013 (parser.py - parsing logic)
- T014 (renderer.py - depends on parser)

```

**Phase 3 Example**:

```bash
# Parallel Group 1 (independent files):

- T017 (config/loader.py)
- T018 (llm/client.py)
- T019 (llm/providers/openai_compat.py)
- T020 (cli/main.py - just argparse setup)

# Then:

- T021 (extractor.py - Stage 1)

# Then parallel:

- T022-T026 (all can run together after Stage 1 done)

# Then:

- T027-T029 (preview and wiring)

```

### Testing Notes

**Note**: This spec does not explicitly request TDD or comprehensive test coverage. Tests are marked as optional throughout.

**If you want to add tests**, consider:

- Unit tests for each model (validation, helper methods)
- Unit tests for parser/renderer (round-trip verification)
- Integration test for full extraction workflow
- Integration test for full integration workflow
- Fixtures in tests/fixtures/journals/ and tests/fixtures/pages/

**Test execution** (if implemented):

```bash
pytest tests/unit/           # Fast unit tests
pytest tests/integration/    # Slower workflow tests
pytest --cov=logsqueak      # With coverage

```

### Validation Checklist

Before considering a user story complete:

**User Story 1**:

- [ ] Can extract knowledge from test journal
- [ ] Preview shows all knowledge blocks
- [ ] RAG matching identifies correct target pages
- [ ] Similarity scores displayed
- [ ] Activity logs correctly skipped
- [ ] No files modified (dry-run)
- [ ] Error handling works (bad file path, LLM failure)

**User Story 2**:

- [ ] Can approve extraction with 'y'
- [ ] Knowledge added to correct page locations
- [ ] Provenance links present on all blocks
- [ ] Existing page content unchanged
- [ ] Property order preserved
- [ ] Can reject with 'n' (no changes)
- [ ] PageIndex refreshed after changes

**User Story 3**:

- [ ] Detects when new section needed
- [ ] Creates structure matching page convention
- [ ] Groups multiple blocks under same new section
- [ ] Preview shows section creation
- [ ] Property order still preserved

---

## Quick Reference

### File Paths Summary

**Models**: src/logsqueak/models/

- config.py (Configuration)
- journal.py (JournalEntry)
- knowledge.py (KnowledgeBlock)
- page.py (TargetPage, PageIndex)
- preview.py (ExtractionPreview, ProposedAction)

**Logseq**: src/logsqueak/logseq/

- parser.py (LogseqOutline, LogseqBlock, parsing logic)
- renderer.py (rendering logic)
- graph.py (path operations)

**RAG**: src/logsqueak/rag/

- index.py (PageIndex implementation)
- embedder.py (embedding utilities)

**Extraction**: src/logsqueak/extraction/

- extractor.py (main orchestrator, Stage 1 + Stage 2)
- classifier.py (activity vs knowledge)

**Integration**: src/logsqueak/integration/

- integrator.py (main orchestrator)
- matcher.py (RAG-based page matching)
- writer.py (provenance links, child bullets, section creation)

**LLM**: src/logsqueak/llm/

- client.py (interface)
- providers/openai_compat.py (OpenAI-compatible provider)

**CLI**: src/logsqueak/cli/

- main.py (entry point, argument parsing, extract command)
- interactive.py (y/n/e prompts)

**Config**: src/logsqueak/config/

- loader.py (YAML loading, validation)

### Dependencies Flow

```
Configuration → LLM Client → Extractor
                           ↓
JournalEntry → Parser → LogseqOutline
                           ↓
PageIndex (RAG) → Extractor (Stage 2) → KnowledgeBlock
                                            ↓
                                   ExtractionPreview
                                            ↓
                                      User Approval
                                            ↓
                              TargetPage ← Integrator → Writer

```

### Key Design Principles

1. **Property Order**: NEVER reorder (insertion order sacred)
2. **Children Placement**: Can insert at appropriate locations (targeted, minimal)
3. **Non-Destructive**: All changes are additive (no deletions/modifications)
4. **Dry-Run First**: Always preview before applying
5. **Provenance**: Every knowledge block links back to source journal
6. **RAG Matching**: Two-stage LLM (extract, then match to candidates)
7. **Per-Page Caching**: Embeddings cached with mtime validation
8. **Graceful Degradation**: LLM errors don't corrupt files

---

**Generated by**: /speckit.tasks
**Next Step**: Start with Phase 1 (Setup), then Phase 2 (Foundational), then MVP (User Story 1)
