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

- **Total Tasks**: 81
- **Phase 1 (Setup)**: 5 tasks ✅ COMPLETE
- **Phase 2 (Foundational)**: 11 tasks ✅ COMPLETE
- **Phase 2.5 (Testing - Foundational)**: 10 tasks ✅ COMPLETE
- **Phase 3 (User Story 1)**: 12 tasks
- **Phase 3.5 (Testing - LLM Integration)**: 7 tasks
- **Phase 4 (User Story 2)**: 11 tasks
- **Phase 4.5 (Testing - Integration Safety)**: 9 tasks
- **Phase 5 (User Story 3)**: 6 tasks
- **Phase 5.5 (Testing - Section Creation)**: 6 tasks
- **Phase 6 (Polish)**: 2 tasks
- **Phase 6.5 (Testing - End-to-End)**: 7 tasks

**Parallel Execution Opportunities**: 42 tasks marked [P] can run in parallel within their phase

**Current Progress**: 21/81 tasks complete (26%)

## Dependencies

### User Story Dependencies

```
Phase 1 (Setup) ✅ COMPLETE
  ↓
Phase 2 (Foundational - models, parsing, RAG index) ✅ COMPLETE
  ↓
Phase 2.5 (Testing - Foundational) ✅ COMPLETE
  ↓
Phase 3 (User Story 1 - P1) ← READY TO START
  ↓
Phase 3.5 (Testing - LLM Integration)
  ↓
Phase 4 (User Story 2 - P2)
  ↓
Phase 4.5 (Testing - Integration Safety)
  ↓
Phase 5 (User Story 3 - P3)
  ↓
Phase 5.5 (Testing - Section Creation)
  ↓
Phase 6 (Polish)
  ↓
Phase 6.5 (Testing - End-to-End)

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

- [X] T001 Create project structure per plan.md at /home/twaugh/devel/logsqueak/src/logsqueak/ with all subdirectories (cli/, config/, extraction/, integration/, rag/, llm/, logseq/, models/)
- [X] T002 Create requirements.txt with 8 core dependencies: httpx>=0.27.0, markdown-it-py>=3.0.0, PyYAML>=6.0.1, pydantic>=2.0.0, click>=8.1.0, python-dateutil>=2.8.0, sentence-transformers>=2.2.0, numpy>=1.24.0
- [X] T003 Create pyproject.toml with project metadata, entry points for CLI (logsqueak = logsqueak.cli.main:main), and Python 3.11+ requirement
- [X] T004 Create tests/ directory structure with unit/, integration/, and fixtures/ subdirectories
- [X] T005 Create LICENSE file with GPLv3 text (already exists, verify it's present)

---

## Phase 2: Foundational Components

**Goal**: Build core data models, Logseq parsing, and RAG infrastructure that all user stories depend on

**Duration**: ~4-6 hours

**Why before user stories**: These are blocking dependencies - every user story needs models, parsing, and page indexing.

### Tasks

#### Data Models (Parallel Group 1)

- [X] T006 [P] Implement Configuration model in src/logsqueak/models/config.py with Pydantic validation for LLM config (endpoint, api_key, model) and Logseq config (graph_path)
- [X] T007 [P] Implement JournalEntry model in src/logsqueak/models/journal.py with attributes (date, file_path, raw_content, outline, line_count) and load() classmethod
- [X] T008 [P] Implement KnowledgeBlock model in src/logsqueak/models/knowledge.py with attributes (content, source_date, confidence, target_page, target_section: Optional[List[str]], suggested_action) and helper methods (content_hash(), provenance_link(), with_provenance(), section_path())
- [X] T009 [P] Implement TargetPage model in src/logsqueak/models/page.py with attributes (name, file_path, outline, organizational_convention) and methods (load(), find_section(), has_duplicate())
- [X] T010 [P] Implement ExtractionPreview model in src/logsqueak/models/preview.py with ProposedAction and display() method for terminal output
- [X] T011 [P] Implement LogseqOutline and LogseqBlock models in src/logsqueak/logseq/parser.py with attributes preserving property order (dict in Python 3.7+), _original_line for round-trip, and add_child(content, position) method
- [X] T012 [P] Implement PageIndex model in src/logsqueak/models/page.py with per-page caching (cache_dir, build(), find_similar(), refresh(), _load_cached_embedding(), _save_embedding())

#### Logseq File Handling (Sequential Group)

- [X] T013 Implement Logseq markdown parser in src/logsqueak/logseq/parser.py using markdown-it-py to parse outline structure with property order preservation
- [X] T014 Implement Logseq markdown renderer in src/logsqueak/logseq/renderer.py with minimal changes guarantee (preserve property order, use _original_line, only append children)
- [X] T015 Implement graph path operations in src/logsqueak/logseq/graph.py with methods to locate journals/ and pages/ directories

#### RAG Infrastructure (Depends on PageIndex model)

- [X] T016 Implement PageIndex build logic in src/logsqueak/rag/index.py with per-page caching to ~/.cache/logsqueak/embeddings/, mtime validation, and 1000-character preview

---

## Phase 2.5: Testing Foundational Components

**Goal**: Comprehensive test coverage for all foundational components to ensure correctness before proceeding to user story implementation

**Duration**: ~3-4 hours

**Status**: ✅ COMPLETE

**Test Results**:

- **Core Tests**: 97/97 passing (100%) ✅
- **PageIndex Tests**: 4 additional tests (require sentence-transformers)
- **Total Coverage**: 111 tests when sentence-transformers installed
- **Test Execution Time**: ~0.15 seconds (fast!)

### Tasks

#### Test Infrastructure (Parallel Group 1)

- [X] T016a [P] Create tests/conftest.py with shared fixtures for journal content, page content, and graph structures
- [X] T016b [P] Configure pytest in pyproject.toml with pythonpath = ["src"] to run tests from source without installation
- [X] T016c [P] Create setup-dev.sh script for automated virtual environment setup with Python 3.11+ detection

#### Unit Tests (Parallel Group 2)

- [X] T016d [P] Create tests/unit/test_config.py (10 tests) - Configuration validation, YAML loading, error handling
- [X] T016e [P] Create tests/unit/test_journal.py (12 tests) - FR-019 2000-line limit, FR-018 malformed handling, date parsing, nested structure
- [X] T016f [P] Create tests/unit/test_knowledge.py (12 tests) - FR-003 provenance links, FR-017 duplicate detection, content hashing
- [X] T016g [P] Create tests/unit/test_parser.py (22 tests) - FR-008 property order preservation, round-trip parsing, nested structures
- [X] T016h [P] Create tests/unit/test_graph.py (16 tests) - Path operations, validation, directory listing
- [X] T016i [P] Create tests/unit/test_preview.py (14 tests) - Preview formatting, action status, display
- [X] T016j [P] Create tests/unit/test_page.py (14 tests) - TargetPage operations (10 tests passing), PageIndex with RAG (4 tests require sentence-transformers)

#### Integration Tests

- [X] T016k Create tests/integration/test_parsing_roundtrip.py (11 tests) - Full parse → modify → render workflows, property preservation, complex scenarios

#### Documentation

- [X] T016l Create TEST_RESULTS.md documenting all test results, coverage, known limitations (PageIndex tests require sentence-transformers), and CI readiness
- [X] T016m Update README.md with virtual environment setup instructions and test execution commands
- [X] T016n Update quickstart.md with venv warnings and development workflow

### Key Achievements

**Functional Requirements Validated**:

- ✅ FR-002: Dry-run mode (preview) - test_preview.py
- ✅ FR-003: Provenance links - test_knowledge.py
- ✅ FR-006: Additive operations only - test_parser.py (round-trip)
- ✅ FR-008: Preserve existing content - test_parser.py
- ✅ FR-017: Duplicate detection - test_knowledge.py
- ✅ FR-018: Malformed entry handling - test_journal.py
- ✅ FR-019: 2000-line limit - test_journal.py
- ✅ FR-009: Missing page detection - test_page.py (requires sentence-transformers)

**Success Criteria Validated**:

- ✅ SC-003: 100% provenance links - test_knowledge.py
- ✅ SC-004: 100% data safety - test_parser.py (round-trip)
- ✅ SC-007: Dry-run mode mandatory - test_preview.py

**Test Quality**:

- ✅ Comprehensive edge case coverage (empty files, malformed content, boundary conditions)
- ✅ Round-trip verification (parse → render → parse)
- ✅ Property order preservation validated
- ✅ Fast execution (<1 second for all core tests)
- ✅ No external dependencies needed (except sentence-transformers for PageIndex)
- ✅ CI-ready with clear pass/fail status

### Known Limitations

**PageIndex Tests Require sentence-transformers**:

- **Reason**: PageIndex uses sentence-transformers library (~500MB download)
- **Impact**: 4 PageIndex tests require the library to be installed in venv
- **Workaround**: All TargetPage tests (10 tests) pass without it
- **To install**: `pip install sentence-transformers>=2.2.0` in activated venv

### Test Execution Commands

```bash
# Activate venv first!
source venv/bin/activate

# Run core tests (no heavy dependencies)
pytest tests/unit/test_config.py \
       tests/unit/test_journal.py \
       tests/unit/test_knowledge.py \
       tests/unit/test_parser.py \
       tests/unit/test_graph.py \
       tests/unit/test_preview.py \
       tests/integration/ -v
# Result: 97 passed ✅

# Run all tests including PageIndex (requires sentence-transformers)
pytest tests/ -v
# Result: 111 passed ✅ (if sentence-transformers installed)

```

### Fixes Applied

**Issue 1: ModuleNotFoundError**

- **Problem**: Tests couldn't import logsqueak modules
- **Fix**: Added `pythonpath = ["src"]` to pyproject.toml
- **Result**: Tests now run from source without installation

**Issue 2: setup-dev.sh Python Version Detection**

- **Problem**: Script only looked for python3.11, failed on python3.13
- **Fix**: Added version detection loop checking python3.13, python3.12, python3.11, python3
- **Result**: Works with any Python 3.11+ installation

**Issue 3: PageIndex Import Failures**

- **Problem**: Module imported sentence-transformers at top level, failing all imports
- **Fix**: Made imports lazy using TYPE_CHECKING and importing inside methods only
- **Result**: 10/14 test_page.py tests pass without sentence-transformers, all 14 with it

### Current Task

**Next Step**: Phase 3 implementation can begin - all foundational components will be fully tested and validated ✅

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

## Phase 3.5: Testing LLM Integration & Extraction

**Goal**: Comprehensive testing of LLM-dependent extraction workflows and RAG matching logic

**Duration**: ~2-3 hours

**Why Important**: Phase 3 introduces LLM integration and RAG matching - the most complex and error-prone parts. Validates that extraction works correctly before adding file write operations.

### Tasks

#### LLM Client Tests (Parallel Group 1)

- [ ] T032a [P] [Test] Create tests/unit/test_llm_client.py (15 tests) - Mock LLM responses, JSON parsing, error handling (network errors, invalid JSON, rate limits, API key validation)

- [ ] T032b [P] [Test] Create tests/unit/test_openai_provider.py (12 tests) - OpenAI-compatible provider with mocked httpx responses, JSON mode validation, request formatting

#### Extraction Workflow Tests (Parallel Group 2)

- [ ] T032c [P] [Test] Create tests/unit/test_extractor.py (18 tests) - Stage 1 extraction with mocked LLM, Stage 2 page selection, confidence threshold logic, duplicate detection

- [ ] T032d [P] [Test] Create tests/unit/test_classifier.py (10 tests) - Activity vs knowledge classification, confidence scoring, edge cases (ambiguous content)

#### RAG & Matching Tests (Parallel Group 3)

- [ ] T032e [P] [Test] Create tests/unit/test_matcher.py (14 tests) - Semantic similarity calculations, top-5 candidate selection, section matching within pages, similarity score validation

- [ ] T032f [P] [Test] Create tests/integration/test_extraction_workflow.py (16 tests) - End-to-end extraction with mocked LLM, RAG index → extract → match → preview pipeline, error propagation

#### Preview Generation Tests

- [ ] T032g [Test] Update tests/unit/test_preview.py with additional tests (8 new tests) - Preview rendering with RAG scores, action status display, warning messages, multi-block previews

**Test Coverage Goals**:

- ✅ FR-001: Knowledge extraction validated
- ✅ FR-004: RAG-based page selection
- ✅ FR-010: Activity log filtering (classifier)
- ✅ FR-011: LLM API error handling
- ✅ FR-017: Duplicate detection
- ✅ SC-001: Progress feedback rendering
- ✅ SC-005: Top-5 RAG candidates

**Deliverable**: Complete test coverage for extraction pipeline before file writes are introduced

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

## Phase 4.5: Testing Integration Safety & File Writing

**Goal**: Validate data safety guarantees and file writing operations - the highest risk area of the system

**Duration**: ~2-3 hours

**Why Important**: Phase 4 introduces file writes - must validate SC-004 (100% data safety) before proceeding to Phase 5. Critical for user trust.

### Tasks

#### File Safety Tests (Parallel Group 1)

- [ ] T043a [P] [Test] Create tests/integration/test_file_safety.py (20 tests) - Round-trip verification after integration, property order preservation under modification, no data loss scenarios, atomic write verification

- [ ] T043b [P] [Test] Create tests/unit/test_writer.py (16 tests) - Child bullet addition, provenance link formatting, fallback logic, targeted placement accuracy

#### Provenance & Integrity Tests (Parallel Group 2)

- [ ] T043c [P] [Test] Create tests/integration/test_provenance.py (12 tests) - Verify 100% of integrated blocks have valid [[date]] provenance links (SC-003), link format validation, date parsing

- [ ] T043d [P] [Test] Create tests/unit/test_integrator.py (14 tests) - Integration orchestration, ProposedAction handling, error recovery, partial failure handling

#### Interactive & Approval Tests (Parallel Group 3)

- [ ] T043e [P] [Test] Create tests/unit/test_interactive.py (10 tests) - Mock user input (y/n/e), approval workflow, edit mode, cancel scenarios

- [ ] T043f [P] [Test] Create tests/integration/test_pageindex_refresh.py (8 tests) - Verify embeddings update after page modifications, cache invalidation, mtime tracking

#### Error Recovery & Edge Cases

- [ ] T043g [Test] Create tests/integration/test_error_recovery.py (15 tests) - Partial failures don't corrupt files, disk full scenarios, permission errors, concurrent modification detection

- [ ] T043h [Test] Create tests/integration/test_duplicate_prevention.py (10 tests) - FR-017 duplicate detection prevents re-adding same content, content_hash() validation, multiple runs safety

- [ ] T043i [Test] Create tests/integration/test_integration_workflow.py (18 tests) - Full workflow: extract → preview → approve → integrate → verify, multi-block integration, rollback on error

**Test Coverage Goals**:

- ✅ SC-004: 100% data safety (no file corruption)
- ✅ SC-003: 100% provenance links
- ✅ FR-003: Provenance links mandatory
- ✅ FR-006: Additive operations only
- ✅ FR-008: Preserve existing content
- ✅ FR-012: User approval required
- ✅ FR-017: Duplicate detection
- ✅ Property order preservation under all scenarios

**Deliverable**: Complete safety validation before section creation features are added

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

## Phase 5.5: Testing Section Creation & Convention Matching

**Goal**: Test organizational bullet creation and convention detection - validate structural modifications don't corrupt pages

**Duration**: ~1-2 hours

**Why Important**: Phase 5 adds structural modifications (creating new sections). Must verify it respects page conventions and doesn't corrupt structure.

### Tasks

#### Convention Detection Tests (Parallel Group 1)

- [ ] T049a [P] [Test] Create tests/unit/test_convention_detector.py (16 tests) - Plain bullets detection, heading bullets detection, mixed styles, empty pages, property-only pages

- [ ] T049b [P] [Test] Create tests/unit/test_section_creator.py (14 tests) - Section creation matching conventions, structure formatting, nesting levels, placement logic

#### Section Creation Integration Tests (Parallel Group 2)

- [ ] T049c [P] [Test] Create tests/integration/test_section_creation.py (18 tests) - End-to-end section creation, convention matching validation, grouping multiple blocks under same section

- [ ] T049d [P] [Test] Create tests/integration/test_convention_preservation.py (12 tests) - Verify plain bullet pages stay plain, heading pages stay heading-based, mixed pages handle correctly

#### Edge Cases & Safety

- [ ] T049e [Test] Create tests/integration/test_section_edge_cases.py (10 tests) - Empty pages, deeply nested structures, pages with only properties, conflicting conventions

- [ ] T049f [Test] Update tests/integration/test_file_safety.py (8 new tests) - Round-trip verification after section creation, no corruption from structural changes

**Test Coverage Goals**:

- ✅ FR-007: New organizational bullets created when needed
- ✅ Convention detection accuracy (plain vs heading bullets)
- ✅ Grouping logic (multiple blocks → one section)
- ✅ Preview shows "Will create new bullet:" correctly
- ✅ Property order still preserved with structural changes

**Deliverable**: Validated section creation before final polish

---

## Phase 6: Polish & Cross-Cutting Concerns

**Goal**: Final touches for production readiness

**Duration**: ~2-3 hours

### Tasks

- [ ] T050 Add comprehensive error messages in src/logsqueak/llm/client.py for common issues (API key invalid, network error, rate limit, invalid JSON response)

- [ ] T051 Add logging configuration in src/logsqueak/cli/main.py with --verbose flag support and file logging to ~/.cache/logsqueak/logs/

**Deliverable**: Production-ready CLI tool with excellent error handling and logging

---

## Phase 6.5: End-to-End & Regression Testing

**Goal**: Final validation with full workflow tests and regression testing before release

**Duration**: ~2-3 hours

**Why Important**: Final gate before release. Ensures all components work together, no regressions, and real-world scenarios work correctly.

### Tasks

#### Full Workflow Tests (Parallel Group 1)

- [ ] T051a [P] [Test] Create tests/e2e/test_full_workflow.py (20 tests) - Real journal entries → extract → preview → approve → verify integrated, multi-step workflows, complex scenarios

- [ ] T051b [P] [Test] Create tests/e2e/test_multi_entry.py (12 tests) - Process multiple journal entries in sequence, state management, cache consistency across runs

#### Real World Scenarios (Parallel Group 2)

- [ ] T051c [P] [Test] Create tests/e2e/test_real_world.py (15 tests) - Realistic Logseq graph data, complex page structures, mixed content types, real-world patterns

- [ ] T051d [P] [Test] Create tests/performance/test_scale.py (10 tests) - 2000-line journals, 1000+ page graphs, large PageIndex performance, memory usage validation

#### Error Scenarios & Edge Cases

- [ ] T051e [Test] Create tests/e2e/test_error_scenarios.py (18 tests) - Network failures, malformed LLM responses, corrupted files, timeout handling, rate limiting

- [ ] T051f [Test] Create tests/e2e/test_cli_integration.py (14 tests) - Full CLI command execution, argument parsing, flag combinations, error message clarity

#### Regression & Validation

- [ ] T051g [Test] Run full regression suite (automated) - Re-run all previous test suites (Phases 2.5, 3.5, 4.5, 5.5), verify no regressions, generate final test report

**Test Coverage Goals**:

- ✅ All user stories work end-to-end
- ✅ Real-world Logseq graphs processed correctly
- ✅ Performance meets SC-002 goals (566 pages in ~20s)
- ✅ FR-019: 2000-line limit enforced
- ✅ SC-006: Acceptable LLM latency
- ✅ All error scenarios handled gracefully
- ✅ No regressions from earlier phases

**Deliverable**: Release-ready system with full validation

**Test Execution**:

```bash
# Run full test suite
pytest tests/ -v --cov=logsqueak --cov-report=html

# Run only E2E tests
pytest tests/e2e/ -v

# Run performance tests
pytest tests/performance/ -v

```

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
