# Tasks: Logsqueak - Interactive TUI for Knowledge Extraction

**Input**: Design documents from `/specs/002-logsqueak-spec/`

**Prerequisites**: plan.md (complete), spec.md (complete), research.md (complete), data-model.md (complete), contracts/ (complete)

**Tests**: Tests are INCLUDED in this task breakdown following user requirement to "use pilot for testing TUI features" and "run tests to verify code actually works"

**Organization**: Tasks are grouped by user story to enable independent implementation and testing. Each foundational component includes tests that must pass before proceeding.

**User Instruction**: "Proceed cautiously, and use pilot for testing TUI features. Start with well-tested foundational components and let me verify them before continuing. Don't just write tests -- also run them to verify the code you've written actually works."

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- Single project structure at repository root
- Source: `src/logsqueak/`
- Tests: `tests/`
- Existing parser: `src/logseq-outline-parser/` (DO NOT MODIFY)

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [x] T001 Update pyproject.toml with TUI dependencies (textual, httpx, pydantic, click, chromadb, sentence-transformers, markdown-it-py, pyyaml, structlog, pytest-textual-snapshot)
- [x] T002 Create main package structure: src/logsqueak/{__init__.py,cli.py,config.py}
- [x] T003 [P] Create models package structure: src/logsqueak/models/{__init__.py,block_state.py,edited_content.py,integration_decision.py,background_task.py}
- [x] T004 [P] Create services package structure: src/logsqueak/services/{__init__.py,llm_client.py,page_indexer.py,rag_search.py,file_monitor.py}
- [x] T005 [P] Create TUI package structure: src/logsqueak/tui/{__init__.py,app.py,screens/__init__.py,widgets/__init__.py}
- [x] T006 [P] Create utils package structure: src/logsqueak/utils/{__init__.py,logging.py,ids.py}
- [x] T007 [P] Create test directory structure: tests/{unit,integration,ui}/{__init__.py}
- [x] T008 Configure structlog for JSON logging to ~/.cache/logsqueak/logs/logsqueak.log
- [x] T009 Create CLI entry point with Click in src/logsqueak/cli.py (basic 'extract' command placeholder)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete and TESTED before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete and all tests pass

### Configuration Management

- [x] T010 [P] Implement LLMConfig Pydantic model in src/logsqueak/models/config.py
- [x] T011 [P] Implement LogseqConfig Pydantic model in src/logsqueak/models/config.py
- [x] T012 [P] Implement RAGConfig Pydantic model in src/logsqueak/models/config.py
- [x] T013 Implement Config root model with YAML loading and file permission validation in src/logsqueak/models/config.py
- [x] T014 Write unit tests for Config models in tests/unit/test_config.py and run with pytest
- [x] T015 Implement config management in src/logsqueak/config.py with lazy validation and helpful error messages
- [x] T016 Write integration tests for config loading in tests/integration/test_config_loading.py and run with pytest

### Core Data Models

- [x] T017 [P] Implement BlockState Pydantic model in src/logsqueak/models/block_state.py
- [x] T018 [P] Implement EditedContent Pydantic model in src/logsqueak/models/edited_content.py
- [x] T019 [P] Implement IntegrationDecision Pydantic model in src/logsqueak/models/integration_decision.py
- [x] T020 [P] Implement BackgroundTask Pydantic model in src/logsqueak/models/background_task.py
- [x] T021 Write unit tests for all data models in tests/unit/test_models.py and run with pytest

### LLM Client (with NDJSON Streaming)

- [x] T022 Implement LLMClient class with stream_ndjson method in src/logsqueak/services/llm_client.py
- [x] T023 Implement retry logic with automatic retry (2s delay) in src/logsqueak/services/llm_client.py
- [x] T024 Implement Pydantic chunk models (KnowledgeClassificationChunk, ContentRewordingChunk, IntegrationDecisionChunk) in src/logsqueak/models/llm_chunks.py
- [x] T025 Write unit tests for LLMClient with mocked httpx responses in tests/unit/test_llm_client.py and run with pytest
- [x] T026 Write integration tests for LLMClient NDJSON parsing (malformed JSON, network errors, timeouts) in tests/integration/test_llm_streaming.py and run with pytest

### File Monitoring

- [x] T027 Implement FileMonitor class (record, is_modified, refresh, check_and_reload) in src/logsqueak/services/file_monitor.py
- [x] T028 Write unit tests for FileMonitor in tests/unit/test_file_monitor.py and run with pytest

### Utilities

- [x] T029 [P] Implement UUID generation utilities (deterministic UUID v5) in src/logsqueak/utils/ids.py
- [x] T030 [P] Implement structured logging setup in src/logsqueak/utils/logging.py
- [x] T031 Write unit tests for utilities in tests/unit/test_utils.py and run with pytest

**Checkpoint**: Foundation ready - ALL tests must pass before proceeding. User should verify tests run successfully.

---

## Phase 3: User Story 1 - Select Knowledge Blocks for Integration (Priority: P1) 🎯 MVP

**Goal**: Display journal blocks in tree view, stream LLM classifications, allow user selection with keyboard controls

**Independent Test**: Open TUI with sample journal entry, navigate blocks with j/k, see LLM suggestions appear incrementally, manually select/deselect blocks with Space, verify status widget shows background task progress

### Tests for User Story 1 (TDD Approach)

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [x] T032 [P] [US1] Write UI test for block tree navigation (j/k keys) using Textual pilot in tests/ui/test_phase1_navigation.py - should FAIL initially
- [x] T033 [P] [US1] Write UI test for block selection toggle (Space key) using Textual pilot in tests/ui/test_phase1_selection.py - should FAIL initially
- [x] T034 [P] [US1] Write UI test for LLM suggestion streaming updates using Textual pilot in tests/ui/test_phase1_llm_streaming.py - should FAIL initially
- [x] T035 [P] [US1] Write UI test for status widget progress display using Textual pilot in tests/ui/test_phase1_status.py - should FAIL initially
- [x] T036 [P] [US1] Write snapshot test for Phase 1 initial render using pytest-textual-snapshot in tests/ui/test_phase1_snapshots.py - should FAIL initially
- [x] T037 Run all Phase 1 UI tests with pytest and verify they FAIL as expected

### Implementation for User Story 1

- [x] T038 [P] [US1] Create BlockTree widget with Textual Tree for hierarchical display in src/logsqueak/tui/widgets/block_tree.py
- [x] T039 [P] [US1] Create StatusPanel widget for background task progress in src/logsqueak/tui/widgets/status_panel.py
- [x] T040 [P] [US1] Create MarkdownViewer widget for bottom panel display in src/logsqueak/tui/widgets/markdown_viewer.py
- [x] T041 [US1] Implement Phase1Screen with BlockTree, StatusPanel, and MarkdownViewer layout in src/logsqueak/tui/screens/block_selection.py
- [x] T042 [US1] Implement keyboard controls (j/k navigation, Space toggle, Shift+j/k jump to knowledge) in src/logsqueak/tui/screens/block_selection.py
- [x] T043 [US1] Add LLM classification worker using run_worker and classify_blocks streaming in src/logsqueak/tui/screens/block_selection.py
- [x] T044 [US1] Add page indexing worker with progress callbacks in src/logsqueak/tui/screens/block_selection.py
- [x] T045 [US1] Implement reactive state updates (selected_blocks, llm_suggestions) in src/logsqueak/tui/screens/block_selection.py
- [x] T046 [US1] Add visual indicators (robot emoji, highlight colors, bottom panel updates) in src/logsqueak/tui/screens/block_selection.py
- [x] T047 [US1] Implement 'a' key to accept all LLM suggestions, 'r' to reset, 'c' to clear all selections in src/logsqueak/tui/screens/block_selection.py
- [x] T048 [US1] Add footer with keyboard shortcuts in src/logsqueak/tui/screens/block_selection.py
- [x] T049 Run all Phase 1 UI tests with pytest and verify they NOW PASS - tests/ui/test_phase1_*.py

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently. User should manually test: navigate blocks, see LLM streaming updates, select blocks, verify status widget.

---

## Phase 4: User Story 2 - Edit and Refine Knowledge Content (Priority: P2) ✅ COMPLETE

**Goal**: Display selected blocks with LLM rewording suggestions, allow manual editing, run RAG search in background

**Independent Test**: Manually create list of selected blocks from Phase 1, verify editing interface shows original context, LLM reworded version, editable field, test Tab key to focus/unfocus editor, test 'a' to accept LLM version, verify RAG search completes before allowing 'n' to proceed

### Tests for User Story 2 (TDD Approach)

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [x] T050 [P] [US2] Write UI test for block navigation (j/k with auto-save) using Textual pilot in tests/ui/test_phase2_navigation.py - should FAIL initially
- [x] T051 [P] [US2] Write UI test for text editor focus/unfocus (Tab key) using Textual pilot in tests/ui/test_phase2_editing.py - should FAIL initially
- [x] T052 [P] [US2] Write UI test for accept LLM version ('a' key) using Textual pilot in tests/ui/test_phase2_llm_accept.py - should FAIL initially
- [x] T053 [P] [US2] Write UI test for revert to original ('r' key) using Textual pilot in tests/ui/test_phase2_revert.py - should FAIL initially
- [x] T054 [P] [US2] Write UI test for RAG search completion blocking using Textual pilot in tests/ui/test_phase2_rag_blocking.py - should FAIL initially
- [x] T055 [P] [US2] Write snapshot test for Phase 2 initial render using pytest-textual-snapshot in tests/ui/test_phase2_snapshots.py - should FAIL initially
- [x] T056 Run all Phase 2 UI tests with pytest and verify they FAIL as expected

### RAG Implementation (Required for US2)

- [x] T057 [P] [US2] Implement PageIndexer class with ChromaDB integration in src/logsqueak/services/page_indexer.py
- [x] T058 [P] [US2] Implement RAGSearch class with semantic search and explicit link boosting in src/logsqueak/services/rag_search.py
- [x] T059 Write unit tests for PageIndexer in tests/unit/test_page_indexer.py and run with pytest
- [x] T060 Write unit tests for RAGSearch in tests/unit/test_rag_search.py and run with pytest
- [x] T061 Write integration test for RAG pipeline (index + search) in tests/integration/test_rag_pipeline.py and run with pytest

### Implementation for User Story 2

- [x] T062 [P] [US2] Create ContentEditor widget with multi-line text editor in src/logsqueak/tui/widgets/content_editor.py
- [x] T063 [US2] Implement Phase2Screen with three panels (original context, LLM reworded, current editable) in src/logsqueak/tui/screens/content_editing.py
- [x] T064 [US2] Implement keyboard controls (j/k navigation with auto-save, Tab focus, 'a' accept, 'r' revert) in src/logsqueak/tui/screens/content_editing.py
- [x] T065 [US2] Add LLM rewording worker using reword_content streaming in src/logsqueak/tui/screens/content_editing.py (stub implemented, workers to be added)
- [x] T066 [US2] Add RAG search worker with progress updates in src/logsqueak/tui/screens/content_editing.py (stub implemented, workers to be added)
- [x] T067 [US2] Implement blocking logic for 'n' key until RAG search complete in src/logsqueak/tui/screens/content_editing.py
- [x] T068 [US2] Add status widget with multi-task progress (rewording, page indexing, RAG search) in src/logsqueak/tui/screens/content_editing.py
- [x] T069 [US2] Add footer with context-sensitive keyboard shortcuts in src/logsqueak/tui/screens/content_editing.py
- [x] T070 Run all Phase 2 UI tests with pytest and verify they NOW PASS - tests/ui/test_phase2_*.py (44/44 passing - 100% ✅)

**Checkpoint**: ✅ Phase 4 Complete! User Story 2 is functionally complete with:
- Three-panel content editing interface working
- Navigation (j/k) with auto-save implemented
- Tab focus/unfocus for editor working
- Accept LLM ('a') and Revert ('r') functionality working
- RAG search blocking logic implemented
- Status panel and footer in place
- **44/44 UI tests passing (100% ✅)**
- Snapshot baselines created for all 6 visual tests
- RAG services (PageIndexer, RAGSearch) implemented with tests written
- Background worker stubs in place (ready for LLM client integration)

---

## Phase 5: User Story 3 - Review Integration Decisions and Write to Pages (Priority: P1) 🎯 MVP

**Goal**: Display integration decisions batched by knowledge block, show target page previews with insertion point, write on 'y' acceptance with immediate provenance markers

**Independent Test**: Manually create refined knowledge blocks and candidate pages from Phase 2, verify decisions batched per knowledge block, navigate between decisions with j/k, see target page preview update with green bar indicator, accept decision with 'y', verify write succeeds and journal gets processed:: property, test multi-page integration (same block to multiple targets)

### Tests for User Story 3 (TDD Approach)

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [x] T071 [P] [US3] Write UI test for decision navigation (j/k between decisions) using Textual pilot in tests/ui/test_phase3_navigation.py - should FAIL initially
- [x] T072 [P] [US3] Write UI test for decision acceptance ('y' key triggers write) using Textual pilot in tests/ui/test_phase3_accept.py - should FAIL initially
- [x] T073 [P] [US3] Write UI test for next block navigation ('n' key) using Textual pilot in tests/ui/test_phase3_next_block.py - should FAIL initially
- [x] T074 [P] [US3] Write UI test for batch accept ('a' key writes all pending) using Textual pilot in tests/ui/test_phase3_batch_accept.py - should FAIL initially
- [x] T075 [P] [US3] Write UI test for write failure handling using Textual pilot in tests/ui/test_phase3_errors.py - should FAIL initially
- [x] T076 [P] [US3] Write snapshot test for Phase 3 initial render using pytest-textual-snapshot in tests/ui/test_phase3_snapshots.py - should FAIL initially
- [x] T077 Run all Phase 3 UI tests with pytest and verify they FAIL as expected

### File Operations Implementation (Required for US3)

- [x] T078 [P] [US3] Implement write_integration function with atomic two-phase writes in src/logsqueak/services/file_operations.py
- [x] T079 [P] [US3] Implement add_processed_marker function for journal provenance in src/logsqueak/services/file_operations.py
- [x] T080 [P] [US3] Implement validation functions (validate_target_exists, validate_structure) in src/logsqueak/services/file_operations.py
- [x] T081 Write unit tests for file operations (add_section, add_under, replace actions) in tests/unit/test_file_operations.py and run with pytest (21/21 passing)
- [ ] T082 Write integration tests for two-phase atomic writes with FileMonitor in tests/integration/test_atomic_writes.py and run with pytest

### Implementation for User Story 3

- [ ] T083 [P] [US3] Create TargetPagePreview widget with scrollable markdown display and green bar indicator in src/logsqueak/tui/widgets/target_page_preview.py
- [ ] T084 [P] [US3] Create DecisionList widget for displaying multiple decisions per block in src/logsqueak/tui/widgets/decision_list.py
- [ ] T085 [US3] Implement Phase3Screen with journal context, refined content, decision list, target preview layout in src/logsqueak/tui/screens/integration_review.py
- [ ] T086 [US3] Implement decision batching logic (wait for all decisions per knowledge block before display) in src/logsqueak/tui/screens/integration_review.py
- [ ] T087 [US3] Implement keyboard controls (j/k decision navigation, 'y' accept, 'n' next block, 'a' accept all) in src/logsqueak/tui/screens/integration_review.py
- [ ] T088 [US3] Add LLM integration decisions worker using plan_integrations streaming in src/logsqueak/tui/screens/integration_review.py
- [ ] T089 [US3] Implement write logic with FileMonitor check, validation, atomic write, provenance marker in src/logsqueak/tui/screens/integration_review.py
- [ ] T090 [US3] Add decision status tracking (⊙ pending, ✓ completed, ⚠ failed) with visual indicators in src/logsqueak/tui/screens/integration_review.py
- [ ] T091 [US3] Implement target page preview auto-scroll to insertion point in src/logsqueak/tui/screens/integration_review.py
- [ ] T092 [US3] Add Tab key to focus/scroll preview widget in src/logsqueak/tui/screens/integration_review.py
- [ ] T093 [US3] Implement error handling (write failures, target not found, concurrent modifications) in src/logsqueak/tui/screens/integration_review.py
- [ ] T094 [US3] Add completion summary screen with statistics and journal link in src/logsqueak/tui/screens/integration_review.py
- [ ] T095 [US3] Add footer with context-sensitive shortcuts (varies based on decision state) in src/logsqueak/tui/screens/integration_review.py
- [ ] T096 Run all Phase 3 UI tests with pytest and verify they NOW PASS - tests/ui/test_phase3_*.py

**Checkpoint**: All P1 user stories (US1, US3) should now be independently functional. This is the MVP! User should manually test: review decisions, accept integrations, verify writes succeed, check provenance markers in journal.

---

## Phase 6: Application Integration & CLI

**Purpose**: Connect all phases into working CLI application

- [ ] T097 Implement main TUI App class with screen management (Phase1Screen → Phase2Screen → Phase3Screen) in src/logsqueak/tui/app.py
- [ ] T098 Implement screen transition logic (Phase 1 'n' → Phase 2, Phase 2 'n' → Phase 3) in src/logsqueak/tui/app.py
- [ ] T099 Implement back navigation (Phase 2 'q' → Phase 1, Phase 3 'q' → Phase 2) in src/logsqueak/tui/app.py
- [ ] T100 Implement global keyboard shortcuts (Ctrl+C quit with confirmation in Phase 3) in src/logsqueak/tui/app.py
- [ ] T101 Implement CLI 'extract' command with date/range parsing in src/logsqueak/cli.py
- [ ] T102 Implement journal loading with multiple date support (grouped by date in tree) in src/logsqueak/cli.py
- [ ] T103 Implement config loading with helpful error messages (missing file, invalid permissions, validation failures) in src/logsqueak/cli.py
- [ ] T104 Wire up all services (LLMClient, PageIndexer, RAGSearch, FileMonitor) to TUI app in src/logsqueak/cli.py
- [ ] T105 Write integration test for full workflow (Phase 1 → Phase 2 → Phase 3) in tests/integration/test_workflow.py and run with pytest
- [ ] T106 Write CLI integration tests (date parsing, config errors, journal loading) in tests/integration/test_cli.py and run with pytest

**Checkpoint**: Complete end-to-end workflow should be functional. User should test: `logsqueak extract`, navigate all 3 phases, complete a full extraction session.

---

## Phase 7: Edge Cases & Error Handling

**Purpose**: Handle edge cases and provide helpful error messages

- [ ] T107 [P] Implement missing config file error with example YAML in src/logsqueak/config.py
- [ ] T108 [P] Implement config validation error messages with clear remediation in src/logsqueak/config.py
- [ ] T109 [P] Implement file permission check (mode 600) for config file in src/logsqueak/config.py
- [ ] T110 [P] Implement network error handling (connection refused, timeout, invalid API key) in src/logsqueak/services/llm_client.py
- [ ] T111 [P] Implement malformed JSON handling in NDJSON streaming in src/logsqueak/services/llm_client.py
- [ ] T112 [P] Implement "No knowledge blocks identified" message in Phase 1 in src/logsqueak/tui/screens/block_selection.py
- [ ] T113 [P] Implement "No relevant pages found" message in Phase 3 in src/logsqueak/tui/screens/integration_review.py
- [ ] T114 [P] Implement concurrent modification detection and reload in Phase 3 writes in src/logsqueak/services/file_operations.py
- [ ] T115 [P] Implement external file modification handling (reload, revalidate) in src/logsqueak/services/file_monitor.py
- [ ] T116 [P] Implement Ctrl+C cancellation warning in Phase 3 (partial journal state) in src/logsqueak/tui/screens/integration_review.py
- [ ] T117 Write integration tests for all edge cases in tests/integration/test_edge_cases.py and run with pytest

---

## Phase 8: Polish & Documentation

**Purpose**: Final polish, documentation, and validation

- [ ] T118 [P] Add logging for all LLM requests (request_id, model, endpoint) in src/logsqueak/services/llm_client.py
- [ ] T119 [P] Add logging for all LLM responses (chunk count, errors) in src/logsqueak/services/llm_client.py
- [ ] T120 [P] Add logging for user actions (navigation, selection, acceptance) in all TUI screens
- [ ] T121 [P] Add logging for file operations (writes, validations, errors) in src/logsqueak/services/file_operations.py
- [ ] T122 [P] Code cleanup and refactoring across all modules
- [ ] T123 [P] Add type hints and docstrings to all public functions
- [ ] T124 [P] Update README.md with installation, configuration, and usage instructions
- [ ] T125 Run full test suite (all unit, integration, UI tests) with pytest and verify 100% pass
- [ ] T126 Manual validation of quickstart.md walkthrough (Phase 1, Phase 2, Phase 3)
- [ ] T127 Test with Ollama local model (verify num_ctx config, connection, streaming)
- [ ] T128 Test with OpenAI API (verify API key handling, rate limits, errors)
- [ ] T129 Test edge case: very large journal entry (>100 blocks, verify UI responsiveness)
- [ ] T130 Test edge case: concurrent file modification (edit journal in Logseq while TUI running)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
  - **CRITICAL**: All foundational tests must pass before proceeding
  - User should verify: `pytest tests/unit/ tests/integration/test_config*.py tests/integration/test_llm*.py -v`
- **User Stories (Phase 3, 4, 5)**: All depend on Foundational phase completion
  - US1 (Block Selection): Can start after Foundational - No dependencies on other stories
  - US2 (Content Editing): Can start after Foundational - No dependencies on other stories (but logically follows US1)
  - US3 (Integration Review): Can start after Foundational - No dependencies on other stories (but logically follows US2)
- **Integration (Phase 6)**: Depends on US1 + US3 (MVP) or all user stories being complete
- **Edge Cases (Phase 7)**: Can run in parallel with Phase 6
- **Polish (Phase 8)**: Depends on all phases complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - Fully independent
  - Tests MUST FAIL before implementation (T032-T037)
  - Tests MUST PASS after implementation (T049)
  - User should manually verify before proceeding
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - Fully independent
  - Tests MUST FAIL before implementation (T050-T056)
  - Tests MUST PASS after implementation (T070)
  - User should manually verify before proceeding
- **User Story 3 (P1)**: Can start after Foundational (Phase 2) - Fully independent
  - Tests MUST FAIL before implementation (T071-T077)
  - Tests MUST PASS after implementation (T096)
  - User should manually verify before proceeding

### Within Each User Story

1. Write tests FIRST (should FAIL)
2. Verify tests fail with pytest
3. Implement features
4. Run tests again (should NOW PASS)
5. Manually test in TUI
6. Get user verification before proceeding

### Parallel Opportunities

- **Phase 1 (Setup)**: All tasks marked [P] can run in parallel
- **Phase 2 (Foundational)**:
  - T010, T011, T012 (config models) can run in parallel
  - T017, T018, T019, T020 (data models) can run in parallel
  - T029, T030 (utilities) can run in parallel
- **Phase 3 (US1 Tests)**: T032, T033, T034, T035, T036 can run in parallel
- **Phase 3 (US1 Widgets)**: T038, T039, T040 can run in parallel
- **Phase 4 (US2 Tests)**: T050, T051, T052, T053, T054, T055 can run in parallel
- **Phase 4 (US2 Services)**: T057, T058 can run in parallel after their tests
- **Phase 4 (US2 Widgets)**: T062 can start immediately
- **Phase 5 (US3 Tests)**: T071, T072, T073, T074, T075, T076 can run in parallel
- **Phase 5 (US3 Services)**: T078, T079, T080 can run in parallel
- **Phase 5 (US3 Widgets)**: T083, T084 can run in parallel
- **Phase 7 (Edge Cases)**: All tasks marked [P] can run in parallel
- **Phase 8 (Polish)**: T118, T119, T120, T121, T122, T123, T124 can run in parallel

---

## Parallel Example: User Story 1

```bash
# Step 1: Write all tests in parallel
Task T032: "Write UI test for block tree navigation using Textual pilot"
Task T033: "Write UI test for block selection toggle using Textual pilot"
Task T034: "Write UI test for LLM streaming updates using Textual pilot"
Task T035: "Write UI test for status widget progress using Textual pilot"
Task T036: "Write snapshot test for Phase 1 initial render"

# Step 2: Verify tests FAIL
pytest tests/ui/test_phase1_*.py -v
# Expected: All tests FAIL

# Step 3: Implement widgets in parallel
Task T038: "Create BlockTree widget"
Task T039: "Create StatusPanel widget"
Task T040: "Create MarkdownViewer widget"

# Step 4: Implement screen (sequential, depends on widgets)
Task T041-T048: "Implement Phase1Screen with all features"

# Step 5: Verify tests NOW PASS
pytest tests/ui/test_phase1_*.py -v
# Expected: All tests PASS

# Step 6: Manual verification by user
logsqueak extract  # User tests navigation, selection, LLM streaming
```

---

## Implementation Strategy

### MVP First (US1 + US3 Only - Fastest Path to Value)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - all tests must pass, user verifies)
3. Complete Phase 3: User Story 1 (tests fail → implement → tests pass → user verifies)
4. Complete Phase 5: User Story 3 (tests fail → implement → tests pass → user verifies)
5. Complete Phase 6: Integration (wire up US1 → US3, skip US2 editing for MVP)
6. **STOP and VALIDATE**: Test full workflow (select blocks → write directly with original content)
7. Deploy/demo if ready

**MVP Value**: Users can extract journal blocks to pages with LLM suggestions, without manual editing

### Full Feature Set (US1 + US2 + US3 - Complete Workflow)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - all tests must pass, user verifies)
3. Complete Phase 3: User Story 1 (TDD: tests → implement → verify)
4. User verifies US1 works independently before proceeding
5. Complete Phase 4: User Story 2 (TDD: tests → implement → verify)
6. User verifies US2 works independently before proceeding
7. Complete Phase 5: User Story 3 (TDD: tests → implement → verify)
8. User verifies US3 works independently before proceeding
9. Complete Phase 6: Integration (wire up US1 → US2 → US3)
10. Complete Phase 7: Edge Cases
11. Complete Phase 8: Polish
12. **Final Validation**: Full test suite + manual quickstart walkthrough

**Full Value**: Complete knowledge extraction workflow with LLM rewording and editing

### Cautious Approach (User Requirement: "Proceed cautiously... let me verify")

**After each major checkpoint, STOP and get user verification:**

1. ✅ **Checkpoint 1**: Phase 2 Foundational - Run `pytest tests/unit/ tests/integration/test_config*.py tests/integration/test_llm*.py -v` and verify all pass
2. ✅ **Checkpoint 2**: US1 Tests Written - Run `pytest tests/ui/test_phase1_*.py -v` and verify all FAIL
3. ✅ **Checkpoint 3**: US1 Implementation - Run `pytest tests/ui/test_phase1_*.py -v` and verify all PASS, then manually test TUI
4. ✅ **Checkpoint 4**: US2 Tests Written - Run `pytest tests/ui/test_phase2_*.py -v` and verify all FAIL
5. ✅ **Checkpoint 5**: US2 Implementation - Run `pytest tests/ui/test_phase2_*.py -v` and verify all PASS, then manually test TUI
6. ✅ **Checkpoint 6**: US3 Tests Written - Run `pytest tests/ui/test_phase3_*.py -v` and verify all FAIL
7. ✅ **Checkpoint 7**: US3 Implementation - Run `pytest tests/ui/test_phase3_*.py -v` and verify all PASS, then manually test TUI
8. ✅ **Checkpoint 8**: Full Integration - Run `pytest -v` (all tests) and verify all PASS, then test `logsqueak extract` end-to-end

---

## Test Execution Commands

```bash
# Phase 2: Verify foundational tests pass
pytest tests/unit/ tests/integration/test_config*.py tests/integration/test_llm*.py -v

# Phase 3: Verify US1 tests (should FAIL before implementation)
pytest tests/ui/test_phase1_*.py -v

# Phase 3: Verify US1 tests (should PASS after implementation)
pytest tests/ui/test_phase1_*.py -v

# Phase 4: Verify US2 tests (should FAIL before implementation)
pytest tests/ui/test_phase2_*.py -v

# Phase 4: Verify US2 tests (should PASS after implementation)
pytest tests/ui/test_phase2_*.py -v

# Phase 5: Verify US3 tests (should FAIL before implementation)
pytest tests/ui/test_phase3_*.py -v

# Phase 5: Verify US3 tests (should PASS after implementation)
pytest tests/ui/test_phase3_*.py -v

# Phase 6: Verify all tests pass
pytest -v

# Phase 8: Final validation with coverage
pytest --cov=logsqueak --cov-report=html -v
```

---

## Notes

- **[P] tasks**: Different files, no dependencies, can run in parallel
- **[Story] label**: Maps task to specific user story for traceability
- **Test-Driven Development**: Write tests FIRST, verify they FAIL, implement, verify they PASS
- **Textual Pilot**: All UI tests use Textual's pilot API for automated testing
- **User Verification**: After each checkpoint, user manually tests functionality before proceeding
- **Commit Strategy**: Commit after each task or logical group (e.g., after T021 "all data models and tests")
- **Stop Points**: After T031, T049, T070, T096, T106, T117, T130 - user should verify before continuing
- **Avoid**: Vague tasks, same file conflicts, cross-story dependencies that break independence
- **Property Order**: NEVER reorder properties in Logseq files (constitutional requirement)
- **Non-Destructive**: All operations preserve provenance and use atomic writes
