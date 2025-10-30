# Tasks: Interactive TUI for Knowledge Extraction

**Feature**: 002-interactive-tui | **Branch**: `002-interactive-tui`

**Input**: Design documents from `/home/twaugh/devel/logsqueak/specs/002-interactive-tui/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/ndjson-streaming.md

**Tests**: Tests are NOT explicitly requested in the spec - focused on implementation tasks

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

Single project structure:
- Source: `src/logsqueak/`
- Tests: `tests/`
- Specs: `specs/002-interactive-tui/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and TUI framework setup

- [X] T001 Add Textual \^0.47.0 to pyproject.toml dependencies
- [X] T002 [P] Create TUI module structure (src/logsqueak/tui/__init__.py, screens/, widgets/)
- [X] T003 [P] Create TUI models module (src/logsqueak/tui/models.py)

---

## Phase 2: Foundational (Blocking Prerequisites) ✅ COMPLETE

**Purpose**: Core infrastructure that MUST be complete before ANY user story screen can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T004 Implement generic NDJSON stream parser in src/logsqueak/llm/streaming.py
- [X] T005 Add HTTP streaming protocol handling to src/logsqueak/llm/providers/openai_compat.py (Ollama + OpenAI formats)
- [X] T006 [P] Define ScreenState dataclass in src/logsqueak/tui/models.py (current_phase, journal_entry, block_states, candidates, decisions, config, llm_client)
- [X] T007 [P] Define BlockState dataclass in src/logsqueak/tui/models.py (block_id, classification, confidence, source, llm_classification, llm_confidence)
- [X] T008 [P] Define CandidatePage dataclass in src/logsqueak/tui/models.py (page_name, similarity_score, included, blocks, search_method)
- [X] T009 [P] Define IntegrationDecision dataclass in src/logsqueak/tui/models.py (knowledge_block_id, target_page, action, target_block_id, target_block_title, confidence, refined_text, source, skip_reason)
- [X] T010 Add stream_extract_ndjson() method to LLMClient abstract class in src/logsqueak/llm/client.py
- [X] T011 Add stream_decisions_ndjson() method to LLMClient abstract class in src/logsqueak/llm/client.py
- [X] T012 Add stream_rewording_ndjson() method to LLMClient abstract class in src/logsqueak/llm/client.py
- [X] T013 Implement stream_extract_ndjson() in OpenAICompatibleProvider (src/logsqueak/llm/providers/openai_compat.py)
- [X] T014 Implement stream_decisions_ndjson() in OpenAICompatibleProvider (src/logsqueak/llm/providers/openai_compat.py)
- [X] T015 Implement stream_rewording_ndjson() in OpenAICompatibleProvider (src/logsqueak/llm/providers/openai_compat.py)
- [X] T016 Create ExtractionApp main Textual app in src/logsqueak/tui/app.py (initializes ScreenState, handles navigation)
- [X] T017 Update CLI entry point in src/logsqueak/cli/main.py to launch ExtractionApp for extract command

**Checkpoint**: ✅ Foundation ready - user story screen implementation can now begin in parallel

---

## Phase 3: User Story 1 - Review and Approve Knowledge Extraction (Priority: P1) 🎯 MVP

**Goal**: Provide visibility into LLM knowledge classification with user override capability

**Independent Test**: Run extraction on journal entry with mixed content, verify classifications display with confidence scores, verify user can override any classification, verify user-marked blocks are locked from LLM updates

### Implementation for User Story 1

- [X] T018 [P] [US1] Create BlockTree custom widget in src/logsqueak/tui/widgets/block_tree.py (hierarchical tree with classification icons and confidence scores)
- [X] T019 [P] [US1] Add block tree label formatting helper to BlockTree widget (icons: ✓/✗/⊙/⊗/?, confidence percentages, low-confidence warnings)
- [X] T020 [US1] Create Phase1Screen in src/logsqueak/tui/screens/phase1.py (compose with Header, BlockTree, Footer)
- [X] T021 [US1] Implement Phase1Screen.on_mount() to initialize all block_states as "pending" (src/logsqueak/tui/screens/phase1.py)
- [X] T022 [US1] Add async LLM streaming task to Phase1Screen (_stream_extraction method)
- [X] T023 [US1] Implement Phase1Screen keyboard bindings (j/k/↑/↓ navigation, K mark knowledge, A mark activity, R reset, Enter continue)
- [X] T024 [US1] Add action_mark_knowledge() to Phase1Screen (sets source="user", classification="knowledge", preserves llm_classification/llm_confidence)
- [X] T025 [US1] Add action_mark_activity() to Phase1Screen (sets source="user", classification="activity", preserves llm_classification/llm_confidence)
- [X] T026 [US1] Add action_reset() to Phase1Screen (restore llm_classification if exists, else reset to "pending")
- [X] T027 [US1] Add smart defaults logic to Phase1Screen (when parent marked, cascade to children unless individually overridden)
- [X] T028 [US1] Implement LLM classification update handler in Phase1Screen (check source != "user" before updating, post message for UI refresh)
- [X] T029 [US1] Add action_continue() to Phase1Screen (cancel LLM tasks, validate at least one knowledge block, push Phase2Screen)
- [X] T030 [US1] Add async on_unmount() to Phase1Screen to cancel background LLM task
- [X] T030b [US1] Add LLM task cancellation to Phase1Screen.action_continue() (store asyncio.Task reference from _stream_extraction, call task.cancel() when proceeding to Phase 2)
- [X] T031 [US1] Update ExtractionApp.on_mount() in src/logsqueak/tui/app.py to push Phase1Screen as initial screen

**Checkpoint**: ✅ User Story 1 COMPLETE - user can classify journal blocks interactively with LLM assistance

---

## Phase 4: User Story 2 - See LLM Integration Decisions with Streaming Feedback (Priority: P1) 🎯 MVP ✅ COMPLETE

**Goal**: Provide real-time transparency into where knowledge will be integrated and how it will be reworded

**Independent Test**: Run extraction through Phase 3, verify streaming display of decisions and reworded content, verify all proposed integrations show target pages/actions/refined text

### Implementation for User Story 2

- [X] T032 [P] [US2] Create Phase2Screen in src/logsqueak/tui/screens/phase2.py (auto-proceed screen with progress display)
- [X] T033 [P] [US2] Implement Phase2Screen.on_mount() to start RAG candidate retrieval for each knowledge block
- [X] T034 [US2] Add async _retrieve_candidates() task to Phase2Screen (semantic + hinted search, populate candidates dict)
- [X] T035 [US2] Add progress indicator to Phase2Screen showing "Retrieving candidates for block X/Y"
- [X] T036 [US2] Add optional review mode binding (R key) to Phase2Screen to enable candidate page filtering
- [X] T037 [US2] Implement auto-proceed logic in Phase2Screen (when all candidates retrieved and no review requested, push Phase3Screen)
- [X] T038 [P] [US2] Create DecisionList custom widget in src/logsqueak/tui/widgets/decision_list.py (grouped by destination page)
- [X] T039 [P] [US2] Add decision item formatting to DecisionList widget (action labels, confidence, refined text preview)
- [X] T040 [US2] Create Phase3Screen in src/logsqueak/tui/screens/phase3.py (compose with Header, DecisionList, Footer)
- [X] T041 [US2] Implement Phase3Screen.on_mount() to initialize decision streaming
- [X] T042 [US2] Add async _stream_decisions() task to Phase3Screen (Phase 3.1 Decider for each knowledge block + candidate page pair)
- [X] T043 [US2] Add async _stream_rewording() task to Phase3Screen (Phase 3.2 Reworder for non-skip decisions)
- [X] T044 [US2] Implement decision update handler in Phase3Screen (populate decisions dict, update DecisionList UI)
- [X] T045 [US2] Implement rewording update handler in Phase3Screen (update refined_text in decisions dict, show streaming refined text)
- [X] T046 [US2] Add page grouping logic to Phase3Screen (collapse pages with all skipped decisions, show explanatory text)
- [X] T047 [US2] Add low-confidence warning indicators to Phase3Screen (⚠ for decisions <60% confidence)
- [X] T048 [US2] Add keyboard binding for continuing to Phase 4 (Enter key pushes Phase4Screen)

**Checkpoint**: ✅ User Story 2 COMPLETE - user can see full LLM decision process with streaming feedback

---

## Phase 5: User Story 3 - Override Integration Decisions and Edit Content (Priority: P2)

**Goal**: Enable user to change integration actions, target locations, and edit refined text before writing

**Independent Test**: Navigate to proposed integration in Phase 3, change action from "Add as new section" to "Add under [block]", edit refined text, verify changes reflected in final write operations

### Implementation for User Story 3

- [ ] T049 [US3] Add action cycling binding (Space key) to Phase3Screen (rotate through skip → add_section → add_under → replace)
- [ ] T050 [US3] Implement action_cycle_action() in Phase3Screen (update decision.action, set source="user", refresh UI)
- [ ] T051 [US3] Add target location picker dialog to Phase3Screen (shows all blocks in target page, user selects with arrow keys)
- [ ] T052 [US3] Add target picking binding (T key) to Phase3Screen (opens picker dialog for add_under/replace actions)
- [ ] T053 [US3] Implement action_pick_target() in Phase3Screen (show picker, update target_block_id and target_block_title)
- [ ] T054 [US3] Create TextEditor dialog widget in src/logsqueak/tui/widgets/text_editor.py (multi-line text editing)
- [ ] T055 [US3] Add text editing binding (E key) to Phase3Screen (opens TextEditor with refined_text)
- [ ] T056 [US3] Implement action_edit_text() in Phase3Screen (show both original and refined text, allow editing, update refined_text)
- [ ] T057 [US3] Add decision locking logic to Phase3Screen (when source="user", prevent LLM from overriding in subsequent updates)
- [ ] T058 [US3] Update streaming handlers in Phase3Screen to check decision.source != "user" before updating

**Checkpoint**: All P1 and P2 user stories should now be independently functional - full interactive control over integration decisions

---

## Phase 6: User Story 4 - Review Candidate Pages Before Integration (Priority: P3)

**Goal**: Allow users to exclude irrelevant candidate pages to reduce unnecessary LLM evaluations

**Independent Test**: Press 'R' during Phase 2, view candidate pages with match percentages, toggle some pages off, verify excluded pages are not considered in Phase 3

### Implementation for User Story 4

- [ ] T059 [US4] Add candidate review mode UI to Phase2Screen (list of pages with similarity percentages)
- [ ] T060 [US4] Implement review mode navigation in Phase2Screen (up/down to select page, Space to toggle inclusion)
- [ ] T061 [US4] Add N key binding to Phase2Screen in review mode (navigate to next knowledge block's candidates)
- [ ] T062 [US4] Add visual indicators to Phase2Screen for included/excluded pages (✓/✗ icons)
- [ ] T063 [US4] Update action_toggle_included() in Phase2Screen (flip CandidatePage.included, refresh UI)
- [ ] T064 [US4] Update Phase3Screen to filter decisions by candidates where included=True
- [ ] T065 [US4] Add review mode exit binding (Enter key) to Phase2Screen (proceed to Phase3 with filtered candidates)

**Checkpoint**: User Story 4 complete - optional candidate page filtering works independently

---

## Phase 7: User Story 5 - Monitor Write Progress and Handle Errors Gracefully (Priority: P2)

**Goal**: Provide closure to extraction workflow with clear success/error reporting

**Independent Test**: Run extraction through Phase 4, observe per-page write progress, verify completion summary shows what was written including any errors

### Implementation for User Story 5

- [ ] T066 [P] [US5] Create ProgressBar custom widget in src/logsqueak/tui/widgets/progress_bar.py (reactive updates for current/total)
- [ ] T067 [P] [US5] Create ErrorBanner widget in src/logsqueak/tui/widgets/error_banner.py (dismissible error display)
- [ ] T068 [US5] Create Phase4Screen in src/logsqueak/tui/screens/phase4.py (compose with Header, ProgressBar, status list, Footer)
- [ ] T069 [US5] Implement Phase4Screen.on_mount() to group decisions by target page
- [ ] T070 [US5] Add async _execute_writes() task to Phase4Screen (process each page sequentially)
- [ ] T071 [US5] Implement per-page write logic in Phase4Screen (convert decisions to WriteOperations, call executor)
- [ ] T072 [US5] Add atomic journal update logic to Phase4Screen with rollback (if page write succeeds but journal marker addition fails, roll back the page write to maintain consistency)
- [ ] T072b [US5] Implement rollback mechanism in Phase4Screen (save pre-write page state, restore on journal update failure)
- [ ] T073 [US5] Add error handling to Phase4Screen (continue with other pages if one fails, collect errors)
- [ ] T074 [US5] Add progress display updates to Phase4Screen (show pending → writing → complete status per page)
- [ ] T075 [US5] Implement completion summary in Phase4Screen (total blocks added, pages updated, link to journal entry)
- [ ] T076 [US5] Add error details view to Phase4Screen (E key binding to expand error messages with suggested actions)
- [ ] T077 [US5] Add exit binding to Phase4Screen (Q key or Enter to close app after completion)

**Checkpoint**: All user stories complete - full end-to-end interactive extraction workflow functional

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories and production readiness

- [ ] T078 [P] Add comprehensive error recovery to all screens (network timeout retry with exponential backoff)
- [ ] T079 [P] Add logging statements to all LLM streaming tasks (src/logsqueak/tui/screens/*.py)
- [ ] T080 Add validation for all state transitions (validate_phase1_complete, validate_phase2_complete, validate_phase3_complete)
- [ ] T081 [P] Add CSS styling to ExtractionApp for visual polish (src/logsqueak/tui/app.py)
- [ ] T082 [P] Update CLAUDE.md with Textual framework and NDJSON streaming pattern documentation
- [ ] T083 Add unit tests for parse_ndjson_stream() in tests/unit/test_streaming.py (malformed JSON, incomplete streams, empty lines)
- [ ] T084 [P] Add unit tests for BlockState lifecycle in tests/unit/test_tui_models.py (state transitions, validation rules)
- [ ] T085 [P] Add unit tests for IntegrationDecision validation in tests/unit/test_tui_models.py
- [ ] T086 Add integration test for Phase 1 workflow in tests/integration/test_phase1_screen.py
- [ ] T087 Add integration test for Phase 3 workflow in tests/integration/test_phase3_screen.py
- [ ] T088 Add integration test for full extraction workflow in tests/integration/test_tui_full_workflow.py
- [ ] T089 [P] Add prompt templates for NDJSON format in src/logsqueak/llm/prompts.py (extraction, decisions, rewording)
- [ ] T090 Update quickstart.md validation script if needed
- [ ] T091 [P] Add terminal capability detection for Unicode fallbacks (ASCII alternatives for limited terminals)
- [ ] T092 Performance optimization: ensure UI updates <500ms, input handling <100ms

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-7)**: All depend on Foundational phase completion
  - US1 + US2 are both P1 (MVP) - can proceed in parallel but US2 depends on US1 screens existing
  - US3 (P2) extends US2 - should start after US2 complete
  - US4 (P3) extends Phase 2 - can proceed independently
  - US5 (P2) is final phase - should start after US1+US2 complete
- **Polish (Phase 8)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P1)**: Can start after US1 complete (needs Phase1Screen to exist for navigation flow)
- **User Story 3 (P2)**: Can start after US2 complete (extends Phase3Screen)
- **User Story 4 (P3)**: Can start after Foundational (independent of other stories, extends Phase2Screen)
- **User Story 5 (P2)**: Can start after US1+US2 complete (final phase of workflow)

### Within Each User Story

- **US1**: Widget creation tasks (T018-T019) can run in parallel before screen implementation (T020-T031)
- **US2**: Widget creation (T038-T039) and screen creation (T032-T037, T040-T048) can partially overlap
- **US3**: All tasks sequential (extend existing Phase3Screen)
- **US4**: All tasks sequential (extend existing Phase2Screen)
- **US5**: Widget creation tasks (T066-T067) can run in parallel before screen implementation (T068-T077)
- **Polish**: Test writing tasks (T083-T088) and documentation tasks (T082, T090) can run in parallel

### Parallel Opportunities

- **Within Setup**: T002 and T003 can run in parallel (different modules)
- **Within Foundational**: T004-T005 (streaming), T006-T009 (models), T010-T015 (LLM methods) can all run in parallel
- **Within US1**: T018-T019 (BlockTree widget) can run in parallel
- **Within US2**: T032-T037 (Phase2Screen), T038-T039 (DecisionList), T040-T048 (Phase3Screen) can partially overlap
- **Within US5**: T066-T067 (widgets) can run in parallel
- **Within Polish**: T078-T092 many tasks can run in parallel (different files/concerns)

---

## Parallel Example: Foundational Phase

```bash
# Launch all model definitions together:
Task: "Define ScreenState dataclass in src/logsqueak/tui/models.py"
Task: "Define BlockState dataclass in src/logsqueak/tui/models.py"
Task: "Define CandidatePage dataclass in src/logsqueak/tui/models.py"
Task: "Define IntegrationDecision dataclass in src/logsqueak/tui/models.py"

# Launch streaming infrastructure in parallel:
Task: "Implement generic NDJSON stream parser in src/logsqueak/llm/streaming.py"
Task: "Add HTTP streaming protocol handling to src/logsqueak/llm/providers/openai_compat.py"
```

---

## Parallel Example: User Story 1

```bash
# Launch widget creation tasks in parallel:
Task: "Create BlockTree custom widget in src/logsqueak/tui/widgets/block_tree.py"
Task: "Add block tree label formatting helper to BlockTree widget"

# After widgets complete, implement Phase1Screen:
Task: "Create Phase1Screen in src/logsqueak/tui/screens/phase1.py"
# ... then sequential tasks for screen implementation
```

---

## Implementation Strategy

### MVP First (User Stories 1 + 2 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1 (knowledge classification)
4. Complete Phase 4: User Story 2 (decision streaming + display)
5. **STOP and VALIDATE**: Test US1+US2 end-to-end
6. Deploy/demo if ready

This delivers a working interactive TUI with:

- Phase 1: Interactive knowledge classification
- Phase 2: Auto RAG retrieval
- Phase 3: Streaming decision display
- Phase 4: Write execution (US5 needed for polish)

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test Phase 1 independently
3. Add User Story 2 → Test Phase 1-3 flow → Deploy/Demo (MVP!)
4. Add User Story 5 → Complete Phase 4 → Test full workflow → Deploy/Demo
5. Add User Story 3 → Enhanced Phase 3 editing → Deploy/Demo
6. Add User Story 4 → Optional candidate filtering → Deploy/Demo
7. Polish phase → Production ready

### Parallel Team Strategy

With multiple developers after Foundational phase completes:

1. **Developer A**: User Story 1 (Phase1Screen + BlockTree widget)
2. **Developer B**: User Story 2 Phase 2 portion (Phase2Screen)
3. **Developer C**: User Story 2 Phase 3 portion (Phase3Screen + DecisionList widget)

Then integrate and test before moving to US3, US4, US5.

---

## Task Summary

**Total Tasks**: 92

**By Phase**:
- Phase 1 (Setup): 3 tasks
- Phase 2 (Foundational): 14 tasks
- Phase 3 (US1 - P1 MVP): 14 tasks
- Phase 4 (US2 - P1 MVP): 17 tasks
- Phase 5 (US3 - P2): 10 tasks
- Phase 6 (US4 - P3): 7 tasks
- Phase 7 (US5 - P2): 12 tasks
- Phase 8 (Polish): 15 tasks

**Parallel Opportunities**:
- Foundational phase: 10 parallel tasks
- User Story 1: 2 parallel tasks
- User Story 2: 4 parallel tasks
- User Story 5: 2 parallel tasks
- Polish phase: ~10 parallel tasks

**MVP Scope** (Recommended first delivery):
- Phase 1 + Phase 2 + Phase 3 (US1) + Phase 4 (US2) = 48 tasks
- Delivers: Interactive classification + streaming decision display + basic execution

**Notes**:

- All tasks follow strict checklist format with Task ID, [P] marker, [Story] label, and file paths
- Tests are not included as they were not explicitly requested in spec.md
- Each user story is independently implementable and testable
- Clear checkpoints after each story completion
- Parallel opportunities identified for team efficiency
- MVP scope clearly defined for incremental delivery
