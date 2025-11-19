# Tasks: Interactive Setup Wizard

**Input**: Design documents from `/specs/003-setup-wizard/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: This feature includes comprehensive unit and integration tests as specified in the contracts and quickstart documentation.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `- [ ] [ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3, US4)
- Include exact file paths in descriptions

## Path Conventions

Using single project structure (from plan.md):
- **Source**: `src/logsqueak/` for application code
- **Tests**: `tests/unit/` and `tests/integration/` at repository root
- **Config**: `src/logsqueak/models/config.py` (existing, to be extended)

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create wizard module structure and extend Config model

- [X] T001 Create wizard package directory structure at src/logsqueak/wizard/
- [X] T002 Create src/logsqueak/wizard/__init__.py with public API exports
- [X] T003 [P] Create empty module files: src/logsqueak/wizard/validators.py, src/logsqueak/wizard/providers.py, src/logsqueak/wizard/prompts.py, src/logsqueak/wizard/wizard.py
- [X] T004 [P] Create test directory structure: tests/unit/wizard/, tests/integration/

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core data structures and Config model extension that ALL user stories depend on

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T005 Add llm_providers field to Config model in src/logsqueak/models/config.py (type: dict[str, dict] | None, default: None)
- [X] T006 [P] Implement ValidationResult dataclass in src/logsqueak/wizard/validators.py
- [X] T007 [P] Implement WizardState dataclass in src/logsqueak/wizard/wizard.py
- [X] T008 [P] Implement OllamaModel dataclass in src/logsqueak/wizard/providers.py
- [X] T009 Write unit tests for Config model extension in tests/unit/test_config_wizard.py (test backwards compatibility, llm_providers field)

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - First-Time Setup (Priority: P1) 🎯 MVP

**Goal**: Enable new users to run `logsqueak init` and complete guided configuration for Ollama provider, creating a valid config.yaml with mode 600 permissions

**Independent Test**: Run `logsqueak init` on fresh system without existing config, complete wizard prompts for Ollama provider, verify `logsqueak extract` works afterward

### Validation Functions for User Story 1

- [X] T010 [P] [US1] Implement validate_graph_path() in src/logsqueak/wizard/validators.py (check path exists, has journals/ and logseq/ subdirectories)
- [X] T011 [P] [US1] Implement check_disk_space() in src/logsqueak/wizard/validators.py (use shutil.disk_usage, return ValidationResult with available_mb)
- [X] T012 [P] [US1] Implement validate_ollama_connection() in src/logsqueak/wizard/validators.py (async HTTP GET to /api/tags, parse models, return ValidationResult)
- [X] T013 [P] [US1] Implement check_embedding_model_cached() in src/logsqueak/wizard/validators.py (check ~/.cache/torch/sentence_transformers/ path)
- [X] T014 [P] [US1] Implement validate_embedding_model() in src/logsqueak/wizard/validators.py (async, downloads all-mpnet-base-v2 if needed, progress callback support)

### Provider Helpers for User Story 1

- [X] T015 [P] [US1] Implement fetch_ollama_models() in src/logsqueak/wizard/providers.py (call validate_ollama_connection, extract OllamaModel list)
- [X] T016 [P] [US1] Implement get_recommended_ollama_model() in src/logsqueak/wizard/providers.py (search for mistral 7b instruct)
- [X] T017 [P] [US1] Implement format_model_size() in src/logsqueak/wizard/providers.py (convert bytes to GB/MB/KB with 1024 base)
- [X] T018 [P] [US1] Implement get_provider_key() in src/logsqueak/wizard/providers.py (generate unique key for llm_providers dict)

### Prompts for User Story 1

- [X] T019 [P] [US1] Implement prompt_graph_path() in src/logsqueak/wizard/prompts.py (Rich Prompt.ask with default, expand ~)
- [X] T020 [P] [US1] Implement prompt_provider_choice() in src/logsqueak/wizard/prompts.py (Rich Prompt.ask with choices: ollama, openai, custom)
- [X] T021 [P] [US1] Implement prompt_ollama_endpoint() in src/logsqueak/wizard/prompts.py (default http://localhost:11434)
- [X] T022 [P] [US1] Implement prompt_ollama_model() in src/logsqueak/wizard/prompts.py (display Rich table, highlight recommended, Prompt.ask with choices)
- [X] T023 [P] [US1] Implement prompt_retry_on_failure() in src/logsqueak/wizard/prompts.py (choices: retry, skip, abort)
- [X] T024 [P] [US1] Implement prompt_continue_on_timeout() in src/logsqueak/wizard/prompts.py (choices: continue, retry, skip)
- [X] T025 [P] [US1] Implement prompt_confirm_overwrite() in src/logsqueak/wizard/prompts.py (Rich Confirm.ask for config overwrite)
- [X] T025a [P] [US1] Implement prompt_advanced_settings() in src/logsqueak/wizard/prompts.py (ask if user wants advanced settings, default No)
- [X] T025b [P] [US1] Implement prompt_num_ctx() in src/logsqueak/wizard/prompts.py (default 32768, only for Ollama)
- [X] T025c [P] [US1] Implement prompt_top_k() in src/logsqueak/wizard/prompts.py (default 10, all providers)

### Wizard Orchestration for User Story 1

- [X] T026 [US1] Implement load_existing_config() in src/logsqueak/wizard/wizard.py (try Config.load, return None on any error)
- [X] T027 [US1] Implement configure_graph_path() in src/logsqueak/wizard/wizard.py (prompt, validate, retry loop, update WizardState)
- [X] T028 [US1] Implement configure_ollama() in src/logsqueak/wizard/wizard.py (try existing endpoint, try localhost, prompt for custom, fetch models, select model, update WizardState)
- [X] T029 [US1] Implement configure_provider() in src/logsqueak/wizard/wizard.py (prompt provider choice, branch to configure_ollama, optionally prompt advanced settings)
- [X] T030 [US1] Implement validate_llm_connection() in src/logsqueak/wizard/wizard.py (call validate_ollama_connection with timeout handling, retry logic)
- [X] T031 [US1] Implement validate_embedding() in src/logsqueak/wizard/wizard.py (check disk space, check cache, download with progress if needed, timeout handling)
- [X] T032 [US1] Implement assemble_config() in src/logsqueak/wizard/wizard.py (create LLMConfig from WizardState, merge llm_providers, return Config)
- [X] T033 [US1] Implement write_config() in src/logsqueak/wizard/wizard.py (create directory, atomic temp file write, chmod 600, rename)
- [X] T034 [US1] Implement show_success_message() in src/logsqueak/wizard/wizard.py (Rich panel with next steps)
- [X] T035 [US1] Implement run_setup_wizard() in src/logsqueak/wizard/wizard.py (orchestrate full flow: load config, configure graph, configure provider, validate LLM, validate embedding, assemble, prompt overwrite, write, show success)

### CLI Integration for User Story 1

- [X] T036 [US1] Add init command to src/logsqueak/cli.py (import run_setup_wizard, call with asyncio.run, handle result)
- [X] T037 [US1] Add CLI help text for init command in src/logsqueak/cli.py

### Tests for User Story 1

- [X] T038 [P] [US1] Unit tests for validators in tests/unit/wizard/test_validators.py (validate_graph_path, check_disk_space, validate_ollama_connection with mocked httpx)
- [X] T039 [P] [US1] Unit tests for providers in tests/unit/wizard/test_providers.py (fetch_ollama_models, get_recommended_ollama_model, format_model_size, get_provider_key)
- [X] T040 [P] [US1] Unit tests for prompts in tests/unit/wizard/test_prompts.py (mock Rich Prompt.ask and Confirm.ask, verify all prompt functions)
- [X] T040a [P] [US1] Unit tests for advanced settings prompts in tests/unit/wizard/test_prompts.py (test defaults, validation, Ollama-specific num_ctx)
- [X] T041 [US1] Integration test for first-time Ollama setup in tests/integration/test_wizard.py (wizard config generation, file permissions, write operations)

**Checkpoint**: At this point, User Story 1 should be fully functional - users can run `logsqueak init` and configure Ollama

---

## Phase 4: User Story 2 - Fixing Broken Configuration (Priority: P2)

**Goal**: Enable users with existing broken configs to run `logsqueak init` and fix issues while preserving valid settings

**Independent Test**: Create broken config file (wrong permissions or invalid values), run `logsqueak init`, verify wizard loads valid defaults and allows fixes

### Implementation for User Story 2

- [X] T042 [US2] Enhance load_existing_config() in src/logsqueak/wizard/wizard.py to handle partial config extraction (catch validation errors, log them, extract valid fields)
- [X] T043 [US2] Enhance configure_graph_path() in src/logsqueak/wizard/wizard.py to show current value from existing config as default
- [X] T044 [US2] Enhance configure_ollama() in src/logsqueak/wizard/wizard.py to use existing config values as defaults
- [X] T045 [US2] Enhance validate_embedding() in src/logsqueak/wizard/wizard.py to skip download if model is already cached and loads successfully
- [X] T046 [US2] Enhance run_setup_wizard() in src/logsqueak/wizard/wizard.py to detect wrong permissions and offer to fix

### Tests for User Story 2

- [X] T047 [P] [US2] Integration test for updating existing valid config in tests/integration/test_wizard.py (existing config loaded as defaults, user can change individual settings)
- [X] T048 [P] [US2] Integration test for fixing config with wrong permissions in tests/integration/test_wizard.py (wizard warns and creates new config with mode 600)
- [X] T049 [P] [US2] Integration test for handling partially invalid config in tests/integration/test_wizard.py (wizard extracts valid fields, prompts for missing ones)
- [X] T050 [US2] Integration test for cached embedding model skip in tests/integration/test_wizard.py (mock check_embedding_model_cached=True, verify download skipped)

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently - first-time setup AND config fixes both work

---

## Phase 5: User Story 3 - Remote Ollama and Custom Endpoints (Priority: P2)

**Goal**: Enable users to configure remote Ollama instances or custom OpenAI-compatible endpoints

**Independent Test**: Run `logsqueak init` with Ollama not running locally, enter remote Ollama URL, verify wizard connects and retrieves models

### Validation Functions for User Story 3

- [X] T051 [P] [US3] Implement validate_openai_connection() in src/logsqueak/wizard/validators.py (async HTTP POST to /chat/completions, minimal test request, return ValidationResult)

### Provider Helpers for User Story 3

- [X] T052 [P] [US3] Implement mask_api_key() in src/logsqueak/wizard/providers.py (first 8 + "..." + last 4)

### Prompts for User Story 3

- [X] T053 [P] [US3] Implement prompt_openai_api_key() in src/logsqueak/wizard/prompts.py (visible input for new keys, show masked version for existing keys)
- [X] T054 [P] [US3] Implement prompt_openai_model() in src/logsqueak/wizard/prompts.py (choices: gpt-4o, gpt-4-turbo, gpt-3.5-turbo, custom)
- [X] T055 [P] [US3] Implement prompt_custom_endpoint() in src/logsqueak/wizard/prompts.py (URL validation)
- [X] T056 [P] [US3] Implement prompt_custom_api_key() in src/logsqueak/wizard/prompts.py (visible input)
- [X] T057 [P] [US3] Implement prompt_custom_model() in src/logsqueak/wizard/prompts.py (freeform text)

### Wizard Orchestration for User Story 3

- [X] T058 [US3] Implement configure_openai() in src/logsqueak/wizard/wizard.py (prompt for API key, prompt for model, update WizardState)
- [X] T059 [US3] Implement configure_custom() in src/logsqueak/wizard/wizard.py (prompt for endpoint, API key, model, update WizardState)
- [X] T060 [US3] Enhance configure_provider() in src/logsqueak/wizard/wizard.py to branch to configure_openai and configure_custom
- [X] T061 [US3] Enhance validate_llm_connection() in src/logsqueak/wizard/wizard.py to call validate_openai_connection for openai and custom providers
- [X] T062 [US3] Enhance assemble_config() in src/logsqueak/wizard/wizard.py to handle openai and custom provider types

### Tests for User Story 3

- [X] T063 [P] [US3] Unit tests for validate_openai_connection in tests/unit/wizard/test_validators.py (mock httpx, test success and various errors)
- [X] T064 [P] [US3] Unit tests for mask_api_key in tests/unit/wizard/test_providers.py (various key formats)
- [X] T065 [P] [US3] Integration test for remote Ollama setup in tests/integration/test_wizard.py (mock remote endpoint, verify connection and model retrieval)
- [X] T066 [P] [US3] Integration test for OpenAI provider setup in tests/integration/test_wizard.py (mock API key validation, verify config created)
- [X] T067 [P] [US3] Integration test for custom endpoint setup in tests/integration/test_wizard.py (mock custom endpoint validation, verify config created)

**Checkpoint**: At this point, User Stories 1, 2, AND 3 should all work independently - Ollama (local/remote), OpenAI, and custom endpoints all supported

---

## Phase 6: User Story 4 - Updating Existing Configuration (Priority: P3)

**Goal**: Enable users with working config to quickly update one setting while preserving others, with fast completion when embedding model is cached

**Independent Test**: Run `logsqueak init` with existing valid config, change only API key, verify other settings unchanged

### Implementation for User Story 4

- [X] T068 [US4] Enhance assemble_config() in src/logsqueak/wizard/wizard.py to preserve all provider settings from existing config when switching providers
- [X] T069 [US4] Add validation to assemble_config() in src/logsqueak/wizard/wizard.py to ensure both old and new provider credentials are in llm_providers dict
- [X] T070 [US4] Enhance run_setup_wizard() in src/logsqueak/wizard/wizard.py to optimize flow for updates (skip unnecessary prompts when user accepts defaults)

### Tests for User Story 4

- [X] T071 [P] [US4] Integration test for switching providers in tests/integration/test_wizard.py (OpenAI → Ollama, verify both sets of credentials preserved in config)
- [X] T072 [P] [US4] Integration test for single setting update in tests/integration/test_wizard.py (change only API key, verify all other settings unchanged)
- [X] T073 [US4] Integration test for fast update with cached embedding in tests/integration/test_wizard.py (verify completion in <30 seconds when skipping embedding download)

**Checkpoint**: All user stories should now be independently functional - complete wizard supports all use cases

---

## Phase 7: Edge Cases & Error Handling

**Purpose**: Handle all edge cases and failure scenarios from spec

- [X] T074 [P] Add timeout handling wrapper in src/logsqueak/wizard/wizard.py for LLM connection (30s timeout, prompt user: continue/retry/skip)
- [X] T075 [P] Add timeout handling wrapper in src/logsqueak/wizard/wizard.py for embedding download (5min timeout, prompt user: continue/retry/skip)
- [X] T076 [P] Add disk space warning logic in src/logsqueak/wizard/wizard.py (warn if <1GB, allow proceeding if user confirms)
- [X] T077 [P] Add error handling for missing journals/ subdirectory in src/logsqueak/wizard/validators.py (clear error message stating which directory missing)
- [X] T078 [P] Add error handling for missing logseq/ subdirectory in src/logsqueak/wizard/validators.py (clear error message stating which directory missing)
- [X] T079 [P] Add error handling for Ollama with no models in src/logsqueak/wizard/wizard.py (suggest installing Mistral 7B Instruct)
- [X] T080 [P] Add error handling for empty/whitespace config file in src/logsqueak/wizard/wizard.py (treat as no config, start fresh)
- [X] T081 [P] Add error handling for invalid YAML syntax in src/logsqueak/wizard/wizard.py (log error, treat as no config, start fresh)
- [X] T082 [P] Add error handling for config directory creation failure in src/logsqueak/wizard/wizard.py (permission denied, show helpful error with suggestions)
- [X] T083 [P] Add error handling for corrupted embedding model download in src/logsqueak/wizard/validators.py (detect via load failure, prompt retry)
- [X] T084 [P] Add error handling for disk space running out during download in src/logsqueak/wizard/validators.py (clear error message with disk space info)
- [X] T085 Add KeyboardInterrupt handling in src/logsqueak/wizard/wizard.py (catch Ctrl+C, show abort message, return False)
- [X] T086 Add validation for graph path with spaces/special characters in src/logsqueak/wizard/validators.py (ensure Path handles correctly)

### Edge Case Tests

- [X] T087 [P] Integration test for LLM connection timeout in tests/integration/test_wizard.py (mock timeout, verify user prompted)
- [X] T088 [P] Integration test for embedding download timeout in tests/integration/test_wizard.py (mock timeout, verify user prompted)
- [X] T089 [P] Integration test for low disk space warning in tests/integration/test_wizard.py (mock disk_usage, verify warning shown)
- [X] T090 [P] Integration test for Ollama with no models in tests/integration/test_wizard.py (empty models list, verify suggestion shown)
- [X] T091 [P] Integration test for wizard abort in tests/integration/test_wizard.py (abort at various stages, verify no config written)
- [X] T092 [P] Integration test for invalid YAML config in tests/integration/test_wizard.py (malformed YAML, verify treated as fresh setup)
- [X] T093 [P] Integration test for path with spaces in tests/integration/test_wizard.py (verify path validation handles correctly)

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Logging, progress indicators, and final validation

- [X] T094 [P] Add structured logging for all wizard actions in src/logsqueak/wizard/wizard.py (config loaded, prompts shown, validations, errors)
- [X] T095 [P] Add Rich Status spinner for network operations in src/logsqueak/wizard/wizard.py (LLM connection test, model fetching)
- [X] T096 [P] Add Rich Progress bar for embedding model download in src/logsqueak/wizard/validators.py (show download progress)
- [X] T097 [P] Verify all error messages follow contract format in src/logsqueak/wizard/ ([ERROR TYPE] + description + actionable guidance)
- [X] T098 [P] Add success message with Rich Panel in src/logsqueak/wizard/wizard.py (show next steps, config file path)
- [ ] T099 Run manual testing scenarios from quickstart.md (all provider types, all error cases, all edge cases)
- [ ] T100 Verify all success criteria from spec.md (setup time <3 minutes, config updates <30 seconds, mode 600 permissions, etc.)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-6)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P2 → P3)
- **Edge Cases (Phase 7)**: Can start in parallel with user stories, integrates with all stories
- **Polish (Phase 8)**: Depends on all user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - Enhances US1 but independent
- **User Story 3 (P2)**: Can start after Foundational (Phase 2) - Extends US1 with new providers but independent
- **User Story 4 (P3)**: Can start after Foundational (Phase 2) - Optimizes all previous stories but independent

### Within Each User Story

- Validators before wizard orchestration (validators are pure functions)
- Providers before prompts (providers are pure functions)
- Prompts before wizard orchestration (wizard calls prompts)
- Wizard orchestration before CLI integration (CLI calls wizard)
- Implementation before tests (write tests alongside, but implementation must exist first)

### Parallel Opportunities

**Phase 1 (Setup):**
- T003 (create module files) can run in parallel
- T004 (create test directories) can run in parallel

**Phase 2 (Foundational):**
- T006, T007, T008 (all dataclasses) can run in parallel after T005

**Phase 3 (User Story 1):**
- All validators (T010-T014) can run in parallel
- All providers (T015-T018) can run in parallel
- All prompts (T019-T025) can run in parallel
- All tests (T038-T040) can run in parallel

**Phase 4 (User Story 2):**
- All tests (T047-T050) can run in parallel

**Phase 5 (User Story 3):**
- Validators (T051), providers (T052), and prompts (T053-T057) can run in parallel
- All tests (T063-T067) can run in parallel

**Phase 6 (User Story 4):**
- All tests (T071-T073) can run in parallel

**Phase 7 (Edge Cases):**
- All error handling tasks (T074-T086) can run in parallel (different error cases)
- All edge case tests (T087-T093) can run in parallel

**Phase 8 (Polish):**
- All polish tasks (T094-T098) can run in parallel

---

## Parallel Example: User Story 1

```bash
# Launch all validators for User Story 1 together:
Task T010: "Implement validate_graph_path() in src/logsqueak/wizard/validators.py"
Task T011: "Implement check_disk_space() in src/logsqueak/wizard/validators.py"
Task T012: "Implement validate_ollama_connection() in src/logsqueak/wizard/validators.py"
Task T013: "Implement check_embedding_model_cached() in src/logsqueak/wizard/validators.py"
Task T014: "Implement validate_embedding_model() in src/logsqueak/wizard/validators.py"

# Launch all provider helpers for User Story 1 together:
Task T015: "Implement fetch_ollama_models() in src/logsqueak/wizard/providers.py"
Task T016: "Implement get_recommended_ollama_model() in src/logsqueak/wizard/providers.py"
Task T017: "Implement format_model_size() in src/logsqueak/wizard/providers.py"
Task T018: "Implement get_provider_key() in src/logsqueak/wizard/providers.py"

# Launch all unit tests for User Story 1 together:
Task T038: "Unit tests for validators in tests/unit/wizard/test_validators.py"
Task T039: "Unit tests for providers in tests/unit/wizard/test_providers.py"
Task T040: "Unit tests for prompts in tests/unit/wizard/test_prompts.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001-T004)
2. Complete Phase 2: Foundational (T005-T009) - CRITICAL, blocks all stories
3. Complete Phase 3: User Story 1 (T010-T041)
4. **STOP and VALIDATE**: Test User Story 1 independently
   - Run `logsqueak init` on fresh system
   - Complete Ollama setup flow
   - Verify `logsqueak extract` works
5. Deploy/demo if ready

**Estimated Time**: 4-6 hours (from quickstart.md)

### Incremental Delivery

1. **Foundation** (Setup + Foundational) → Config model extended, wizard structure ready
2. **MVP** (User Story 1) → New users can run `logsqueak init` and configure Ollama
3. **Fixes** (User Story 2) → Existing users can fix broken configs
4. **Flexibility** (User Story 3) → Remote Ollama and custom endpoints supported
5. **Polish** (User Story 4 + Edge Cases + Polish) → Fast updates, robust error handling

Each increment adds value without breaking previous functionality.

### Parallel Team Strategy

With multiple developers:

1. **Team completes Setup + Foundational together** (T001-T009)
2. **Once Foundational is done:**
   - Developer A: User Story 1 validators & providers (T010-T018)
   - Developer B: User Story 1 prompts (T019-T025)
   - Developer C: User Story 3 validation functions (T051-T057)
3. **After parallel work converges:**
   - Developer A: User Story 1 wizard orchestration (T026-T035)
   - Developer B: User Story 2 enhancements (T042-T046)
   - Developer C: User Story 3 wizard integration (T058-T062)
4. **Stories complete and integrate independently**

---

## Total Tasks Summary

- **Phase 1 (Setup)**: 4 tasks
- **Phase 2 (Foundational)**: 5 tasks
- **Phase 3 (User Story 1 - P1)**: 36 tasks (includes advanced settings: T025a-T025c, T040a)
- **Phase 4 (User Story 2 - P2)**: 9 tasks
- **Phase 5 (User Story 3 - P2)**: 17 tasks
- **Phase 6 (User Story 4 - P3)**: 6 tasks
- **Phase 7 (Edge Cases)**: 20 tasks
- **Phase 8 (Polish)**: 7 tasks

**Total**: 104 tasks

**Parallelization**: ~42 tasks can run in parallel (40% parallelizable)

---

## Notes

- [P] tasks = different files, no dependencies on incomplete work
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Comprehensive tests included per contract requirements
- Follow quickstart.md for detailed implementation patterns
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- All file paths are absolute from repository root
