# Implementation Plan: Interactive TUI for Knowledge Extraction

**Branch**: `002-interactive-tui` | **Date**: 2025-10-29 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/002-interactive-tui/spec.md`
**Design Reference**: `/specs/002-interactive-tui/interactive-tui-design.md`

## Summary

Replace the current batch extraction workflow with an interactive text-based UI that provides real-time visibility into the LLM's knowledge extraction process and enables user control at every decision point. The TUI will use NDJSON streaming for incremental UI updates, allowing users to review and override LLM classifications, candidate page selections, and integration decisions across 4 phases before committing changes to their Logseq graph.

**Key Technical Approach**:

- Textual framework for TUI (built on Rich, async-first)
- NDJSON (newline-delimited JSON) for incremental LLM response parsing
- Screen-based navigation (Phase1Screen → Phase2Screen → Phase3Screen → Phase4Screen)
- Shared state management across screens (BlockState, IntegrationDecision)
- Real-time streaming updates while maintaining UI responsiveness

## Technical Context

**Language/Version**: Python 3.11+ (already project requirement)

**Primary Dependencies**:

- Textual \^0.47.0 (TUI framework)
- Rich (already in project, Textual dependency)
- httpx (already in project for async LLM streaming)
- Existing logsqueak modules (extraction, integration, RAG, LLM client)

**Storage**: File-based I/O (Logseq markdown files) - no change from current architecture

**Testing**: pytest (existing test infrastructure)

**Target Platform**: Linux/macOS terminal emulators with Unicode + 256-color support

**Project Type**: Single project (CLI tool with TUI interface)

**Performance Goals**:

- <500ms UI response time to user input
- <100ms input handling during LLM streaming
- Process typical journal entry (11 root blocks, 22 total blocks) in <3 minutes
  on nVidia RTX 2060 Super
- Support up to 100 blocks per journal entry

**Constraints**:

- Keyboard-only interface (no mouse required)
- Must maintain responsive UI during LLM streaming (async tasks)
- Must support terminals with limited Unicode (ASCII fallbacks)
- NDJSON parsing must handle incomplete chunks gracefully

**Scale/Scope**:

- 4 TUI screens (one per phase)
- ~10-20 interactive widgets per screen
- Support journal entries up to 2000 lines (existing JournalEntry limit)
- Typical use case: 3-15 root blocks, 2-3 nesting levels

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### I. Proof-of-Concept First ✅

**Status**: PASS

- This feature demonstrates feasibility of interactive LLM-driven knowledge extraction
- NDJSON streaming validated against live Ollama server (see spec clarifications)
- Replaces existing batch mode (no backward compatibility burden)
- Textual framework enables rapid TUI iteration
- Incremental implementation path: Phase 1 (extraction) → Phase 3 (decisions) → Phase 2 (optional review) → Phase 4 (execution)

### II. Non-Destructive Operations ✅

**Status**: PASS

- User approval at every phase before writes (FR-044: can quit at any phase)
- Phase 4 shows preview before execution
- Atomic journal updates (FR-036: provenance markers only after successful writes)
- All operations traceable via `processed::` markers
- Error recovery preserves partial successes (FR-034: continue with other pages if one fails)
- User can override any LLM decision before commit (FR-003: users have final say)

### III. Simplicity and Transparency ✅

**Status**: PASS

- File-based I/O (no new databases)
- NDJSON for structured LLM outputs (simpler than complex JSON schema parsers)
- Real-time visibility of LLM work (streaming classifications, decisions, reworded text)
- User-friendly terminology (FR-022: "Add as new section" vs "APPEND_ROOT")
- Minimal abstraction: Screen classes map directly to phases
- Clear error messages with actionable guidance (FR-041)

### Development Workflow ✅

**Status**: PASS

- No dry-run mode needed (user approval built into interactive flow)
- Graceful failures: malformed LLM JSON → skip item, continue (see edge cases in spec)
- Prompt logging already implemented in project (PromptLogger)
- Incremental iteration strategy: Build P1 Screen → P3 Screen → P2 Screen → P4 Screen

**Re-check after Phase 1**: Will verify data models maintain simplicity and don't introduce unnecessary complexity

## Project Structure

### Documentation (this feature)

```text
specs/002-interactive-tui/
├── spec.md                    # Feature specification (requirements, success criteria)
├── interactive-tui-design.md  # UI/UX design reference (mockups, flows, architecture)
├── plan.md                    # This file (implementation plan)
├── research.md                # Phase 0: Technical decisions and rationale
├── data-model.md              # Phase 1: TUI state models and data flow
├── quickstart.md              # Phase 1: Developer guide for TUI development
├── contracts/                 # Phase 1: LLM streaming interfaces
│   └── ndjson-streaming.md    # NDJSON parsing contract
├── checklists/
│   └── requirements.md        # Spec validation checklist
└── tasks.md                   # Phase 2: Implementation tasks (created by /speckit.tasks)
```

### Source Code (repository root)

```text
src/logsqueak/
├── tui/                       # NEW: Interactive TUI module
│   ├── __init__.py
│   ├── app.py                 # ExtractionApp: Main Textual app
│   ├── screens/               # Screen implementations
│   │   ├── __init__.py
│   │   ├── phase1.py          # Phase1Screen: Knowledge extraction
│   │   ├── phase2.py          # Phase2Screen: Candidate page review
│   │   ├── phase3.py          # Phase3Screen: Integration decisions
│   │   └── phase4.py          # Phase4Screen: Write operations
│   ├── widgets/               # Custom Textual widgets
│   │   ├── __init__.py
│   │   ├── block_tree.py      # Hierarchical block display with confidence
│   │   ├── decision_list.py   # Grouped integration decisions
│   │   └── progress_bar.py    # Phase progress indicator
│   └── models.py              # TUI-specific state models
│
├── llm/                       # MODIFIED: Add NDJSON streaming support
│   ├── client.py              # Add async streaming methods
│   ├── streaming.py           # NEW: Generic parse_ndjson_stream() utility
│   └── providers/
│       └── openai_compat.py   # MODIFIED: HTTP streaming protocol handling
│
├── cli/
│   └── main.py                # MODIFIED: Replace batch mode with TUI
│
├── extraction/
│   └── extractor.py           # MINIMAL CHANGES: Extract logic reused by TUI
│
├── integration/
│   └── executor.py            # MINIMAL CHANGES: Execution logic reused by TUI
│
└── rag/
    └── vector_store.py        # NO CHANGES: Reused as-is

tests/
├── tui/                       # NEW: TUI-specific tests
│   ├── test_app.py
│   ├── test_screens.py
│   └── test_models.py
│
├── integration/               # MODIFIED: Update integration tests for TUI
│   └── test_tui_workflow.py
│
└── unit/                      # EXISTING: Reuse existing unit tests
    ├── test_config.py
    ├── test_llm.py
    ├── test_streaming.py      # NEW: NDJSON streaming parser tests
    └── ...
```

**Structure Decision**: Single project with new `tui/` module. The TUI layer sits on top of existing extraction, integration, and RAG modules, reusing their logic without modification. This maintains the "Simplicity" principle by avoiding architectural reorganization while cleanly separating interactive UI concerns.

**NDJSON Streaming Architecture**: The `parse_ndjson_stream()` utility lives in `llm/streaming.py` (not `tui/`) because it's provider-agnostic LLM infrastructure. The function handles generic line-buffering and JSON parsing, while provider-specific HTTP streaming protocol handling (Ollama's `{"message": {"content": "..."}}` vs OpenAI's `data: {...}` SSE format) lives in `llm/providers/openai_compat.py`. This separation enables future reuse by non-TUI features (e.g., batch processing with progress logging).

## Complexity Tracking

> **All constitution checks passed - no violations to justify**

## Phase 0: Research & Unknowns

**Prerequisites**: Constitution Check passed (see above)

### Research Topics

The following technical decisions need investigation and documentation:

1. **NDJSON Streaming Implementation**
   - **Question**: How to robustly parse NDJSON from chunked async streams?
   - **Context**: Validated that Ollama outputs NDJSON, but need production-ready parser
   - **Investigate**: Buffering strategy, error handling for incomplete lines, performance

2. **Textual App Architecture**
   - **Question**: Best practices for managing shared state across multiple Textual screens?
   - **Context**: Need to pass BlockState, IntegrationDecision, LLM client across 4 screens
   - **Investigate**: Screen lifecycle, data passing patterns, reactive bindings

3. **Async LLM Streaming + UI Responsiveness**
   - **Question**: How to update Textual UI from background async tasks without blocking?
   - **Context**: LLM streaming tasks run concurrently with user keyboard input
   - **Investigate**: Textual's async event loop integration, message passing, worker threads

4. **Tree Widget Customization**
   - **Question**: Can Textual's built-in Tree widget display custom icons + confidence scores?
   - **Context**: Need to show ✓/✗/⊙/⊗ icons with percentage confidence per block
   - **Investigate**: Tree node rendering, custom formatters, color styling

5. **Keyboard Shortcut Conflicts**
   - **Question**: How to handle conflicting keybindings (e.g., 'k' for vim-up vs 'K' for mark-knowledge)?
   - **Context**: Support both vim-style navigation and single-letter actions
   - **Investigate**: Textual's Binding priority, modal keymaps, case sensitivity

6. **Error Recovery in Streaming**
   - **Question**: How to handle malformed NDJSON lines without crashing the TUI?
   - **Context**: LLM might output invalid JSON, network errors, etc.
   - **Investigate**: Try/except strategies, user notification, graceful degradation

**Output**: `research.md` with decisions, rationale, and code examples for each topic

## Phase 1: Design & Contracts

**Prerequisites**: `research.md` complete

### 1.1 Data Model (`data-model.md`)

Extract and formalize the state models from the design document:

**Entities**:

1. **BlockState** (TUI state tracking)
   - `block_id: str` - Hybrid ID from journal block
   - `classification: Literal["pending", "knowledge", "activity"]` - Current classification
   - `confidence: float | None` - LLM confidence score (0.0-1.0) or None if user-marked
   - `source: Literal["user", "llm"]` - Who made the classification decision
   - **Lifecycle**: Created on Phase 1 mount → Updated during streaming → Read in Phase 2/3

2. **IntegrationDecision** (Phase 3 state)
   - `knowledge_block_id: str` - Source knowledge block
   - `target_page: str` - Destination page name
   - `action: Literal["skip", "add_section", "add_under", "replace"]` - User-friendly action
   - `target_block_id: str | None` - Target block for "add_under"/"replace"
   - `target_block_title: str | None` - Human-readable target name
   - `confidence: float` - LLM decision confidence
   - `refined_text: str` - Reworded content for integration
   - `source: Literal["user", "llm"]` - Who made the decision
   - **Lifecycle**: Created in Phase 3.1 → Updated in Phase 3.2 → Executed in Phase 4

3. **CandidatePage** (Phase 2 state)
   - `page_name: str` - Page name from RAG search
   - `similarity_score: float` - Semantic match score
   - `included: bool` - User selection (default: True)
   - `blocks: list[dict]` - Blocks within page for targeting
   - **Lifecycle**: Created in Phase 2 RAG search → Filtered by user → Used in Phase 3

4. **ScreenState** (TUI navigation)
   - `current_phase: int` - Active phase (1-4)
   - `journal_entry: JournalEntry` - Loaded journal data
   - `block_states: dict[str, BlockState]` - Map of block_id → state
   - `candidates: dict[str, list[CandidatePage]]` - Map of knowledge_block_id → pages
   - `decisions: dict[tuple[str, str], IntegrationDecision]` - Map of (block_id, page) → decision
   - **Lifecycle**: Created on app mount → Passed to each screen → Persisted until quit

**Validation Rules**:

- BlockState.confidence must be 0.0-1.0 if present
- IntegrationDecision.action must map to valid ActionType enum
- User-marked blocks (source="user") cannot be overridden by LLM

**State Transitions**:

```
BlockState:
  pending → knowledge/activity (LLM classification)
  pending → knowledge/activity (User override)
  knowledge ↔ activity (User toggle)

IntegrationDecision:
  created with action="skip" (Decider Phase 3.1)
  updated with refined_text (Reworder Phase 3.2)
  action changed by user (User override)
```

### 1.2 API Contracts (`contracts/`)

#### `contracts/ndjson-streaming.md`

Document the NDJSON streaming contract between LLM client and TUI:

**LLM Client Interface**:

```python
class LLMClient(ABC):
    @abstractmethod
    async def stream_extract_ndjson(
        self,
        blocks: list[LogseqBlock]
    ) -> AsyncIterator[dict]:
        """
        Stream knowledge extraction results as NDJSON.

        Yields one JSON object per line:
        {"block_id": "abc123", "is_knowledge": true, "confidence": 0.85}
        {"block_id": "def456", "is_knowledge": false, "confidence": 0.92}

        Each object is complete and parseable immediately upon arrival.
        """
        pass

    @abstractmethod
    async def stream_decisions_ndjson(
        self,
        knowledge_block: LogseqBlock,
        candidate_pages: list[CandidatePage]
    ) -> AsyncIterator[dict]:
        """
        Stream integration decisions as NDJSON.

        Yields one decision per page:
        {"page": "Software Architecture", "action": "add_section", "confidence": 0.88}
        {"page": "Microservices", "action": "skip", "reason": "duplicate"}
        """
        pass
```

**NDJSON Parser Contract**:

```python
async def parse_ndjson_stream(
    stream: AsyncIterator[str]
) -> AsyncIterator[dict]:
    """
    Parse NDJSON from chunked stream.

    Input: Token-by-token chunks ('{"', 'foo', '": ', '"bar', '"}', '\n', ...)
    Output: Complete JSON objects as they become parseable

    Error Handling:
    - Invalid JSON on a line → Log error, skip line, continue
    - Incomplete line at stream end → Log warning, discard
    - Empty lines → Skip silently
    """
```

### 1.3 Developer Quickstart (`quickstart.md`)

```markdown
# Interactive TUI Development Guide

## Running the TUI

```bash
# Activate virtual environment
source venv/bin/activate

# Run interactive extraction (default: today's journal)
logsqueak extract

# Extract specific date
logsqueak extract 2025-10-29

# Extract date range
logsqueak extract 2025-10-25..2025-10-29
```

## TUI Architecture

### Screen Flow
1. Phase1Screen: Classify blocks as knowledge/activity
2. Phase2Screen: Review candidate pages (optional)
3. Phase3Screen: Decide where to integrate knowledge
4. Phase4Screen: Execute writes and show results

### Shared State
All screens access `ExtractionApp.block_states`, `candidates`, `decisions`

### Adding a New Screen

1. Create `src/logsqueak/tui/screens/my_screen.py`
2. Inherit from `textual.screen.Screen`
3. Define `BINDINGS` for keyboard shortcuts
4. Implement `compose()` for layout
5. Add async `on_mount()` for initialization
6. Wire into app flow in `app.py`

## Testing the TUI

```bash
# Unit tests for individual screens
pytest tests/tui/test_screens.py

# Integration test for full workflow
pytest tests/tui/test_app.py

# Test NDJSON parsing
pytest tests/unit/test_streaming.py
```

## Debugging

### Enable Textual Dev Console
```bash
textual console
# In another terminal:
logsqueak extract
```

### View LLM Prompts
```bash
# Prompts logged to ~/.cache/logsqueak/prompts/
tail -f ~/.cache/logsqueak/prompts/latest.log
```

### 1.4 Update Agent Context

**Script**: `.specify/scripts/bash/update-agent-context.sh claude`

**Updates**: `.specify/memory/claude-context.md` (or appropriate agent file)

**New Technology to Add**:

- Textual ^0.47.0 (TUI framework)
- NDJSON streaming pattern
- Async/await for UI + LLM concurrency

**Manual Preservation**: Existing project context remains (5-phase pipeline, LLM client, etc.)

## Phase 2: Task Generation

**Prerequisites**: Phase 1 complete (data-model.md, contracts/, quickstart.md)

This phase is handled by the `/speckit.tasks` command and is NOT part of `/speckit.plan`.

The `/speckit.tasks` command will:

1. Read this plan.md
2. Read data-model.md and contracts/
3. Generate dependency-ordered tasks in tasks.md
4. Create atomic, testable implementation tasks

**Example tasks that will be generated**:

- Task: Implement NDJSON streaming parser (`src/logsqueak/llm/streaming.py`)
- Task: Add HTTP streaming protocol handling to OpenAI provider (`src/logsqueak/llm/providers/openai_compat.py`)
- Task: Add async streaming methods to LLM client interface (`src/logsqueak/llm/client.py`)
- Task: Create BlockState and ScreenState models (`src/logsqueak/tui/models.py`)
- Task: Build Phase1Screen with Tree widget (`src/logsqueak/tui/screens/phase1.py`)
- Task: Integrate TUI into CLI (`src/logsqueak/cli/main.py`)

## Design References

This implementation follows the detailed UI/UX design in:

- **Primary**: `specs/002-interactive-tui/interactive-tui-design.md`
  - Phase-by-phase screen layouts
  - Keyboard shortcuts and bindings
  - Visual indicators (✓/✗/⊙/⊗)
  - Streaming update behaviors
  - Error handling flows

**Key Design Principles**:

1. User always has final say (source="user" blocks locked)
2. Real-time feedback via NDJSON streaming
3. Grouped display by destination page (Phase 3)
4. Smart defaults with override (parent→children cascading)
5. Friendly terminology ("Add as new section" vs technical terms)

**Technical Stack** (from design doc):

- Textual App with 4 Screen subclasses
- Tree widget for hierarchical block display
- Async tasks for LLM streaming
- Reactive data binding for UI updates
- CSS-like styling for visual indicators

## Success Criteria Mapping

Implementation must satisfy these measurable outcomes from spec.md:

- **SC-001**: Users navigate 4 phases without documentation → Discoverable keyboard shortcuts in Footer
- **SC-002**: Visual feedback <500ms → Async UI updates, no blocking operations
- **SC-003**: Override any LLM decision → BlockState.source="user" locks decisions
- **SC-004**: Complete typical entry <3 minutes → NDJSON streaming, no unnecessary waits
- **SC-005**: Understand UI elements from labels → Friendly terminology in FR-022
- **SC-006**: 90% success with partial failures → Continue on error (FR-034)
- **SC-007**: Identify low confidence (<75%) → Color coding + ⚠ indicator
- **SC-008**: Responsive during streaming (<100ms input) → Async event loop separation

## Next Steps

After this plan is complete:

1. **Run `/speckit.tasks`** to generate detailed implementation tasks in `tasks.md`
2. **Implement in priority order**: P1 user stories first (Phase 1 + Phase 3 screens)
3. **Iterate on UX**: Test with real journal entries, refine based on feedback
4. **Validate assumptions**: Confirm NDJSON streaming performance at scale

---

**Plan Status**: Ready for Phase 0 (Research)
**Command to proceed**: `/speckit.tasks` (after Phase 0 and Phase 1 complete)
