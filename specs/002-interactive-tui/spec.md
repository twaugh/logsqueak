# Feature Specification: Interactive TUI for Knowledge Extraction

**Feature Branch**: `002-interactive-tui`
**Created**: 2025-10-29
**Status**: Draft
**Input**: User description: "Interactive TUI. Use the design written up in specs/002-interactive-tui/interactive-tui-design.md."

## Clarifications

### Session 2025-10-29

- Q: Do LLM providers actually support streaming JSON-structured outputs for the interactive TUI? → A: **VALIDATED with better approach discovered!** Testing against live Ollama server (qwen2.5:7b-instruct) revealed **NDJSON (newline-delimited JSON) enables true incremental parsing**. By prompting for NDJSON format (without `format='json'` parameter), each knowledge block arrives as a complete JSON object on its own line. Successfully parsed 6 objects incrementally across 129 chunks (objects appeared at chunks 21, 44, 67, 86, 106, final). This enables real-time UI updates as each block is generated, providing much better UX than waiting for complete response.

- Q: Are the journal entry size assumptions realistic? → A: **VALIDATED against actual Logseq graph** - Analyzed 10 most recent journal entries. Results: 11.4 avg root blocks (range: 3-26), 22.2 avg total blocks (max: 71), 1.2 avg nesting depth (max: 2 levels), 27.2 avg lines (max: 77). All metrics well within assumed ranges (3-15 root blocks, 2-3 nesting levels, <100 total blocks, <2000 lines). Performance optimization targeting this range is appropriate.

- Q: Do we need to maintain backward compatibility with existing batch mode? → A: **No** - This feature will **replace** the current batch extraction workflow entirely. The CLI interface can be changed freely. No need to preserve `--no-interactive` flag or maintain the old non-interactive behavior. This simplifies implementation significantly.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Review and Approve Knowledge Extraction (Priority: P1)

As a Logsqueak user, I want to see what the system identifies as "lasting knowledge" versus "daily activity" in my journal entries, so that I can ensure only meaningful insights are extracted and integrated into my knowledge base.

**Why this priority**: This is the foundation of user control and transparency. Without the ability to review and correct knowledge extraction decisions, users may lose trust in the system or end up with low-quality knowledge integration. This story delivers immediate value by giving users visibility and control over the most critical decision point.

**Independent Test**: Can be fully tested by running the extraction command on a journal entry with mixed content (knowledge + activities), reviewing the classification UI, and verifying that user overrides are respected in the final output.

**Acceptance Scenarios**:

1. **Given** a journal entry with 5 blocks (3 knowledge, 2 activities), **When** I run the interactive extraction command, **Then** I see a tree view showing all 5 blocks with their LLM-assigned classifications and confidence scores
2. **Given** the LLM has classified a block as "activity" with 85% confidence, **When** I navigate to that block and mark it as "knowledge", **Then** the block's classification changes to "knowledge" with a visual indicator showing it was user-marked
3. **Given** I've marked a parent block as "activity", **When** the system applies smart defaults, **Then** all child blocks are also marked as "activity" unless I individually override them
4. **Given** the LLM is still analyzing some blocks, **When** I press Enter to continue, **Then** the system proceeds to the next phase with the current classifications (user-marked blocks take precedence over pending LLM decisions)

---

### User Story 2 - See LLM Integration Decisions with Streaming Feedback (Priority: P1)

As a Logsqueak user, I want to see in real-time how the system decides where to integrate each piece of knowledge and how it rewords the content, so that I can understand the system's reasoning and make informed decisions about accepting or modifying the proposals.

**Why this priority**: This provides transparency for the most complex phase (integration decisions). Users need to see where their knowledge is going and how it's being reworded before committing changes to their graph. This is critical for trust and quality control.

**Independent Test**: Can be fully tested by running extraction through Phase 3, observing the streaming display of decisions and reworded content, and verifying that all proposed integrations are shown with their target pages, actions, and refined text.

**Acceptance Scenarios**:

1. **Given** the system has found 3 candidate pages for a knowledge block, **When** the LLM evaluates each page, **Then** I see streaming decisions appear showing the action (e.g., "Add as new section", "Skip - already covered") and confidence scores
2. **Given** the LLM decides to integrate a knowledge block into a page, **When** the rewording phase starts, **Then** I see "Refining... [streaming]" followed by the progressively appearing refined text
3. **Given** the decisions are grouped by destination page, **When** I view the Phase 3 screen, **Then** I see pages organized with their associated knowledge blocks, and pages with no accepted blocks are collapsed with explanatory text
4. **Given** the LLM suggests an action with 62% confidence, **When** the decision appears in the UI, **Then** I see a warning indicator (⚠) suggesting I review this decision more carefully

---

### User Story 3 - Override Integration Decisions and Edit Content (Priority: P2)

As a Logsqueak user, I want to change where knowledge gets integrated and edit the reworded content before it's written to my graph, so that I have final control over what goes into my knowledge base and where it's placed.

**Why this priority**: This completes the user control loop. While P1 stories provide visibility, this story enables active intervention. It's P2 because the system can still function with just viewing and accepting LLM decisions, but this adds critical editing capabilities for power users.

**Independent Test**: Can be fully tested by navigating to a proposed integration in Phase 3, changing its action from "Add as new section" to "Add under [specific block]", editing the refined text, and verifying those changes are reflected in the final write operations.

**Acceptance Scenarios**:

1. **Given** the LLM suggests "Add as new section" for a knowledge block, **When** I press Space to cycle through actions, **Then** I see options rotate through "Skip", "Add as new section", "Add under [block title]", and "Replace [block title]"
2. **Given** I've selected "Add under [block title]" action, **When** I press 'T' to pick a different location, **Then** I see a dialog showing all available blocks in that page where I can choose the target location
3. **Given** the LLM has reworded a knowledge block, **When** I press 'E' to edit the text, **Then** I see a text editor showing both the original journal text and the refined version, and I can modify the refined text
4. **Given** I've changed an integration decision, **When** the system continues processing other blocks, **Then** my modified decision is "locked" and the LLM does not override it

---

### User Story 4 - Review Candidate Pages Before Integration (Priority: P3)

As a Logsqueak user, I want to see which pages the system is considering for each knowledge block and optionally exclude irrelevant ones, so that I can reduce unnecessary LLM evaluations and ensure knowledge only goes to appropriate pages.

**Why this priority**: This is an optimization feature that reduces costs and improves precision. Most users will trust the RAG system's candidate selection, making this optional. It's valuable for power users who want fine-grained control, but not essential for the core workflow.

**Independent Test**: Can be fully tested by pressing 'R' during Phase 2 to enter review mode, viewing the candidate pages with match percentages, toggling some pages off, and verifying those pages are excluded from Phase 3 decision-making.

**Acceptance Scenarios**:

1. **Given** the system has retrieved 12 candidate pages for a knowledge block, **When** I press 'R' during Phase 2, **Then** I see a list of all 12 pages with their semantic match percentages
2. **Given** I'm reviewing candidate pages, **When** I press Space on a page with 45% match that seems irrelevant, **Then** that page is marked as excluded (✗) and will not be considered in Phase 3
3. **Given** I'm in candidate review mode, **When** I press 'N' to move to the next knowledge block, **Then** I see the candidate pages for the next block with all pages initially selected
4. **Given** I don't interact during Phase 2, **When** candidate retrieval completes, **Then** the system automatically proceeds to Phase 3 with all candidates selected

---

### User Story 5 - Monitor Write Progress and Handle Errors Gracefully (Priority: P2)

As a Logsqueak user, I want to see the progress of write operations to my knowledge graph and understand what succeeded or failed, so that I can track what was integrated and address any errors that occur.

**Why this priority**: This provides closure to the extraction workflow and ensures users understand what happened. Error visibility is critical for maintaining graph integrity. It's P2 because write operations happen after all user decisions, so it's less interactive, but still important for confidence and troubleshooting.

**Independent Test**: Can be fully tested by running extraction through Phase 4, observing the per-page write progress, and verifying that completion summaries accurately reflect what was written (including any errors that occurred).

**Acceptance Scenarios**:

1. **Given** the system is writing 18 blocks to 8 pages, **When** Phase 4 executes, **Then** I see a progress display showing each page's status (pending → writing → complete) with the number of blocks being written
2. **Given** all write operations complete successfully, **When** Phase 4 finishes, **Then** I see a completion summary showing total blocks added, pages updated, and a link to my journal entry with provenance markers
3. **Given** a write operation fails for one page, **When** Phase 4 continues with remaining pages, **Then** I see the error marked (⚠) with details about what failed, and the system continues processing other pages
4. **Given** the completion summary shows errors, **When** I press 'E' to view error details, **Then** I see specific error messages with suggested actions (e.g., "Rebuild your index: logsqueak index rebuild")

---

### Edge Cases

- What happens when a journal entry has no blocks classified as "knowledge" after Phase 1? (System should show completion message indicating no knowledge to extract)
- How does the system handle streaming interruptions (network issues, LLM API errors)? (Should preserve partial state, show error message, allow user to retry or cancel)
- What happens if a user marks a parent block as "knowledge" but then individually marks all children as "activity"? (System respects individual child decisions, only the parent classification applies to the parent block itself)
- How does the system behave when a user presses Enter during Phase 1 while LLM is still analyzing blocks? (System cancels pending LLM tasks and proceeds with current classifications)
- What happens when the LLM response includes malformed JSON during streaming? (System logs error, shows user-friendly message, skips that particular item, continues with remaining items)
- What happens if a target block specified in an integration decision no longer exists in the destination page? (Write operation fails for that specific integration, error is logged with details, other integrations for that page continue)
- How does the system handle very large journal entries (e.g., approaching the 2000-line limit)? (UI should remain responsive with virtualized scrolling; streaming should handle blocks progressively)
- What happens when a user tries to cancel (Ctrl+C) during Phase 4 write operations? (System warns that canceling mid-write may leave partial updates, asks for confirmation before terminating)

## Requirements *(mandatory)*

### Functional Requirements

#### Phase 1: Knowledge Extraction

- **FR-001**: System MUST display journal blocks in a hierarchical tree view showing the parent-child relationships from the original journal entry
- **FR-002**: System MUST visually indicate the classification state of each block (pending, LLM-classified as knowledge, LLM-classified as activity, user-marked as knowledge, user-marked as activity)
- **FR-003**: System MUST display confidence scores as percentages for LLM-classified blocks
- **FR-004**: System MUST highlight blocks with confidence scores below 75% with warning indicators to suggest user review
- **FR-005**: Users MUST be able to navigate between blocks using keyboard controls (arrow keys, j/k)
- **FR-006**: Users MUST be able to manually mark any block as "knowledge" or "activity", overriding LLM decisions
- **FR-007**: System MUST apply smart defaults when a parent block is marked (all children default to the same classification)
- **FR-008**: Users MUST be able to override individual child block classifications after a parent default has been applied
- **FR-009**: System MUST prevent the LLM from overriding any user-marked block classification
- **FR-010**: System MUST update the UI progressively as LLM classification results stream in
- **FR-011**: Users MUST be able to proceed to Phase 2 at any time by pressing Enter, even if LLM analysis is incomplete
- **FR-012**: System MUST cancel ongoing LLM analysis tasks when the user proceeds to the next phase

#### Phase 2: Candidate Page Retrieval

- **FR-013**: System MUST display real-time progress of candidate page retrieval for each knowledge block
- **FR-014**: System MUST automatically proceed to Phase 3 when all candidate retrieval completes and user has not requested review
- **FR-015**: Users MUST be able to enter review mode by pressing 'R' during candidate retrieval
- **FR-016**: System MUST display candidate pages with semantic match percentages in descending order (highest match first)
- **FR-017**: Users MUST be able to toggle individual pages for inclusion/exclusion from consideration
- **FR-018**: Users MUST be able to navigate between knowledge blocks in review mode to see candidates for each
- **FR-019**: System MUST exclude user-deselected pages from Phase 3 decision-making

#### Phase 3: Integration Decisions

- **FR-020**: System MUST group integration decisions by destination page (not by knowledge block)
- **FR-021**: System MUST display streaming LLM decisions for each (knowledge block, candidate page) pair as they arrive
- **FR-022**: System MUST translate technical action types to user-friendly labels (e.g., "APPEND_ROOT" → "Add as new section")
- **FR-023**: System MUST show confidence scores for each integration decision
- **FR-024**: System MUST display "Refining... [streaming]" status while reworded content is being generated
- **FR-025**: System MUST show progressively appearing refined text as it streams from the LLM
- **FR-026**: System MUST collapse pages with no accepted integrations and show explanatory text (e.g., "All blocks skipped - already covered")
- **FR-027**: Users MUST be able to cycle through available actions (Skip, Add as new section, Add under, Replace) using keyboard controls
- **FR-028**: Users MUST be able to edit refined text using an inline text editor
- **FR-029**: Users MUST be able to select different target block locations for "Add under" and "Replace" actions
- **FR-030**: System MUST lock user-modified decisions against LLM overrides
- **FR-031**: Users MUST be able to collapse/expand page sections to manage screen space

#### Phase 4: Write Operations

- **FR-032**: System MUST display per-page write progress showing pending, in-progress, and completed status
- **FR-033**: System MUST show the number of blocks being written to each page
- **FR-034**: System MUST continue processing remaining pages if a write operation fails for one page
- **FR-035**: System MUST log detailed error information for any failed write operations
- **FR-036**: System MUST atomically update journal entries with provenance markers only after successful page writes
- **FR-037**: System MUST display a completion summary showing total blocks added, pages updated, and link to journal entry

#### Completion & Error Handling

- **FR-038**: System MUST provide a completion summary distinguishing between fully successful extractions and those with errors
- **FR-039**: Users MUST be able to view detailed changes (which blocks were added to which pages)
- **FR-040**: Users MUST be able to view detailed error information with suggested remediation actions
- **FR-041**: System MUST provide actionable error messages (e.g., "Rebuild your index: logsqueak index rebuild")

#### General UI Requirements

- **FR-042**: System MUST display consistent keyboard shortcuts in a footer area on all screens
- **FR-043**: System MUST support both standard (arrow keys, Enter, Esc) and vim-style (j/k/h/l) navigation
- **FR-044**: System MUST allow users to quit at any phase using Ctrl+C or 'Q' key
- **FR-045**: System MUST maintain responsive UI performance while streaming LLM responses (UI updates should not block user input)

### Key Entities

- **BlockState**: Represents the classification state of a journal block throughout the pipeline (block ID, classification type, confidence score, source of classification)
- **IntegrationDecision**: Represents a decision about integrating a specific knowledge block into a target page (knowledge block ID, target page, action type, target block ID if applicable, confidence score, refined text, source of decision)
- **CandidatePage**: Represents a page retrieved during RAG search (page name, semantic match score, included/excluded status, blocks within the page for targeting)
- **ProgressState**: Represents the current phase and completion status (phase number, completion percentage, streaming status indicators)

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can navigate through all four phases of knowledge extraction without requiring any external documentation for basic operations
- **SC-002**: Users see visual feedback within 500ms of any interaction (keyboard input, phase transition)
- **SC-003**: Users can override any LLM decision at any phase before final write operations execute
- **SC-004**: System completes the extraction workflow for a typical 10-block journal entry (with 5 knowledge blocks, 3 candidate pages per block) in under 3 minutes
- **SC-005**: Users can understand the purpose of each UI element from visible labels and indicators (no hidden keyboard shortcuts required for core functionality)
- **SC-006**: When write operations fail for one page, at least 90% of successful integrations still complete successfully
- **SC-007**: Users can identify low-confidence LLM decisions (< 75%) through visual indicators without reading documentation
- **SC-008**: The UI remains responsive (accepts user input within 100ms) even while processing streaming LLM responses

## Assumptions

1. **Streaming LLM Support**: We assume the LLM providers support streaming responses for JSON-structured outputs. **Validated 2025-10-29 - NDJSON streaming discovered**: Tested with Ollama (qwen2.5:7b-instruct). By prompting for NDJSON (newline-delimited JSON) format, the LLM outputs one complete JSON object per line, enabling true incremental parsing. Successfully parsed 6 objects incrementally as they arrived across 129 chunks. The UI will use NDJSON streaming (buffer chunks, split by newlines, parse each complete line) to update in real-time as each knowledge block/decision is generated. This provides significantly better UX than waiting for the complete response, allowing users to review early results while the LLM continues processing.

2. **Terminal Capabilities**: We assume users have modern terminal emulators supporting 256 colors and Unicode characters (for visual indicators like ✓, ⚠, ○, etc.). Fallback ASCII characters will be used in limited environments.

3. **Typical Journal Entry Size**: We assume most journal entries will have 3-15 root blocks with 2-3 levels of nesting, totaling under 100 blocks. **Validated 2025-10-29**: Analyzed 10 recent journal entries from actual Logseq graph. Measured 11.4 avg root blocks (max: 26), 22.2 avg total blocks (max: 71), 1.2 avg nesting depth (max: 2 levels), 27.2 avg lines (max: 77). All metrics confirm assumptions are realistic. Performance optimization will target this range.

4. **User Familiarity**: We assume users are comfortable with basic keyboard-driven interfaces (similar to common CLI tools, text editors, or terminal UIs like htop, vim). No mouse support is planned initially.

5. **Single Extraction Session**: We assume users will process one journal entry at a time in a single session. Batch processing multiple dates in interactive mode is out of scope.

6. **Error Recovery**: We assume users can resolve errors (like stale index issues) by following suggested actions and re-running the extraction. Mid-session error recovery with retry is not required initially.

## Dependencies

- **Textual Framework**: The feature requires the Textual Python library (version 0.47.0+) for building the TUI interface. **Note**: Not currently installed; will be added to pyproject.toml dependencies during implementation.
- **Existing 5-Phase Pipeline**: The feature depends on the current extraction, RAG, decision, and write pipeline infrastructure remaining functionally stable. **Validated 2025-10-29**: All pipeline components (extraction, integration, RAG, LLM client) are importable and functional.
- **LLM Streaming APIs**: The feature requires LLM providers to support streaming responses (OpenAI-compatible streaming endpoints)
- **Async Support**: The feature requires Python 3.11+ with async/await support for concurrent LLM streaming and UI updates

## Out of Scope

- Mouse/click-based interactions (keyboard-only interface)
- Saving user preferences or learning from past decisions
- Batch processing multiple journal entries in a single interactive session
- Custom themes or extensive visual customization
- Undo/redo functionality for decisions made during the session
- Split-screen or multi-pane views (single screen per phase)
- Export of extraction decisions or integration plans to external formats
- Real-time collaboration features (multiple users in same extraction session)
- Backward compatibility with existing batch mode (this feature replaces it entirely)
- Preserving old CLI interface or command structure
