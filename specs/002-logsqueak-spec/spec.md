# Feature Specification: Logsqueak - Interactive TUI for Knowledge Extraction

**Feature Branch**: `002-logsqueak-spec`

**Created**: 2025-11-04

**Status**: Specification

**Input**: User description: "Logsqueak. Use specs/001-logsqueak/spec.md as the specification/design."

## Clarifications

### Session 2025-11-04

- Q: Can the system handle multiple journal entries in a single session? → A: Yes - the hierarchical tree view naturally supports multiple journal entries by using date nodes (e.g., "2025-10-15", "2025-10-16") as top-level grouping nodes with journal blocks nested underneath. This requires minimal changes to the existing design.
- Q: How should API credentials (keys, tokens) be stored and accessed by the application? → A: User manages credentials in dedicated config file with file permissions (600), application reads on startup
- Q: What should the automatic retry behavior be for LLM service network failures? → A: Auto-retry once (2s delay), then prompt user
- Q: What should the log file retention and rotation policy be? → A: No automatic cleanup - logs accumulate indefinitely (user manually deletes if needed)
- Q: How should the system handle concurrent file modifications (user edits journal/page externally while TUI is running)? → A: Automatically reload modified files and re-validate current operation before proceeding
- Q: What timeout value should be used for LLM service requests? → A: 60s
- Q: How should the configuration file be initialized on first run? → A: Require user to create the configuration file before first run (show helpful error with example if missing); file specifies LLM server endpoint, credentials, model, num_ctx, and Logseq graph location
- Q: What format should the configuration file use? → A: YAML format with explicit structure (sections: llm:, logseq:, rag:) matching existing Logsqueak config format
- Q: When should configuration validation occur (startup vs lazy)? → A: Validate configuration lazily (only when each setting is first used)
- Q: How should the system respond to configuration validation failures? → A: Show error message and exit immediately (user must manually edit config file and restart)

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Select Knowledge Blocks for Integration (Priority: P1)

A Logseq user has been journaling daily and wants to identify which blocks from their journal entries contain lasting knowledge worth integrating into their knowledge base. They want to review LLM suggestions while maintaining full control over what gets extracted.

**Why this priority**: This is the foundation of the entire workflow - users must be able to select knowledge blocks before anything else can happen. Without this, the feature cannot function.

**Independent Test**: Can be fully tested by opening a journal entry in the TUI, navigating through blocks, seeing LLM suggestions appear as they stream in, and manually selecting/deselecting blocks. Delivers immediate value by showing users what the system considers knowledge vs activity logs.

**Acceptance Scenarios**:

1. **Given** one or more journal entries with blocks (mix of tasks, activities, and knowledge), **When** user opens the extraction screen, **Then** system displays all blocks in a tree view (fully expanded) showing the original journal hierarchy, with multiple journal entries grouped by date as top-level nodes
2. **Given** the tree view is displayed, **When** user navigates with arrow keys, **Then** the selected block's full content is shown at the bottom in rendered markdown with text wrapping
3. **Given** the LLM is analyzing blocks in the background, **When** the LLM identifies a block as knowledge, **Then** that block is highlighted with a distinct color (different from user-selected highlight) and the bottom panel shows the LLM's reasoning
4. **Given** the LLM has suggested a block as knowledge, **When** user views the tree, **Then** that block shows a robot emoji (🤖) indicator (positioned far left or far right without shifting the block text) and LLM-suggested highlight color
5. **Given** several blocks have been suggested by the LLM, **When** user presses Space on a block, **Then** the block is marked as user-selected (different highlight color) for extraction, and the robot emoji remains visible
6. **Given** background tasks are running, **When** the user views the screen, **Then** a status widget shows which tasks are active ("Analyzing knowledge blocks...", "Building page index: 45%")
7. **Given** the LLM has identified 5 knowledge blocks scattered among 20 total blocks, **When** user presses Shift+j repeatedly, **Then** the selection jumps directly to each of the 5 LLM-identified knowledge blocks in sequence, skipping all non-knowledge blocks, and wraps back to the first knowledge block after the last one
8. **Given** the user has selected at least one block, **When** they press 'n' (next), **Then** the system proceeds to the content editing screen

**Keyboard Controls**:

- `j` / `k` / `↓` / `↑`: Navigate blocks
- `Shift+j` / `Shift+k` / `Shift+↓` / `Shift+↑`: Jump to next/previous LLM-identified knowledge block (skips non-knowledge blocks, wraps around)
- `Space`: Toggle knowledge selection for current block
- `a`: Accept all LLM knowledge suggestions
- `r`: Reset current block to LLM suggestion (clear user selection)
- `c`: Clear all selections
- `n`: Continue to next screen
- `q`: Quit application

---

### User Story 2 - Edit and Refine Knowledge Content (Priority: P2)

After selecting knowledge blocks, the user wants to review and refine the content before integration. They want to see LLM-suggested rewordings that remove journal-specific context, compare them with the original, and choose which version to use or edit it themselves.

**Why this priority**: Essential for ensuring quality of integrated knowledge. Users need to remove temporal context and make content evergreen before it goes into their knowledge base. Depends on P1 (block selection).

**Independent Test**: Can be tested independently by manually creating a list of selected blocks from Phase 1, then testing the editing interface with LLM rewording suggestions, manual edits, and comparison views. Delivers value by improving knowledge quality.

**Acceptance Scenarios**:

1. **Given** 5 knowledge blocks were selected in the previous screen, **When** user enters the editing screen, **Then** system displays a list of those blocks with each block's original full hierarchical context (showing parent blocks with proper indentation) in a read-only display area
2. **Given** the LLM is generating reworded versions in the background, **When** a reworded version becomes available, **Then** the reworded content is displayed alongside the original hierarchical context
3. **Given** the LLM-reworded version is displayed, **When** user reviews it, **Then** they can see the original hierarchical context, the LLM-reworded text, and the current editable version to compare before choosing
4. **Given** a knowledge block is displayed, **When** user presses Tab to focus the text editor, **Then** the editor border highlights and cursor appears, allowing them to directly edit the content that will be integrated
5. **Given** the LLM has provided a reworded version and editor is unfocused, **When** user presses 'a' (Accept LLM version), **Then** the text field is updated with the reworded content
6. **Given** the user has edited a block and editor is unfocused, **When** they press 'r' (Revert to original), **Then** the text field returns to the original block content from the journal
7. **Given** background tasks are running, **When** user views the status widget, **Then** it shows which tasks are active: "Refining knowledge content: 3/5 complete", "Building page index: 78%", "Finding relevant pages: 2/5 complete"
8. **Given** page indexing is still running, **When** user tries to proceed, **Then** system shows "Waiting for page index..." message and blocks progression
9. **Given** page indexing completes, **When** RAG search begins for the selected blocks, **Then** status widget shows "Finding relevant pages: 0/5 complete" with progress updates
10. **Given** RAG search completes, **When** user presses 'n', **Then** system proceeds to integration decisions screen

**Keyboard Controls**:

- `j` / `k` / `↓` / `↑`: Navigate between knowledge blocks (auto-saves current edits, only when editor unfocused)
- `Tab`: Focus text editor for editing (or unfocus to enable navigation)
- `a`: Accept LLM reworded version (replaces editor content, only available when LLM response received and editor unfocused)
- `r`: Revert to original content (restores journal content, only when editor unfocused)
- `n`: Continue to next screen (only enabled when RAG search complete)
- `q`: Back to block selection screen

---

### User Story 3 - Review Integration Decisions and Write to Pages (Priority: P1)

The user wants to review integration suggestions for each knowledge block, see where each will be integrated in the context of the target page, and approve integrations. The user focuses on one knowledge block at a time, seeing all relevant page suggestions for that block together.

**Why this priority**: This is the core value delivery - actually integrating knowledge into pages. Without this, no knowledge gets added to the knowledge base. Must be P1 alongside block selection because it's the essential outcome.

**Independent Test**: Can be tested independently by manually creating refined knowledge blocks and candidate pages from Phase 2, then testing the decision review interface with preview, approval/skip actions, and immediate writes with provenance links. Delivers the fundamental value proposition.

**Acceptance Scenarios**:

1. **Given** 5 knowledge blocks need integration and the LLM is evaluating candidate pages from RAG search, **When** the user enters Phase 3, **Then** they see a "Processing knowledge blocks..." status while the system waits for ALL decisions for the first knowledge block to arrive (the system does NOT display any decisions until all decisions for that block are ready)
2. **Given** all decisions for a knowledge block have arrived from the LLM, **When** the system displays them, **Then** the user sees only the active/relevant decisions (e.g., if 10 candidate pages were searched but only 2 are relevant, only 2 decisions are shown; if no pages are relevant, the block shows "No relevant pages found")
3. **Given** a knowledge block has multiple relevant target pages, **When** displaying decisions, **Then** the system shows at most 2 decisions per (knowledge block, target page) pair (the LLM filters to the most relevant integration points)
4. **Given** a knowledge block is displayed with its decisions, **When** user reviews it, **Then** they see:

   - The knowledge block number and total (e.g., "Knowledge Block 1 of 5")
   - Hierarchical journal context (where the knowledge came from, shown dimmed)
   - The refined content from Phase 2 (what will be integrated)
   - A list of suggested target pages with decisions (e.g., "Projects/Acme", "Reading Notes")
   - For the currently selected decision: target page preview showing existing page structure with new content highlighted with green bar (`┃`) in its target location
   - Action description, confidence score, and LLM reasoning for the selected decision

5. **Given** a knowledge block has 3 relevant pages with decisions, **When** user presses 'j' or 'k', **Then** they navigate between the different page decisions for this knowledge block, and the preview panel updates to show the selected decision's target page context
6. **Given** a decision shows "Add under 'Project Timeline'" with the target page preview visible, **When** user presses 'y' (accept), **Then** system immediately writes that knowledge block to the target page at the specified location and atomically adds to the `processed::` property on the journal block (creating it if it doesn't exist, or appending to the comma-separated list if it does), where each link uses markdown link format `[Page Name](((uuid)))` with the target page name as link text (with '___' converted to '/' for hierarchical pages) and Logseq block reference `((uuid))` as the link target
7. **Given** a write operation succeeds, **When** the system updates the display, **Then** that decision is marked as completed with a checkmark (✓) and remains visible in the list
8. **Given** user has accepted one decision for a knowledge block and other decisions remain, **When** they press 'y' on another decision for the same block, **Then** system writes to that additional page as well (the user can integrate the same knowledge block to multiple pages)
9. **Given** a write operation fails, **When** the error occurs, **Then** system shows error details (e.g., "Target block not found"), marks that decision as failed (⚠), and allows user to continue with remaining decisions
10. **Given** user has reviewed all decisions for a knowledge block, **When** they press 'n' (next block), **Then** system advances to the next knowledge block and waits for all its decisions to arrive before displaying (same batching behavior)
11. **Given** the status display shows "Block 2 of 5 (3 decisions: ✓ 1 completed, ⊙ 2 pending)", **When** user views the screen, **Then** they can see progress at a glance and available actions are shown in the footer
12. **Given** all knowledge blocks are reviewed and processed, **When** the last block completes, **Then** system shows completion summary with total blocks integrated, total writes performed, failed count, and link to journal entry with provenance markers

**Keyboard Controls**:

- `j` / `k` / `↓` / `↑`: Navigate between decisions for the current knowledge block
- `y`: Accept current decision and write immediately (can accept multiple decisions for same block)
- `n`: Skip to next knowledge block without writing any more decisions for current block
- `a`: Accept and write all pending decisions for current knowledge block, then advance to next block
- `Enter`: Advance to next knowledge block (when current block processing complete)
- `q`: Back to content editing screen

---

### Edge Cases

- What happens if configuration file is missing on first run? (System displays error message with expected path `~/.config/logsqueak/config.yaml` and example YAML format, then exits with non-zero status)
- What happens if a configuration setting is invalid when first accessed? (System displays clear error message indicating which setting is invalid and expected format/value, then exits immediately with non-zero status; user must manually edit config file and restart)
- What happens when a journal entry has no blocks identified as knowledge by the LLM? (System shows "No knowledge blocks identified" message, user can manually select blocks anyway)
- How does the system handle streaming interruptions (network issues, LLM API errors)? (Shows error message, preserves partial state, allows user to retry or cancel)
- What happens if page indexing fails? (Shows error, offers to retry, blocks progression to Phase 3)
- What happens if RAG search returns candidate pages but LLM determines none are relevant for a knowledge block? (Phase 3 shows "No relevant pages found" message for that block, user presses 'n' to skip to next knowledge block)
- What happens when LLM service request fails due to network error? (System automatically retries once after 2-second delay; if still failing, prompts user to retry or cancel)
- What if a target block specified in an integration decision no longer exists when user presses 'y'? (Write operation fails for that decision, error shown with details, decision marked as failed (⚠), user can continue with remaining decisions)
- How does the system handle very large journal entries (approaching 2000-line limit)? (UI remains responsive with virtualized scrolling; warns user if limit approached)
- What happens when user tries to cancel (Ctrl+C) during a write operation in Phase 3? (Shows warning about potential partial journal state with some blocks already marked as processed, asks for confirmation)
- What if the LLM response includes malformed JSON during streaming? (Logs error, shows user-friendly message, skips that item, continues with remaining items)
- What happens if user edits journal or target page file externally (e.g., in Logseq app) while TUI session is active? (System detects file modification timestamp changes before write operations, automatically reloads modified files, re-validates that target blocks/structure still exist, and either proceeds with write or shows error if validation fails)
- What happens if user accepts a decision ('y') but the write takes a long time? (UI shows "Writing..." status, blocks navigation until write completes or fails, then auto-advances to next decision)
- What if user skips all knowledge blocks without accepting any decisions? (Completion summary shows "0 blocks integrated, 5 blocks skipped" with link to journal entry - no processed:: markers added since nothing was written)
- What happens if user advances to next knowledge block ('n' or 'a') while decisions are still streaming in for that next block? (System shows "Processing knowledge blocks..." status and blocks interaction until all decisions for that block arrive)

## Requirements *(mandatory)*

### Functional Requirements

#### Phase 1: Block Selection

- **FR-001**: System MUST display journal blocks in a hierarchical tree view fully expanded, showing all parent-child relationships from the original journal entries, with multiple journal entries grouped by date as top-level nodes (e.g., "2025-10-15", "2025-10-16") with markdown rendering of the first line only of each block (see FR-014 for supported markdown features)
- **FR-002**: System MUST display the selected block's full content at the bottom of the screen in rendered markdown with text wrapping, showing all lines in the block content without indentation
- **FR-003**: System MUST display LLM reasoning for why a block was identified as knowledge in the bottom panel when that block is selected
- **FR-004**: System MUST run two background tasks in parallel when this screen loads: (1) LLM classification of knowledge blocks with incremental results, (2) page index building for semantic search
- **FR-005**: System MUST display a status widget showing which background tasks are active and their progress (percentage when calculable, on/off status otherwise)
- **FR-006**: System MUST visually highlight blocks identified as knowledge by the LLM with a distinct color (different from user-selected highlight) as they arrive via streaming
- **FR-007**: System MUST allow users to manually select/deselect any block as knowledge using keyboard controls, independent of LLM suggestions
- **FR-008**: System MUST use distinct highlight colors to differentiate between LLM-suggested blocks and user-selected blocks, ensuring clear visual distinction between the two selection states
- **FR-009**: Users MUST be able to navigate between blocks using keyboard controls:
  - **(a)** Basic navigation: arrow keys (↑/↓) or vim-style (j/k) to move sequentially between blocks
  - **(b)** Jump-to-knowledge navigation: Shift+j/k/↓/↑ to jump to next/previous LLM-identified knowledge blocks, skipping non-knowledge blocks, with wrap-around behavior
- **FR-010**: Users MUST be able to accept all LLM suggestions at once using 'a' key
- **FR-011**: System MUST allow users to reset a user-selected block back to LLM suggestion state using 'r' key
- **FR-012**: System MUST allow users to clear all selections using 'c' key
- **FR-013**: System MUST allow progression to Phase 2 only when at least one block is selected
- **FR-014**: System MUST support rich markdown rendering in both the tree view (first line only) and bottom panel (all lines), including links, bold, code, checkboxes, and text wrapping

#### Phase 2: Content Editing

- **FR-015**: System MUST display a list of all selected knowledge blocks from Phase 1
- **FR-016**: System MUST display each block's original full hierarchical context (including parent blocks with proper indentation) in a read-only display area
- **FR-017**: System MUST display each block's content in an editable text field, starting with the original block content (without parent context)
- **FR-018**: System MUST run three background tasks in parallel when this screen loads: (1) LLM generation of reworded versions with incremental results, (2) continue page indexing if not complete, (3) semantic search after indexing completes
- **FR-019**: System MUST display a waiting indicator before reworded version is available, then display the LLM-reworded content alongside the original hierarchical context
- **FR-020**: System MUST provide keyboard shortcuts to accept LLM version ('a' key) and revert to original ('r' key)
- **FR-021**: System MUST visually indicate when "Accept LLM version" action is available versus unavailable based on whether the LLM-reworded version has arrived
- **FR-022**: System MUST allow users to directly edit the text field content using a multi-line text editor with cursor movement, selection, and standard editing features
- **FR-023**: System MUST update the text field with LLM-reworded content when user presses 'a'
- **FR-024**: System MUST restore the original block content when user presses 'r'
- **FR-025**: System MUST display status widget showing progress for all active background tasks
- **FR-026**: System MUST block progression to Phase 3 until page indexing is complete
- **FR-027**: System MUST start RAG search for selected blocks when page indexing completes
- **FR-028**: System MUST block progression to Phase 3 until RAG search is complete
- **FR-029**: System MUST display status messages when blocking progression
- **FR-029a**: When a background task fails (page indexing, LLM classification, RAG search), system MUST display error details and offer user options to retry the failed task or cancel the operation
- **FR-030**: Users MUST be able to navigate between knowledge blocks using keyboard controls

#### Phase 3: Integration Decisions

- **FR-031**: System MUST perform semantic search (combining similarity search and explicit page link hints) using the original hierarchical context of each knowledge block
- **FR-032**: System MUST batch LLM integration decisions by knowledge block - all decisions for a given knowledge block must arrive before displaying any decisions for that block
- **FR-032a**: LLM MUST return only active/relevant decisions via prompt-based filtering (e.g., if 10 candidate pages are searched but only 2 are relevant, return only 2 decisions). See contracts/llm-api.md Phase 3 "Relevance Filtering" section for filtering criteria (confidence ≥ 30%, clear semantic connection).
- **FR-032b**: LLM MUST return at most 2 decisions per (knowledge block, target page) pair
- **FR-033**: System MUST display all decisions for a knowledge block together in a list format, with one knowledge block visible at a time
- **FR-034**: System MUST show confidence scores as percentages for each integration decision
- **FR-035**: System MUST display hierarchical journal context with proper indentation, all markdown rendered and displayed in dimmed style
- **FR-036**: System MUST display the refined content from Phase 2 (not the original journal content) as the content that will be integrated
- **FR-037**: System MUST display target page preview showing the existing page structure with the new content visually integrated for the currently selected decision
- **FR-038**: System MUST show new content with green bar (`┃`) visual indicator in its target location
- **FR-039**: System MUST show replaced content with strikethrough styling, with new content shown below it with green bar
- **FR-040**: System MUST provide navigation controls (j/k/arrows) to move between decisions for the current knowledge block
- **FR-041**: System MUST write the knowledge block to the target page immediately when user presses 'y' on a pending decision
- **FR-042**: System MUST allow user to accept multiple decisions for the same knowledge block (integrating the same content to multiple pages)
- **FR-043**: System MUST atomically add to the `processed::` property on the journal block after each successful page write (appending to comma-separated list)
- **FR-044**: System MUST mark completed integrations with checkmark (✓) and keep them visible in the decision list
- **FR-045**: System MUST display error details if a write operation fails, mark decision as failed (⚠), and allow continuation with remaining decisions
- **FR-046**: System MUST allow advancing to next knowledge block using 'n' key (skipping remaining decisions for current block)
- **FR-047**: System MUST provide batch action to accept and write all pending decisions for current knowledge block using 'a' key, then advance to next block
- **FR-048**: System MUST display status showing current knowledge block number, total blocks, and decision status breakdown for current block
- **FR-049**: System MUST display context-sensitive footer showing available actions based on current decision state
- **FR-050**: System MUST display LLM reasoning for the currently selected decision
- **FR-051**: System MUST support full Logseq markdown rendering in all displayed content
- **FR-052**: System MUST show "Processing knowledge blocks..." status while waiting for all decisions for a knowledge block to arrive
- **FR-053**: System MUST show "No relevant pages found" message if LLM returns no active decisions for a knowledge block

#### Completion & Error Handling

- **FR-054**: System MUST display a completion summary showing total blocks integrated, failed count, and link to journal entry with provenance markers
- **FR-055**: Users MUST be able to view detailed error information with suggested remediation actions
- **FR-056**: System MUST provide actionable error messages for common failure modes

#### General UI Requirements

- **FR-057**: System MUST display consistent keyboard shortcuts in a footer area on all screens
- **FR-058**: System MUST support both standard (arrow keys, Enter) and vim-style (j/k) navigation
- **FR-059**: System MUST allow users to quit at any phase using Ctrl+C or 'q' key
- **FR-060**: System MUST execute all background tasks in a way that allows UI interaction to continue uninterrupted
- **FR-061**: System MUST notify users when malformed data is encountered during LLM streaming while continuing to process remaining items
- **FR-062**: System MUST use consistent key bindings across all phases for similar actions

#### Logging & Debugging

- **FR-063**: System MUST log complete LLM request/response lifecycle for debugging and troubleshooting:
  - **(a)** Log complete LLM requests at the start of each request for debugging and audit purposes
  - **(b)** Log LLM response data incrementally as it arrives for troubleshooting interrupted sessions
- **FR-065**: System MUST log detailed structured information including block IDs, confidence scores, and reasoning
- **FR-066**: System MUST ensure logging occurs even on errors or cancellation
- **FR-067a**: System MUST NOT automatically delete or rotate log files (logs accumulate indefinitely until user manually cleans up cache directory)

#### Data Model Requirements

- **FR-067**: System MUST use existing Logseq parsing capabilities for all Logseq markdown reading, writing, and block manipulation
- **FR-068**: System MUST preserve exact property order when reading and writing Logseq files (property order is significant in Logseq)
- **FR-069**: System MUST use stable block identifiers (explicit ID properties OR content-based identifiers) for consistent block targeting
- **FR-070**: System MUST generate deterministic identifiers (UUID v5) for all integrated blocks to enable precise future references and idempotent retry operations
- **FR-071**: System MUST maintain provenance links from journal to integrated blocks via property that lists target page and block references

#### Concurrent Modification Handling

- **FR-071a**: System MUST check file modification timestamps before performing any write operation to journal or page files
- **FR-071b**: System MUST automatically reload files that have been modified externally since initial load
- **FR-071c**: System MUST re-validate that target blocks and structure referenced in integration decisions still exist after reloading modified files
- **FR-071d**: If validation succeeds after reload, system MUST proceed with write operation; if validation fails (target block deleted, structure changed), system MUST mark operation as failed with descriptive error message

#### Configuration Management

- **FR-072a**: System MUST require configuration file at `~/.config/logsqueak/config.yaml` to exist before first run
- **FR-072b**: If configuration file is missing, system MUST display helpful error message with expected file path and example YAML format with structured sections (llm:, logseq:, rag:) showing required fields: llm.endpoint, llm.api_key, llm.model, llm.num_ctx (optional), logseq.graph_path, and rag.top_k (optional)
- **FR-072c**: System MUST NOT automatically create configuration file with default or placeholder values
- **FR-072d**: Configuration file MUST use YAML format with explicit hierarchical structure matching existing Logsqueak config conventions (llm: section for LLM service settings, logseq: section for graph settings, rag: section for search settings)
- **FR-072e**: System MUST validate configuration settings lazily (only when each setting is first accessed during operation), not eagerly at startup
- **FR-072f**: When lazy validation fails for a setting (e.g., invalid graph path, missing required field), system MUST display clear error message indicating which setting is invalid and what the valid format or expected value should be, then exit immediately with non-zero status
- **FR-072g**: System MUST NOT provide in-TUI configuration editing or retry mechanisms; user must manually edit configuration file and restart application after validation failures

#### Security & Privacy

- **FR-072**: System MUST read API credentials from dedicated configuration file with restrictive file permissions (mode 600 or equivalent platform restriction)
- **FR-073**: System MUST load credentials only on application startup (not stored in memory beyond active session)
- **FR-074**: System MUST display clear error message if configuration file has overly permissive permissions (readable by group/world) and refuse to start

#### Network Resilience

- **FR-075**: System MUST automatically retry failed LLM service requests once after a 2-second delay for transient network errors (connection timeout, connection refused, temporary service unavailability)
- **FR-076**: System MUST preserve partial results received before network failure and allow user to continue with available data or retry the failed operation
- **FR-077**: After automatic retry fails, system MUST prompt user with options to: retry manually, cancel operation, or continue with partial results (where applicable)
- **FR-078**: System MUST apply a 60-second timeout to all LLM service requests (for initial connection and ongoing streaming response)
- **FR-079**: When timeout is reached, system MUST treat it as a network failure and follow the retry behavior specified in FR-075 through FR-077

### Key Entities

- **BlockState**: Represents the selection state of a journal block (block_id: str, classification: Literal["pending", "knowledge"], confidence: float | None, source: Literal["user", "llm"], llm_classification: Literal["knowledge"] | None, llm_confidence: float | None, reason: str | None)

- **EditedContent**: Represents the edited content for a knowledge block in Phase 2 (block_id: str, original_content: str, reworded_content: str | None, current_content: str, rewording_complete: bool)

- **IntegrationDecision**: Represents a decision about integrating a knowledge block to a specific page (knowledge_block_id: str, target_page: str, action: Literal["add_section", "add_under", "replace"], target_block_id: str | None, target_block_title: str | None, confidence: float, refined_text: str, reasoning: str, write_status: Literal["pending", "completed", "failed"])

- **BackgroundTask**: Represents a long-running background task (task_type: Literal["llm_classification", "page_indexing", "rag_search", "llm_rewording", "llm_decisions"], status: Literal["running", "completed", "failed"], progress_percentage: float | None)

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can navigate through all phases of knowledge extraction without requiring external documentation for basic operations
- **SC-002**: Users see visual feedback within 500ms of any interaction (keyboard input, phase transition)
- **SC-003**: Users can override any LLM suggestion at any phase before final write operations
- **SC-004**: System completes the extraction workflow for a typical journal entry (11 avg root blocks, 22 avg total blocks, ~5 knowledge blocks, ~2 relevant decisions per block) in under 3 minutes
- **SC-005**: Users can identify LLM-suggested knowledge blocks vs user-selected blocks through visual indicators
- **SC-006**: When write operations fail for one page, other integrations still complete successfully
- **SC-007**: The UI remains responsive (accepts user input within 100ms) even while background analysis is running
- **SC-008**: 100% of integrated knowledge blocks include valid provenance links to source journal
- **SC-009**: System handles LLM API errors gracefully in 100% of cases (no crashes, clear error messages)

## Assumptions

1. **LLM Analysis Support**: The system can leverage LLM services that support incremental result streaming for responsive user experience.

2. **Terminal Capabilities**: Users have modern terminal emulators supporting 256 colors and Unicode characters. Fallback ASCII characters available for limited environments.

3. **Typical Journal Entry Size**: Most journal entries have 3-15 root blocks with 2-3 levels of nesting, totaling under 100 blocks. Validated against actual Logseq graph.

4. **User Familiarity**: Users are comfortable with keyboard-driven interfaces (similar to vim, htop, other TUIs).

5. **Multiple Journal Entries**: Users can process multiple journal entries in a single session. Tree view displays entries grouped by date (e.g., "2025-10-15", "2025-10-16") with journal blocks nested underneath, enabling efficient batch processing of date ranges.

6. **Error Recovery**: Users can resolve errors by following suggested actions and re-running extraction. Mid-session retry not required initially.

## Dependencies

- **Text User Interface Framework**: Requires a TUI library for building interactive terminal applications.

- **Logseq Parser**: Depends on existing Logseq markdown parsing and rendering capabilities for reading and writing Logseq files.

- **Existing Pipeline Infrastructure**: Depends on extraction, semantic search, integration, and LLM client modules remaining functionally stable.

- **LLM Service APIs**: Requires LLM services that support streaming responses for incremental results.

- **Async Runtime**: Requires async/await support for concurrent background tasks and UI updates.

- **Vector Store**: Requires persistent block-level semantic search with vector embeddings.

- **Embedding Model**: Requires text embedding capabilities for semantic similarity search.

## Out of Scope

- Mouse/click-based interactions (keyboard-only interface)
- Saving user preferences or learning from past decisions
- Batch processing multiple journal entries
- Custom themes or extensive visual customization
- Undo/redo functionality for decisions made during session
- Split-screen or multi-pane views (single screen per phase)
- Export of extraction decisions to external formats
- Real-time collaboration features
- Creating new Logseq pages (target pages must exist)
- Automatic extraction without user review (always requires explicit approval)

## Project Constitution

**See `.specify/memory/constitution.md` for complete project constitution.**

Key principles:

### I. Proof-of-Concept First

- Prioritize working software over perfection
- Ship iteratively, demonstrate feasibility
- No backwards compatibility guarantees (POC stage)

### II. Non-Destructive Operations (NON-NEGOTIABLE)

- All operations traceable via `processed::` markers in journal entries
- UPDATE operations replace content but preserve structure/IDs
- APPEND operations add new blocks without modifying existing content
- Every integrated block generates deterministic `id::` property (UUID v5 based on knowledge_block_id + target_page + action)
- Deterministic IDs enable idempotent retry (same inputs → same UUID → detectable duplicate)
- Journal entries atomically marked with block references to integrated knowledge

### III. Simplicity and Transparency

- Prefer file-based I/O over databases (except ChromaDB for RAG)
- Use JSON for structured LLM outputs
- Show the user what the LLM is doing
- Avoid premature abstraction

### Commit Message Requirements

All AI-assisted commits MUST include:

```
Assisted-by: Claude Code

```
