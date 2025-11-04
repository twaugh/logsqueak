# Feature Specification: Interactive TUI for Knowledge Extraction

**Feature Branch**: `002-interactive-tui`
**Created**: 2025-10-29
**Status**: Specification
**Input**: User description: "Interactive TUI. Use the design written up in specs/002-interactive-tui/interactive-tui-design.md."

## Clarifications

### Session 2025-10-29

- Q: Do LLM providers actually support streaming JSON-structured outputs for the interactive TUI? → A: **VALIDATED with better approach discovered!** Testing against live Ollama server (qwen2.5:7b-instruct) revealed **NDJSON (newline-delimited JSON) enables true incremental parsing**. By prompting for NDJSON format (without `format='json'` parameter), each knowledge block arrives as a complete JSON object on its own line. Successfully parsed 6 objects incrementally across 129 chunks (objects appeared at chunks 21, 44, 67, 86, 106, final). This enables real-time UI updates as each block is generated, providing much better UX than waiting for complete response.

- Q: Are the journal entry size assumptions realistic? → A: **VALIDATED against actual Logseq graph** - Analyzed 10 most recent journal entries. Results: 11.4 avg root blocks (range: 3-26), 22.2 avg total blocks (max: 71), 1.2 avg nesting depth (max: 2 levels), 27.2 avg lines (max: 77). All metrics well within assumed ranges (3-15 root blocks, 2-3 nesting levels, <100 total blocks, <2000 lines). Performance optimization targeting this range is appropriate.

- Q: Do we need to maintain backward compatibility with existing batch mode? → A: **No** - This feature will **replace** the current batch extraction workflow entirely. The CLI interface can be changed freely. No need to preserve `--no-interactive` flag or maintain the old non-interactive behavior. This simplifies implementation significantly.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Review and Accept Knowledge Extraction Suggestions (Priority: P1)

As a Logsqueak user, I want to see what the system suggests as "lasting knowledge" in my journal entries and explicitly accept those suggestions, so that I maintain full control over what gets extracted and integrated into my knowledge base.

**Why this priority**: This is the foundation of user control and transparency. By requiring explicit acceptance of LLM suggestions rather than auto-classification, users maintain full control and won't accidentally integrate incorrect classifications. This story delivers immediate value by giving users visibility and requiring conscious approval of the most critical decision point.

**Independent Test**: Can be fully tested by running the extraction command on a journal entry with mixed content (knowledge + activities), reviewing the LLM suggestions in the UI, explicitly accepting suggestions, and verifying that only accepted blocks proceed to integration.

**Acceptance Scenarios**:

1. **Given** a journal entry with 5 blocks where LLM identifies 3 as knowledge, **When** I run the interactive extraction command, **Then** I see a tree view showing all 5 blocks in "pending" state with LLM knowledge suggestions displayed with dark blue background
2. **Given** the LLM has suggested a block as "knowledge" with 92% confidence, **When** I navigate to that block and press Space (toggle), **Then** the block's selection state changes to "knowledge" with a dark green background indicating user acceptance
3. **Given** I have several LLM knowledge suggestions available, **When** I press 'a' (Accept AI), **Then** all LLM knowledge suggestions are accepted at once and marked with dark green backgrounds
4. **Given** the LLM is still analyzing some blocks, **When** I press Enter to continue, **Then** the system proceeds to the next phase with currently accepted knowledge blocks (blocks not suggested by LLM are not included in integration)

**Keyboard Controls (Phase 1)**:

- `j` / `k` / `↓` / `↑`: Navigate up/down through blocks
- `Space`: Toggle knowledge selection for current block
- `a`: Accept all AI knowledge suggestions
- `r`: Reset current block to AI suggestion (clear user selection)
- `c`: Clear all selections (deselect all blocks)
- `n`: Continue to Phase 2 (Next →)
- `q`: Quit application

---

### User Story 2 - Review and Accept Reworded Knowledge Content (Priority: P1)

As a Logsqueak user, I want to see how the LLM rewords my knowledge blocks to remove journal-specific context and decide whether to keep the original or use the reworded version, so that I can ensure the content is appropriate for my evergreen knowledge base.

**Why this priority**: This provides transparency and control over content transformation before integration. Users need to see and approve how their journal entries are being rephrased as standalone knowledge before they're added to pages. This is critical for maintaining voice and accuracy.

**Independent Test**: Can be fully tested by running extraction through Phase 2, observing the streaming display of reworded content for each knowledge block, and verifying that user choices (original vs reworded) are preserved for Phase 3.

**Acceptance Scenarios**:

1. **Given** I have 3 knowledge blocks accepted in Phase 1, **When** I enter Phase 2, **Then** I see a tree view showing the journal hierarchy with knowledge blocks (collapsed by default, showing selected version) and their parents, with a help panel at the bottom
2. **Given** a knowledge block is collapsed, **When** I press Enter to expand it, **Then** I see two child nodes showing "○ Original: ..." and "✓ Reworded: ..." with checkmarks indicating the selected version
3. **Given** I navigate through the tree and select a knowledge block, **When** I press Space to toggle, **Then** the tree node's main label updates to show the newly selected version, and the child node indicators swap (○ ↔ ✓)
4. **Given** I have multiple knowledge blocks with reworded versions, **When** I press 'a' (Accept all rewording), **Then** all tree nodes update their main labels to show reworded content with ✓ indicators

**Keyboard Controls (Phase 2)**:

- `j` / `k` / `↓` / `↑`: Navigate between knowledge blocks (skips parent nodes and version children)
- `Enter`: Expand/collapse knowledge block to see both original and reworded versions
- `Space`: Toggle between original and reworded version
- `a`: Accept all reworded versions
- `o`: Use original for all blocks
- `n`: Continue to Phase 3 (Next →)
- `q`: Back to Phase 1

---

### User Story 3 - See LLM Integration Decisions with Streaming Feedback (Priority: P1)

As a Logsqueak user, I want to see in real-time how the system decides where to integrate each piece of knowledge, so that I can understand the system's reasoning and accept or reject each integration proposal.

**Why this priority**: This provides transparency for the integration phase. Users need to see where their knowledge is going and why before committing changes to their graph. This is critical for trust and quality control.

**Independent Test**: Can be fully tested by running extraction through Phase 3, observing the streaming display of integration decisions, and verifying that all proposed integrations are shown with their target pages, actions, and reasoning.

**Acceptance Scenarios**:

1. **Given** Phase 3 starts with 3 knowledge blocks to integrate, **When** the system begins processing, **Then** I see a status indicating the index is being updated (if needed) before RAG search begins
2. **Given** RAG search completes for a knowledge block, **When** the LLM evaluates candidate pages, **Then** I see decisions stream in one at a time, showing the action (e.g., "Add new block", "Skip - already covered"), confidence scores, and reasoning
3. **Given** the decisions are shown one at a time, **When** I view each decision, **Then** I see the hierarchical journal context in tree format (e.g., "Journal 2025-01-15 +-Parent block +-Current block"), the selected content version (original or reworded), target page preview, and navigation controls to move between decisions
4. **Given** the LLM suggests an action with low confidence, **When** the decision appears in the UI, **Then** I see the confidence percentage displayed and can review the reasoning

**Keyboard Controls (Phase 3)**:

- `j` / `k` / `↓` / `↑` / `n` / `p`: Navigate between decisions (next/previous)
- `Space`: Toggle accept/skip for current decision
- `a`: Accept all remaining decisions
- `s`: Skip all remaining decisions
- `Enter`: Continue to Phase 4 (write operations)
- `q`: Back to Phase 2

---

### User Story 4 - Override Integration Decisions and Edit Content (Priority: P2)

As a Logsqueak user, I want to change where knowledge gets integrated and edit the content before it's written to my graph, so that I have final control over what goes into my knowledge base and where it's placed.

**Why this priority**: This completes the user control loop. While P1 stories provide visibility, this story enables active intervention. It's P2 because the system can still function with just viewing and accepting LLM decisions, but this adds critical editing capabilities for power users.

**Independent Test**: Can be fully tested by navigating to a proposed integration in Phase 3, changing its action from "Add as new section" to "Add under [specific block]", editing the content, and verifying those changes are reflected in the final write operations.

**Acceptance Scenarios**:

1. **Given** the LLM suggests "Add as new section" for a knowledge block, **When** I press 'T' to change target, **Then** I see options to change the action type ("Skip", "Add as new section", "Add under [block title]", "Replace [block title]")
2. **Given** I've selected "Add under [block title]" action, **When** I confirm, **Then** I can select which specific block to add under from the target page's hierarchy
3. **Given** the system is showing a knowledge block (original or reworded version selected in Phase 2), **When** I press 'E' to edit the text, **Then** I see a text editor where I can modify the content that will be integrated
4. **Given** I've changed an integration decision, **When** the system continues processing other blocks, **Then** my modified decision is "locked" and not overridden

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
- What happens if a user accepts a parent block as "knowledge" but then individually deselects all children? (System respects individual child decisions, only the parent selection applies to the parent block itself)
- How does the system behave when a user presses Enter during Phase 1 while LLM is still analyzing blocks? (System cancels pending LLM tasks and proceeds with current classifications)
- What happens when the LLM response includes malformed JSON during streaming? (System logs error, shows user-friendly message, skips that particular item, continues with remaining items)
- What happens if a target block specified in an integration decision no longer exists in the destination page? (Write operation fails for that specific integration, error is logged with details, other integrations for that page continue)
- How does the system handle very large journal entries (e.g., approaching the 2000-line limit)? (UI should remain responsive with virtualized scrolling; streaming should handle blocks progressively)
- What happens when a user tries to cancel (Ctrl+C) during Phase 4 write operations? (System warns that canceling mid-write may leave partial updates, asks for confirmation before terminating)

## Requirements *(mandatory)*

### Functional Requirements

#### Phase 1: Knowledge Extraction

- **FR-001**: System MUST display journal blocks in a hierarchical tree view showing the parent-child relationships from the original journal entry
- **FR-002**: System MUST visually indicate the selection state of each block using background colors: dark blue for LLM-suggested knowledge (pending), dark green for user-selected knowledge, very dark green for LLM-suggested knowledge that was accepted, no background for unselected blocks
- **FR-003**: System MUST display a knowledge count `[N]` showing the number of knowledge blocks in each subtree (including self)
- **FR-004**: System MUST display confidence scores as percentages for LLM-suggested blocks
- **FR-005**: System MUST highlight blocks with confidence scores below 75% with warning indicators to suggest user review
- **FR-006**: Users MUST be able to navigate between blocks using keyboard controls (arrow keys, j/k)
- **FR-007**: Users MUST be able to explicitly accept blocks as "knowledge" using keyboard controls (Space to toggle)
- **FR-008**: Users MUST be able to accept all LLM knowledge suggestions at once using a single keyboard command ('a' key)
- **FR-009**: System MUST keep blocks in "pending" state until user explicitly accepts them, even when LLM suggestions are available
- **FR-010**: System MUST prevent the LLM from overriding any user-accepted block classification
- **FR-011**: System MUST update the UI progressively as LLM suggestions stream in, showing dark blue backgrounds for suggested knowledge blocks
- **FR-012**: Users MUST be able to proceed to Phase 2 at any time by pressing 'n' key, even if LLM analysis is incomplete
- **FR-013**: System MUST cancel ongoing LLM analysis tasks when the user proceeds to the next phase
- **FR-014**: System MUST validate that at least one knowledge block has been explicitly accepted before allowing continuation to Phase 2
- **FR-015**: System MUST display LLM reasoning for the currently highlighted block in a reason bar at the bottom of the screen
- **FR-016**: System MUST support rich markdown rendering in block content, including links, bold, code, and checkboxes
- **FR-016a**: System MUST display full block content without truncation; multi-line content will automatically expand tree node height vertically, and long single lines will wrap or scroll horizontally depending on terminal width
- **FR-017**: Toggle actions (Space key) MUST only affect the selected block, not cascade to children
- **FR-018**: System MUST allow users to reset accepted blocks back to pending state using the 'r' key
- **FR-019**: System MUST allow users to clear all selections (deselect all blocks) using the 'c' key

#### Phase 2: Content Rewording

- **FR-020**: System MUST display a tree view showing the journal hierarchy with knowledge blocks and their parent blocks, showing only blocks relevant to the knowledge extraction (not the entire journal)
- **FR-021**: System MUST display knowledge blocks as expandable tree nodes that are collapsed by default, showing the currently selected version (original or reworded) in the main label
- **FR-022**: System MUST show version indicators in knowledge block labels (○ for original, ✓ for reworded)
- **FR-022a**: System MUST make only knowledge blocks selectable in the tree (parent nodes and version children are shown for context but cannot be selected)
- **FR-022b**: System MUST skip over parent nodes and version children during navigation with all navigation keys (j/k/↓/↑), ensuring users can only navigate between knowledge blocks
- **FR-022c**: System MUST set navigation bindings with high priority to override default Tree widget behavior
- **FR-023**: System MUST allow users to expand knowledge blocks using Enter key to reveal two child nodes showing both original and reworded versions
- **FR-024**: System MUST display expanded children with labels "○ Original: ..." and "✓ Reworded: ..." with checkmarks indicating the currently selected version
- **FR-025**: System MUST stream LLM-reworded content for each knowledge block, displaying "(Refining...)" status during streaming
- **FR-026**: System MUST allow users to toggle between original and reworded versions for each knowledge block using Space key
- **FR-027**: System MUST immediately update the knowledge block's main label to show the newly selected version when toggling
- **FR-028**: System MUST update the child node indicators when toggling, swapping which child shows ✓ and which shows ○
- **FR-029**: System MUST provide a way to accept all reworded versions at once using 'a' key (updates all main labels to show reworded content with ✓)
- **FR-030**: System MUST provide a way to use original content for all blocks at once using 'o' key (updates all main labels to show original content with ○)
- **FR-031**: System MUST support full Logseq markdown rendering in all tree labels (main nodes and children), including bold (`**text**`), page links (`[[Page]]`), code (`` `code` ``), TODO/DONE markers, and external links
- **FR-031a**: System MUST display full block content in tree labels without truncation; multi-line content will automatically expand tree node height vertically, and long single lines will wrap or scroll horizontally depending on terminal width
- **FR-032**: System MUST preserve user's content selection choices when proceeding to Phase 3
- **FR-033**: System MUST display a help panel at the bottom showing current selection status and available keyboard shortcuts
- **FR-034**: System MUST allow users to proceed to Phase 3 at any time by pressing 'n' key, even if rewording is incomplete
- **FR-035**: System MUST allow users to return to Phase 1 using 'q' key
- **FR-036**: System MUST update the help panel automatically when navigating between tree nodes to show the current block's selection status

#### Phase 3: Integration Decisions

- **FR-037**: System MUST display a status indicator when updating the page index (if needed) before RAG search begins
- **FR-038**: System MUST perform RAG search (semantic + hinted) for each knowledge block to find candidate pages
- **FR-039**: System MUST stream LLM integration decisions for each (knowledge block, candidate page) pair as they arrive
- **FR-040**: System MUST display decisions one at a time in single-decision-per-screen format with clear status indicators (⊙ PENDING, ✓ ACCEPTED, ✗ SKIPPED)
- **FR-041**: System MUST translate technical action types to user-friendly labels (e.g., "APPEND_ROOT" → "Add as new section", "APPEND_CHILD" → "Add under 'block title'", "UPDATE" → "Replace 'block title'")
- **FR-042**: System MUST show confidence scores as percentages for each integration decision
- **FR-043**: System MUST display hierarchical journal context for each knowledge block with proper indentation showing parent-child relationships, with all markdown rendered and displayed in dimmed style
- **FR-044**: System MUST display the content version selected in Phase 2 (original or reworded) as the content that will be integrated, with a label indicating which version
- **FR-045**: System MUST display target page name and integration action details in a dedicated section
- **FR-046**: System MUST provide navigation controls to move between decisions (j/k/↓/↑/n/p for next/previous)
- **FR-047**: System MUST allow toggling decision state with Space key (pending → accepted → skipped → pending)
- **FR-048**: System MUST provide batch actions: 'a' to accept all remaining decisions, 's' to skip all remaining decisions
- **FR-049**: System MUST display LLM reasoning for each decision
- **FR-050**: System MUST support full Logseq markdown rendering in all displayed content (knowledge blocks, parent context)
- **FR-050a**: System MUST display full block content without truncation; multi-line content will automatically expand node display height vertically, and long single lines will wrap or scroll horizontally depending on terminal width
- **FR-051**: System MUST allow users to return to Phase 2 using 'q' key
- **FR-052**: Users MAY be able to cycle through available actions (Skip, Add as new section, Add under, Replace) using keyboard controls (optional feature)
- **FR-053**: Users MAY be able to edit the content that will be integrated using an inline text editor (optional feature)
- **FR-054**: Users MAY be able to select different target block locations for "Add under" and "Replace" actions (optional feature)
- **FR-055**: System MAY lock user-modified decisions against LLM overrides (optional feature)

#### Phase 4: Write Operations

- **FR-056**: System MUST display per-page write progress showing pending, in-progress, and completed status
- **FR-057**: System MUST show the number of blocks being written to each page
- **FR-058**: System MUST continue processing remaining pages if a write operation fails for one page
- **FR-059**: System MUST log detailed error information for any failed write operations
- **FR-060**: System MUST atomically update journal entries with provenance markers only after successful page writes
- **FR-061**: System MUST display a completion summary showing total blocks added, pages updated, and link to journal entry
- **FR-062**: System MUST support full Logseq markdown rendering in the completion summary, showing integrated blocks with their original formatting

#### Completion & Error Handling

- **FR-063**: System MUST provide a completion summary distinguishing between fully successful extractions and those with errors
- **FR-064**: Users MUST be able to view detailed changes (which blocks were added to which pages)
- **FR-065**: Users MUST be able to view detailed error information with suggested remediation actions
- **FR-066**: System MUST provide actionable error messages (e.g., "Rebuild your index: logsqueak index rebuild")

#### General UI Requirements

- **FR-067**: System MUST display consistent keyboard shortcuts in a footer area on all screens
- **FR-068**: System MUST support both standard (arrow keys, Enter) and vim-style (j/k) navigation
- **FR-069**: System MUST allow users to quit at any phase using Ctrl+C or 'q' key
- **FR-070**: System MUST maintain responsive UI performance while streaming LLM responses (UI updates should not block user input)
- **FR-071**: System MUST notify users when malformed JSON is encountered during LLM streaming (via visual indicator or status message) while continuing to process remaining items
- **FR-072**: System MUST use consistent key bindings across all phases for similar actions (e.g., 'a' for accept all)
- **FR-073**: System MUST display user-friendly phase labels instead of technical "Phase X" labels (e.g., "Pick blocks to integrate into pages" for Phase 1, "Review reworded content" for Phase 2, "Accept or reject integrations" for Phase 3)

#### Logging & Debugging

- **FR-074**: System MUST log complete raw LLM prompts and responses to `~/.cache/logsqueak/prompts/*.log` for all phases (extraction, rewording, integration decisions)
- **FR-075**: System MUST capture partial LLM responses when user cancels streaming for debugging purposes
- **FR-076**: System MUST log detailed structured information about rewording and decisions to `~/.cache/logsqueak/logs/tui_*.log` including block IDs, confidence scores, reasoning, and before/after text
- **FR-077**: System MUST use try/finally patterns to ensure logging occurs even on errors or cancellation

### Key Entities

- **BlockState**: Represents the selection state of a journal block throughout the pipeline (block_id: str, classification: Literal["pending", "knowledge"], confidence: float | None, source: Literal["user", "llm"], llm_classification: Literal["knowledge"] | None, llm_confidence: float | None, reason: str | None) - Preserves original LLM suggestion even after user acceptance for potential reset functionality. The `reason` field stores LLM reasoning for why a block was suggested as knowledge. Blocks are either pending (not yet decided), or knowledge (selected for integration).

- **ContentVersion**: Represents the content selection for a knowledge block in Phase 2 (block_id: str, original_content: str, reworded_content: str | None, selected_version: Literal["original", "reworded"], rewording_complete: bool) - Tracks which version of content the user has chosen to integrate. The `reworded_content` field is populated by streaming LLM response, and `selected_version` defaults to "original" until user changes it.

- **IntegrationDecision**: Represents a decision about integrating a specific knowledge block into a target page (knowledge block ID, target page, action type, target block ID if applicable, confidence score, source of decision, reasoning) - The actual content to integrate comes from the ContentVersion entity.

- **CandidatePage**: Represents a page retrieved during RAG search (page name, semantic match score, included/excluded status, blocks within the page for targeting)

- **ProgressState**: Represents the current phase and completion status (phase number, completion percentage, streaming status indicators)

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can navigate through all four phases of knowledge extraction without requiring any external documentation for basic operations
- **SC-002**: Users see visual feedback within 500ms of any interaction (keyboard input, phase transition) - measured as absolute wall-clock time including any background network/LLM operations
- **SC-003**: Users can override any LLM decision at any phase before final write operations execute
- **SC-004**: System completes the extraction workflow for a typical journal entry (11 avg root blocks, 22 avg total blocks as validated in Assumption #3, with ~5 knowledge blocks, 3 candidate pages per block) in under 3 minutes
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
