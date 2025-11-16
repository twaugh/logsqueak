# Feature Specification: Logsqueak - Knowledge Extraction from Journals

**Feature Branch**: `001-logsqueak`
**Created**: 2025-11-04
**Status**: Specification
**Description**: Interactive TUI for extracting lasting knowledge from Logseq journal entries and integrating it into relevant pages using LLM-powered analysis with RAG-based semantic search.

## Clarifications

### Combined from Previous Sessions

- Q: Do LLM providers support streaming JSON-structured outputs? → A: **VALIDATED with NDJSON approach** - By prompting for NDJSON format (newline-delimited JSON), each knowledge block/decision arrives as a complete JSON object on its own line, enabling true incremental parsing and real-time UI updates.

- Q: Are journal entry size assumptions realistic? → A: **VALIDATED against actual Logseq graph** - 11.4 avg root blocks (range: 3-26), 22.2 avg total blocks (max: 71), 1.2 avg nesting depth (max: 2 levels), 27.2 avg lines (max: 77). All within assumed ranges.

- Q: Should we maintain backward compatibility with batch mode? → A: **No** - TUI is the only mode. No batch processing, no CLI-only workflow.

- Q: Where should configuration be stored? → A: `~/.config/logsqueak/config.yaml` (XDG Base Directory specification)

- Q: Does the LLM classify blocks as "knowledge" vs "activity"? → A: **No** - The LLM only identifies blocks that contain lasting knowledge. Blocks are either identified as knowledge or not identified at all.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Select Knowledge Blocks for Integration

A Logseq user has been journaling daily and wants to identify which blocks from their journal entries contain lasting knowledge worth integrating into their knowledge base. They want to review LLM suggestions while maintaining full control over what gets extracted.

**Acceptance Scenarios**:

1. **Given** a journal entry with 15 blocks (mix of tasks, activities, and knowledge), **When** user opens the extraction screen, **Then** system displays all blocks in a tree view (fully expanded) showing the original journal hierarchy
2. **Given** the tree view is displayed, **When** user navigates with arrow keys, **Then** the selected block's full content is shown at the bottom in rendered markdown with text wrapping
3. **Given** the LLM is analyzing blocks in the background, **When** the LLM identifies a block as knowledge, **Then** that block is highlighted with a distinct color (different from user-selected highlight) and the bottom panel shows the LLM's reasoning
4. **Given** the LLM has suggested a block as knowledge, **When** user views the tree, **Then** that block shows a robot emoji (🤖) indicator (positioned far left or far right without shifting the block text) and LLM-suggested highlight color
5. **Given** several blocks have been suggested by the LLM, **When** user presses Space on a block, **Then** the block is marked as user-selected (different highlight color) for extraction, and the robot emoji remains visible
6. **Given** background tasks are running, **When** the user views the screen, **Then** a status widget shows which tasks are active ("Analyzing knowledge blocks...", "Building page index: 45%")
7. **Given** the user has selected at least one block, **When** they press 'n' (next), **Then** the system proceeds to the content editing screen

**Keyboard Controls**:

- `j` / `k` / `↓` / `↑`: Navigate blocks
- `Space`: Toggle knowledge selection for current block
- `a`: Accept all LLM knowledge suggestions
- `r`: Reset current block to LLM suggestion (clear user selection)
- `c`: Clear all selections
- `n`: Continue to next screen
- `q`: Quit application

---

### User Story 2 - Edit and Refine Knowledge Content

After selecting knowledge blocks, the user wants to review and refine the content before integration. They want to see LLM-suggested rewordings that remove journal-specific context, compare them with the original, and choose which version to use or edit it themselves.

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

### User Story 3 - Review Integration Decisions and Write to Pages

The user wants to see where each refined knowledge block will be integrated in the context of the target page, understand the LLM's reasoning, and approve each integration. When approved, the system writes immediately and shows the result.

**Acceptance Scenarios**:

1. **Given** 5 knowledge blocks need integration, **When** the LLM evaluates candidate pages from RAG search, **Then** decisions stream in one at a time, showing the action (e.g., "Add as new section", "Add under 'Project Timeline'"), confidence score, and reasoning
2. **Given** a decision is displayed, **When** user reviews it, **Then** they see:
   - Hierarchical journal context (where the knowledge came from, shown dimmed)
   - The refined content from Phase 2 (what will be integrated)
   - Target page preview showing existing page structure
   - New content shown with green bar (`┃`) in its target location
   - If replacing: old content shown with strikethrough, new content below it with green bar
3. **Given** a decision shows "Add under 'Project Timeline'" with the target page preview visible, **When** user presses 'y' (accept), **Then** system immediately writes that knowledge block to the target page at the specified location and atomically adds to the `extracted-to::` property on the journal block (creating it if it doesn't exist, or appending to the comma-separated list if it does), where each link uses the target page name as link text (with '___' converted to '/' for hierarchical pages) and block reference `((uuid))` as the link target (e.g., `extracted-to:: [Plans/Project X](((abcde-1234-...)))` or `extracted-to:: [Page A](((uuid1))), [Page B](((uuid2)))` for multiple integrations)
4. **Given** a write operation succeeds, **When** the system updates the display, **Then** that decision is marked as completed with a checkmark and the next decision is shown
5. **Given** a write operation fails, **When** the error occurs, **Then** system shows error details (e.g., "Target block not found"), marks that decision as failed, and allows user to continue with remaining decisions
6. **Given** the status display shows "Decision 3 of 8 (✓ 2 completed, ⊙ 5 pending, ✗ 0 skipped)", **When** user views the screen, **Then** they can see progress at a glance and available actions are shown in the footer ("Press y to accept, n to skip, j/k to navigate")
7. **Given** user navigates to a decision with status ⊙ PENDING, **When** they press 'n', **Then** that decision is marked as skipped (✗), auto-advances to next decision, and status count updates
8. **Given** user navigates to a decision with status ✓ COMPLETED, **When** they view it, **Then** the 'y' action is not shown in the footer (cannot re-write), only navigation keys are available
9. **Given** all decisions are reviewed and processed, **When** the last decision completes, **Then** system shows completion summary with total blocks integrated, failed count, and link to journal entry with provenance markers

**Keyboard Controls**:

- `y`: Accept current decision and write immediately, auto-advance to next (hidden for already-written decisions)
- `n`: Skip current decision (no write), auto-advance to next
- `j` / `k` / `↓` / `↑`: Navigate between decisions without taking action
- `a`: Accept and write all remaining pending decisions sequentially
- `s`: Skip all remaining pending decisions
- `Enter`: View completion summary when all decisions processed
- `q`: Back to content editing screen

---

### Edge Cases

- What happens when a journal entry has no blocks identified as knowledge by the LLM? (System shows "No knowledge blocks identified" message, user can manually select blocks anyway)
- How does the system handle streaming interruptions (network issues, LLM API errors)? (Shows error message, preserves partial state, allows user to retry or cancel)
- What happens if page indexing fails? (Shows error, offers to retry, blocks progression to Phase 3)
- What happens if RAG search returns no candidate pages? (Phase 3 shows "No relevant pages found" message, can skip or wait for more decisions to stream in)
- What if a target block specified in an integration decision no longer exists when user presses 'y'? (Write operation fails for that decision, error shown with details, decision marked as failed (⚠), user can continue with remaining decisions)
- How does the system handle very large journal entries (approaching 2000-line limit)? (UI remains responsive with virtualized scrolling; warns user if limit approached)
- What happens when user tries to cancel (Ctrl+C) during a write operation in Phase 3? (Shows warning about potential partial journal state with some blocks already marked as processed, asks for confirmation)
- What if the LLM response includes malformed JSON during streaming? (Logs error, shows user-friendly message, skips that item, continues with remaining items)
- What happens if user accepts a decision ('y') but the write takes a long time? (UI shows "Writing..." status, blocks navigation until write completes or fails, then auto-advances to next decision)
- What if all decisions are skipped or fail? (Completion summary shows "0 blocks integrated, X skipped, Y failed" with link to journal entry - no extracted-to:: markers added since nothing was written)

## Requirements *(mandatory)*

### Functional Requirements

#### Phase 1: Block Selection

- **FR-001**: System MUST display journal blocks in a hierarchical tree view fully expanded, showing all parent-child relationships from the original journal entry, with rich markdown rendering of the first line only of each block
- **FR-002**: System MUST display the selected block's full content at the bottom of the screen in rendered markdown with text wrapping, showing all lines in the block content without indentation (block-centric view, not hierarchical)
- **FR-003**: System MUST display LLM reasoning for why a block was identified as knowledge in the bottom panel when that block is selected
- **FR-004**: System MUST run two background tasks in parallel when this screen loads: (1) LLM streaming classification of knowledge blocks, (2) ChromaDB page index building (with incremental caching)
- **FR-004a**: System MUST index pages as block-level chunks using full-context text (block content with all parent context) via `generate_chunks()` from `logseq-outline-parser`, storing each chunk with its hybrid ID for precise retrieval
- **FR-004b**: System MUST implement incremental indexing using a manifest file (`~/.cache/logsqueak/manifest.json`) that tracks each page's modification time (mtime), detecting three states: (1) **Deletions** - pages in manifest but not on disk (remove from index), (2) **Updates** - pages with changed mtime since last index (re-index), (3) **Additions** - pages not in manifest (index for first time)
- **FR-004c**: System MUST store the manifest as JSON with format `{"page_name": mtime_timestamp}` and save atomically after each indexing operation
- **FR-005**: System MUST display a status widget showing which background tasks are active and their progress (percentage when calculable, on/off status otherwise)
- **FR-006**: System MUST visually highlight blocks identified as knowledge by the LLM with a distinct color (different from user-selected highlight) as they arrive via streaming
- **FR-006a**: System MUST display a robot emoji (🤖) indicator next to LLM-suggested blocks, positioned in a fixed-width reserved space (minimum 2 characters) at the far left of each tree line, ensuring all block text aligns consistently regardless of emoji presence
- **FR-006b**: System MUST keep the robot emoji indicator visible even after user manually selects the block, showing that it was originally an LLM suggestion
- **FR-007**: System MUST allow users to manually select/deselect any block as knowledge using keyboard controls, independent of LLM suggestions
- **FR-008**: System MUST use a different highlight color for user-selected blocks vs LLM-suggested blocks
- **FR-009**: Users MUST be able to navigate between blocks using keyboard controls (arrow keys, j/k)
- **FR-010**: Users MUST be able to accept all LLM suggestions at once using 'a' key
- **FR-011**: System MUST allow users to reset a user-selected block back to LLM suggestion state using 'r' key
- **FR-012**: System MUST allow users to clear all selections using 'c' key
- **FR-013**: System MUST allow progression to Phase 2 only when at least one block is selected
- **FR-014**: System MUST support rich markdown rendering in both the tree view (first line only) and bottom panel (all lines), including links, bold, code, checkboxes, and text wrapping

#### Phase 2: Content Editing

- **FR-015**: System MUST display a list of all selected knowledge blocks from Phase 1
- **FR-016**: System MUST display each block's original full hierarchical context (including parent blocks with proper indentation) in a read-only display area, showing the block in its journal context
- **FR-017**: System MUST display each block's content in an editable text field, starting with the original block content (without parent context)
- **FR-018**: System MUST run three background tasks using non-blocking workers when this screen loads: (1) LLM streaming reworded versions of each block, (2) continue page indexing if not yet complete from Phase 1, (3) RAG search after indexing completes
- **FR-019**: System MUST display a waiting indicator (e.g., "⏳ Waiting for LLM response...") in the LLM reworded section before the reworded version is available, then display the LLM-reworded content alongside the original hierarchical context when a reworded version becomes available, allowing user to compare before choosing
- **FR-020**: System MUST provide keyboard shortcuts to accept LLM version ('a' key) and revert to original ('r' key) for each knowledge block
- **FR-021**: System MUST visually indicate when "Accept LLM version" action is available (normal display) versus unavailable (dimmed with "waiting..." indicator) based on whether the LLM-reworded version has arrived
- **FR-022**: System MUST allow users to directly edit the text field content for any block using a multi-line text editor with cursor movement, selection, and standard editing features
- **FR-023**: System MUST update the text field with LLM-reworded content when user presses 'a' (accept LLM version)
- **FR-024**: System MUST restore the original block content (without parent context) when user presses 'r' (revert to original)
- **FR-024a**: System MUST preserve user edits to the text field when navigating between blocks (edits auto-saved on navigation)
- **FR-024b**: System MUST visually indicate the current state of the editable content: "(LLM version)" when matching the LLM reworded text, "(modified)" when manually edited, or no label when unchanged from original
- **FR-024c**: System MUST require explicit focus on the text editor (via Tab key) before accepting text input, and MUST visually indicate when the editor has focus (e.g., border highlight, cursor visible)
- **FR-024d**: System MUST disable navigation keys (j/k/↓/↑) and action keys (a/r) when the text editor has focus, allowing Tab or Escape to unfocus and re-enable those controls
- **FR-025**: System MUST display status widget showing progress for all active background tasks: "Refining knowledge content: X/Y complete", "Building page index: Z%", "Finding relevant pages: X/Y complete"
- **FR-026**: System MUST block progression to Phase 3 until page indexing is complete
- **FR-027**: System MUST start RAG search for selected blocks (using original hierarchical context) when page indexing completes
- **FR-028**: System MUST block progression to Phase 3 until RAG search is complete
- **FR-029**: System MUST display status messages when blocking progression ("Waiting for page index...", "Finding relevant pages...")
- **FR-030**: Users MUST be able to navigate between knowledge blocks using keyboard controls

#### Phase 3: Integration Decisions

- **FR-031**: System MUST perform RAG search (semantic + hinted) using the original hierarchical context of each knowledge block to find candidate pages (completed in Phase 2)
- **FR-032**: System MUST stream LLM integration decisions for each (knowledge block, candidate page) pair as they arrive
- **FR-033**: System MUST display decisions one at a time in single-decision-per-screen format with status indicators (⊙ PENDING, ✓ COMPLETED, ✗ SKIPPED, ⚠ FAILED)
- **FR-034**: System MUST show confidence scores as percentages for each integration decision
- **FR-035**: System MUST display hierarchical journal context with proper indentation, all markdown rendered and displayed in dimmed style
- **FR-036**: System MUST display the refined content from Phase 2 (not the original journal content) as the content that will be integrated
- **FR-037**: System MUST display target page preview showing the existing page structure with the new content visually integrated
- **FR-038**: System MUST show new content with green bar (`┃`) visual indicator in its target location within the page preview
- **FR-039**: System MUST show replaced content with strikethrough styling, with new content shown below it with green bar
- **FR-040**: System MUST provide navigation controls to move between decisions without taking action (j/k/↓/↑)
- **FR-041**: System MUST write the knowledge block to the target page immediately when user presses 'y' (accept) on a pending decision
- **FR-042**: System MUST automatically advance to the next decision after accepting (y) or skipping (n) the current decision
- **FR-043**: System MUST atomically add to the `extracted-to::` property on the journal block only after successful page write, appending to the existing comma-separated list if the property already exists, where each link uses the target page name as link text (converting '___' to '/' for hierarchical pages) and block reference `((uuid))` as the target (e.g., `extracted-to:: [Plans/Project X](((abcde-1234...))), [Daily Notes](((fghij-5678...)))`)
- **FR-044**: System MUST mark completed integrations with checkmark (✓) and update the status count display
- **FR-045**: System MUST display error details if a write operation fails (e.g., "Target block not found"), mark decision as failed (⚠), and allow continuation with remaining decisions
- **FR-046**: System MUST allow skipping a decision using 'n' key (git-style no/skip) without writing and advance to next decision, marking as skipped (✗)
- **FR-047**: System MUST provide batch action to accept and write all remaining pending decisions using 'a' key (writes happen sequentially with status updates)
- **FR-048**: System MUST provide batch action to skip all remaining pending decisions using 's' key
- **FR-049**: System MUST display status summary showing decision count and breakdown (e.g., "Decision 3 of 8 (✓ 2 completed, ⊙ 5 pending, ✗ 1 skipped)")
- **FR-050**: System MUST hide 'y' action from footer when viewing a completed decision (status ✓ COMPLETED) since it cannot be re-written
- **FR-051**: System MUST display context-sensitive footer showing available actions based on current decision state
- **FR-052**: System MUST display LLM reasoning for each decision
- **FR-053**: System MUST support full Logseq markdown rendering in all displayed content

#### Completion & Error Handling

- **FR-054**: System MUST display a completion summary showing total blocks integrated, failed count, and link to journal entry with provenance markers
- **FR-055**: Users MUST be able to view detailed error information with suggested remediation actions (e.g., "Rebuild your index: logsqueak index rebuild")
- **FR-056**: System MUST provide actionable error messages for common failure modes

#### General UI Requirements

- **FR-057**: System MUST display consistent keyboard shortcuts in a footer area on all screens
- **FR-058**: System MUST support both standard (arrow keys, Enter) and vim-style (j/k) navigation
- **FR-059**: System MUST allow users to quit at any phase using Ctrl+C or 'q' key
- **FR-060**: System MUST execute all background tasks (LLM streaming, page indexing, RAG search) using non-blocking workers that allow UI interaction to continue uninterrupted, maintaining responsive UI performance where keyboard input is processed within 100ms
- **FR-060a**: Phase 1 screen MUST display immediately upon application start, showing the journal block tree BEFORE sending the LLM classification request (LLM streaming starts in background after UI is visible)
- **FR-060b**: Phase 2 screen MUST display immediately when user presses 'n' to continue from Phase 1, showing the selected knowledge blocks BEFORE sending the LLM rewording request (LLM streaming starts in background after UI is visible)
- **FR-060c**: Phase 3 screen MUST display immediately when user presses 'n' to continue from Phase 2, showing waiting message BEFORE sending the LLM decision request (LLM streaming starts in background after UI is visible)
- **FR-061**: System MUST notify users when malformed JSON is encountered during LLM streaming while continuing to process remaining items
- **FR-062**: System MUST use consistent key bindings across all phases for similar actions (e.g., 'j/k' for navigation, 'q' for back/quit, 'a' for accept)

#### Logging & Debugging

- **FR-063**: System MUST log complete raw LLM prompts to `~/.cache/logsqueak/prompts/*.log` at the start of each LLM request for all phases
- **FR-064**: System MUST log raw LLM response chunks incrementally as they arrive (streaming mode), appending to the log file in real-time so partial responses are captured even if streaming is interrupted or cancelled
- **FR-065**: System MUST log detailed structured information to `~/.cache/logsqueak/logs/tui_*.log` including block IDs, confidence scores, reasoning, and before/after text
- **FR-066**: System MUST use try/finally patterns to ensure logging occurs even on errors or cancellation

#### Data Model Requirements

- **FR-067**: System MUST use the `logseq-outline-parser` package for all Logseq markdown parsing, rendering, and block manipulation
- **FR-068**: System MUST preserve exact property order when reading and writing Logseq files (insertion order is sacred)
- **FR-069**: System MUST use hybrid ID system (explicit `id::` properties OR content hashes) for block identification, using `generate_chunks()` from `logseq-outline-parser` which implements `generate_content_hash()` for content-based IDs
- **FR-070**: System MUST generate unique UUIDs for all integrated blocks (via `id::` property)
- **FR-071**: System MUST maintain provenance links from journal to integrated blocks via `extracted-to::` property containing comma-separated markdown links with page names as link text (converting file name '___' to '/' for hierarchical pages) and block references `((uuid))` as targets (backward links from pages to journal are automatically provided by Logseq's backlinks feature)

### Key Entities

- **BlockState**: Represents the selection state of a journal block (block_id: str, classification: Literal["pending", "knowledge"], confidence: float | None, source: Literal["user", "llm"], llm_classification: Literal["knowledge"] | None, llm_confidence: float | None, reason: str | None)

- **EditedContent**: Represents the edited content for a knowledge block in Phase 2 (block_id: str, original_content: str, reworded_content: str | None, current_content: str, rewording_complete: bool)

- **IntegrationDecision**: Represents a decision about integrating a knowledge block (knowledge_block_id: str, target_page: str, action: Literal["skip", "add_section", "add_under", "replace"], target_block_id: str | None, target_block_title: str | None, confidence: float, refined_text: str, source: Literal["user", "llm"], skip_reason: str | None, write_status: Literal["pending", "completed", "failed", "skipped"])

- **BackgroundTask**: Represents a long-running background task (task_type: Literal["llm_classification", "page_indexing", "rag_search", "llm_rewording", "llm_decisions"], status: Literal["running", "completed", "failed"], progress_percentage: float | None)

## NDJSON Streaming Protocol

The system uses NDJSON (Newline-Delimited JSON) for real-time streaming of LLM responses. This enables incremental UI updates as results arrive, improving perceived responsiveness and allowing graceful handling of network interruptions.

### Protocol Overview

**Format**: Each complete JSON object is written on a single line, terminated by `\n`. Multiple objects are separated by newlines.

**Example stream**:

```
{"block_id": "abc123", "confidence": 0.85, "reason": "documents key architectural decision"}
{"block_id": "def456", "confidence": 0.92, "reason": "captures important insight about system behavior"}
```

**Key Properties**:

- Each line is a complete, independently parseable JSON object
- Objects arrive in arbitrary order (not necessarily input order)
- Stream may end early on network errors (partial results are preserved)
- Empty lines are skipped silently
- Malformed lines are logged and skipped (doesn't break the stream)

### Parser Implementation

The NDJSON parser must implement line-buffering with incremental parsing:

```python
async def parse_ndjson_stream(stream: AsyncIterator[str]) -> AsyncIterator[dict]:
    """
    Parse NDJSON from chunked async stream.

    Args:
        stream: Async iterator yielding string chunks (token-by-token)

    Yields:
        Complete JSON objects as they become parseable

    Error Handling:
        - Invalid JSON on a line → Log warning, skip line, continue
        - Incomplete line at stream end → Log warning, discard
        - Empty lines → Skip silently
    """
    buffer = ""
    line_number = 0

    async for chunk in stream:
        buffer += chunk

        # Process all complete lines in buffer
        while '\n' in buffer:
            line_end = buffer.index('\n')
            complete_line = buffer[:line_end].strip()
            buffer = buffer[line_end + 1:]
            line_number += 1

            if not complete_line:
                continue

            try:
                obj = json.loads(complete_line)
                if not isinstance(obj, dict):
                    logger.warning(f"Line {line_number}: Expected dict, got {type(obj)}")
                    continue
                yield obj
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_number}: Malformed JSON at position {e.pos}: {e.msg}")
                continue

    # Handle incomplete final line
    if buffer.strip():
        line_number += 1
        try:
            obj = json.loads(buffer.strip())
            if isinstance(obj, dict):
                yield obj
        except json.JSONDecodeError as e:
            logger.warning(f"Line {line_number} (incomplete): {e.msg}")
```

**Performance Characteristics**:

- Memory usage: O(longest line), typically <1KB per line
- Latency: Yields immediately upon line completion (no batching)

### Phase 1: Knowledge Extraction Stream

**LLM Client Method**:

```python
async def stream_extract_ndjson(self, journal_content: str) -> AsyncIterator[dict]
```

**Input**: Full journal markdown with `id::` properties injected for all blocks

**Output Schema** (per line):
```json
{
  "block_id": "string (id:: property value or content hash)",
  "confidence": "float (0.0-1.0)",
  "reason": "string (7-10 words explaining why this is knowledge)"
}
```

**Example**:

```json
{"block_id": "abc123", "confidence": 0.85, "reason": "documents key architectural decision"}
{"block_id": "def456", "confidence": 0.92, "reason": "captures important insight about system behavior"}
```

**Required Fields**: `block_id`, `confidence`, `reason`

**Behavior**:

- LLM outputs ONLY blocks classified as knowledge (no activity logs)
- Objects may arrive in any order
- `confidence < 0.5` indicates borderline cases
- `reason` should be 7-10 words explaining the classification

**Error Handling**:

- Unknown `block_id` → Log warning, skip object
- Missing required field → Log warning, skip object
- Invalid `confidence` range → Log warning, clamp to [0.0, 1.0]

### Phase 2: Content Rewording Stream

**LLM Client Method**:

```python
async def stream_rewording_ndjson(self, decisions: List[dict]) -> AsyncIterator[dict]
```

**Input**:

- `decisions`: List of dicts with `block_id`, `full_text`, `hierarchical_text`

**Output Schema** (per line):

```json
{
  "block_id": "string (source block ID from Phase 1)",
  "refined_text": "string (reworded content as plain block text)"
}
```

**Example**:

```json
{
  "block_id": "abc123",
  "refined_text": "Bounded contexts matter more than service size.\nEach context should have clear ownership.\nAvoid sharing databases across contexts."
}
```

**Required Fields**: `block_id`, `refined_text`

**Format Requirements**:

- `refined_text` MUST be plain block content (no bullet markers like `- `, no indentation)
- Multi-line content uses `\n` for continuation lines
- Preserve `[[Page Links]]` and `((block-refs))` from original
- Remove temporal references ("Today", dates, "I learned")
- Remove first-person narrative
- Make content standalone (no pronouns like "it", "this" without context)

**Behavior**:

- Each unique knowledge block is reworded once
- The same refined text is applied to all decisions referencing that `block_id`
- Results may arrive in any order

**Error Handling**:

- Missing required field → Log warning, skip object
- Unmatched `block_id` → Log warning, skip object
- Empty `refined_text` → Log warning, skip object

### Phase 3: Integration Decisions Stream

**LLM Client Method**:

```python
async def stream_decisions_ndjson(
    self,
    knowledge_block: dict,
    candidate_pages: List[dict]
) -> AsyncIterator[dict]
```

**Input**:

- `knowledge_block`: Dict with `block_id`, `content`, `hierarchical_text`
- `candidate_pages`: List of dicts with `page_name`, `similarity_score`, `chunks` (list of blocks with `target_id` and `hierarchical_text`)

**Output Schema** (per line):

```json
{
  "page": "string (target page name)",
  "action": "string (skip | add_section | add_under | replace)",
  "target_id": "string | null (hybrid ID of target block)",
  "target_title": "string | null (human-readable block name)",
  "confidence": "float (0.0-1.0)",
  "reasoning": "string (LLM explanation for decision)"
}
```

**Example (add_section)**:

```json
{
  "page": "Software Architecture",
  "action": "add_section",
  "target_id": null,
  "target_title": null,
  "confidence": 0.88,
  "reasoning": "Core architectural principle deserving its own section"
}
```

**Example (add_under)**:

```json
{
  "page": "Microservices",
  "action": "add_under",
  "target_id": "block-uuid-123",
  "target_title": "Benefits and Tradeoffs",
  "confidence": 0.91,
  "reasoning": "Fits naturally under existing Benefits discussion"
}
```

**Example (skip)**:

```json
{
  "page": "Docker",
  "action": "skip",
  "target_id": null,
  "target_title": null,
  "confidence": 0.95,
  "reasoning": "Already covered in this page"
}
```

**Required Fields**: `page`, `action`, `confidence`, `reasoning`

**Conditional Fields**:

- If `action` is `"skip"` or `"add_section"`: `target_id` and `target_title` MUST be `null`
- If `action` is `"add_under"` or `"replace"`: `target_id` and `target_title` MUST be non-null strings

**Behavior**:

- One decision per candidate page (order may not match input)
- LLM may decide "skip" for all pages (valid outcome)
- Results stream in as soon as each decision is made

**Error Handling**:

- Invalid `action` value → Log warning, default to "skip"
- Missing `target_id` when required → Log warning, change action to "add_section"
- Invalid `confidence` range → Log warning, clamp to [0.0, 1.0]

### Network Error Handling

**Retry Strategy** (automatic, transparent to user):

```python
async def stream_with_retry(endpoint, payload, max_retries=3) -> AsyncIterator[str]:
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                async with client.stream("POST", endpoint, json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        yield line
                    return  # Success
        except (httpx.TimeoutException, httpx.RequestError) as e:
            if attempt < max_retries - 1:
                backoff = 1.0 * (2 ** attempt)  # 1s, 2s, 4s
                await asyncio.sleep(backoff)
            else:
                raise LLMTimeoutError(f"Failed after {max_retries} attempts: {e}")
        except httpx.HTTPStatusError as e:
            # Don't retry 4xx/5xx errors
            raise LLMAPIError(f"HTTP {e.response.status_code}", status_code=e.response.status_code)
```

**Automatic Retries** (transient errors):

- Network timeout: 3 retries with exponential backoff (1s, 2s, 4s)
- Connection refused: 3 retries
- Transient HTTP errors (502, 503, 504): 3 retries

**No Retry** (permanent errors):

- HTTP 4xx client errors (invalid request, auth failure)
- HTTP 500 server errors (unless 502/503/504)

**Manual Retries** (user-initiated):

- User can press `R` key to retry failed blocks
- TUI sends only failed blocks to LLM (not all blocks)
- Results merged with existing successful classifications

### Incomplete Stream Handling

**Scenario**: Stream ends before all expected objects are received.

**Causes**:

- Network interruption mid-stream
- LLM stops generating (API timeout, rate limit)
- Incomplete final line (no trailing newline)

**Handling**:

- Parser attempts to parse incomplete final line (if valid JSON without newline)
- Otherwise discards and logs warning
- UI compares expected vs received objects
- Shows user summary: "Classified 8/10 blocks. 2 blocks need retry."
- User can continue with partial results or retry failed blocks

**Example**:

```
# Stream received:
{"block_id": "abc123", "confidence": 0.85, "reason": "..."}
{"block_id": "def456", "confidence": 0.92, "reason": "..."}
# Network timeout here - blocks 3-5 never received
```

UI response:

```
Classified 2/5 blocks. 3 pending. Press R to retry or Enter to continue.
```

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can navigate through all phases of knowledge extraction without requiring external documentation for basic operations
- **SC-002**: Users see visual feedback within 500ms of any interaction (keyboard input, phase transition)
- **SC-003**: Users can override any LLM suggestion at any phase before final write operations
- **SC-004**: System completes the extraction workflow for a typical journal entry (11 avg root blocks, 22 avg total blocks, ~5 knowledge blocks, 3 candidate pages per block) in under 3 minutes
- **SC-005**: Users can identify LLM-suggested knowledge blocks vs user-selected blocks through visual indicators
- **SC-006**: When write operations fail for one page, other integrations still complete successfully
- **SC-007**: The UI remains responsive (accepts user input within 100ms) even while processing streaming LLM responses
- **SC-008**: 100% of integrated knowledge blocks include valid provenance links to source journal
- **SC-009**: System handles LLM API errors gracefully in 100% of cases (no crashes, clear error messages)

## Assumptions

1. **Streaming LLM Support**: LLM providers support NDJSON streaming (newline-delimited JSON) for incremental parsing. **Validated with Ollama.**

2. **Terminal Capabilities**: Users have modern terminal emulators supporting 256 colors and Unicode characters. Fallback ASCII characters available for limited environments.

3. **Typical Journal Entry Size**: Most journal entries have 3-15 root blocks with 2-3 levels of nesting, totaling under 100 blocks. **Validated against actual Logseq graph.**

4. **User Familiarity**: Users are comfortable with keyboard-driven interfaces (similar to vim, htop, other TUIs).

5. **Single Extraction Session**: Users process one journal entry at a time. No batch processing multiple dates.

6. **Error Recovery**: Users can resolve errors (like stale index issues) by following suggested actions and re-running extraction. Mid-session retry not required initially.

## Dependencies

- **Textual Framework**: Requires Textual Python library (version 0.47.0+) for TUI interface. **Note**: Will be added to pyproject.toml dependencies during implementation.

- **logseq-outline-parser Package**: MUST use existing `logseq-outline-parser` package (located in `src/logseq-outline-parser/`) for all Logseq markdown parsing, rendering, and block manipulation. Provides:

  - `LogseqOutline.parse()` and `LogseqOutline.render()` for parsing/rendering markdown
  - `LogseqBlock` data structures with property order preservation
  - Hybrid ID system (`id::` properties and content hashing)
  - `find_block_by_id()` for precise block targeting
  - `GraphPaths` for navigating Logseq directory structure

- **Existing Pipeline Infrastructure**: Depends on extraction, RAG, integration, and LLM client modules remaining functionally stable.

- **LLM Streaming APIs**: Requires LLM providers to support streaming responses (OpenAI-compatible streaming endpoints).

- **Async Support**: Requires Python 3.11+ with async/await support for concurrent LLM streaming and UI updates.

- **ChromaDB**: Requires ChromaDB for persistent block-level semantic search with vector embeddings.

- **sentence-transformers**: Requires sentence-transformers for generating embeddings (large dependency, ~500MB).

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

- All operations traceable via `extracted-to::` markers in journal entries
- UPDATE operations replace content but preserve structure/IDs
- APPEND operations add new blocks without modifying existing content
- Every integrated block generates unique `id::` property (UUID)
- Journal entries atomically marked with block references to integrated knowledge
- Logseq's automatic backlinks provide reverse navigation (pages → journals)

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

## Active Technologies

- Python 3.11+
- Textual 0.47.0+ (TUI framework)
- ChromaDB (persistent vector store)
- sentence-transformers (embeddings)
- httpx (LLM client)
- markdown-it-py (parsing)
- pydantic (validation)
- click (CLI)
- File-based I/O (Logseq markdown files)
