# Interactive TUI Design

**Status:** Design
**Created:** 2025-10-29
**Goal:** Replace the current batch processing workflow with an interactive text-based UI that gives users control over the extraction and integration process while providing real-time feedback from streaming LLM responses.

---

## Table of Contents

- [Overview](#overview)
- [User Experience Goals](#user-experience-goals)
- [Core Interaction Model](#core-interaction-model)
- [Phase-by-Phase UI Flow](#phase-by-phase-ui-flow)
  - [Phase 1: Knowledge Extraction](#phase-1-knowledge-extraction)
  - [Phase 2: Candidate Page Review](#phase-2-candidate-page-review)
  - [Phase 3: Integration Decisions](#phase-3-integration-decisions)
  - [Phase 4: Writing Changes](#phase-4-writing-changes)
  - [Completion Summary](#completion-summary)
- [Design Decisions](#design-decisions)
- [Technical Implementation](#technical-implementation)
- [Terminology Mappings](#terminology-mappings)

---

## Overview

The current logsqueak extraction workflow is entirely automated - the user runs a command and the LLM makes all decisions about what to extract and where to integrate it. While this works, it provides limited visibility and no opportunity for user input during the process.

This design introduces an **interactive TUI (Text-based User Interface)** that:

1. **Shows the LLM's work in real-time** as streaming responses arrive
2. **Allows user override at every decision point** - users have final say
3. **Provides confidence scores** so users know when to review more carefully
4. **Supports granular block-level selection** matching the extractor's actual behavior
5. **Uses friendly, approachable language** instead of technical terms

---

## User Experience Goals

### Primary Goals

1. **Transparency**: Users should see what the LLM is thinking and why
2. **Control**: Users can override any LLM decision at any phase
3. **Efficiency**: Optional review - users can skip interaction if they trust the LLM
4. **Feedback**: Real-time progress updates as LLM streams responses
5. **Approachability**: Friendly language, helpful guidance, clear next steps

### Non-Goals

- **Not a full IDE**: We're not building a Logseq editor in the terminal
- **Not for preferences**: We won't save user preferences for future runs (yet)
- **Not batch editing**: Focus on one journal entry at a time

---

## Core Interaction Model

### Block Selection Granularity

Users can select **individual blocks** at any level of the hierarchy:

```
○ Parent block
  ○ Child block 1
  ○ Child block 2
    ○ Grandchild block
```

**Smart Defaults with Override:**
- Marking a parent block defaults all children to the same state
- Users can override individual children after setting parent
- Maximum flexibility while maintaining intuitive behavior

**Example:**
1. User marks parent as "activity" → all children default to "activity"
2. User then marks one child as "knowledge" → only that child changes
3. LLM won't override any user-marked blocks

### User Always Has Final Say

At every phase, user decisions override LLM decisions:

- **Phase 1**: User can mark blocks as knowledge/activity before or during LLM analysis
- **Phase 2**: User can exclude candidate pages from consideration
- **Phase 3**: User can change integration actions, edit reworded text, change target locations

Once a user makes a decision, it's "locked" and the LLM won't override it.

### Streaming Updates

The UI updates progressively as LLM responses stream in:

- Block classifications appear as LLM analyzes them
- Decisions and rewordings stream in during Phase 3
- Users can interact with the UI while LLM is still processing
- Clear visual indicators show what's "streaming" vs "complete"

---

## Phase-by-Phase UI Flow

### Phase 1: Knowledge Extraction

**Goal:** Classify each journal block as "knowledge" (lasting insight) or "activity" (daily log).

#### Initial Display (LLM Running)

```
┌─ Analyzing Journal: 2025-10-29 ──────────────────────────────────────┐
│ 4 top-level entries • 7 blocks total                                 │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│ ? Had an interesting discussion with Sarah about microservices       │
│   ? The key insight is that bounded contexts matter more than size   │
│   ? We should split by domain responsibility, not technical layers   │
│                                                                       │
│ ⊗ Dentist appointment at 3pm                                         │
│   └─ You marked this as a daily activity                             │
│                                                                       │
│ ✓ Read paper on vector databases                     [analyzing...]  │
│   ✓ ChromaDB uses HNSW for approximate nearest neighbor search  92%  │
│   ? Trade-off between recall and query speed via ef_construction     │
│   ? paper:: https://arxiv.org/abs/2103.00123                        │
│                                                                       │
│ ? Call mom                                                           │
│                                                                       │
├───────────────────────────────────────────────────────────────────────┤
│ ↑/↓ Move    K Keep as knowledge    A Mark as activity    R Reset     │
│ Enter Continue to next step        Ctrl+C Cancel                     │
├───────────────────────────────────────────────────────────────────────┤
│ Analyzing... • 2 knowledge blocks • 1 activity • 4 waiting           │
└───────────────────────────────────────────────────────────────────────┘
```

#### Visual Indicators

- `?` = Waiting for LLM analysis
- `✓` = LLM classified as knowledge (shows confidence %)
- `✗` = LLM classified as activity (shows confidence %)
- `⊙` = User marked as knowledge
- `⊗` = User marked as activity
- `[analyzing...]` = LLM currently processing this block

#### Confidence Score Display

- **90-100%**: Normal display (high confidence)
- **75-89%**: Normal display (good confidence)
- **60-74%**: Yellow/warning color (suggest user review)
- **<60%**: `⚠` warning indicator (definitely review)

Example:
```
│ ⚠ "Read the paper"                                               [2] │
│   Skip (too vague)                                              58%  │
│   ← Low confidence - you may want to review this                     │
```

#### Keyboard Controls

- `↑/↓` or `j/k` - Navigate between blocks
- `K` - Mark current block as knowledge
- `A` - Mark current block as activity
- `R` - Reset current block (let LLM decide)
- `Enter` - Continue to next phase
- `Ctrl+C` - Cancel extraction

#### Behavior

1. LLM starts analyzing all blocks immediately when phase begins
2. User can navigate and mark blocks while LLM is running
3. User-marked blocks are "locked" - LLM skips them
4. As LLM classifications arrive, unmarked blocks update with results
5. When all blocks are classified (or user presses Enter), proceed to Phase 2

---

### Phase 2: Candidate Page Review

**Goal:** Review which pages the system will consider for integrating each knowledge block.

This phase is **optional** - users can press Enter to skip review and accept all candidates.

#### Display (Auto-Progress Mode)

```
┌─ Finding Pages for Your Knowledge ───────────────────────────────────┐
│                                                                       │
│ Searching for relevant pages...                                      │
│                                                                       │
│ ✓ Block 1: Found 12 candidate pages                                 │
│ ✓ Block 2: Found 8 candidate pages                                  │
│ ⏳ Block 3: Searching...                                             │
│ ○ Block 4: Waiting                                                   │
│ ○ Block 5: Waiting                                                   │
│                                                                       │
│ Press R to review candidates, or wait...                             │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

#### Display (Review Mode - if user presses R)

```
┌─ Finding Pages for Your Knowledge ───────────────────────────────────┐
│ Block 1 of 5                                                          │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│ "The key insight is that bounded contexts matter more than size"     │
│                                                                       │
│ Found 12 potentially relevant pages:                                 │
│                                                                       │
│ ✓ Software Architecture                       89% match      [1]     │
│ ✓ Microservices                                85% match      [2]     │
│ ✓ Domain-Driven Design                         76% match      [3]     │
│ ✓ System Design Principles                     71% match      [4]     │
│ ✗ Daily Standup Notes                          45% match      [5]     │
│                                                                       │
│ ... 7 more pages (↓ to scroll, M to show all)                        │
│                                                                       │
├───────────────────────────────────────────────────────────────────────┤
│ Space Toggle page    M Show all    N Next block    Enter Continue    │
├───────────────────────────────────────────────────────────────────────┤
│ Block 1/5 • 11 pages selected                                        │
└───────────────────────────────────────────────────────────────────────┘
```

#### Keyboard Controls (Review Mode)

- `↑/↓` - Navigate between candidate pages
- `Space` - Toggle page inclusion
- `M` - Show all pages (expand beyond top N)
- `N` - Next knowledge block
- `P` - Previous knowledge block
- `Enter` - Continue to Phase 3

#### Behavior

1. For each knowledge block, run RAG search (semantic + hinted)
2. If user presses `R`, show review UI for each block
3. User can exclude irrelevant pages (reduces LLM evaluation cost)
4. If user doesn't press `R`, auto-proceed when all searches complete
5. Proceed to Phase 3 with filtered candidate list

---

### Phase 3: Integration Decisions

**Goal:** For each (knowledge block, candidate page) pair, decide the integration action and refine the content.

This is the most complex phase - the LLM makes decisions about where and how to integrate knowledge, then rewords content for evergreen storage.

#### Display (Streaming Decisions)

```
┌─ Deciding Where to Integrate ────────────────────────────────────────┐
│ Reviewing 5 knowledge blocks with 47 possible page matches           │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│ ▼ Software Architecture                                    2 blocks  │
│                                                                       │
│   ⊙ "The key insight is that bounded contexts matter..."        [1] │
│     Add as new section                                          88%  │
│     Refined: "Bounded contexts should drive microservice            │
│               boundaries rather than technical layer separation."    │
│                                                                       │
│   ✓ "We should split by domain responsibility..."               [2] │
│     Add as new section                                          84%  │
│     Refining... [streaming]                                          │
│                                                                       │
│ ▼ Microservices                                                1 block│
│                                                                       │
│   ✓ "The key insight is that bounded contexts matter..."        [1] │
│     Add under "Bounded Context Pattern"                         92%  │
│     Refined: "Bounded contexts are more significant than service    │
│               size when defining microservice boundaries."           │
│                                                                       │
│ ▶ Domain-Driven Design                                      0 blocks │
│   All blocks skipped (already covered)                               │
│                                                                       │
│ ▶ Databases                                                 [pending]│
│                                                                       │
├───────────────────────────────────────────────────────────────────────┤
│ ↑/↓ Move    Space Change action    E Edit text    T Pick location   │
│ ←/→ Collapse/expand page           Enter Continue                    │
├───────────────────────────────────────────────────────────────────────┤
│ Analyzing... • 35/47 reviewed • 18 will be added • 17 skipped       │
└───────────────────────────────────────────────────────────────────────┘
```

#### Action Labels (User-Friendly)

| Technical Term | User-Facing Display |
|---------------|---------------------|
| `APPEND_ROOT` | "Add as new section" |
| `APPEND_CHILD` | "Add under 'Block Title'" |
| `UPDATE` | "Replace 'Block Title'" |
| `IGNORE_DUPLICATE` | "Skip (already covered)" |
| `IGNORE_IRRELEVANT` | "Skip (not relevant)" |
| `IGNORE_TOO_VAGUE` | "Skip (too general)" |

#### Visual Indicators

- `✓` = LLM decision, accepted
- `⊙` = User modified this decision
- `✗` = Skipped (LLM or user)
- `[streaming]` = Currently processing
- `▼` = Page section expanded
- `▶` = Page section collapsed

#### Keyboard Controls

- `↑/↓` - Navigate between items
- `Space` - Cycle through actions: Skip ↔ Add as new section ↔ Add under... ↔ Replace...
- `E` - Edit refined text (opens inline editor)
- `T` - Pick location (for "Add under" or "Replace" actions)
- `←/→` - Collapse/expand page sections
- `Enter` - Continue to Phase 4

#### Change Location Dialog (Press T)

```
┌─ Choose Location in "Microservices" ─────────────────────────────────┐
│                                                                       │
│ Where should this go?                                                │
│                                                                       │
│ ○ Add as new section (top level)                                     │
│ ● Add under "Bounded Context Pattern"                                │
│ ○ Add under "Service Decomposition Strategies"                       │
│ ○ Add under "Inter-Service Communication"                            │
│                                                                       │
│ ↑/↓ Move    Enter Confirm    Esc Cancel                              │
└───────────────────────────────────────────────────────────────────────┘
```

#### Edit Text Dialog (Press E)

```
┌─ Edit Refined Text ──────────────────────────────────────────────────┐
│                                                                       │
│ Original:                                                             │
│ "The key insight is that bounded contexts matter more than size"     │
│                                                                       │
│ Edit your refined version:                                           │
│ ┌──────────────────────────────────────────────────────────────────┐ │
│ │Bounded contexts should drive microservice boundaries rather     │ │
│ │than technical layer separation.█                                │ │
│ │                                                                  │ │
│ └──────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│ Ctrl+S Save    Esc Cancel                                            │
└───────────────────────────────────────────────────────────────────────┘
```

#### Streaming Behavior

1. **Decider (Phase 3.1)**: For each (block, page) pair, stream decision + confidence
2. **Reworder (Phase 3.2)**: If action != Skip, immediately stream refined content
3. UI updates as each decision arrives:
   - First shows action + confidence
   - Then shows "Refining... [streaming]"
   - Finally shows complete refined text
4. User can interact during streaming - mark blocks, change actions, etc.
5. User-modified items are locked (LLM won't override)

#### Grouping Strategy

**Group by destination page** - users think "what's going into my Software Architecture page?" not "where's this block going?"

Pages with no accepted blocks are collapsed by default and show "(0 blocks)" or reason like "(all blocks skipped - already covered)".

---

### Phase 4: Writing Changes

**Goal:** Execute the integration operations and update journal with provenance links.

#### Display (Writing in Progress)

```
┌─ Saving Your Knowledge ──────────────────────────────────────────────┐
│ Writing 18 blocks to 8 pages...                                      │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│ ✓ Software Architecture                              2/2 complete    │
│   ✓ Added 2 new sections                                             │
│   ✓ Journal updated with links                                       │
│                                                                       │
│ ✓ Microservices                                      1/1 complete    │
│   ✓ Added content under "Bounded Context Pattern"                    │
│   ✓ Journal updated with links                                       │
│                                                                       │
│ ⏳ Databases                                          1/3 writing     │
│   ✓ Added new section                                                │
│   ⏳ Updating "Vector Databases Overview"...                          │
│   ○ Pending...                                                        │
│                                                                       │
│ ○ Vector Search                                       pending         │
│ ○ System Design                                       pending         │
│ ... 3 more pages                                                      │
│                                                                       │
├───────────────────────────────────────────────────────────────────────┤
│ Progress: 3/8 pages complete                                         │
└───────────────────────────────────────────────────────────────────────┘
```

#### Error Handling

If errors occur, continue with remaining pages and show summary:

```
┌─ Saving Your Knowledge ──────────────────────────────────────────────┐
│ Writing 18 blocks to 8 pages...                                      │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│ ✓ Software Architecture                              2/2 complete    │
│ ✓ Microservices                                      1/1 complete    │
│ ⚠ Databases                                          1/3 failed      │
│   ✓ Added new section                                                │
│   ✗ Failed to update "Vector Databases Overview"                     │
│       Error: Block not found (id:: abc123)                           │
│   ⊘ Skipped remaining operations for this page                       │
│ ✓ Vector Search                                      2/2 complete    │
│ ... continuing with remaining pages                                  │
│                                                                       │
├───────────────────────────────────────────────────────────────────────┤
│ Progress: 7/8 pages • 1 page had errors                              │
└───────────────────────────────────────────────────────────────────────┘
```

#### Behavior

1. Group all write operations by page (minimize file I/O)
2. For each page:
   - Parse AST
   - Apply all operations (UPDATE, APPEND_CHILD, APPEND_ROOT)
   - Generate UUIDs for new blocks
   - Serialize and write file
   - Atomically update journal with `processed::` marker
3. If any page write fails:
   - Log error details
   - Continue with remaining pages
   - Show summary of failures at end
4. Proceed to completion summary

---

### Completion Summary

#### Success Case

```
┌─ All Done! ───────────────────────────────────────────────────────────┐
│                                                                       │
│ ✓ Successfully added 16 blocks of knowledge to your graph            │
│                                                                       │
│ Updated pages:                                                        │
│   Software Architecture        +2 sections                           │
│   Microservices                +1 block                              │
│   Databases                    +2 sections, updated 1 block          │
│   Vector Search                +2 sections                           │
│   System Design                +3 sections                           │
│   ... and 3 more pages                                               │
│                                                                       │
│ Your journal entry has been marked with links:                       │
│   journals/2025_10_29.md                                             │
│                                                                       │
│ Press V to view changes, Q to quit                                   │
└───────────────────────────────────────────────────────────────────────┘
```

#### With Errors

```
┌─ All Done! ───────────────────────────────────────────────────────────┐
│                                                                       │
│ ✓ Successfully added 16 blocks of knowledge to your graph            │
│                                                                       │
│ Updated pages:                                                        │
│   Software Architecture        +2 sections                           │
│   Microservices                +1 block                              │
│   Databases                    +2 sections, updated 1 block          │
│   Vector Search                +2 sections                           │
│   System Design                +3 sections                           │
│   ... and 3 more pages                                               │
│                                                                       │
│ ⚠ 1 page had errors:                                                 │
│   Databases: 1 block failed to update (2 successful)                 │
│                                                                       │
│ Your journal entry has been marked with links:                       │
│   journals/2025_10_29.md                                             │
│                                                                       │
│ Press V to view changes, E to see errors, Q to quit                  │
└───────────────────────────────────────────────────────────────────────┘
```

#### View Changes (Press V)

```
┌─ Changes Made ────────────────────────────────────────────────────────┐
│                                                                       │
│ Software Architecture (pages/Software Architecture.md)               │
│                                                                       │
│   + Bounded contexts should drive microservice boundaries rather     │
│     than technical layer separation.                                 │
│     id:: 12ab34cd-5678-90ef-ghij-klmnopqrstuv                        │
│                                                                       │
│   + Split microservices by domain responsibility, not technical      │
│     layers.                                                           │
│     id:: 56ef78gh-1234-56ab-cdef-ghijklmnopqrst                      │
│                                                                       │
│ Microservices (pages/Microservices.md)                               │
│                                                                       │
│   Under "Bounded Context Pattern":                                   │
│   + Bounded contexts are more significant than service size when     │
│     defining microservice boundaries.                                │
│     id:: 90ab12cd-efgh-3456-ijkl-mnopqrstuvwxyz                      │
│                                                                       │
│ ↑/↓ Scroll    Q Back to summary                                      │
└───────────────────────────────────────────────────────────────────────┘
```

#### View Errors (Press E)

```
┌─ Errors During Save ──────────────────────────────────────────────────┐
│                                                                       │
│ Databases (pages/Databases.md)                                       │
│                                                                       │
│ ✗ Failed to update block: "Vector Databases Overview"                │
│   Block ID: abc123def456                                             │
│   Error: Block not found in page                                     │
│   This usually means:                                                │
│   • The block was moved or deleted since indexing                    │
│   • The page structure changed significantly                         │
│                                                                       │
│   Original content that couldn't be integrated:                      │
│   "ChromaDB uses HNSW (Hierarchical Navigable Small World) graphs    │
│    for approximate nearest neighbor search, trading exact recall     │
│    for query speed via the ef_construction parameter."               │
│                                                                       │
│   Suggested action:                                                  │
│   • Rebuild your index: logsqueak index rebuild                      │
│   • Or manually add this content to the page                         │
│                                                                       │
│ Q Back to summary                                                     │
└───────────────────────────────────────────────────────────────────────┘
```

---

## Design Decisions

### 1. Parent/Child Block Selection

**Decision:** Smart defaults with override (Option B)

- Marking a parent block defaults all children to the same state
- Users can override individual children after setting parent
- Provides intuitive behavior while preserving maximum flexibility

**Rationale:** Balances ease of use (mark parent, all children follow) with precision (can still override individual blocks).

### 2. Phase 2 User Interaction

**Decision:** Optional review (auto-proceed if user doesn't interact)

- Show progress of candidate retrieval
- User can press `R` to review and exclude pages
- If user doesn't press `R`, auto-proceed when complete

**Rationale:** Most users will trust the RAG system, but power users can fine-tune. Don't force interaction.

### 3. Phase 3 Grouping Strategy

**Decision:** Group by destination page (Option C)

- Show which knowledge blocks are going into each page
- Users think "what's being added to Software Architecture?" not "where is block X going?"

**Rationale:** Matches mental model of reviewing changes to their knowledge graph.

### 4. Streaming Granularity

**Decision:** Block-by-block streaming

- Stream individual decisions as they arrive
- Immediately trigger rewording for accepted blocks
- Update UI progressively during LLM processing

**Rationale:** Provides engaging feedback, allows early user intervention, feels responsive.

### 5. Error Handling

**Decision:** Continue with other pages, show summary at end

- Don't stop entire process if one page fails
- Log all errors with context
- Show actionable suggestions for resolution

**Rationale:** Partial success is better than complete failure. User gets most of their work done even if one page has issues.

### 6. Keyboard Shortcuts

**Decision:** Friendly, discoverable bindings

- Use common keys: arrows, Enter, Esc, Space
- Show all bindings in footer
- Support vim-style alternatives (j/k) but don't require them
- Single-letter shortcuts for common actions (K=knowledge, A=activity)

**Rationale:** Accessible to all users, not just vim users. Footer makes commands discoverable.

### 7. Visual Design

**Decision:** Start with Textual defaults, iterate based on feedback

- Use Textual's built-in themes and widgets
- Consistent with terminal conventions
- Don't over-customize until we validate UX

**Rationale:** Ship faster, get user feedback, then refine visuals.

---

## Technical Implementation

### Technology Stack

**Primary Library:** [Textual](https://textual.textualize.io/)

- Modern Python TUI framework
- Built on top of Rich (which logsqueak already uses)
- Excellent async support (critical for streaming LLM responses)
- Built-in widgets: Tree, TextArea, Footer, Header, etc.
- Reactive data binding
- CSS-like styling

**Dependencies:**
```python
# Add to pyproject.toml
textual = "^0.47.0"  # Latest stable version
```

### App Structure

```python
from textual.app import App, ComposeResult
from textual.screen import Screen
from textual.widgets import Tree, Static, Footer, Header, TextArea
from textual.containers import Container, VerticalScroll
from textual.binding import Binding
from dataclasses import dataclass

@dataclass
class BlockState:
    """Track state of each block through pipeline"""
    block_id: str
    classification: str  # "pending" | "knowledge" | "activity"
    confidence: float | None
    source: str  # "user" | "llm"

@dataclass
class IntegrationDecision:
    """Track decision for each (knowledge, page) pair"""
    knowledge_block_id: str
    target_page: str
    action: str  # "skip" | "add_section" | "add_under" | "replace"
    target_block_id: str | None  # For "add_under" and "replace"
    target_block_title: str | None  # Human-readable target name
    confidence: float
    refined_text: str
    source: str  # "user" | "llm"

class ExtractionApp(App):
    """Main TUI application for interactive extraction"""

    CSS = """
    /* Textual CSS for styling */
    .knowledge { color: $success; }
    .activity { color: $text-muted; }
    .pending { color: $warning; }
    .low-confidence { color: $warning; }
    .user-override { color: $accent; }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("ctrl+c", "quit", "Quit", show=False),
    ]

    def __init__(self, journal_entry: JournalEntry, config: Config):
        super().__init__()
        self.journal_entry = journal_entry
        self.config = config
        self.llm_client = get_llm_client(config)

        # Shared state across screens
        self.block_states: dict[str, BlockState] = {}
        self.candidates: dict[str, list[CandidatePage]] = {}
        self.decisions: dict[tuple[str, str], IntegrationDecision] = {}

        self.current_phase = 1

    def on_mount(self) -> None:
        # Start with Phase 1
        self.push_screen(Phase1Screen(self))

class Phase1Screen(Screen):
    """Phase 1: Knowledge extraction with streaming"""

    BINDINGS = [
        ("up,k", "cursor_up", "Move up"),
        ("down,j", "cursor_down", "Move down"),
        ("k", "mark_knowledge", "Keep as knowledge"),
        ("a", "mark_activity", "Mark as activity"),
        ("r", "reset", "Reset"),
        ("enter", "continue", "Continue"),
    ]

    def __init__(self, app: ExtractionApp):
        super().__init__()
        self.extraction_app = app

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Static(f"Analyzing Journal: {self.extraction_app.journal_entry.date}",
                   id="phase-title"),
            Tree("Journal Entry", id="block-tree"),
            Static("", id="status-bar"),
        )
        yield Footer()

    async def on_mount(self) -> None:
        # Build tree from journal blocks
        tree = self.query_one("#block-tree", Tree)
        self._populate_tree(tree, self.extraction_app.journal_entry.root_blocks)

        # Initialize block states
        for block in self._all_blocks():
            self.extraction_app.block_states[block.id] = BlockState(
                block_id=block.id,
                classification="pending",
                confidence=None,
                source="llm",
            )

        # Start LLM extraction in background
        self.extraction_task = asyncio.create_task(self._stream_extraction())

    async def _stream_extraction(self) -> None:
        """Stream LLM classification decisions"""
        llm_client = self.extraction_app.llm_client

        # Get blocks that user hasn't manually classified
        blocks_to_analyze = [
            block for block in self._all_blocks()
            if self.extraction_app.block_states[block.id].source != "user"
        ]

        async for block_id, is_knowledge, confidence in llm_client.stream_extract(blocks_to_analyze):
            # Only update if user hasn't overridden this block in the meantime
            if self.extraction_app.block_states[block_id].source != "user":
                self.extraction_app.block_states[block_id] = BlockState(
                    block_id=block_id,
                    classification="knowledge" if is_knowledge else "activity",
                    confidence=confidence,
                    source="llm",
                )
                self._update_tree_display(block_id)
                self._update_status_bar()

    def action_mark_knowledge(self) -> None:
        """User manually marks block as knowledge"""
        block_id = self._get_selected_block_id()
        self.extraction_app.block_states[block_id] = BlockState(
            block_id=block_id,
            classification="knowledge",
            confidence=None,
            source="user",
        )
        self._update_tree_display(block_id)

        # Smart default: mark all children as knowledge too
        self._default_children(block_id, "knowledge")

    def action_continue(self) -> None:
        """Proceed to Phase 2"""
        # Cancel ongoing LLM task if still running
        if self.extraction_task and not self.extraction_task.done():
            self.extraction_task.cancel()

        # Switch to Phase 2
        self.app.push_screen(Phase2Screen(self.extraction_app))

class Phase2Screen(Screen):
    """Phase 2: Candidate page retrieval (optional review)"""
    # ... implementation

class Phase3Screen(Screen):
    """Phase 3: Integration decisions with streaming"""

    BINDINGS = [
        ("up,k", "cursor_up", "Move up"),
        ("down,j", "cursor_down", "Move down"),
        ("space", "toggle_action", "Change action"),
        ("e", "edit_text", "Edit text"),
        ("t", "pick_location", "Pick location"),
        ("left,h", "collapse", "Collapse"),
        ("right,l", "expand", "Expand"),
        ("enter", "continue", "Continue"),
    ]

    async def _stream_decisions(self) -> None:
        """Stream LLM decisions for each (block, page) pair"""
        llm_client = self.extraction_app.llm_client

        # Get knowledge blocks from Phase 1
        knowledge_blocks = [
            block for block in self._all_blocks()
            if self.extraction_app.block_states[block.id].classification == "knowledge"
        ]

        for knowledge_block in knowledge_blocks:
            candidates = self.extraction_app.candidates[knowledge_block.id]

            for candidate_page in candidates:
                pair_key = (knowledge_block.id, candidate_page.name)

                # Phase 3.1: Get decision (streaming)
                decision = await llm_client.decide_action(
                    knowledge_block, candidate_page
                )

                if decision.action != "IGNORE":
                    # Show decision, mark as "streaming rewording"
                    self._update_display(pair_key, decision, status="streaming")

                    # Phase 3.2: Stream rewording
                    refined_text = await llm_client.rephrase_content(
                        knowledge_block, candidate_page
                    )

                    # Update with final refined text
                    decision.refined_text = refined_text
                    self._update_display(pair_key, decision, status="complete")

                # Store decision
                self.extraction_app.decisions[pair_key] = decision

class Phase4Screen(Screen):
    """Phase 4: Execute write operations"""
    # ... implementation
```

### Streaming JSON Parsing

Since LLM responses are structured JSON, we need robust streaming parsing:

```python
import json
from typing import AsyncIterator

async def stream_json_objects(response_stream: AsyncIterator[str]) -> AsyncIterator[dict]:
    """
    Parse streaming JSON response into individual objects.

    Handles both:
    - JSON array: [{"obj1": ...}, {"obj2": ...}]
    - Newline-delimited JSON: {"obj1": ...}\n{"obj2": ...}\n
    """
    buffer = ""
    decoder = json.JSONDecoder()

    async for chunk in response_stream:
        buffer += chunk

        # Try to parse complete objects
        while buffer:
            buffer = buffer.lstrip()
            if not buffer:
                break

            # Skip array/object delimiters
            if buffer[0] in "[,":
                buffer = buffer[1:]
                continue

            try:
                obj, idx = decoder.raw_decode(buffer)
                yield obj
                buffer = buffer[idx:]
            except json.JSONDecodeError:
                # Incomplete object, wait for more data
                break
```

### LLM Client Streaming Methods

Extend the LLM client with streaming support:

```python
class LLMClient:
    async def stream_extract(
        self,
        blocks: list[LogseqBlock]
    ) -> AsyncIterator[tuple[str, bool, float]]:
        """
        Stream knowledge extraction decisions.

        Yields: (block_id, is_knowledge, confidence)
        """
        # Implementation depends on LLM provider
        # For OpenAI-compatible APIs with streaming:
        response = await self.client.chat.completions.create(
            model=self.extractor_model,
            messages=[...],
            stream=True,
        )

        async for obj in stream_json_objects(response):
            yield (
                obj["block_id"],
                obj["is_knowledge"],
                obj["confidence"],
            )

    async def decide_action(
        self,
        knowledge_block: LogseqBlock,
        candidate_page: CandidatePage,
    ) -> DecisionResult:
        """Phase 3.1: Decide integration action (non-streaming)"""
        # ... existing implementation

    async def rephrase_content(
        self,
        knowledge_block: LogseqBlock,
        candidate_page: CandidatePage,
    ) -> str:
        """Phase 3.2: Rephrase content (streaming)"""
        # Stream the refined text token by token
        # UI can show progressive text generation
```

### CLI Integration

Add new command to run interactive mode:

```python
@cli.command()
@click.argument("date", required=False)
@click.option("--interactive/--no-interactive", default=True,
              help="Run in interactive TUI mode")
def extract(date: str | None, interactive: bool):
    """Extract knowledge from journal entry"""

    config = load_config()
    journal_entry = load_journal_entry(date, config)

    if interactive:
        # Run TUI
        app = ExtractionApp(journal_entry, config)
        app.run()
    else:
        # Run existing batch mode
        run_batch_extraction(journal_entry, config)
```

---

## Terminology Mappings

All user-facing language should be friendly and descriptive, not technical.

### Action Types

| Internal Code | User Display |
|--------------|-------------|
| `APPEND_ROOT` | "Add as new section" |
| `APPEND_CHILD` (with target) | "Add under 'Target Block Title'" |
| `UPDATE` (with target) | "Replace 'Target Block Title'" |
| `IGNORE_DUPLICATE` | "Skip (already covered)" |
| `IGNORE_IRRELEVANT` | "Skip (not relevant)" |
| `IGNORE_TOO_VAGUE` | "Skip (too general)" |

### Phase Names

| Internal | User Display |
|---------|-------------|
| Phase 1: Extraction | "Analyzing Journal" |
| Phase 2: Candidate Retrieval | "Finding Pages for Your Knowledge" |
| Phase 3: Decision & Rewording | "Deciding Where to Integrate" |
| Phase 4: Execution | "Saving Your Knowledge" |

### Block States

| Internal | User Display |
|---------|-------------|
| `classification: "knowledge"` | "Keep as knowledge" / "✓ [confidence]%" |
| `classification: "activity"` | "Daily activity" / "✗ [confidence]%" |
| `classification: "pending"` | "?" (waiting for analysis) |
| `source: "user"` | "⊙" (user marked as knowledge) / "⊗" (user marked as activity) |
| `source: "llm"` | "✓" / "✗" (LLM classified) |

### Other Terms

| Technical | User-Friendly |
|-----------|---------------|
| Knowledge extraction | Analyzing journal |
| Confidence score | % match / percentage |
| RAG / Semantic search | Finding relevant pages |
| Vector store | (hidden from user) |
| `processed::` marker | "Journal updated with links" |
| Hybrid ID | (hidden from user) |
| Write operation | "Adding" / "Updating" / "Replacing" |

---

## Next Steps

1. **Prototype Phase 1** - Build basic extraction screen with tree view and streaming
2. **Test streaming JSON parsing** - Validate with real LLM responses
3. **Implement Phase 3** - Most complex phase with grouped display
4. **Add Phase 2 & 4** - Simpler phases once core UX is validated
5. **User testing** - Get feedback on real journal entries
6. **Iterate on visual design** - Refine colors, layout, terminology based on feedback

---

**End of Design Document**
