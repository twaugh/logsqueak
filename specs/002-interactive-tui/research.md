# Research: Interactive TUI Technical Decisions

**Branch**: `feature/interactive-tui-design` | **Date**: 2025-10-29 | **Status**: Complete

This document addresses the 6 research topics identified in `plan.md` and provides technical decisions, rationale, and implementation guidance for building the Interactive TUI feature.

---

## Table of Contents

1. [NDJSON Streaming Implementation](#1-ndjson-streaming-implementation)
2. [Textual App Architecture](#2-textual-app-architecture)
3. [Async LLM Streaming + UI Responsiveness](#3-async-llm-streaming--ui-responsiveness)
4. [Tree Widget Customization](#4-tree-widget-customization)
5. [Keyboard Shortcut Conflicts](#5-keyboard-shortcut-conflicts)
6. [Error Recovery in Streaming](#6-error-recovery-in-streaming)

---

## 1. NDJSON Streaming Implementation

### Research Question

How to robustly parse NDJSON (Newline-Delimited JSON) from chunked async streams with proper buffering and error handling?

### Decision

**Use line-buffering with incremental JSON parsing.**

LLM providers will output one complete JSON object per line, and the parser will:

1. Buffer incoming chunks until a complete line (ending with `\n`) is received
2. Parse each complete line as a standalone JSON object
3. Skip empty lines silently
4. Log and skip malformed JSON lines without crashing
5. Handle incomplete final lines gracefully

### Rationale

**Validated with live testing**: The test script at `/tmp/test_ndjson_streaming_incremental.py` successfully streamed and parsed 6 JSON objects incrementally across 128 chunks from an Ollama server.

**Key advantages**:

- **Simplicity**: Each line is a complete, parseable JSON object (no complex state machine)
- **Robustness**: Parsing errors on one line don't affect subsequent lines
- **Real-time UX**: UI can update immediately as each object arrives (not waiting for entire response)
- **Predictable buffering**: Only need to buffer up to the next newline character

**Why not alternatives**:

- ❌ **Streaming JSON arrays** `[{...}, {...}]`: Requires parsing incomplete JSON, complex buffering
- ❌ **Progressive completion**: Can't parse until entire response arrives, no real-time updates
- ❌ **Custom delimiters**: Non-standard, harder to debug

### Implementation Guidance

#### Core Parser Function

```python
from typing import AsyncIterator
import json


async def parse_ndjson_stream(
    stream: AsyncIterator[str]
) -> AsyncIterator[dict]:
    """
    Parse NDJSON from chunked async stream.

    Input: Token-by-token chunks ('{"', 'foo', '": ', '"bar', '"}', '\n', ...)
    Output: Complete JSON objects as they become parseable

    Error Handling:
    - Invalid JSON on a line → Log error, skip line, continue
    - Incomplete line at stream end → Log warning, discard
    - Empty lines → Skip silently
    """
    buffer = ""

    async for chunk in stream:
        buffer += chunk

        # Process all complete lines in buffer
        while '\n' in buffer:
            line_end = buffer.index('\n')
            complete_line = buffer[:line_end].strip()
            buffer = buffer[line_end + 1:]  # Remove processed line + newline

            if not complete_line:
                # Empty line, skip silently
                continue

            try:
                obj = json.loads(complete_line)
                yield obj
            except json.JSONDecodeError as e:
                # Log error but continue processing
                import logging
                logging.warning(
                    f"Skipping malformed NDJSON line: {complete_line[:60]}... "
                    f"Error: {e}"
                )
                # Don't yield anything, just continue to next line

    # Handle any remaining content in buffer at stream end
    if buffer.strip():
        try:
            obj = json.loads(buffer.strip())
            yield obj
        except json.JSONDecodeError:
            import logging
            logging.warning(
                f"Discarding incomplete NDJSON at end of stream: {buffer[:60]}..."
            )
```

#### Integration with Ollama/OpenAI Streaming

```python
import httpx


async def stream_llm_ndjson(
    endpoint: str,
    model: str,
    messages: list[dict],
    api_key: str | None = None
) -> AsyncIterator[dict]:
    """
    Stream NDJSON responses from LLM provider.

    Works with:
    - Ollama (endpoint: http://localhost:11434/api/chat)
    - OpenAI-compatible APIs (endpoint: https://api.openai.com/v1/chat/completions)
    """
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        # CRITICAL: Do NOT use format='json' - let model output NDJSON naturally
    }

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    async with httpx.AsyncClient(timeout=30.0) as client:
        async with client.stream(
            "POST",
            endpoint,
            json=payload,
            headers=headers
        ) as response:
            response.raise_for_status()

            # For Ollama: Each line is a JSON chunk with "message.content"
            # For OpenAI: Each line is "data: {json}" with "choices[0].delta.content"

            content_buffer = ""

            async for line in response.aiter_lines():
                if not line.strip():
                    continue

                # Parse the streaming protocol wrapper
                if endpoint.endswith("/api/chat"):
                    # Ollama format
                    chunk_data = json.loads(line)
                    content = chunk_data.get("message", {}).get("content", "")
                    done = chunk_data.get("done", False)
                else:
                    # OpenAI format: "data: {...}" or "data: [DONE]"
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            done = True
                            content = ""
                        else:
                            chunk_data = json.loads(data)
                            content = chunk_data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            done = False

                # Accumulate content and parse NDJSON
                if content:
                    async for obj in parse_ndjson_stream(async_iter([content])):
                        yield obj

                if done:
                    break


async def async_iter(items):
    """Helper to convert list to async iterator"""
    for item in items:
        yield item
```

#### Example Prompt for NDJSON Output

```python
SYSTEM_PROMPT = """You are a knowledge extraction assistant.

OUTPUT FORMAT: Return newline-delimited JSON (NDJSON).
Each knowledge block MUST be on its own line as a complete JSON object.

Example:
{"block_id": "abc123", "is_knowledge": true, "confidence": 0.85}
{"block_id": "def456", "is_knowledge": false, "confidence": 0.92}

CRITICAL RULES:

1. One JSON object per line
2. Each line must be complete and valid JSON
3. No trailing commas
4. No wrapping in array brackets []
"""
```

### Best Practices

1. **Always buffer by lines**: Don't try to parse partial JSON objects
2. **Graceful degradation**: Skip malformed lines, don't crash entire stream
3. **Log parsing errors**: Help debug LLM output issues
4. **Test with slow streams**: Simulate token-by-token arrival to verify buffering
5. **Handle empty content**: Some chunks may have no content (control messages)

### Performance Characteristics

From test results (`/tmp/test_ndjson_streaming_incremental.py`):

- **128 chunks** → **6 parsed objects** (average 21 chunks per object)
- **Incremental parsing**: Each object available immediately upon line completion
- **Zero blocking**: UI can update as soon as `yield obj` returns
- **Memory efficient**: Only buffers up to next newline (typically <1KB)

---

## 2. Textual App Architecture

### Research Question

Best practices for managing shared state across multiple Textual screens while maintaining clean separation of concerns?

### Decision

**Use a central App class with shared state dictionaries, passed to screens via constructor.**

The `ExtractionApp` will:

1. Store all shared state as instance attributes (not global variables)
2. Pass `self` (the app instance) to each screen's constructor
3. Screens access state via `self.extraction_app.block_states`, etc.
4. Use Textual's reactive data binding for automatic UI updates
5. Navigate between screens using `app.push_screen()` and `app.pop_screen()`

### Rationale

**Textual's design philosophy**: Screens are meant to be stateful components that access app-level data.

**Why this approach**:

- ✅ **Clear data ownership**: App owns state, screens are views
- ✅ **Type safety**: IDE can autocomplete `app.block_states`
- ✅ **Testability**: Can construct screens with mock app instances
- ✅ **No global state**: All state scoped to app instance
- ✅ **Reactive updates**: Textual automatically re-renders when reactive attributes change

**Why not alternatives**:

- ❌ **Global variables**: Breaks testability, not thread-safe
- ❌ **Message passing only**: Too verbose, harder to reason about data flow
- ❌ **External state management (Redux-like)**: Overkill for single-user CLI app

### Implementation Guidance

#### App Class with Shared State

```python
from textual.app import App
from textual.screen import Screen
from textual.reactive import reactive
from dataclasses import dataclass, field


@dataclass
class BlockState:
    """State of each journal block"""
    block_id: str
    classification: str  # "pending" | "knowledge" | "activity"
    confidence: float | None
    source: str  # "user" | "llm"


@dataclass
class IntegrationDecision:
    """Decision for each (knowledge block, target page) pair"""
    knowledge_block_id: str
    target_page: str
    action: str  # "skip" | "add_section" | "add_under" | "replace"
    target_block_id: str | None
    target_block_title: str | None
    confidence: float
    refined_text: str
    source: str  # "user" | "llm"


@dataclass
class CandidatePage:
    """Candidate page from RAG search"""
    page_name: str
    similarity_score: float
    included: bool = True  # User can exclude
    blocks: list[dict] = field(default_factory=list)


class ExtractionApp(App):
    """Main TUI application for interactive extraction"""

    CSS = """
    .knowledge { color: $success; }
    .activity { color: $text-muted; }
    .pending { color: $warning; }
    .low-confidence { color: $warning; }
    .user-override { color: $accent; }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("ctrl+c", "quit", "Quit", False),
    ]

    def __init__(self, journal_entry, config):
        super().__init__()
        self.journal_entry = journal_entry
        self.config = config
        self.llm_client = get_llm_client(config)

        # Shared state (accessed by all screens)
        self.block_states: dict[str, BlockState] = {}
        self.candidates: dict[str, list[CandidatePage]] = {}
        self.decisions: dict[tuple[str, str], IntegrationDecision] = {}

        # Current phase (for navigation)
        self.current_phase = 1

    def on_mount(self) -> None:
        """Start with Phase 1 screen"""
        self.push_screen(Phase1Screen(self))
```

#### Screen Lifecycle and State Access

```python
from textual.screen import Screen
from textual.widgets import Tree, Static, Footer, Header
from textual.containers import Container


class Phase1Screen(Screen):
    """Phase 1: Knowledge extraction"""

    BINDINGS = [
        ("k", "mark_knowledge", "Keep as knowledge"),
        ("a", "mark_activity", "Mark as activity"),
        ("enter", "continue", "Continue"),
    ]

    def __init__(self, app: ExtractionApp):
        super().__init__()
        self.extraction_app = app  # Store reference to app

    def compose(self):
        """Build UI layout"""
        yield Header()
        yield Container(
            Static(f"Analyzing Journal: {self.extraction_app.journal_entry.date}"),
            Tree("Journal Entry", id="block-tree"),
        )
        yield Footer()

    async def on_mount(self) -> None:
        """Initialize screen state"""
        # Access shared state via app
        for block in self._get_all_blocks():
            self.extraction_app.block_states[block.id] = BlockState(
                block_id=block.id,
                classification="pending",
                confidence=None,
                source="llm"
            )

        # Start background LLM task
        self.extraction_task = asyncio.create_task(self._stream_extraction())

    async def _stream_extraction(self):
        """Background task that updates shared state"""
        async for block_id, is_knowledge, confidence in self.extraction_app.llm_client.stream_extract(...):
            # Update shared state (app owns it)
            if self.extraction_app.block_states[block_id].source != "user":
                self.extraction_app.block_states[block_id] = BlockState(
                    block_id=block_id,
                    classification="knowledge" if is_knowledge else "activity",
                    confidence=confidence,
                    source="llm"
                )
                # Trigger UI update
                self._update_tree_display(block_id)

    def action_mark_knowledge(self):
        """User action that modifies shared state"""
        block_id = self._get_selected_block_id()

        # Modify shared state via app
        self.extraction_app.block_states[block_id] = BlockState(
            block_id=block_id,
            classification="knowledge",
            confidence=None,
            source="user"
        )

        # Update UI
        self._update_tree_display(block_id)

    def action_continue(self):
        """Navigate to next screen"""
        # Cancel ongoing tasks
        if hasattr(self, 'extraction_task') and not self.extraction_task.done():
            self.extraction_task.cancel()

        # Navigate to Phase 2
        self.app.push_screen(Phase2Screen(self.extraction_app))
```

#### Reactive Data Binding (Optional)

For more advanced automatic updates, use Textual's `reactive` descriptor:

```python
from textual.reactive import reactive


class Phase1Screen(Screen):
    # Reactive attribute: UI auto-updates when this changes
    blocks_analyzed = reactive(0)

    def watch_blocks_analyzed(self, new_value: int):
        """Called automatically when blocks_analyzed changes"""
        status = self.query_one("#status-bar", Static)
        status.update(f"Analyzed {new_value} / {len(self.extraction_app.block_states)} blocks")

    async def _stream_extraction(self):
        async for block_id, is_knowledge, confidence in ...:
            # Update state
            self.extraction_app.block_states[block_id] = ...

            # Increment reactive counter (triggers watch_blocks_analyzed)
            self.blocks_analyzed += 1
```

#### Screen Navigation Patterns

```python
# Push a new screen (adds to stack)
self.app.push_screen(Phase2Screen(self.extraction_app))

# Pop current screen (go back)
self.app.pop_screen()

# Replace current screen (no back navigation)
self.app.switch_screen(Phase2Screen(self.extraction_app))

# Modal dialog
self.app.push_screen(ConfirmDialog("Are you sure?"), callback=self.on_confirm)

def on_confirm(self, confirmed: bool):
    if confirmed:
        # Proceed with action
        pass
```

### Best Practices

1. **App owns state, screens are views**: Never duplicate state in screens
2. **Pass app to screen constructor**: `Phase1Screen(self.extraction_app)`
3. **Use type hints**: `self.extraction_app: ExtractionApp` helps IDE autocomplete
4. **Cancel async tasks on screen exit**: Prevent memory leaks from background tasks
5. **Minimize screen-to-screen communication**: Use app state as single source of truth
6. **Use reactive attributes for dynamic UI**: Auto-update status bars, counters, etc.

### Testing Strategy

```python
import pytest
from textual.app import App


@pytest.mark.asyncio
async def test_phase1_screen_state():
    """Test screen correctly accesses and modifies app state"""

    # Create mock app
    app = ExtractionApp(journal_entry=mock_journal, config=mock_config)

    # Create screen
    screen = Phase1Screen(app)

    # Simulate mounting
    await screen.on_mount()

    # Verify initial state
    assert len(app.block_states) == 5  # Mock journal has 5 blocks
    assert all(s.classification == "pending" for s in app.block_states.values())

    # Simulate user action
    screen.action_mark_knowledge()  # Marks currently selected block

    # Verify state was updated
    selected_id = screen._get_selected_block_id()
    assert app.block_states[selected_id].classification == "knowledge"
    assert app.block_states[selected_id].source == "user"
```

---

## 3. Async LLM Streaming + UI Responsiveness

### Research Question

How to update Textual UI from background async tasks (LLM streaming) without blocking user keyboard input?

### Decision

**Use Textual's `asyncio.create_task()` for background LLM work + message posting for UI updates.**

The pattern:

1. Screen's `on_mount()` starts background task: `self.task = asyncio.create_task(self._stream_llm())`
2. Background task streams LLM responses and posts messages to update UI
3. Message handlers run on main event loop, update widgets reactively
4. User keyboard input is handled independently on main event loop (never blocked)
5. Cancel background tasks when screen is unmounted or user proceeds

### Rationale

**Textual is async-first**: Built on `asyncio`, designed for concurrent operations.

**Why this approach**:

- ✅ **Non-blocking UI**: User can navigate, type, press keys while LLM streams
- ✅ **Thread-safe updates**: Message handlers run on main event loop
- ✅ **Built-in pattern**: Textual's `Worker` class wraps this pattern
- ✅ **Cancellation support**: `task.cancel()` cleanly stops background work
- ✅ **Input latency <100ms**: Keyboard handling runs independently of LLM work

**Why not alternatives**:

- ❌ **Threading**: Harder to synchronize UI updates, not Textual's design
- ❌ **Synchronous polling**: Blocks UI, adds latency
- ❌ **Queue-based patterns**: More complex than necessary

### Implementation Guidance

#### Basic Pattern: Background Task + Message Posting

```python
from textual.screen import Screen
from textual.message import Message
from textual.widgets import Tree, TreeNode
import asyncio


class LLMClassificationComplete(Message):
    """Posted when LLM finishes classifying a block"""

    def __init__(self, block_id: str, classification: str, confidence: float):
        super().__init__()
        self.block_id = block_id
        self.classification = classification
        self.confidence = confidence


class Phase1Screen(Screen):

    async def on_mount(self) -> None:
        """Start background LLM task"""
        # Start task but don't await (runs in background)
        self.llm_task = asyncio.create_task(self._stream_llm_classifications())

    async def _stream_llm_classifications(self) -> None:
        """Background task: Stream LLM responses"""
        try:
            async for block_id, is_knowledge, confidence in self.extraction_app.llm_client.stream_extract(...):
                # Update shared state
                classification = "knowledge" if is_knowledge else "activity"
                self.extraction_app.block_states[block_id].classification = classification
                self.extraction_app.block_states[block_id].confidence = confidence

                # Post message to main event loop (thread-safe)
                self.post_message(
                    LLMClassificationComplete(block_id, classification, confidence)
                )
        except asyncio.CancelledError:
            # Task was cancelled (user moved to next screen)
            pass
        except Exception as e:
            # Handle errors
            self.post_message(ErrorOccurred(str(e)))

    async def on_llm_classification_complete(self, message: LLMClassificationComplete) -> None:
        """Handle message (runs on main event loop)"""
        # Update UI (safe to modify widgets here)
        tree = self.query_one("#block-tree", Tree)
        node = self._find_tree_node(tree, message.block_id)

        if message.classification == "knowledge":
            node.label = f"✓ {node.data['text']} ({message.confidence:.0%})"
        else:
            node.label = f"✗ {node.data['text']} ({message.confidence:.0%})"

        tree.refresh()

    async def on_unmount(self) -> None:
        """Clean up when screen is removed"""
        if hasattr(self, 'llm_task') and not self.llm_task.done():
            self.llm_task.cancel()
            try:
                await self.llm_task  # Wait for cancellation
            except asyncio.CancelledError:
                pass
```

#### Using Textual's Worker API (Higher-Level)

```python
from textual.worker import Worker, WorkerState


class Phase1Screen(Screen):

    async def on_mount(self) -> None:
        """Start background work using Worker API"""
        self.run_worker(self._stream_llm_classifications(), exclusive=True)

    @work(exclusive=True, thread=False)
    async def _stream_llm_classifications(self) -> None:
        """Worker decorator handles task management"""
        async for block_id, is_knowledge, confidence in self.extraction_app.llm_client.stream_extract(...):
            # Update state
            self.extraction_app.block_states[block_id].classification = ...

            # Call UI update method directly (Worker handles thread safety)
            self.call_from_thread(self._update_tree_node, block_id)

    def _update_tree_node(self, block_id: str) -> None:
        """UI update (called safely from worker)"""
        tree = self.query_one("#block-tree", Tree)
        node = self._find_tree_node(tree, block_id)
        # Update node label...
        tree.refresh()
```

#### Handling User Input During Streaming

```python
class Phase1Screen(Screen):

    BINDINGS = [
        ("k", "mark_knowledge", "Mark as knowledge"),
        ("down", "cursor_down", "Move down"),
    ]

    def action_mark_knowledge(self) -> None:
        """User action: runs immediately on main event loop"""
        block_id = self._get_selected_block_id()

        # Mark as user-overridden (LLM won't change this)
        self.extraction_app.block_states[block_id] = BlockState(
            block_id=block_id,
            classification="knowledge",
            confidence=None,
            source="user"  # LOCK this decision
        )

        # Update UI immediately (no waiting for LLM)
        self._update_tree_node(block_id)

    async def _stream_llm_classifications(self) -> None:
        """Background LLM task respects user overrides"""
        async for block_id, is_knowledge, confidence in ...:
            # Check if user has already decided
            if self.extraction_app.block_states[block_id].source == "user":
                # Skip this block, user has final say
                continue

            # Update LLM decision
            self.extraction_app.block_states[block_id] = ...
            self.post_message(...)
```

#### Guaranteeing <100ms Input Response Time

```python
from textual.app import App


class ExtractionApp(App):
    """App with responsive input handling"""

    async def on_key(self, event):
        """Global key handler (runs before screen bindings)"""
        # Log input latency
        import time
        start = time.perf_counter()

        # Let screen handle key
        result = await super().on_key(event)

        # Verify latency
        latency_ms = (time.perf_counter() - start) * 1000
        if latency_ms > 100:
            self.log.warning(f"Input latency {latency_ms:.1f}ms exceeded target")

        return result
```

### Performance Characteristics

**Verified design**:

- **Main event loop**: Handles keyboard input, screen updates, message passing
- **Background tasks**: Stream LLM responses, parse NDJSON, update state
- **Message queue**: Thread-safe communication between async tasks and UI

**Expected latencies**:

- Keyboard input → Action handler: **<10ms** (main event loop)
- LLM chunk arrives → UI update: **<50ms** (message posting + handler)
- User action → Visual feedback: **<20ms** (synchronous widget update)

**Concurrency model**:

```
Main Event Loop (async)
├── Keyboard input handlers ──────────> <10ms latency
├── Message handlers ─────────────────> <50ms latency
├── Widget updates (reactive) ────────> <20ms latency
└── Background tasks
    ├── LLM streaming (async) ────────> Non-blocking
    ├── NDJSON parsing ───────────────> Non-blocking
    └── State updates ────────────────> Post messages to main loop
```

### Best Practices

1. **Always use `asyncio.create_task()`**: Never block with `await` in `on_mount()`
2. **Post messages for UI updates**: Don't modify widgets directly from background tasks
3. **Cancel tasks on unmount**: Prevent memory leaks and zombie tasks
4. **Check user overrides before LLM updates**: User has final say
5. **Log long-running operations**: Help debug performance issues
6. **Use `Worker` API for complex tasks**: Better lifecycle management

### Testing Responsiveness

```python
import pytest
import asyncio
from textual.app import App
from textual.pilot import Pilot


@pytest.mark.asyncio
async def test_keyboard_input_during_streaming():
    """Verify UI remains responsive during LLM streaming"""

    app = ExtractionApp(journal_entry=mock_journal, config=mock_config)

    async with app.run_test() as pilot:
        # Start streaming (slow, takes 5 seconds)
        screen = app.screen
        screen.llm_task = asyncio.create_task(slow_stream())

        # User presses key immediately
        await pilot.press("k")

        # Verify action was processed without waiting for stream
        assert app.block_states["block1"].source == "user"

        # Stream should still be running
        assert not screen.llm_task.done()


async def slow_stream():
    """Simulates slow LLM streaming"""
    for i in range(100):
        await asyncio.sleep(0.05)  # 50ms per chunk = 5s total
```

---

## 4. Tree Widget Customization

### Research Question

Can Textual's built-in `Tree` widget display custom icons (✓/✗/⊙/⊗) with confidence scores and hierarchical indentation?

### Decision

**Yes, use `Tree` widget with custom labels and styling.**

The approach:

1. Use Textual's `Tree` widget (supports hierarchical data natively)
2. Set node labels to Rich-formatted text with icons and confidence: `Text("✓ Block content", style="green") + Text(" 92%", style="dim")`
3. Use Tree's `data` attribute to store block metadata (ID, state)
4. Update node labels dynamically as state changes
5. Apply CSS classes for color-coded confidence levels

### Rationale

**Textual's Tree is flexible**: Supports Rich text formatting, custom styling, metadata storage.

**Why this approach**:

- ✅ **Built-in**: No custom widget needed
- ✅ **Rich text support**: Icons, colors, bold, dim all work
- ✅ **Hierarchical display**: Matches journal block structure
- ✅ **Keyboard navigation**: Up/down/expand/collapse built-in
- ✅ **Reactive updates**: Can refresh labels dynamically

**Why not alternatives**:

- ❌ **Custom widget from scratch**: Reinventing the wheel, more bugs
- ❌ **Static widget**: No built-in keyboard navigation
- ❌ **ListView**: Doesn't support hierarchy

### Implementation Guidance

#### Building Tree from Journal Blocks

```python
from textual.widgets import Tree, TreeNode
from rich.text import Text


class Phase1Screen(Screen):

    def compose(self):
        yield Tree("Journal Entry", id="block-tree")

    async def on_mount(self):
        tree = self.query_one("#block-tree", Tree)
        tree.show_root = False  # Hide root, show only journal blocks

        # Build tree from journal structure
        self._populate_tree(tree.root, self.extraction_app.journal_entry.root_blocks)

    def _populate_tree(self, parent: TreeNode, blocks: list[LogseqBlock]) -> None:
        """Recursively build tree from journal blocks"""
        for block in blocks:
            # Create node with initial state
            label = self._format_block_label(block.id)
            node = parent.add(label, data={"block_id": block.id, "block": block})

            # Add children recursively
            if block.children:
                self._populate_tree(node, block.children)

    def _format_block_label(self, block_id: str) -> Text:
        """Format label based on current state"""
        state = self.extraction_app.block_states[block_id]
        block = self._get_block_by_id(block_id)

        # Truncate long text
        text = block.content.split('\n')[0][:60]
        if len(block.content) > 60:
            text += "..."

        # Build formatted label
        label = Text()

        # Icon
        if state.source == "user":
            if state.classification == "knowledge":
                label.append("⊙ ", style="bold cyan")  # User-marked knowledge
            else:
                label.append("⊗ ", style="bold yellow")  # User-marked activity
        elif state.classification == "knowledge":
            label.append("✓ ", style="green")
        elif state.classification == "activity":
            label.append("✗ ", style="dim")
        else:
            label.append("? ", style="yellow")  # Pending

        # Block text
        label.append(text, style="default")

        # Confidence score (if LLM-classified)
        if state.confidence is not None:
            confidence_pct = f"{state.confidence:.0%}"

            # Color-code by confidence level
            if state.confidence >= 0.90:
                style = "green dim"
            elif state.confidence >= 0.75:
                style = "dim"
            elif state.confidence >= 0.60:
                style = "yellow"
            else:
                style = "red"

            label.append(f"  {confidence_pct}", style=style)

            # Warning for low confidence
            if state.confidence < 0.60:
                label.append(" ⚠", style="red bold")

        return label

    def _update_tree_node(self, block_id: str) -> None:
        """Update node label when state changes"""
        tree = self.query_one("#block-tree", Tree)
        node = self._find_tree_node(tree.root, block_id)

        if node:
            node.label = self._format_block_label(block_id)
            tree.refresh()

    def _find_tree_node(self, parent: TreeNode, block_id: str) -> TreeNode | None:
        """Recursively find node by block ID"""
        for node in parent.children:
            if node.data and node.data.get("block_id") == block_id:
                return node

            # Search children
            found = self._find_tree_node(node, block_id)
            if found:
                return found

        return None
```

#### Handling Tree Navigation and Selection

```python
class Phase1Screen(Screen):

    BINDINGS = [
        ("up,k", "cursor_up", "Move up"),
        ("down,j", "cursor_down", "Move down"),
        ("right,l", "expand", "Expand"),
        ("left,h", "collapse", "Collapse"),
    ]

    def action_cursor_up(self) -> None:
        tree = self.query_one("#block-tree", Tree)
        tree.action_cursor_up()

    def action_cursor_down(self) -> None:
        tree = self.query_one("#block-tree", Tree)
        tree.action_cursor_down()

    def action_expand(self) -> None:
        tree = self.query_one("#block-tree", Tree)
        if tree.cursor_node:
            tree.cursor_node.expand()

    def action_collapse(self) -> None:
        tree = self.query_one("#block-tree", Tree)
        if tree.cursor_node:
            tree.cursor_node.collapse()

    def _get_selected_block_id(self) -> str:
        """Get currently selected block ID"""
        tree = self.query_one("#block-tree", Tree)
        if tree.cursor_node and tree.cursor_node.data:
            return tree.cursor_node.data["block_id"]
        raise ValueError("No block selected")
```

#### Styling with CSS

```python
class ExtractionApp(App):

    CSS = """
    Tree {
        height: 100%;
        padding: 1 2;
    }

    Tree:focus {
        border: solid $accent;
    }

    TreeNode {
        padding: 0 1;
    }

    TreeNode.-hover {
        background: $boost;
    }
    """
```

### Visual Examples

**Pending blocks** (waiting for LLM):

```
? Had interesting discussion with Sarah about microservices
  ? The key insight is that bounded contexts matter more than size
```

**LLM-classified blocks**:

```
✓ Read paper on vector databases  92%
  ✓ ChromaDB uses HNSW for approximate nearest neighbor search  90%
  ✓ Trade-off between recall and query speed via ef_construction  85%
```

**User-overridden blocks**:

```
⊙ Important insight from team meeting
  ⊙ This is definitely knowledge
⊗ Dentist appointment at 3pm
```

**Low-confidence warning**:

```
✗ Read the paper  58% ⚠
  ← Review this: low confidence
```

### Best Practices

1. **Use Rich `Text` objects**: More powerful than plain strings
2. **Truncate long content**: Keep labels readable (max 60-80 chars)
3. **Store metadata in `node.data`**: Don't parse labels to get IDs
4. **Color-code confidence**: Visual hierarchy (green > yellow > red)
5. **Update labels, don't rebuild tree**: More efficient
6. **Support keyboard navigation**: Bind to Tree's built-in actions

---

## 5. Keyboard Shortcut Conflicts

### Research Question

How to handle conflicting keybindings (e.g., 'k' for vim-up vs 'K' for mark-knowledge) while supporting both vim-style navigation and single-letter actions?

### Decision

**Use case-sensitive bindings with explicit priority ordering.**

The approach:

1. Lowercase `k` = vim-style "move up" (cursor navigation)
2. Uppercase `K` = "mark as knowledge" (state change action)
3. Use Textual's binding priority system to resolve conflicts
4. Provide arrow key alternatives for all navigation (accessibility)
5. Show only the most common/important bindings in Footer (avoid clutter)

### Rationale

**Textual supports case sensitivity**: `"k"` and `"K"` are different bindings.

**Why this approach**:

- ✅ **Vim users happy**: `j/k` for navigation works as expected
- ✅ **Discoverable**: Arrow keys always work, shown in Footer
- ✅ **Mnemonic actions**: `K` = Knowledge, `A` = Activity
- ✅ **Explicit priority**: Later bindings override earlier ones
- ✅ **Accessible**: Don't require vim knowledge

**Why not alternatives**:

- ❌ **Modal keymaps**: Too complex, confusing for users
- ❌ **Conflicting bindings**: Users press wrong key by accident
- ❌ **No vim support**: Alienates power users

### Implementation Guidance

#### Case-Sensitive Bindings

```python
from textual.screen import Screen
from textual.binding import Binding


class Phase1Screen(Screen):

    BINDINGS = [
        # Navigation (lowercase, vim-style)
        Binding("j", "cursor_down", "Move down", show=False),  # Don't clutter footer
        Binding("k", "cursor_up", "Move up", show=False),
        Binding("h", "collapse", "Collapse", show=False),
        Binding("l", "expand", "Expand", show=False),

        # Navigation (arrows, always shown)
        Binding("up", "cursor_up", "Move up", show=True),
        Binding("down", "cursor_down", "Move down", show=True),
        Binding("left", "collapse", "Collapse", show=True),
        Binding("right", "expand", "Expand", show=True),

        # Actions (uppercase, mnemonic)
        Binding("K", "mark_knowledge", "Keep as knowledge", show=True),
        Binding("A", "mark_activity", "Mark as activity", show=True),
        Binding("R", "reset", "Reset", show=True),

        # Special keys
        Binding("enter", "continue", "Continue", show=True),
        Binding("escape,q", "quit", "Quit", show=True),
    ]

    def action_cursor_up(self) -> None:
        """Move cursor up (triggered by 'k' or '↑')"""
        tree = self.query_one("#block-tree", Tree)
        tree.action_cursor_up()

    def action_mark_knowledge(self) -> None:
        """Mark block as knowledge (triggered by 'K')"""
        block_id = self._get_selected_block_id()
        # ... update state
```

#### Binding Priority and Conflicts

```python
class Phase1Screen(Screen):

    # Bindings are processed in REVERSE order (last wins)
    BINDINGS = [
        # Low priority: Vim navigation
        ("k", "cursor_up", "Move up"),

        # High priority: Mark as knowledge
        # This would conflict, but we use uppercase 'K' instead
        ("K", "mark_knowledge", "Keep as knowledge"),
    ]

    # If there were a real conflict, Textual would:
    # 1. Warn in console
    # 2. Use the LAST defined binding (higher in list)
```

#### Conditional Bindings (Advanced)

```python
from textual.binding import Binding


class Phase3Screen(Screen):
    """Phase 3: Different bindings based on UI state"""

    def get_bindings(self):
        """Dynamically compute bindings based on state"""
        bindings = [
            Binding("up,k", "cursor_up", "Move up"),
            Binding("down,j", "cursor_down", "Move down"),
        ]

        # Only show "Edit text" if a non-skipped decision is selected
        selected = self._get_selected_decision()
        if selected and selected.action != "skip":
            bindings.append(Binding("E", "edit_text", "Edit text", show=True))

        # Only show "Pick location" for APPEND_CHILD/UPDATE actions
        if selected and selected.action in ("add_under", "replace"):
            bindings.append(Binding("T", "pick_location", "Pick location", show=True))

        return bindings
```

#### Footer Display Strategy

```python
class Phase1Screen(Screen):

    BINDINGS = [
        # SHOWN in footer (show=True or default)
        ("up", "cursor_up", "↑ Move up"),
        ("down", "cursor_down", "↓ Move down"),
        ("K", "mark_knowledge", "K Keep"),
        ("A", "mark_activity", "A Skip"),
        ("enter", "continue", "Enter Continue"),

        # HIDDEN in footer (show=False) but still work
        ("k", "cursor_up", "Move up", False),
        ("j", "cursor_down", "Move down", False),
        ("h", "collapse", "Collapse", False),
        ("l", "expand", "Expand", False),
    ]
```

Expected footer:

```
↑ Move up    ↓ Move down    K Keep    A Skip    Enter Continue
```

Vim users can still use `j/k`, but it's not shown (reduces clutter).

### Key Binding Conventions

**Navigation** (always lowercase, vim-style optional):

- `j` / `↓` = Move down
- `k` / `↑` = Move up
- `h` / `←` = Collapse / Move left
- `l` / `→` = Expand / Move right

**Actions** (uppercase, mnemonic):

- `K` = Keep as knowledge
- `A` = Mark as activity
- `R` = Reset / Review
- `E` = Edit
- `T` = Pick target/location
- `S` = Save
- `Q` = Quit

**Special** (standard terminal conventions):

- `Enter` = Confirm / Continue
- `Escape` = Cancel / Back
- `Ctrl+C` = Quit (always)
- `Space` = Toggle / Select
- `Tab` = Switch focus

### Best Practices

1. **Always provide arrow key alternatives**: Accessibility and discoverability
2. **Use uppercase for actions**: Avoids vim navigation conflicts
3. **Show only essential bindings in footer**: Max 5-7 items
4. **Group related bindings**: Navigation together, actions together
5. **Use `show=False` for power-user shortcuts**: Keep footer clean
6. **Test both lowercase and uppercase**: Verify no conflicts

### Testing Keyboard Conflicts

```python
import pytest
from textual.pilot import Pilot


@pytest.mark.asyncio
async def test_vim_navigation_and_actions():
    """Verify lowercase 'k' and uppercase 'K' work as expected"""

    app = ExtractionApp(journal_entry=mock_journal, config=mock_config)

    async with app.run_test() as pilot:
        screen = app.screen
        tree = screen.query_one("#block-tree", Tree)

        # Initial cursor position
        initial_node = tree.cursor_node

        # Press lowercase 'k' → should move cursor up
        await pilot.press("k")
        assert tree.cursor_node != initial_node  # Cursor moved

        # Press uppercase 'K' → should mark as knowledge
        await pilot.press("K")
        block_id = screen._get_selected_block_id()
        assert app.block_states[block_id].classification == "knowledge"
```

---

## 6. Error Recovery in Streaming

### Research Question

How to handle malformed NDJSON lines, network errors, and LLM failures without crashing the TUI?

### Decision

**Use try/except at multiple levels with graceful degradation and user notification.**

The strategy:

1. **NDJSON parsing errors**: Log warning, skip line, continue with next line
2. **Network errors**: Retry with exponential backoff, show error banner if persistent
3. **LLM response errors**: Mark affected blocks as "error", allow user to retry or skip
4. **Partial successes**: Process all valid responses, summarize errors at end
5. **User control**: Always allow user to proceed despite errors

### Rationale

**LLMs are unreliable**: They may output malformed JSON, incomplete responses, or refuse requests.

**Why this approach**:

- ✅ **Partial success better than total failure**: User gets most of their work done
- ✅ **Clear error communication**: User knows what failed and why
- ✅ **Non-blocking**: Errors don't stop entire pipeline
- ✅ **User control**: Can retry, skip, or manually fix
- ✅ **Traceable**: All errors logged for debugging

**Why not alternatives**:

- ❌ **Crash on first error**: Loses all work
- ❌ **Silent failures**: User doesn't know what's wrong
- ❌ **Automatic retries only**: May hang indefinitely

### Implementation Guidance

#### NDJSON Parsing Error Handling

```python
import logging
import json


async def parse_ndjson_stream(stream: AsyncIterator[str]) -> AsyncIterator[dict]:
    """Parse NDJSON with graceful error recovery"""

    buffer = ""
    line_number = 0

    async for chunk in stream:
        buffer += chunk

        while '\n' in buffer:
            line_end = buffer.index('\n')
            complete_line = buffer[:line_end].strip()
            buffer = buffer[line_end + 1:]
            line_number += 1

            if not complete_line:
                continue  # Skip empty lines silently

            try:
                obj = json.loads(complete_line)

                # Validate expected fields
                if not isinstance(obj, dict):
                    logging.warning(
                        f"Line {line_number}: Expected JSON object, got {type(obj).__name__}"
                    )
                    continue

                yield obj

            except json.JSONDecodeError as e:
                # Log error with context
                logging.warning(
                    f"Line {line_number}: Malformed JSON: {complete_line[:100]}... "
                    f"Error at position {e.pos}: {e.msg}"
                )
                # Don't yield anything, continue to next line

    # Handle incomplete final line
    if buffer.strip():
        line_number += 1
        try:
            obj = json.loads(buffer.strip())
            yield obj
        except json.JSONDecodeError as e:
            logging.warning(
                f"Line {line_number} (incomplete): Could not parse final line: "
                f"{buffer[:100]}... Error: {e.msg}"
            )
```

#### Network Error Handling with Retry

```python
import httpx
import asyncio
from typing import AsyncIterator


async def stream_with_retry(
    endpoint: str,
    payload: dict,
    headers: dict,
    max_retries: int = 3,
    initial_backoff: float = 1.0
) -> AsyncIterator[str]:
    """Stream LLM response with exponential backoff retry"""

    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                async with client.stream("POST", endpoint, json=payload, headers=headers) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        yield line

                    # Success, exit retry loop
                    return

        except httpx.TimeoutException as e:
            logging.warning(f"Attempt {attempt + 1}/{max_retries}: Timeout - {e}")
            if attempt < max_retries - 1:
                backoff = initial_backoff * (2 ** attempt)
                logging.info(f"Retrying in {backoff:.1f}s...")
                await asyncio.sleep(backoff)
            else:
                raise  # Final attempt failed

        except httpx.HTTPStatusError as e:
            logging.error(f"HTTP error {e.response.status_code}: {e.response.text}")
            raise  # Don't retry on 4xx/5xx errors

        except httpx.RequestError as e:
            logging.warning(f"Attempt {attempt + 1}/{max_retries}: Network error - {e}")
            if attempt < max_retries - 1:
                backoff = initial_backoff * (2 ** attempt)
                await asyncio.sleep(backoff)
            else:
                raise
```

#### UI Error Notification

```python
from textual.widgets import Static
from textual.message import Message


class ErrorBanner(Static):
    """Dismissible error banner"""

    DEFAULT_CSS = """
    ErrorBanner {
        background: $error;
        color: $text;
        padding: 1 2;
        dock: top;
        height: auto;
    }
    """


class ErrorOccurred(Message):
    """Posted when an error occurs"""

    def __init__(self, error_message: str, recoverable: bool = True):
        super().__init__()
        self.error_message = error_message
        self.recoverable = recoverable


class Phase1Screen(Screen):

    async def _stream_llm_classifications(self) -> None:
        """Stream with error handling"""
        try:
            async for block_id, is_knowledge, confidence in self.extraction_app.llm_client.stream_extract(...):
                # Process classification
                ...

        except httpx.HTTPError as e:
            # Network error
            self.post_message(
                ErrorOccurred(
                    f"Network error: {e}. Some blocks may not be classified.",
                    recoverable=True
                )
            )

        except Exception as e:
            # Unexpected error
            logging.exception("Unexpected error during LLM streaming")
            self.post_message(
                ErrorOccurred(
                    f"Unexpected error: {e}. You can continue or quit.",
                    recoverable=True
                )
            )

    async def on_error_occurred(self, message: ErrorOccurred) -> None:
        """Show error banner"""
        banner = ErrorBanner(f"⚠ {message.error_message}")
        self.mount(banner)

        # Auto-dismiss after 10 seconds
        await asyncio.sleep(10)
        banner.remove()
```

#### Partial Success Pattern

```python
class Phase1Screen(Screen):

    async def _stream_llm_classifications(self) -> None:
        """Process all blocks, track successes and failures"""

        blocks_to_classify = list(self._get_pending_blocks())
        successes = []
        failures = []

        try:
            async for block_id, is_knowledge, confidence in self.extraction_app.llm_client.stream_extract(blocks_to_classify):
                try:
                    # Update state
                    self.extraction_app.block_states[block_id] = ...
                    successes.append(block_id)

                    # Update UI
                    self._update_tree_node(block_id)

                except Exception as e:
                    logging.exception(f"Error processing block {block_id}")
                    failures.append((block_id, str(e)))

        except Exception as e:
            # Streaming failed partway through
            logging.exception("Streaming error")
            remaining = set(b.id for b in blocks_to_classify) - set(successes)
            for block_id in remaining:
                failures.append((block_id, "Streaming interrupted"))

        # Show summary
        if failures:
            self.post_message(ErrorOccurred(
                f"Classified {len(successes)}/{len(blocks_to_classify)} blocks. "
                f"{len(failures)} failed (marked as pending).",
                recoverable=True
            ))
```

#### Retry Mechanism

```python
class Phase1Screen(Screen):

    BINDINGS = [
        # ... other bindings
        ("R", "retry_failed", "Retry failed blocks"),
    ]

    def action_retry_failed(self) -> None:
        """Retry LLM classification for failed/pending blocks"""
        pending = [
            block_id for block_id, state in self.extraction_app.block_states.items()
            if state.classification == "pending" and state.source == "llm"
        ]

        if pending:
            # Cancel existing task if running
            if hasattr(self, 'llm_task') and not self.llm_task.done():
                self.llm_task.cancel()

            # Start new task for pending blocks
            self.llm_task = asyncio.create_task(
                self._stream_llm_classifications_for_blocks(pending)
            )
        else:
            self.post_message(
                ErrorOccurred("No failed blocks to retry.", recoverable=True)
            )
```

### Error Types and Handling

| Error Type | Handling Strategy | User Impact |
|-----------|------------------|-------------|
| **Malformed JSON line** | Log warning, skip line, continue | One block not classified, others proceed |
| **Network timeout** | Retry 3x with backoff | Brief delay, automatic recovery |
| **Network failure** | Show error banner, mark pending blocks | User can retry or proceed manually |
| **HTTP 4xx/5xx** | Show error, don't retry | User adjusts config or skips |
| **Incomplete stream** | Process partial results | Show summary of what succeeded |
| **Invalid block ID in response** | Log warning, skip item | One block not classified |
| **Missing required field** | Log warning, skip item | One block not classified |
| **Unexpected exception** | Log traceback, show error banner | User can retry or quit |

### Best Practices

1. **Always log errors**: Include context (line number, block ID, etc.)
2. **Show user-friendly messages**: Not raw exception traces
3. **Provide retry mechanism**: Let user recover from transient errors
4. **Track partial successes**: Don't lose work from successful operations
5. **Test error paths**: Simulate network failures, malformed responses
6. **Use exponential backoff**: Don't hammer failing services
7. **Set reasonable timeouts**: Don't hang indefinitely

### Testing Error Scenarios

```python
import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_malformed_json_handling():
    """Verify parser skips malformed lines"""

    async def mock_stream():
        yield '{"valid": "json"}\n'
        yield '{invalid json}\n'  # Malformed
        yield '{"another": "valid"}\n'

    objects = []
    async for obj in parse_ndjson_stream(mock_stream()):
        objects.append(obj)

    # Should have parsed 2 valid objects, skipped 1 malformed
    assert len(objects) == 2
    assert objects[0] == {"valid": "json"}
    assert objects[1] == {"another": "valid"}


@pytest.mark.asyncio
async def test_network_retry():
    """Verify retry logic on network failure"""

    call_count = 0

    async def failing_request():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise httpx.TimeoutException("Timeout")
        else:
            return "Success"

    with patch('httpx.AsyncClient.stream', side_effect=failing_request):
        result = await stream_with_retry(...)

    assert call_count == 3  # Failed twice, succeeded on third attempt
```

---

## Summary of Decisions

### Technology Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| **TUI Framework** | Textual \^0.47.0 | Async-first, Rich integration, production-ready |
| **Streaming Format** | NDJSON | Simple buffering, incremental parsing, robust error handling |
| **Async Pattern** | `asyncio.create_task()` + message posting | Non-blocking UI, <100ms input latency |
| **Tree Display** | Textual `Tree` widget + Rich formatting | Built-in hierarchy, keyboard nav, custom styling |
| **Keyboard Bindings** | Case-sensitive with arrow alternatives | Vim-friendly, accessible, no conflicts |
| **Error Recovery** | Try/except at multiple levels + partial success | Graceful degradation, user control |

### Performance Targets

- ✅ **UI response time**: <500ms visual feedback on user input
- ✅ **Input handling**: <100ms latency during LLM streaming
- ✅ **NDJSON parsing**: <50ms per object (incremental)
- ✅ **Typical workflow**: <3 minutes for 11 root blocks, 22 total blocks

### Next Steps

1. **Implement NDJSON parser** (`src/logsqueak/llm/streaming.py`)
2. **Add HTTP streaming protocol handling** (`src/logsqueak/llm/providers/openai_compat.py`)
3. **Create shared state models** (`src/logsqueak/tui/models.py`)
4. **Build Phase1Screen prototype** with Tree widget and streaming
5. **Extend LLM client** with async streaming methods
6. **Write integration tests** for error recovery scenarios

---

**Research Status**: Complete | **Ready for**: Phase 1 (Design & Contracts)
