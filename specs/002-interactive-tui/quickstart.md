# Interactive TUI Developer Quickstart Guide

**Feature**: Interactive TUI for Knowledge Extraction
**Created**: 2025-10-29
**Status**: Development Guide
**Related**: [spec.md](./spec.md) | [plan.md](./plan.md) | [interactive-tui-design.md](./interactive-tui-design.md) | [data-model.md](./data-model.md)

---

## Table of Contents

- [Running the TUI](#running-the-tui)
- [TUI Architecture](#tui-architecture)
- [Adding a New Screen](#adding-a-new-screen)
- [Custom Widgets](#custom-widgets)
- [Testing the TUI](#testing-the-tui)
- [Debugging](#debugging)
- [Best Practices](#best-practices)

---

## Running the TUI

### Environment Setup

**IMPORTANT**: Always activate the virtual environment before running any commands:

```bash
# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Verify Textual is installed
pip list | grep textual  # Should show textual ^0.47.0
```

### Running the Interactive Extraction

```bash
# Extract from today's journal (interactive mode)
logsqueak extract

# Extract specific date
logsqueak extract 2025-10-29

# Extract date range
logsqueak extract 2025-10-25..2025-10-29

# Enable verbose logging to see detailed pipeline steps
logsqueak extract --verbose 2025-10-29
```

### Configuration Requirements

The TUI uses your existing logsqueak configuration at `~/.config/logsqueak/config.yaml`:

```yaml
llm:
  endpoint: https://api.openai.com/v1
  api_key: sk-your-api-key-here
  model: gpt-4-turbo-preview  # Default model for all phases

  # Optional: Configure separate models for different phases
  # extractor_model: gpt-4-turbo-preview  # Phase 1 (defaults to model)
  # decider_model: gpt-3.5-turbo         # Phase 3.1 - faster/cheaper for decisions
  # reworder_model: gpt-4-turbo-preview  # Phase 3.2 - high quality for content

logseq:
  graph_path: ~/Documents/logseq-graph

rag:
  top_k: 10  # Number of similar chunks to retrieve (default: 10)
```

**First-time setup**: Make sure you've built the vector store index before extraction:

```bash
# Build/rebuild vector store index (required before first extraction)
logsqueak index rebuild

# Check index status
logsqueak index status
```

---

## TUI Architecture

### Screen Flow

The TUI is organized around 4 sequential screens corresponding to the extraction pipeline:

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│  Phase1     │      │  Phase2     │      │  Phase3     │      │  Phase4     │
│  Screen     │ ───> │  Screen     │ ───> │  Screen     │ ───> │  Screen     │
│             │      │             │      │             │      │             │
│ Knowledge   │      │ Candidate   │      │ Integration │      │ Writing     │
│ Extraction  │      │ Pages       │      │ Decisions   │      │ Changes     │
└─────────────┘      └─────────────┘      └─────────────┘      └─────────────┘
      │                     │                     │                     │
      └─────────────────────┴─────────────────────┴─────────────────────┘
                                  │
                          Shared ScreenState
```

**Phase 1 (Phase1Screen)**: Classify blocks as knowledge/activity

- User sees journal blocks in a tree structure
- LLM streams classifications (✓/✗ with confidence scores)
- User can override any classification (⊙/⊗ icons)

**Phase 2 (Phase2Screen)**: Review candidate pages (optional)

- RAG retrieves relevant pages for each knowledge block
- User can review and exclude irrelevant pages
- Auto-proceeds if user doesn't interact

**Phase 3 (Phase3Screen)**: Decide where to integrate knowledge

- LLM decides action for each (block, page) pair
- LLM rewords content for evergreen storage
- User can change actions, edit text, pick locations
- Grouped by destination page

**Phase 4 (Phase4Screen)**: Execute writes and show results

- Groups operations by page
- Shows progress as writes complete
- Displays completion summary with links

### Shared State Management (ScreenState Pattern)

All screens share a single `ScreenState` instance that flows through the entire session:

```python
from dataclasses import dataclass
from logsqueak.models.journal import JournalEntry
from logsqueak.config.config import Config
from logsqueak.llm.client import LLMClient

@dataclass
class ScreenState:
    """Shared state container for entire extraction session"""
    current_phase: int                                    # Active phase (1-4)
    journal_entry: JournalEntry                          # Loaded journal data
    block_states: dict[str, BlockState]                  # Phase 1: Block classifications
    candidates: dict[str, list[CandidatePage]]           # Phase 2: RAG results
    decisions: dict[tuple[str, str], IntegrationDecision] # Phase 3: Integration plans
    config: Config                                        # User configuration
    llm_client: LLMClient                                 # Shared LLM client

# Screens access state via constructor
class Phase1Screen(Screen):
    def __init__(self, state: ScreenState):
        super().__init__()
        self.state = state  # Shared mutable reference

    async def on_mount(self):
        # Initialize block_states (mutates shared state)
        for block in self._all_blocks():
            self.state.block_states[block.id] = BlockState(
                block_id=block.id,
                classification="pending",
                confidence=None,
                source="llm",
                llm_classification=None,  # LLM hasn't classified yet
                llm_confidence=None
            )
```

**Key principles**:

- **App owns state**: `ExtractionApp` creates `ScreenState` on mount
- **Screens mutate state**: Each screen updates its phase's data (block_states, candidates, decisions)
- **State flows forward**: Pass same `ScreenState` instance to each new screen
- **No duplication**: Single source of truth prevents sync issues

### NDJSON Streaming Integration

LLM responses stream as newline-delimited JSON objects, enabling real-time UI updates:

```python
async def parse_ndjson_stream(stream: AsyncIterator[str]) -> AsyncIterator[dict]:
    """
    Parse NDJSON from chunked async stream.

    Input: Token-by-token chunks ('{"', 'foo', '": ', '"bar', '"}', '\n', ...)
    Output: Complete JSON objects as they become parseable
    """
    buffer = ""

    async for chunk in stream:
        buffer += chunk

        # Process all complete lines in buffer
        while '\n' in buffer:
            line_end = buffer.index('\n')
            complete_line = buffer[:line_end].strip()
            buffer = buffer[line_end + 1:]

            if not complete_line:
                continue  # Skip empty lines

            try:
                obj = json.loads(complete_line)
                yield obj  # UI can update immediately
            except json.JSONDecodeError as e:
                logging.warning(f"Skipping malformed JSON: {complete_line[:60]}...")
                # Continue to next line (graceful degradation)
```

**Benefits**:

- Incremental parsing: UI updates as each object arrives
- Error recovery: Malformed lines don't crash the stream
- Simple buffering: Only buffer up to next newline character

### Async Task Management

Background LLM streaming runs concurrently with user input using asyncio:

```python
class Phase1Screen(Screen):
    async def on_mount(self) -> None:
        # Start LLM streaming in background (non-blocking)
        self.llm_task = asyncio.create_task(self._stream_extraction())

    async def _stream_extraction(self) -> None:
        """Background task: streams LLM responses"""
        try:
            async for block_id, is_knowledge, confidence in self.state.llm_client.stream_extract(...):
                # Update shared state
                if self.state.block_states[block_id].source != "user":
                    classification = "knowledge" if is_knowledge else "activity"
                    self.state.block_states[block_id] = BlockState(
                        block_id=block_id,
                        classification=classification,
                        confidence=confidence,
                        source="llm",
                        llm_classification=classification,  # Preserve LLM decision
                        llm_confidence=confidence
                    )
                    # Trigger UI refresh
                    self._update_tree_display(block_id)
        except asyncio.CancelledError:
            pass  # Task was cancelled (user moved to next screen)

    def action_mark_knowledge(self) -> None:
        """User action: runs immediately on main event loop"""
        block_id = self._get_selected_block_id()
        current_state = self.state.block_states[block_id]

        # Mark as user-overridden (LLM won't change this)
        self.state.block_states[block_id] = BlockState(
            block_id=block_id,
            classification="knowledge",
            confidence=None,
            source="user",  # LOCK this decision
            llm_classification=current_state.llm_classification,  # Preserve
            llm_confidence=current_state.llm_confidence
        )
        self._update_tree_display(block_id)

    async def on_unmount(self) -> None:
        """Clean up when screen is removed"""
        if hasattr(self, 'llm_task') and not self.llm_task.done():
            self.llm_task.cancel()
            try:
                await self.llm_task
            except asyncio.CancelledError:
                pass
```

**Performance guarantees**:

- User input handling: <100ms latency (runs on main event loop)
- LLM streaming: Non-blocking (runs in background task)
- UI updates: <50ms per update (message posting + handler)

---

## Adding a New Screen

Follow these steps to create a new TUI screen:

### 1. Create Screen File

Create a new file in `src/logsqueak/tui/screens/`:

```bash
touch src/logsqueak/tui/screens/my_screen.py
```

### 2. Inherit from textual.screen.Screen

```python
from textual.screen import Screen
from textual.widgets import Header, Footer, Static, Tree
from textual.containers import Container
from textual.binding import Binding
from logsqueak.tui.models import ScreenState

class MyScreen(Screen):
    """Description of what this screen does"""

    # Optional: Custom CSS styling
    CSS = """
    MyScreen {
        /* Textual CSS-like styling */
    }
    """

    # Required: Keyboard bindings
    BINDINGS = [
        Binding("up,k", "cursor_up", "Move up", show=True),
        Binding("down,j", "cursor_down", "Move down", show=True),
        Binding("enter", "continue", "Continue", show=True),
        Binding("escape,q", "quit", "Quit", show=True),
    ]

    def __init__(self, state: ScreenState):
        super().__init__()
        self.state = state  # Store shared state reference
```

### 3. Define BINDINGS

Bindings map keyboard shortcuts to action methods:

```python
BINDINGS = [
    # Navigation (lowercase, vim-style optional)
    Binding("j", "cursor_down", "Move down", show=False),  # Hidden in footer
    Binding("k", "cursor_up", "Move up", show=False),

    # Navigation (arrows, always shown)
    Binding("up", "cursor_up", "↑ Move up", show=True),
    Binding("down", "cursor_down", "↓ Move down", show=True),

    # Actions (uppercase, mnemonic)
    Binding("K", "mark_knowledge", "K Keep", show=True),
    Binding("A", "mark_activity", "A Skip", show=True),

    # Special keys
    Binding("enter", "continue", "Enter Continue", show=True),
    Binding("escape,q", "quit", "Quit", show=True),
]
```

**Conventions**:

- Lowercase `j/k/h/l`: Vim-style navigation (hidden in footer)
- Arrow keys: Always shown for discoverability
- Uppercase letters: Mnemonic actions (K=Knowledge, A=Activity)
- `show=True`: Display in footer, `show=False`: hidden power-user shortcut

### 4. Implement compose() Layout

The `compose()` method defines the widget layout:

```python
from textual.containers import Container, VerticalScroll
from textual.widgets import Header, Footer, Static, Tree

def compose(self):
    """Build UI layout using Textual widgets"""
    yield Header()

    yield Container(
        Static(f"Phase Title: {self.state.journal_entry.date}", id="phase-title"),
        Tree("Root Label", id="main-tree"),
        Static("", id="status-bar"),
        id="main-container"
    )

    yield Footer()
```

**Common widgets**:

- `Header()`: App title bar
- `Footer()`: Keyboard shortcuts display
- `Static()`: Non-interactive text display
- `Tree()`: Hierarchical data display (expandable/collapsible)
- `TextArea()`: Multi-line text editing
- `Container()`: Layout container for grouping widgets
- `VerticalScroll()`: Scrollable container

### 5. Add async on_mount() Initialization

The `on_mount()` lifecycle hook runs when the screen is displayed:

```python
async def on_mount(self) -> None:
    """Initialize screen state and start background tasks"""

    # Populate UI widgets
    tree = self.query_one("#main-tree", Tree)
    self._populate_tree(tree, self.state.journal_entry.root_blocks)

    # Initialize shared state for this phase
    # (mutate self.state.block_states, candidates, or decisions)

    # Start background LLM task (if needed)
    self.llm_task = asyncio.create_task(self._stream_llm_data())

    # Update status bar
    self._update_status()
```

**Important**: Use `asyncio.create_task()` for background work, don't `await` it directly or the UI will block.

### 6. Wire into App Navigation

Update `src/logsqueak/tui/app.py` to navigate to your screen:

```python
from logsqueak.tui.screens.my_screen import MyScreen

class ExtractionApp(App):
    def on_mount(self) -> None:
        # Start with your new screen
        self.push_screen(MyScreen(self.screen_state))

# Or from another screen:
class PreviousScreen(Screen):
    def action_continue(self) -> None:
        # Navigate to next screen
        self.app.push_screen(MyScreen(self.state))
```

### Complete Example: Minimal Screen

```python
from textual.screen import Screen
from textual.widgets import Header, Footer, Static
from textual.binding import Binding
from logsqueak.tui.models import ScreenState

class MinimalScreen(Screen):
    """Minimal example screen"""

    BINDINGS = [
        Binding("enter", "continue", "Continue"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, state: ScreenState):
        super().__init__()
        self.state = state

    def compose(self):
        yield Header()
        yield Static(f"Processing journal: {self.state.journal_entry.date}", id="title")
        yield Static("Press Enter to continue", id="help")
        yield Footer()

    async def on_mount(self):
        # Initialize any state
        self.state.current_phase = 2

    def action_continue(self):
        # Navigate to next screen
        from logsqueak.tui.screens.next_screen import NextScreen
        self.app.push_screen(NextScreen(self.state))

    def action_quit(self):
        self.app.exit()
```

---

## Custom Widgets

### When to Create a Widget

Create a custom widget when:

- You need to reuse a UI component across multiple screens
- A component has complex internal state or behavior
- You want to encapsulate widget-specific logic

**Don't create a widget if**: You're just grouping static content (use `Container` instead)

### BlockTree Widget Example

A reusable tree widget for displaying journal blocks with classification icons:

```python
from textual.widgets import Tree, TreeNode
from rich.text import Text
from logsqueak.tui.models import BlockState

class BlockTree(Tree):
    """Tree widget for displaying journal blocks with classification state"""

    def __init__(self, label: str, block_states: dict[str, BlockState]):
        super().__init__(label)
        self.block_states = block_states
        self.show_root = False  # Hide root node

    def populate_from_blocks(self, blocks: list):
        """Build tree from journal block hierarchy"""
        self._populate_recursive(self.root, blocks)

    def _populate_recursive(self, parent: TreeNode, blocks: list):
        for block in blocks:
            label = self._format_block_label(block.id)
            node = parent.add(label, data={"block_id": block.id, "block": block})

            # Add children recursively
            if block.children:
                self._populate_recursive(node, block.children)

    def _format_block_label(self, block_id: str) -> Text:
        """Format label based on current BlockState"""
        state = self.block_states.get(block_id)
        if not state:
            return Text("? Unknown block")

        # Get block content (truncate if long)
        block = self._get_block_by_id(block_id)
        text = block.content.split('\n')[0][:60]
        if len(block.content) > 60:
            text += "..."

        # Build formatted label with icon and confidence
        label = Text()

        # Icon
        if state.source == "user":
            icon = "⊙ " if state.classification == "knowledge" else "⊗ "
            style = "bold cyan" if state.classification == "knowledge" else "bold yellow"
            label.append(icon, style=style)
        elif state.classification == "knowledge":
            label.append("✓ ", style="green")
        elif state.classification == "activity":
            label.append("✗ ", style="dim")
        else:
            label.append("? ", style="yellow")  # Pending

        # Block text
        label.append(text, style="default")

        # Confidence score
        if state.confidence is not None:
            confidence_pct = f"  {state.confidence:.0%}"

            if state.confidence >= 0.90:
                style = "green dim"
            elif state.confidence >= 0.75:
                style = "dim"
            elif state.confidence >= 0.60:
                style = "yellow"
            else:
                style = "red"
                confidence_pct += " ⚠"

            label.append(confidence_pct, style=style)

        return label

    def update_block_label(self, block_id: str):
        """Refresh label for specific block (called after state change)"""
        node = self._find_node_by_id(self.root, block_id)
        if node:
            node.label = self._format_block_label(block_id)
            self.refresh()

    def _find_node_by_id(self, parent: TreeNode, block_id: str) -> TreeNode | None:
        """Recursively find tree node by block ID"""
        for node in parent.children:
            if node.data and node.data.get("block_id") == block_id:
                return node

            found = self._find_node_by_id(node, block_id)
            if found:
                return found

        return None

# Usage in a screen:
class Phase1Screen(Screen):
    def compose(self):
        yield Header()
        yield BlockTree("Journal Entry", self.state.block_states)
        yield Footer()

    async def on_mount(self):
        tree = self.query_one(BlockTree)
        tree.populate_from_blocks(self.state.journal_entry.root_blocks)

    def action_mark_knowledge(self):
        block_id = self._get_selected_block_id()
        current_state = self.state.block_states[block_id]

        # Update state
        self.state.block_states[block_id] = BlockState(
            block_id=block_id,
            classification="knowledge",
            confidence=None,
            source="user",
            llm_classification=current_state.llm_classification,
            llm_confidence=current_state.llm_confidence
        )

        # Refresh widget
        tree = self.query_one(BlockTree)
        tree.update_block_label(block_id)
```

### DecisionList Widget Example

A widget for displaying integration decisions grouped by page:

```python
from textual.widgets import Static
from textual.containers import VerticalScroll
from rich.text import Text

class DecisionList(VerticalScroll):
    """Grouped list of integration decisions by destination page"""

    def __init__(self, decisions: dict):
        super().__init__()
        self.decisions = decisions

    def populate(self):
        """Build grouped decision display"""
        # Group decisions by target page
        by_page = {}
        for (block_id, page), decision in self.decisions.items():
            if page not in by_page:
                by_page[page] = []
            by_page[page].append(decision)

        # Render each page group
        for page_name, page_decisions in sorted(by_page.items()):
            # Page header
            header = Static(
                self._format_page_header(page_name, page_decisions),
                classes="page-header"
            )
            self.mount(header)

            # Decision items
            for decision in page_decisions:
                if decision.action != "skip":
                    item = Static(
                        self._format_decision_item(decision),
                        classes="decision-item"
                    )
                    self.mount(item)

    def _format_page_header(self, page_name: str, decisions: list) -> Text:
        """Format page section header"""
        non_skip = sum(1 for d in decisions if d.action != "skip")

        header = Text()
        header.append("▼ ", style="bold")
        header.append(page_name, style="bold cyan")
        header.append(f"  {non_skip} blocks", style="dim")

        return header

    def _format_decision_item(self, decision) -> Text:
        """Format individual decision item"""
        item = Text()

        # Icon
        if decision.source == "user":
            item.append("  ⊙ ", style="bold cyan")
        else:
            item.append("  ✓ ", style="green")

        # Truncated refined text
        text_preview = decision.refined_text[:80]
        if len(decision.refined_text) > 80:
            text_preview += "..."
        item.append(f'"{text_preview}"', style="default")

        # Action and confidence
        item.append(f"\n    {self._friendly_action(decision.action)}", style="dim")
        item.append(f"  {decision.confidence:.0%}", style="dim")

        return item

    def _friendly_action(self, action: str) -> str:
        """Convert technical action to user-friendly label"""
        mapping = {
            "add_section": "Add as new section",
            "add_under": f"Add under '{decision.target_block_title}'",
            "replace": f"Replace '{decision.target_block_title}'",
            "skip": "Skip"
        }
        return mapping.get(action, action)
```

### Widget Composition Patterns

**Pattern 1: Container Widgets**

Use containers to group related widgets:

```python
class PhaseHeader(Container):
    """Reusable header for phase screens"""

    def __init__(self, phase_num: int, title: str, journal_date: str):
        super().__init__()
        self.phase_num = phase_num
        self.title = title
        self.journal_date = journal_date

    def compose(self):
        yield Static(f"Phase {self.phase_num}: {self.title}", classes="phase-title")
        yield Static(f"Journal: {self.journal_date}", classes="journal-date")

# Usage:
def compose(self):
    yield Header()
    yield PhaseHeader(1, "Knowledge Extraction", self.state.journal_entry.date)
    yield Tree(...)
    yield Footer()
```

**Pattern 2: Reactive Widgets**

Use reactive attributes for automatic UI updates:

```python
from textual.reactive import reactive

class ProgressBar(Static):
    """Progress indicator with reactive updates"""

    current = reactive(0)
    total = reactive(100)

    def watch_current(self, new_value: int):
        """Auto-update when current changes"""
        self.update(self._render_progress())

    def watch_total(self, new_value: int):
        """Auto-update when total changes"""
        self.update(self._render_progress())

    def _render_progress(self) -> str:
        percentage = (self.current / self.total * 100) if self.total > 0 else 0
        filled = int(percentage / 5)  # 20 blocks = 100%
        bar = "█" * filled + "░" * (20 - filled)
        return f"{bar} {self.current}/{self.total} ({percentage:.0f}%)"

# Usage in screen:
async def _stream_extraction(self):
    progress = self.query_one(ProgressBar)
    progress.total = len(blocks)

    async for block_id, classification, confidence in llm_stream:
        # Update state...
        progress.current += 1  # Auto-triggers watch_current()
```

---

## Testing the TUI

### Unit Tests for Screens

Test screen behavior in isolation:

```python
import pytest
from textual.pilot import Pilot
from logsqueak.tui.screens.phase1 import Phase1Screen
from logsqueak.tui.models import ScreenState

@pytest.mark.asyncio
async def test_phase1_screen_initialization():
    """Test Phase1Screen initializes block_states correctly"""

    # Create mock state
    state = ScreenState(
        current_phase=1,
        journal_entry=mock_journal_with_5_blocks,
        block_states={},
        candidates={},
        decisions={},
        config=mock_config,
        llm_client=mock_llm_client
    )

    # Create screen
    screen = Phase1Screen(state)

    # Simulate mounting
    await screen.on_mount()

    # Verify block_states populated
    assert len(state.block_states) == 5
    assert all(bs.classification == "pending" for bs in state.block_states.values())
    assert all(bs.source == "llm" for bs in state.block_states.values())

@pytest.mark.asyncio
async def test_user_override_locks_block():
    """Test user marking a block prevents LLM override"""

    state = create_test_state()
    screen = Phase1Screen(state)
    await screen.on_mount()

    # User marks block as knowledge
    screen._select_block("block-1")
    screen.action_mark_knowledge()

    # Verify state locked
    assert state.block_states["block-1"].classification == "knowledge"
    assert state.block_states["block-1"].source == "user"
    assert state.block_states["block-1"].confidence is None

    # Simulate LLM trying to override
    # (in real code, LLM stream checks source != "user" before updating)
    assert state.block_states["block-1"].source == "user"  # Still locked
```

### Integration Tests for Workflows

Test full screen flows with Textual's `run_test()` context manager:

```python
import pytest
from textual.pilot import Pilot
from logsqueak.tui.app import ExtractionApp

@pytest.mark.asyncio
async def test_full_phase1_workflow():
    """Test complete Phase 1 interaction flow"""

    app = ExtractionApp(
        journal_entry=mock_journal,
        config=mock_config
    )

    async with app.run_test() as pilot:
        # Screen should initialize with pending blocks
        screen = app.screen
        assert isinstance(screen, Phase1Screen)

        # Simulate user marking first block as knowledge
        await pilot.press("k")  # Move to first block (if not already selected)
        await pilot.press("K")  # Mark as knowledge

        # Verify state updated
        first_block_id = get_first_block_id(app.screen_state.journal_entry)
        assert app.screen_state.block_states[first_block_id].classification == "knowledge"
        assert app.screen_state.block_states[first_block_id].source == "user"

        # User continues to Phase 2
        await pilot.press("enter")

        # Verify screen transition
        assert isinstance(app.screen, Phase2Screen)
        assert app.screen_state.current_phase == 2

@pytest.mark.asyncio
async def test_keyboard_navigation():
    """Test vim-style and arrow key navigation"""

    app = ExtractionApp(mock_journal, mock_config)

    async with app.run_test() as pilot:
        tree = app.screen.query_one("#block-tree", Tree)
        initial_cursor = tree.cursor_node

        # Test lowercase 'j' moves down
        await pilot.press("j")
        assert tree.cursor_node != initial_cursor

        # Test lowercase 'k' moves up
        await pilot.press("k")
        assert tree.cursor_node == initial_cursor

        # Test arrow keys work too
        await pilot.press("down")
        assert tree.cursor_node != initial_cursor
```

### NDJSON Parsing Tests

Test streaming parser handles edge cases:

```python
import pytest
from logsqueak.llm.streaming import parse_ndjson_stream

@pytest.mark.asyncio
async def test_ndjson_parsing_incremental():
    """Test parser handles chunked stream correctly"""

    async def chunked_stream():
        # Simulate token-by-token streaming
        yield '{"block_id": "'
        yield 'abc123'
        yield '", "is_knowledge": '
        yield 'true'
        yield ', "confidence": 0.85}\n'
        yield '{"block_id": "def456", "is_knowledge": false, "confidence": 0.92}\n'

    objects = []
    async for obj in parse_ndjson_stream(chunked_stream()):
        objects.append(obj)

    assert len(objects) == 2
    assert objects[0] == {"block_id": "abc123", "is_knowledge": True, "confidence": 0.85}
    assert objects[1] == {"block_id": "def456", "is_knowledge": False, "confidence": 0.92}

@pytest.mark.asyncio
async def test_ndjson_parsing_malformed_line():
    """Test parser gracefully skips malformed JSON"""

    async def stream_with_errors():
        yield '{"valid": "json"}\n'
        yield '{invalid json}\n'  # Malformed
        yield '{"another": "valid"}\n'

    objects = []
    async for obj in parse_ndjson_stream(stream_with_errors()):
        objects.append(obj)

    # Should parse 2 valid objects, skip 1 malformed
    assert len(objects) == 2
    assert objects[0] == {"valid": "json"}
    assert objects[1] == {"another": "valid"}

@pytest.mark.asyncio
async def test_ndjson_parsing_empty_lines():
    """Test parser ignores empty lines"""

    async def stream_with_empty_lines():
        yield '{"first": "object"}\n'
        yield '\n'  # Empty line
        yield '\n'  # Another empty
        yield '{"second": "object"}\n'

    objects = [obj async for obj in parse_ndjson_stream(stream_with_empty_lines())]

    assert len(objects) == 2
```

### Testing Async Behavior

Test background tasks and concurrency:

```python
import pytest
import asyncio
from unittest.mock import AsyncMock

@pytest.mark.asyncio
async def test_llm_streaming_non_blocking():
    """Test LLM streaming doesn't block user input"""

    # Create mock LLM client with slow streaming
    async def slow_stream():
        for i in range(10):
            await asyncio.sleep(0.1)  # 100ms per block = 1s total
            yield f"block-{i}", True, 0.9

    mock_llm = AsyncMock()
    mock_llm.stream_extract = slow_stream

    state = ScreenState(
        llm_client=mock_llm,
        # ... other state
    )

    screen = Phase1Screen(state)
    await screen.on_mount()

    # Start streaming (runs in background)
    assert hasattr(screen, 'llm_task')
    assert not screen.llm_task.done()

    # User input should work immediately (not waiting for stream)
    import time
    start = time.perf_counter()
    screen.action_mark_knowledge()  # Should be instant
    latency = (time.perf_counter() - start) * 1000

    assert latency < 100, f"User input took {latency:.1f}ms (should be <100ms)"

    # Stream should still be running
    assert not screen.llm_task.done()
```

---

## Debugging

### Textual Dev Console

The Textual dev console shows real-time logs, events, and widget hierarchy:

```bash
# Terminal 1: Start the dev console
textual console

# Terminal 2: Run your TUI (logs will appear in console)
logsqueak extract 2025-10-29
```

**What you'll see**:

- All log messages (logging.info, logging.warning, etc.)
- Textual events (key presses, mouse moves, widget updates)
- Widget tree visualization
- CSS styling debug info
- Performance metrics

**Pro tip**: Add strategic logging in your code:

```python
class Phase1Screen(Screen):
    async def _stream_extraction(self):
        self.log("Starting LLM extraction stream")

        async for block_id, is_knowledge, confidence in stream:
            self.log(f"Classified {block_id}: {is_knowledge} ({confidence:.2f})")
            # Update state...

        self.log("LLM extraction stream complete")
```

### LLM Prompt Logging (PromptLogger)

View exactly what prompts are being sent to the LLM:

```bash
# Prompts are automatically logged to timestamped files
ls ~/.cache/logsqueak/prompts/

# View latest prompt log in real-time
tail -f ~/.cache/logsqueak/prompts/20251029_143022.log
```

**Log format**:

```
=== Interaction 1: extraction ===
Timestamp: 2025-10-29 14:30:22.154

[SYSTEM]
You are a knowledge extraction assistant...

[USER]
Analyze these journal blocks:

- Block 1: Had interesting discussion about microservices...
- Block 2: Dentist appointment at 3pm...

[RESPONSE]
{"block_id": "block-1", "is_knowledge": true, "confidence": 0.92}
{"block_id": "block-2", "is_knowledge": false, "confidence": 0.95}
```

**Enable custom log file**:

```bash
# Specify custom log location
logsqueak extract --prompt-log-file /tmp/debug-prompts.log 2025-10-29
```

### Viewing Logs

**Application logs** (if verbose logging enabled):

```bash
# Run with verbose flag
logsqueak extract --verbose 2025-10-29

# Logs show:
# - Phase transitions
# - Block classifications
# - RAG search results
# - Integration decisions
# - Write operations
```

**Python logging configuration** (for development):

```python
# In your screen or test file
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Now all logsqueak logs will appear
```

### Common Issues and Solutions

**Issue 1: Screen doesn't update after state change**

```python
# Problem: Mutated state but didn't refresh UI
self.state.block_states[block_id].classification = "knowledge"
# UI shows old state

# Solution: Trigger widget refresh
self._update_tree_display(block_id)
self.refresh()  # Force screen re-render
```

**Issue 2: Background task not running**

```python
# Problem: Awaited the task (blocks UI)
await self._stream_extraction()  # WRONG - blocks until complete

# Solution: Use create_task (runs in background)
self.llm_task = asyncio.create_task(self._stream_extraction())  # CORRECT
```

**Issue 3: Widget not found with query_one()**

```python
# Problem: Querying before widget is mounted
def compose(self):
    yield Tree("Journal", id="my-tree")

def __init__(self, state):
    super().__init__()
    tree = self.query_one("#my-tree", Tree)  # WRONG - not mounted yet

# Solution: Query in on_mount() or later
async def on_mount(self):
    tree = self.query_one("#my-tree", Tree)  # CORRECT - widget is mounted
```

**Issue 4: Keyboard shortcut conflicts**

```python
# Problem: Lowercase 'k' conflicts with vim-up and mark-knowledge
BINDINGS = [
    ("k", "cursor_up", "Move up"),
    ("k", "mark_knowledge", "Mark knowledge"),  # CONFLICT
]

# Solution: Use case-sensitive bindings
BINDINGS = [
    ("k", "cursor_up", "Move up"),        # Lowercase = navigation
    ("K", "mark_knowledge", "Mark knowledge"),  # Uppercase = action
]
```

**Issue 5: NDJSON parsing hangs on incomplete stream**

```python
# Problem: Stream ends without final newline
async def broken_stream():
    yield '{"incomplete": "object"'  # No closing brace or newline

# Solution: Parser handles incomplete final line gracefully
# (see parse_ndjson_stream implementation - discards incomplete buffer at end)
```

---

## Best Practices

### State Management Patterns

**1. App owns state, screens are views**

```python
# GOOD: App creates and owns ScreenState
class ExtractionApp(App):
    def __init__(self, journal_entry, config):
        super().__init__()
        self.screen_state = ScreenState(...)  # Single source of truth

# BAD: Screen creates its own state copy
class Phase1Screen(Screen):
    def __init__(self, state):
        super().__init__()
        self.my_state = copy.deepcopy(state)  # Duplication, sync issues
```

**2. Mutate state, then refresh UI**

```python
# GOOD: Clear separation of state mutation and UI update
def action_mark_knowledge(self):
    # 1. Update state
    self.state.block_states[block_id] = BlockState(...)

    # 2. Refresh UI
    self._update_tree_display(block_id)

# BAD: Mixing state and UI logic
def action_mark_knowledge(self):
    tree = self.query_one(Tree)
    node = tree.cursor_node
    node.label = "⊙ " + node.label  # Direct UI manipulation, state out of sync
```

**3. Use type hints for IDE support**

```python
# GOOD: Type hints enable autocomplete and error checking
class Phase1Screen(Screen):
    def __init__(self, state: ScreenState):
        super().__init__()
        self.state: ScreenState = state  # IDE knows state structure

    def _update_status(self) -> None:
        status = self.query_one("#status-bar", Static)
        # IDE can autocomplete: self.state.block_states, etc.

# BAD: No type hints
class Phase1Screen(Screen):
    def __init__(self, state):
        self.state = state  # IDE doesn't know what 'state' contains
```

### Error Handling Strategies

**1. Try/except at multiple levels**

```python
async def _stream_extraction(self) -> None:
    """Stream LLM classifications with multi-level error handling"""
    try:
        async for block_id, is_knowledge, confidence in self.state.llm_client.stream_extract(...):
            try:
                # Update state
                self.state.block_states[block_id] = BlockState(...)
                self._update_tree_display(block_id)
            except Exception as e:
                # Block-level error: log and continue with other blocks
                self.log.exception(f"Error processing block {block_id}")
                self.post_message(ErrorOccurred(f"Failed to classify block {block_id}"))

    except httpx.HTTPError as e:
        # Network error: show banner, allow user to retry
        self.post_message(ErrorOccurred(f"Network error: {e}", recoverable=True))

    except asyncio.CancelledError:
        # Task cancelled: clean exit
        self.log("Extraction task cancelled by user")
        raise  # Re-raise to complete cancellation

    except Exception as e:
        # Unexpected error: log traceback, show error screen
        self.log.exception("Unexpected error during extraction")
        self.post_message(ErrorOccurred(f"Unexpected error: {e}", recoverable=False))
```

**2. Graceful degradation for partial failures**

```python
async def _stream_extraction(self) -> None:
    """Process all blocks, track successes and failures"""
    successes = []
    failures = []

    try:
        async for block_id, is_knowledge, confidence in stream:
            try:
                # Process block
                successes.append(block_id)
            except Exception as e:
                # Log but continue
                failures.append((block_id, str(e)))

    except Exception as e:
        # Stream failed partway
        remaining = set(all_block_ids) - set(successes)
        failures.extend((bid, "Streaming interrupted") for bid in remaining)

    # Show summary
    if failures:
        self.post_message(ErrorOccurred(
            f"Classified {len(successes)}/{len(all_block_ids)} blocks. "
            f"{len(failures)} failed.",
            recoverable=True
        ))
```

**3. User-friendly error messages**

```python
# GOOD: Actionable error message
self.post_message(ErrorOccurred(
    "Network timeout connecting to LLM. "
    "Check your internet connection or try again later. "
    "Press R to retry or Enter to skip."
))

# BAD: Technical jargon
self.post_message(ErrorOccurred(
    "httpx.TimeoutException: Read timeout on POST /v1/chat/completions"
))
```

### Performance Considerations

**1. Use asyncio.create_task() for background work**

```python
# GOOD: Non-blocking background task
async def on_mount(self):
    self.llm_task = asyncio.create_task(self._stream_extraction())
    # UI remains responsive

# BAD: Blocking await
async def on_mount(self):
    await self._stream_extraction()  # UI frozen until complete
```

**2. Batch UI updates for efficiency**

```python
# GOOD: Batch multiple state updates, refresh once
def _process_llm_batch(self, results: list):
    for block_id, classification, confidence in results:
        self.state.block_states[block_id] = BlockState(...)

    # Single refresh for all changes
    self._refresh_tree()

# BAD: Refresh after each update
def _process_llm_batch(self, results: list):
    for block_id, classification, confidence in results:
        self.state.block_states[block_id] = BlockState(...)
        self._refresh_tree()  # Expensive refresh in loop
```

**3. Use O(1) lookups, avoid linear scans**

```python
# GOOD: Dict lookup O(1)
block_state = self.state.block_states[block_id]

# BAD: Linear scan O(n)
block_state = next(
    bs for bs in self.state.block_states.values()
    if bs.block_id == block_id
)
```

**4. Cancel tasks on screen exit**

```python
# GOOD: Clean up background tasks
async def on_unmount(self):
    if hasattr(self, 'llm_task') and not self.llm_task.done():
        self.llm_task.cancel()
        try:
            await self.llm_task
        except asyncio.CancelledError:
            pass

# BAD: Leave tasks running (memory leak)
async def on_unmount(self):
    pass  # llm_task keeps running forever
```

### Accessibility (Keyboard Navigation)

**1. Always provide arrow key alternatives**

```python
BINDINGS = [
    # Vim-style (power users)
    Binding("j", "cursor_down", "Move down", show=False),
    Binding("k", "cursor_up", "Move up", show=False),

    # Arrow keys (accessible to all users)
    Binding("down", "cursor_down", "↓ Move", show=True),
    Binding("up", "cursor_up", "↑ Move", show=True),
]
```

**2. Show essential bindings in footer**

```python
# GOOD: 5-7 essential bindings shown
BINDINGS = [
    Binding("up", "cursor_up", "↑ Move", show=True),
    Binding("K", "mark_knowledge", "K Keep", show=True),
    Binding("A", "mark_activity", "A Skip", show=True),
    Binding("enter", "continue", "Enter Continue", show=True),
    Binding("q", "quit", "Quit", show=True),
    # Power-user shortcuts hidden
    Binding("j", "cursor_down", show=False),
]

# BAD: Footer cluttered with 20 bindings
BINDINGS = [
    Binding("j", "cursor_down", "j down", show=True),
    Binding("k", "cursor_up", "k up", show=True),
    # ... 15 more shown bindings (overwhelming)
]
```

**3. Use mnemonic shortcuts**

```python
# GOOD: Memorable single-letter actions
Binding("K", "mark_knowledge", "K Keep as knowledge")
Binding("A", "mark_activity", "A Mark as activity")
Binding("E", "edit_text", "E Edit text")
Binding("T", "pick_target", "T Pick location")

# BAD: Arbitrary or conflicting shortcuts
Binding("X", "mark_knowledge", "X Mark")  # Not memorable
Binding("k", "mark_knowledge", "k Mark")  # Conflicts with vim navigation
```

**4. Provide help text for actions**

```python
def compose(self):
    yield Header()
    yield Static(
        "Use ↑/↓ to navigate, K to keep as knowledge, A to mark as activity",
        classes="help-text"
    )
    yield Tree(...)
    yield Footer()
```

---

## Additional Resources

### Textual Documentation

- [Official Textual Docs](https://textual.textualize.io/)
- [Textual Widget Guide](https://textual.textualize.io/widgets/)
- [Textual CSS Reference](https://textual.textualize.io/styles/)
- [Textual Events & Messages](https://textual.textualize.io/guide/events/)

### Project Documentation

- [spec.md](./spec.md) - Feature requirements and success criteria
- [plan.md](./plan.md) - Implementation plan and phase breakdown
- [interactive-tui-design.md](./interactive-tui-design.md) - UI mockups and design decisions
- [data-model.md](./data-model.md) - State models and data flow
- [research.md](./research.md) - Technical research and decisions
- [CLAUDE.md](/home/twaugh/devel/logsqueak/CLAUDE.md) - Project overview and architecture

### Related Code

- `src/logsqueak/tui/` - TUI implementation (screens, widgets, models)
- `src/logsqueak/llm/client.py` - LLM client with streaming support
- `src/logsqueak/extraction/extractor.py` - Knowledge extraction pipeline
- `src/logsqueak/integration/executor.py` - Write operation execution

---

**Quick Reference Card**

```
RUNNING
  logsqueak extract              # Today's journal
  logsqueak extract 2025-10-29   # Specific date
  logsqueak extract --verbose    # With detailed logging

DEVELOPMENT
  textual console                # Start dev console (Terminal 1)
  logsqueak extract              # Run TUI (Terminal 2)

TESTING
  pytest tests/tui/              # All TUI tests
  pytest tests/tui/test_screens.py -v  # Specific test file

DEBUGGING
  tail -f ~/.cache/logsqueak/prompts/latest.log  # View LLM prompts

KEYBOARD SHORTCUTS (Phase 1)
  ↑/↓ (or j/k)   Navigate blocks
  K              Mark as knowledge
  A              Mark as activity
  R              Undo (restore LLM decision if available, otherwise reset to pending)
  Enter          Continue to next phase
  Q              Quit
```

---

**End of Quickstart Guide**
