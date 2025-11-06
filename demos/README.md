# Logsqueak Demos

Interactive demonstrations of Logsqueak TUI widgets and screens.

## Phase 3 Integration Review Demo

Demonstrates the Phase 3 Integration Review workflow with sample knowledge blocks.

**Run:**
```bash
python demos/demo_phase3.py
```

**Features:**
- Knowledge block display with hierarchical context
- Decision list showing integration targets
- Target page preview with green bar insertion indicator
- Keyboard-driven navigation and approval workflow
- State tracking for completed/pending decisions

**Sample Data:**
- 3 knowledge blocks (Python, Docker, CI/CD topics)
- Multiple integration decisions per block
- Target pages with realistic Logseq structure

**Keyboard Controls:**
- `j` / `k` - Navigate between decisions
- `y` - Accept current decision
- `n` - Move to next knowledge block
- `a` - Accept all pending decisions for current block
- `q` - Quit

**Layout:**
- Left panel: Knowledge block content + decision list
- Right panel: Target page preview with insertion indicator
- Status bar: Progress tracking (completed/pending counts)

## TargetPagePreview Demo

Demonstrates the `TargetPagePreview` widget with Logseq markdown support.

**Run:**
```bash
python demo_target_preview.py
```

**Features:**
- Logseq `[[page links]]` syntax rendering
- Logseq `#tag` syntax rendering
- Standard markdown (headers, bullets, bold, links)
- Insertion point indicator (green bar)
- Interactive sample switching (press 1-4)
- Toggle insertion indicator (press 'i')

**Samples:**
1. Basic - Simple nested bullets
2. Formatted - Mixed formatting with Logseq syntax
3. Nested - Deep hierarchical structure
4. Mixed - Full project example with properties

**Key Learnings:**
- Textual Markdown widgets need to be yielded at the top level for proper sizing
- Use `call_later()` to defer widget updates until after mount
- `load_preview()` must be async and await `update()`
- Custom MarkdownIt parsers work with `parser_factory` parameter
- Line indicators require a composite widget with separate gutter (can't modify markdown content directly)
- Logseq properties need double-space line endings to render on separate lines in markdown
