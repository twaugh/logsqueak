# Logsqueak Demos

Interactive demonstrations of Logsqueak TUI widgets.

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
