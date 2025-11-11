# logseq-outline-parser

Parse and manipulate Logseq outline markdown files.

This package provides a Python parser for Logseq's outline-based markdown format, enabling programmatic manipulation of Logseq pages and journals.

## Features

- **Parse Logseq markdown** into structured AST (Abstract Syntax Tree)
- **Render back to markdown** with minimal changes guarantee
- **Preserve property order** - insertion order is sacred
- **Hybrid ID system** - support for both `id::` properties and content hashes
- **Full-context generation** - for semantic search and block identification
- **Graph path operations** - navigate Logseq directory structure

## Installation

```bash
pip install logseq-outline-parser
```

## Quick Start

```python
from logseq_outline import LogseqOutline

# Parse Logseq markdown
markdown = """- My bullet
  property:: value
  - Child bullet"""

outline = LogseqOutline.parse(markdown)

# Access blocks
root_block = outline.blocks[0]
print(root_block.content[0])  # "My bullet"
print(root_block.get_property("property"))  # "value"
print(root_block.children[0].content[0])  # "Child bullet"

# Modify and render
root_block.add_child("New child")
new_markdown = outline.render()
```

## Key Concepts

### LogseqBlock

Represents a single bullet in the outline with:

- `content`: All block lines (first line, continuation lines, properties)
- `indent_level`: Indentation level (0 = root)
- `block_id`: Persistent ID from `id::` property (if present)
- `children`: Child blocks

### LogseqOutline

The parsed outline structure with:

- `blocks`: Top-level blocks
- `frontmatter`: Lines before first bullet (page-level content)
- `indent_str`: Detected indentation (default: 2 spaces)

### Hybrid ID System

Blocks can be identified by:

1. Explicit `id::` property (UUID format)
2. Content hash (MD5) if no `id::` property exists

This enables precise block targeting while supporting existing Logseq content.

## API Reference

### Parsing

```python
# Parse with default settings (normalized indentation)
outline = LogseqOutline.parse(markdown_string)

# Parse with strict indent preservation (for write operations)
outline = LogseqOutline.parse(markdown_string, strict_indent_preservation=True)
```

**Indent Preservation Modes:**

- **Default mode** (`strict_indent_preservation=False`): Continuation line indentation is normalized. Outdent markers are NOT created. Use this for all read-only operations (LLM prompts, display, analysis).

- **Strict mode** (`strict_indent_preservation=True`): Exact indentation is preserved using internal outdent markers (`\x00N\x00`) for continuation lines with non-standard indentation. Only use when re-parsing content for modification and writing back to files.

### Rendering

```python
markdown = outline.render()  # Preserves exact structure and order
```

### Finding Blocks

```python
# Find by heading text
block = outline.find_heading("My Heading")

# Find by hybrid ID
block = outline.find_block_by_id("65f3a8e0-1234-5678-9abc-def012345678")
block = outline.find_block_by_id("a1b2c3d4...", page_name="My Page")
```

### Block Properties

```python
# Get property
value = block.get_property("property-name")

# Set property (preserves location or appends)
block.set_property("property-name", "value")

# Get hybrid ID
hybrid_id = block.get_hybrid_id(parents=[root, parent])
```

### Graph Operations

```python
from logseq_outline import GraphPaths
from pathlib import Path

graph = GraphPaths(Path("~/Documents/logseq-graph"))

# Navigate directories
journals_dir = graph.journals_dir  # ~/Documents/logseq-graph/journals
pages_dir = graph.pages_dir        # ~/Documents/logseq-graph/pages

# Get file paths
journal_path = graph.get_journal_path("2025_01_15")
page_path = graph.get_page_path("Project X")

# Check existence
if graph.journal_exists("2025_01_15"):
    ...

# List files
all_journals = graph.list_journals()  # Sorted chronologically
all_pages = graph.list_pages()        # Sorted alphabetically
```

## Design Principles

### Property Order Preservation (NON-NEGOTIABLE)

This parser **NEVER reorders block properties**. Insertion order is sacred in Logseq.

- Properties are stored in exact order from source
- Round-trip parsing preserves order exactly
- Uses Python 3.7+ dict ordering (insertion order guaranteed)

### Minimal Changes Guarantee

When rendering back to markdown:
- Exact order of all blocks preserved
- Original indentation maintained
- Properties never reordered
- Only targeted modifications applied

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Type checking
mypy src/
```

## License

GPLv3 - All code is licensed under GPLv3 regardless of authorship method.

## Part of Logsqueak Project

This package was extracted from [Logsqueak](https://github.com/twaugh/logsqueak), a CLI tool that extracts lasting knowledge from Logseq journal entries using LLM-powered analysis.
