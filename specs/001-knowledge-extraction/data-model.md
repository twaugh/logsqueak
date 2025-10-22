# Data Model: Knowledge Extraction from Journals

**Phase**: 1 (Design & Contracts)
**Date**: 2025-10-22
**Source**: Entities from [spec.md](./spec.md)

## Overview

This document defines the core data structures for the knowledge extraction system. All entities are represented as Python dataclasses/Pydantic models for type safety and validation.

## Core Entities

### 1. JournalEntry

Represents a daily Logseq journal file.

**Attributes**:

- `date: date` - The journal date (e.g., 2025-01-15)
- `file_path: Path` - Absolute path to the journal markdown file
- `raw_content: str` - Complete file contents
- `outline: LogseqOutline` - Parsed outline structure (bullets, hierarchy, properties)
- `line_count: int` - Number of lines (for FR-019 2000-line limit enforcement)

**Validation Rules**:

- File must exist and be readable
- Must be valid Logseq markdown (bullets with indentation)
- Line count ≤ 2000 (warn and truncate if exceeded per FR-019)

**State Transitions**:

```
Created → Loaded → Parsed → Processed

```

**Python Representation**:

```python
from dataclasses import dataclass
from datetime import date
from pathlib import Path

@dataclass
class JournalEntry:
    date: date
    file_path: Path
    raw_content: str
    outline: 'LogseqOutline'
    line_count: int

    @classmethod
    def load(cls, file_path: Path) -> 'JournalEntry':
        """Load and parse journal entry from file."""
        raw_content = file_path.read_text()
        lines = raw_content.splitlines()

        if len(lines) > 2000:
            logger.warning(f"Entry exceeds 2000 lines ({len(lines)}), truncating")
            raw_content = '\n'.join(lines[:2000])

        outline = LogseqOutline.parse(raw_content)
        entry_date = parse_date_from_filename(file_path.name)

        return cls(
            date=entry_date,
            file_path=file_path,
            raw_content=raw_content,
            outline=outline,
            line_count=len(lines)
        )

```

---

### 2. KnowledgeBlock

A piece of information extracted from a journal that has lasting value (vs. temporary activity logs).

**Attributes**:

- `content: str` - The knowledge text content
- `source_date: date` - Journal date this was extracted from
- `confidence: float` - LLM confidence score (0.0-1.0) for activity vs. knowledge classification
- `target_page: str` - Page name where this should be integrated (e.g., "Project X")
- `target_section: Optional[List[str]]` - Hierarchical path to target location in outline (e.g., ["Projects", "Active", "Timeline"] for nested structure, or ["Timeline"] for top-level)
- `suggested_action: ActionType` - How to integrate: ADD_CHILD | CREATE_SECTION

**Validation Rules**:

- `content` must not be empty
- `confidence` must be 0.0 ≤ confidence ≤ 1.0
- `target_page` must be valid page name (no forbidden characters)
- `target_section` if provided, must be non-empty list of strings (each representing a level in the hierarchy)
- Source date must match journal entry date

**Uniqueness**: Combination of (content hash, target_page) for duplicate detection (FR-017)

**Python Representation**:

```python
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Optional, List
import hashlib

class ActionType(Enum):
    ADD_CHILD = "add_child"           # Add as child bullet under existing section
    CREATE_SECTION = "create_section" # Create new organizational bullet

@dataclass
class KnowledgeBlock:
    content: str
    source_date: date
    confidence: float
    target_page: str
    target_section: Optional[List[str]]  # Hierarchical path, e.g., ["Projects", "Active", "Timeline"]
    suggested_action: ActionType

    def content_hash(self) -> str:
        """Generate hash for duplicate detection."""
        return hashlib.md5(self.content.encode()).hexdigest()[:8]

    def provenance_link(self) -> str:
        """Generate Logseq page link to source journal."""
        return f"[[{self.source_date.isoformat()}]]"

    def with_provenance(self) -> str:
        """Return content with provenance link appended."""
        return f"{self.content} {self.provenance_link()}"

    def section_path(self) -> str:
        """Format section path for display."""
        if not self.target_section:
            return "(page root)"
        return " > ".join(self.target_section)

```

---

### 3. TargetPage

An existing Logseq page where knowledge will be integrated.

**Attributes**:

- `name: str` - Page name (e.g., "Project X")
- `file_path: Path` - Absolute path to page markdown file
- `outline: LogseqOutline` - Parsed outline structure
- `organizational_convention: ConventionType` - Detected page style (PLAIN_BULLETS | HEADING_BULLETS | MIXED)

**Validation Rules**:

- Page file must exist (FR-009: report if missing)
- Must be valid Logseq outline markdown
- Name must match filename (case-insensitive with URL encoding)

**Relationships**:

- One TargetPage can have many KnowledgeBlocks
- Parent-child relationship via outline structure

**Python Representation**:

```python
from dataclasses import dataclass
from pathlib import Path
from enum import Enum

class ConventionType(Enum):
    PLAIN_BULLETS = "plain"     # Uses "- Section"
    HEADING_BULLETS = "heading" # Uses "- ## Section"
    MIXED = "mixed"              # Mix of both styles

@dataclass
class TargetPage:
    name: str
    file_path: Path
    outline: 'LogseqOutline'
    organizational_convention: ConventionType

    @classmethod
    def load(cls, graph_path: Path, page_name: str) -> Optional['TargetPage']:
        """Load target page from Logseq graph."""
        # Logseq stores pages as "Page Name.md" or URL-encoded variants
        file_path = graph_path / "pages" / f"{page_name}.md"

        if not file_path.exists():
            return None

        outline = LogseqOutline.parse(file_path.read_text())
        convention = detect_convention(outline)

        return cls(
            name=page_name,
            file_path=file_path,
            outline=outline,
            organizational_convention=convention
        )

    def find_section(self, section_name: str) -> Optional['LogseqBlock']:
        """Find section by heading text."""
        return self.outline.find_heading(section_name)

    def has_duplicate(self, knowledge: KnowledgeBlock) -> bool:
        """Check if knowledge already exists on page (FR-017)."""
        # Simple content match - can be enhanced with semantic similarity
        return knowledge.content in self.outline.render()

```

---

### 4. LogseqOutline

Parsed representation of Logseq's outline-based markdown structure.

**Attributes**:

- `blocks: List[LogseqBlock]` - Top-level blocks in document
- `source_text: str` - Original markdown for debugging

**Operations**:

- `parse(markdown: str) -> LogseqOutline` - Parse markdown into outline structure
- `find_heading(text: str) -> Optional[LogseqBlock]` - Locate section by heading
- `add_child(parent: LogseqBlock, content: str)` - Add bullet under parent
- `create_section(heading: str, level: int)` - Insert new organizational bullet
- `render() -> str` - Convert back to Logseq markdown

**Python Representation**:

```python
from dataclasses import dataclass, field
from typing import List, Optional
from markdown_it import MarkdownIt

@dataclass
class LogseqBlock:
    """Single bullet in outline with children.

    IMPORTANT: Preserves exact structure and order from source.
    - Properties must never be reordered (insertion order preserved)
    - Children can be inserted where appropriate (targeted, minimal changes)
    """
    content: str
    indent_level: int
    properties: dict = field(default_factory=dict)  # Preserves insertion order (Python 3.7+)
    children: List['LogseqBlock'] = field(default_factory=list)

    # Preserve original formatting details
    _original_line: Optional[str] = field(default=None, repr=False)  # For exact round-trip

    def add_child(self, content: str, position: Optional[int] = None):
        """Add child bullet with proper indentation.

        Args:
            content: The bullet content
            position: Optional index to insert at (None = append to end)
        """
        child = LogseqBlock(
            content=content,
            indent_level=self.indent_level + 1
        )
        if position is None:
            self.children.append(child)
        else:
            self.children.insert(position, child)
        return child

@dataclass
class LogseqOutline:
    blocks: List[LogseqBlock]
    source_text: str

    @classmethod
    def parse(cls, markdown: str) -> 'LogseqOutline':
        """Parse Logseq markdown into outline structure.

        IMPORTANT: Preserves exact order of properties:
        - Blocks appear in same order as source
        - Properties MUST maintain insertion order (dict preserves order in Python 3.7+)
        - Children preserve their original sequence (can be modified with targeted inserts)
        - Whitespace and formatting captured in _original_line for round-trip
        """
        # Use markdown-it-py with custom Logseq plugin
        md = MarkdownIt()
        tokens = md.parse(markdown)
        blocks = cls._build_hierarchy(tokens)
        return cls(blocks=blocks, source_text=markdown)

    def find_heading(self, text: str) -> Optional[LogseqBlock]:
        """Recursively search for heading containing text."""
        def search(blocks: List[LogseqBlock]) -> Optional[LogseqBlock]:
            for block in blocks:
                if text.lower() in block.content.lower():
                    return block
                if found := search(block.children):
                    return found
            return None
        return search(self.blocks)

    def render(self) -> str:
        """Render outline back to Logseq markdown.

        IMPORTANT: Minimal changes guarantee (FR-008):
        - Preserves exact order of all blocks
        - Maintains original indentation (2 spaces per level)
        - NEVER reorders properties (insertion order sacred)
        - Children modifications are targeted and minimal
        - Uses _original_line when available for perfect round-trip
        """
        lines = []
        def render_block(block: LogseqBlock):
            # Use original line if available (unchanged blocks)
            if block._original_line and not block.children:
                lines.append(block._original_line)
            else:
                # Render block (possibly with new children appended)
                indent = "  " * block.indent_level
                lines.append(f"{indent}- {block.content}")
                # Render children in exact order (new ones at end)
                for child in block.children:
                    render_block(child)

        for block in self.blocks:
            render_block(block)
        return '\n'.join(lines)

```

---

### 5. ExtractionPreview

Summary of proposed changes shown to user before applying (FR-002 dry-run mode).

**Attributes**:

- `journal_date: date` - Source journal being processed
- `knowledge_blocks: List[KnowledgeBlock]` - All extracted knowledge
- `proposed_actions: List[ProposedAction]` - What will happen to each block
- `warnings: List[str]` - Issues found (missing pages, duplicates, etc.)

**Display Format**:

```
Found 3 knowledge blocks in journals/2025_01_15.md:
  1. "Project X deadline moved to May"
     → Target: Project X
     → Section: Projects > Active > Timeline
     → Action: Add child block

  2. "Main competitor is Product Y"
     → Target: Project X
     → Section: (page root)
     → Action: Add to end of page

  3. "Competitor uses pricing model Z"
     ⚠ DUPLICATE: Similar content already exists on "Project X"
     → Action: Skip integration

Apply changes? [y/N/e]

```

**Python Representation**:

```python
from dataclasses import dataclass
from typing import List
from datetime import date
from enum import Enum

class ActionStatus(Enum):
    READY = "ready"
    SKIPPED = "skipped"     # Duplicate or missing page
    WARNING = "warning"      # Will proceed but user should review

@dataclass
class ProposedAction:
    knowledge: KnowledgeBlock
    status: ActionStatus
    reason: Optional[str]  # Why skipped/warned

    def describe(self) -> str:
        """Human-readable description of action."""
        kb = self.knowledge
        parts = [
            f'  "{kb.content[:60]}..."',
            f'  → Target: {kb.target_page}',
            f'  → Section: {kb.section_path()}',  # Uses section_path() helper
            f'  → Action: {kb.suggested_action.value}'
        ]

        if self.status == ActionStatus.SKIPPED:
            parts.append(f'  ⚠ SKIPPED: {self.reason}')
        elif self.status == ActionStatus.WARNING:
            parts.append(f'  ⚠ WARNING: {self.reason}')

        return '\n'.join(parts)

@dataclass
class ExtractionPreview:
    journal_date: date
    knowledge_blocks: List[KnowledgeBlock]
    proposed_actions: List[ProposedAction]
    warnings: List[str]

    def display(self) -> str:
        """Render preview for terminal display."""
        lines = [
            f"Found {len(self.knowledge_blocks)} knowledge blocks in journals/{self.journal_date.isoformat()}.md:",
            ""
        ]

        for i, action in enumerate(self.proposed_actions, 1):
            lines.append(f"{i}. {action.describe()}")
            lines.append("")

        if self.warnings:
            lines.append("Warnings:")
            for warning in self.warnings:
                lines.append(f"  ⚠ {warning}")
            lines.append("")

        return '\n'.join(lines)

```

---

### 6. Configuration

User settings stored at ~/.config/logsqueak/config.yaml.

**Attributes**:

- `llm.endpoint: HttpUrl` - LLM API base URL
- `llm.api_key: str` - API authentication token
- `llm.model: str` - Model name/identifier
- `logseq.graph_path: Path` - Absolute path to Logseq graph directory

**Validation Rules** (via Pydantic):

- `endpoint` must be valid HTTP(S) URL
- `api_key` must not be empty
- `graph_path` must exist and be a directory
- `model` must not be empty

**Example YAML**:

```yaml
llm:
  endpoint: https://api.openai.com/v1
  api_key: sk-...
  model: gpt-4-turbo-preview

logseq:
  graph_path: ~/Documents/logseq-graph

```

**Python Representation**:

```python
from pydantic import BaseModel, Field, HttpUrl
from pathlib import Path

class LLMConfig(BaseModel):
    endpoint: HttpUrl
    api_key: str = Field(..., min_length=1)
    model: str = Field(default="gpt-4-turbo-preview")

class LogseqConfig(BaseModel):
    graph_path: Path

    @validator('graph_path')
    def path_must_exist(cls, v):
        if not v.exists() or not v.is_dir():
            raise ValueError(f"Logseq graph path does not exist: {v}")
        return v

class Config(BaseModel):
    llm: LLMConfig
    logseq: LogseqConfig

    @classmethod
    def load(cls, config_path: Path) -> 'Config':
        """Load from YAML file with validation."""
        import yaml
        with config_path.open() as f:
            data = yaml.safe_load(f)
        return cls(**data)

```

---

### 7. PageIndex

Semantic index of all pages in the Logseq graph for RAG-based page matching. Uses per-page caching to avoid re-embedding unchanged pages.

**Attributes**:

- `pages: List[TargetPage]` - All pages in the graph
- `embeddings: np.ndarray` - Embedding matrix (shape: [num_pages, embedding_dim])
- `model: SentenceTransformer` - Embedding model instance
- `page_texts: List[str]` - Text used for embeddings (name + preview)
- `cache_dir: Path` - Directory for per-page embedding cache (~/.cache/logsqueak/embeddings/)

**Operations**:

- `build(graph_path: Path) -> PageIndex` - Scan graph, embed pages (uses cache)
- `find_similar(text: str, top_k: int) -> List[Tuple[TargetPage, float]]` - Semantic search
- `refresh(page_name: str)` - Update single page embedding after modification
- `_load_cached_embedding(page_path: Path) -> Optional[np.ndarray]` - Load from cache if valid
- `_save_embedding(page_path: Path, embedding: np.ndarray)` - Save to cache

**Python Representation**:

```python
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

@dataclass
class PageIndex:
    """Semantic index for RAG-based page matching.

    Uses per-page caching to avoid re-embedding unchanged pages.
    Cache invalidation: mtime-based (re-embed if page file modified).
    """
    pages: List[TargetPage]
    embeddings: np.ndarray
    model: SentenceTransformer
    page_texts: List[str]
    cache_dir: Path

    @classmethod
    def build(cls, graph_path: Path, cache_dir: Optional[Path] = None) -> 'PageIndex':
        """Build index from all pages in graph.

        Each page is represented as: "{page_name} {first_1000_chars}"
        This captures page title and substantial initial content.

        Uses per-page caching:
        - Cache file: ~/.cache/logsqueak/embeddings/{page_name}.pkl
        - Contains: {'embedding': ndarray, 'mtime': float, 'text': str}
        - Invalidate: if page file mtime > cached mtime
        """
        # Load embedding model (80MB, cached by sentence-transformers)
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # Setup cache directory
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "logsqueak" / "embeddings"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Scan all pages in graph
        pages_dir = graph_path / "pages"
        pages = []
        page_texts = []
        embeddings_list = []

        for page_file in pages_dir.glob("*.md"):
            page = TargetPage.load(graph_path, page_file.stem)
            if not page:
                continue

            pages.append(page)

            # Combine page name + content preview for embedding
            preview = page.outline.render()[:1000]
            page_text = f"{page.name} {preview}"
            page_texts.append(page_text)

            # Try to load from cache
            cached_embedding = cls._load_cached_embedding(
                cache_dir, page_file, page_text
            )

            if cached_embedding is not None:
                embeddings_list.append(cached_embedding)
            else:
                # Cache miss - compute and save
                embedding = model.encode(page_text, convert_to_numpy=True)
                cls._save_embedding(cache_dir, page_file, embedding, page_text)
                embeddings_list.append(embedding)

        # Stack into matrix
        embeddings = np.vstack(embeddings_list)

        return cls(
            pages=pages,
            embeddings=embeddings,
            model=model,
            page_texts=page_texts,
            cache_dir=cache_dir
        )

    @staticmethod
    def _load_cached_embedding(
        cache_dir: Path, page_file: Path, page_text: str
    ) -> Optional[np.ndarray]:
        """Load cached embedding if valid (mtime check).

        Returns:
            Cached embedding if valid, None otherwise
        """
        cache_file = cache_dir / f"{page_file.stem}.pkl"
        if not cache_file.exists():
            return None

        try:
            with cache_file.open('rb') as f:
                cached = pickle.load(f)

            # Validate: mtime must match and text must match
            if (cached['mtime'] >= page_file.stat().st_mtime and
                cached['text'] == page_text):
                return cached['embedding']
        except (pickle.PickleError, KeyError, OSError):
            pass

        return None

    @staticmethod
    def _save_embedding(
        cache_dir: Path, page_file: Path, embedding: np.ndarray, page_text: str
    ):
        """Save embedding to per-page cache file."""
        cache_file = cache_dir / f"{page_file.stem}.pkl"
        try:
            with cache_file.open('wb') as f:
                pickle.dump({
                    'embedding': embedding,
                    'mtime': page_file.stat().st_mtime,
                    'text': page_text
                }, f)
        except (pickle.PickleError, OSError) as e:
            # Cache write failure is non-fatal (just slower next time)
            import logging
            logging.warning(f"Failed to cache embedding for {page_file.stem}: {e}")

    def find_similar(self, text: str, top_k: int = 5) -> List[Tuple[TargetPage, float]]:
        """Find top-K most semantically similar pages.

        Args:
            text: Knowledge block content to match
            top_k: Number of candidates to return

        Returns:
            List of (page, similarity_score) tuples, sorted by score descending
        """
        # Embed the query text
        query_embedding = self.model.encode(text, convert_to_numpy=True)

        # Compute cosine similarity with all pages
        # Normalize embeddings for cosine similarity
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        page_norms = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)

        similarities = np.dot(page_norms, query_norm)

        # Get top-K indices
        top_k_indices = np.argsort(similarities)[-top_k:][::-1]

        # Return (page, score) tuples
        return [
            (self.pages[i], float(similarities[i]))
            for i in top_k_indices
        ]

    def refresh(self, page_name: str):
        """Update embedding for a single page after modification.

        Re-embeds the page and updates cache. Called after we modify a page
        to keep the index current for subsequent extractions in same session.
        """
        # Find page index
        for i, page in enumerate(self.pages):
            if page.name == page_name:
                # Re-embed this page
                preview = page.outline.render()[:1000]
                page_text = f"{page.name} {preview}"
                self.page_texts[i] = page_text

                # Compute new embedding
                embedding = self.model.encode(page_text, convert_to_numpy=True)
                self.embeddings[i] = embedding

                # Update cache
                page_file = page.file_path
                self._save_embedding(self.cache_dir, page_file, embedding, page_text)
                break

```

**Usage in Extraction Workflow**:

```python
# At startup (with caching)
page_index = PageIndex.build(config.logseq.graph_path)
# First run (566 pages): ~20 seconds (embeds all pages)
# Subsequent runs: <1 second (loads from cache)
# After modifying 5 pages: ~1.5 seconds (loads 561 from cache, embeds 5)

# For each knowledge block extracted by LLM
knowledge = KnowledgeBlock(content="Deadline moved to May", ...)

# Find candidate pages
candidates = page_index.find_similar(knowledge.content, top_k=5)
# Returns: [
#   (TargetPage("Project X"), 0.87),
#   (TargetPage("Q1 Planning"), 0.72),
#   ...
# ]

# Pass candidates to LLM for final decision
llm_prompt = f"""
Knowledge: {knowledge.content}

Candidate pages (with similarity scores):
{format_candidates(candidates)}

Pick the most appropriate page and section.
"""

# After integrating knowledge to a page, update its embedding
page_index.refresh("Project X")
```

---

## Data Flow

```
0. At startup:
   Build PageIndex from all pages in graph (embed all pages)
   ↓
1. Load JournalEntry from file
   ↓
2. Extract KnowledgeBlocks via LLM (initial extraction - no page targeting yet)
   ↓
3. For each KnowledgeBlock:
   a. Find candidate pages via PageIndex.find_similar(content, top_k=5)
   b. Pass candidates to LLM for final page + section selection
   c. Load TargetPage from graph
   d. Check for duplicates (FR-017)
   e. Find target section in outline
   f. Create ProposedAction
   ↓
4. Build ExtractionPreview
   ↓
5. Display preview to user
   ↓
6. On approval (y):
   For each ProposedAction (status=READY):
     - Add knowledge with provenance to TargetPage.outline
     - Write modified outline back to file

```

## Validation Summary

| Entity | Key Validations |
|--------|----------------|
| JournalEntry | File exists, ≤2000 lines, valid markdown |
| KnowledgeBlock | Non-empty content, 0≤confidence≤1, valid page name |
| TargetPage | File exists, valid outline, detectable convention |
| LogseqOutline | Parseable markdown, valid bullet hierarchy |
| ExtractionPreview | At least one knowledge block |
| Configuration | Valid URLs, existing paths, non-empty credentials |
| PageIndex | Embeddings shape matches pages count, valid model |

## Error Handling

- **Malformed Journal** (FR-018): Log warning, skip, continue with next entry
- **Missing Target Page** (FR-009): Mark as SKIPPED in preview, inform user
- **Duplicate Knowledge** (FR-017): Mark as SKIPPED in preview with reason
- **LLM API Failure** (FR-011): Graceful error, no file corruption, clear message
- **Invalid Config**: Pydantic validation errors with helpful messages

## Order Preservation Guarantees

**Critical for FR-008 (preserve existing content):**

### What We Preserve

1. **Block Order**: All blocks remain in exact original sequence
2. **Property Order**: Properties MUST maintain insertion order (Python 3.7+ dict guarantee) - NEVER reorder
3. **Whitespace**: Original indentation and formatting preserved via `_original_line`
4. **Unchanged Blocks**: Blocks not modified are written byte-for-byte identical to source

### What We Modify (Minimally)

1. **New Children**: Can be inserted where appropriate (default: append to end)
2. **Targeted Placement**: New content inserted at logical locations (e.g., Timeline section)
3. **Modified Blocks**: Only when adding new child bullets (parent block may need re-render)

### Implementation Strategy

```python
# Good: Targeted child insertion
parent_block.add_child("New item")  # Append to end
parent_block.add_child("Urgent item", position=0)  # Insert at top if needed

# Good: Preserve property order
block.properties = parse_properties(line)  # Dict preserves insertion order

# Bad: NEVER reorder properties
block.properties = dict(sorted(block.properties.items()))  # ❌ FORBIDDEN
block.properties = {k: block.properties[k] for k in sorted(block.properties)}  # ❌ FORBIDDEN

# Bad: Sorting children arbitrarily
parent_block.children.sort(key=lambda x: x.content)  # ❌ Avoid unless intentional
```

### Round-Trip Guarantee

For unchanged blocks:
```
Original: "  - Timeline"
Parsed:   LogseqBlock(content="Timeline", indent_level=1, _original_line="  - Timeline")
Rendered: "  - Timeline"  # Exact match (byte-for-byte)
```

For modified blocks (new child added):
```
Original: "- Projects"
After:    "- Projects\n  - New project [[2025-01-15]]"
         # New child added where appropriate, original "- Projects" preserved
```

## Implementation Notes

- Use Python 3.11+ `dataclasses` for POC simplicity
- Switch to Pydantic models if need more validation later
- `LogseqOutline` custom wrapper uses markdown-it-py internally
- All file paths are `pathlib.Path` (not strings)
- Dates use stdlib `datetime.date` (not strings)
- Configuration uses Pydantic for validation (type safety, great errors)
- **Dict order**: Python 3.7+ guarantees dict preserves insertion order (used for properties)
