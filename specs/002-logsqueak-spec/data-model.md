# Data Model: Logsqueak Interactive TUI

**Date**: 2025-11-05

**Feature**: 002-logsqueak-spec

**Status**: Specification

This document defines all data models used in the Logsqueak Interactive TUI application. All models use Pydantic for validation and serialization.

---

## Table of Contents

1. [Core State Models](#core-state-models)
   - [BlockState](#blockstate)
   - [EditedContent](#editedcontent)
   - [IntegrationDecision](#integrationdecision)
   - [BackgroundTask](#backgroundtask)

2. [Configuration Models](#configuration-models)
   - [LLMConfig](#llmconfig)
   - [LogseqConfig](#logseqconfig)
   - [RAGConfig](#ragconfig)
   - [Config](#config)

3. [Supporting Models](#supporting-models)
   - [FileTracker](#filetracker)
   - [TaskProgress](#taskprogress)

4. [Model Relationships](#model-relationships)
5. [State Transitions](#state-transitions)

---

## Core State Models

### BlockState

Represents the selection state of a journal block during Phase 1 (Block Selection).

**Purpose**: Track whether a block has been classified as knowledge by the LLM and/or selected by the user for extraction.

**Pydantic Model**:

```python
from pydantic import BaseModel, Field
from typing import Literal, Optional

class BlockState(BaseModel):
    """Selection state for a journal block in Phase 1."""

    block_id: str = Field(
        ...,
        description="Stable block identifier (explicit id:: property or content hash)"
    )

    classification: Literal["pending", "knowledge"] = Field(
        default="pending",
        description="Current classification status (user or LLM)"
    )

    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence score for current classification (0.0-1.0)"
    )

    source: Literal["user", "llm"] = Field(
        ...,
        description="Who set the current classification"
    )

    llm_classification: Optional[Literal["knowledge"]] = Field(
        default=None,
        description="LLM's classification if available (only 'knowledge' or None)"
    )

    llm_confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="LLM's confidence score if classification available"
    )

    reason: Optional[str] = Field(
        default=None,
        description="LLM's reasoning for why block is knowledge"
    )

    class Config:
        frozen = False  # Allow mutation during user interaction

```

**Field Descriptions**:

- `block_id`: Unique identifier for the block (from `id::` property or content hash)
- `classification`: Current state - "pending" (not classified yet) or "knowledge" (identified as knowledge)
- `confidence`: Confidence score for current classification (user selections have confidence=1.0)
- `source`: Indicates whether current classification came from "user" or "llm"
- `llm_classification`: Stores LLM's original suggestion (None if LLM hasn't classified, "knowledge" if LLM suggested)
- `llm_confidence`: LLM's confidence score (preserved even if user overrides)
- `reason`: LLM's explanation for why this block contains knowledge

**Example JSON**:

```json
{
  "block_id": "abc123def456",
  "classification": "knowledge",
  "confidence": 0.92,
  "source": "llm",
  "llm_classification": "knowledge",
  "llm_confidence": 0.92,
  "reason": "Contains a reusable insight about Python async patterns that would be valuable in the 'Programming Notes' page."
}

```

**Example JSON (User Override)**:

```json
{
  "block_id": "xyz789",
  "classification": "knowledge",
  "confidence": 1.0,
  "source": "user",
  "llm_classification": null,
  "llm_confidence": null,
  "reason": null
}

```

---

### EditedContent

Represents the edited content for a knowledge block in Phase 2 (Content Editing).

**Purpose**: Track original content, LLM-generated reworded version, and current user edits for each knowledge block.

**Pydantic Model**:

```python
from pydantic import BaseModel, Field
from typing import Optional

class EditedContent(BaseModel):
    """Edited content state for a knowledge block in Phase 2."""

    block_id: str = Field(
        ...,
        description="Stable block identifier matching BlockState"
    )

    original_content: str = Field(
        ...,
        description="Original block content from journal (single block only, without parent context)"
    )

    hierarchical_context: str = Field(
        ...,
        description="Full hierarchical context including all parent blocks (shown in top read-only panel)"
    )

    reworded_content: Optional[str] = Field(
        default=None,
        description="LLM-generated reworded version of original_content (removes temporal context, single block only)"
    )

    current_content: str = Field(
        ...,
        description="Current editable content (starts as original_content, user can modify in bottom panel)"
    )

    rewording_complete: bool = Field(
        default=False,
        description="Whether LLM has finished generating reworded version"
    )

    class Config:
        frozen = False  # Allow mutation during editing

```

**Field Descriptions**:

- `block_id`: Links to corresponding BlockState from Phase 1
- `original_content`: The exact content from the journal entry (single block only, no parent hierarchy)
- `hierarchical_context`: Full hierarchical context including all parent blocks (displayed in top read-only panel for context)
- `reworded_content`: LLM-suggested version with temporal context removed (single block only, None until LLM responds)
- `current_content`: The version that will be integrated (starts as original_content, user can accept LLM version or manually edit)
- `rewording_complete`: Flag indicating whether to show "Accept LLM version" action

**Example JSON**:

```json
{
  "block_id": "abc123def456",
  "original_content": "Today I learned that using `asyncio.create_task()` is better than `await` for concurrent operations in Python.",
  "hierarchical_context": "2025-01-15 - Tuesday\n  Morning coding session\n    Today I learned that using `asyncio.create_task()` is better than `await` for concurrent operations in Python.",
  "reworded_content": "Using `asyncio.create_task()` enables concurrent operations in Python, unlike `await` which executes sequentially.",
  "current_content": "Using `asyncio.create_task()` enables concurrent operations in Python, unlike `await` which executes sequentially.",
  "rewording_complete": true
}

```

---

### IntegrationDecision

Represents a decision about integrating a knowledge block to a specific page location in Phase 3.

**Purpose**: Store LLM's suggestion for where/how to integrate a knowledge block, track user approval and write status.

**Pydantic Model**:

```python
from pydantic import BaseModel, Field
from typing import Literal, Optional

class IntegrationDecision(BaseModel):
    """Integration decision for a knowledge block in Phase 3."""

    knowledge_block_id: str = Field(
        ...,
        description="Block ID of the knowledge being integrated"
    )

    target_page: str = Field(
        ...,
        description="Target page name (hierarchical pages use '/' separator)"
    )

    action: Literal["add_section", "add_under", "replace", "skip_exists"] = Field(
        ...,
        description="Type of integration action"
    )

    target_block_id: Optional[str] = Field(
        default=None,
        description="Target block ID for 'add_under', 'replace', or 'skip_exists' actions"
    )

    target_block_title: Optional[str] = Field(
        default=None,
        description="Human-readable title of target block (for display)"
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="LLM's confidence score for this integration (0.0-1.0)"
    )

    refined_text: str = Field(
        ...,
        description="The content to integrate (from Phase 2 EditedContent)"
    )

    reasoning: str = Field(
        ...,
        description="LLM's explanation for this integration decision"
    )

    write_status: Literal["pending", "completed", "failed"] = Field(
        default="pending",
        description="Status of write operation"
    )

    error_message: Optional[str] = Field(
        default=None,
        description="Error details if write_status is 'failed'"
    )

    class Config:
        frozen = False  # Allow mutation when user accepts/writes

```

**Field Descriptions**:

- `knowledge_block_id`: Links to EditedContent and BlockState
- `target_page`: Name of the Logseq page (e.g., "Projects/Acme" uses "/" for hierarchy)
- `action`: How to integrate:
  - `add_section`: Create new top-level section
  - `add_under`: Add as child under specific block
  - `replace`: Replace existing block content (all content lines) while preserving properties and children
  - `skip_exists`: Knowledge already exists in target page - signals that this entire knowledge block should be skipped (no integration needed, as content is already recorded elsewhere)

- `target_block_id`: Identifies target block for `add_under`, `replace`, or `skip_exists` (None for `add_section`). For `skip_exists`, points to the existing block containing duplicate content.
- `target_block_title`: Human-readable description (e.g., "Under 'Project Timeline'" or "Already exists at 'Async Patterns'")
- `confidence`: LLM's confidence (shown as percentage in UI for actionable decisions)
- `refined_text`: The actual content to write (comes from EditedContent.current_content)
- `reasoning`: LLM's explanation for why this integration makes sense (or why it's a duplicate for `skip_exists`)
- `write_status`: Tracks whether decision has been accepted and written
- `error_message`: Details if write failed (e.g., "Target block not found")

**Example JSON (New Integration)**:

```json
{
  "knowledge_block_id": "abc123def456",
  "target_page": "Programming Notes/Python",
  "action": "add_under",
  "target_block_id": "section-async-patterns",
  "target_block_title": "Async Patterns",
  "confidence": 0.87,
  "refined_text": "Using `asyncio.create_task()` enables concurrent operations in Python, unlike `await` which executes sequentially.",
  "reasoning": "This insight fits well under the 'Async Patterns' section as it directly explains task concurrency.",
  "write_status": "pending",
  "error_message": null
}

```

**Example JSON (Duplicate Detected)**:

```json
{
  "knowledge_block_id": "abc123def456",
  "target_page": "Programming Notes/Python",
  "action": "skip_exists",
  "target_block_id": "block-xyz789",
  "target_block_title": "Already exists at 'Async Patterns'",
  "confidence": 0.95,
  "refined_text": "Using `asyncio.create_task()` enables concurrent operations in Python, unlike `await` which executes sequentially.",
  "reasoning": "This knowledge already exists in the target page under the 'Async Patterns' section with nearly identical wording.",
  "write_status": "pending",
  "error_message": null
}

```

---

### BackgroundTask

Represents a long-running background task that executes while user interacts with UI.

**Purpose**: Track status and progress of asynchronous operations (LLM streaming, page indexing, RAG search).

**Pydantic Model**:

```python
from pydantic import BaseModel, Field
from typing import Literal, Optional

class BackgroundTask(BaseModel):
    """Background task state for async operations."""

    task_type: Literal[
        "llm_classification",
        "page_indexing",
        "rag_search",
        "llm_rewording",
        "llm_decisions"
    ] = Field(
        ...,
        description="Type of background task"
    )

    status: Literal["running", "completed", "failed"] = Field(
        default="running",
        description="Current task status"
    )

    progress_percentage: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Progress percentage if calculable (0.0-100.0)"
    )

    progress_current: Optional[int] = Field(
        default=None,
        description="Current item count (e.g., 3 blocks processed)"
    )

    progress_total: Optional[int] = Field(
        default=None,
        description="Total item count (e.g., 5 total blocks)"
    )

    error_message: Optional[str] = Field(
        default=None,
        description="Error details if status is 'failed'"
    )

    class Config:
        frozen = False  # Allow mutation as task progresses

```

**Field Descriptions**:

- `task_type`: Which background operation this represents
  - `llm_classification`: Phase 1 knowledge block classification
  - `page_indexing`: Building ChromaDB vector index
  - `rag_search`: Semantic search for candidate pages
  - `llm_rewording`: Phase 2 content refinement
  - `llm_decisions`: Phase 3 integration decision generation

- `status`: Current state - "running", "completed", or "failed"
- `progress_percentage`: Optional percentage (e.g., 45.5 for page indexing)
- `progress_current`: Optional current count (e.g., 3 for "3 blocks reworded")
- `progress_total`: Optional total count (e.g., 5 for "5 total blocks")
- `error_message`: Details if task failed

**Example JSON (With Percentage)**:

```json
{
  "task_type": "page_indexing",
  "status": "running",
  "progress_percentage": 67.3,
  "progress_current": null,
  "progress_total": null,
  "error_message": null
}

```

**Example JSON (With Count)**:

```json
{
  "task_type": "llm_rewording",
  "status": "running",
  "progress_percentage": null,
  "progress_current": 3,
  "progress_total": 5,
  "error_message": null
}

```

---

## Configuration Models

### LLMConfig

Configuration for LLM API connection and behavior.

**Purpose**: Store API endpoint, credentials, and model settings for LLM requests.

**Pydantic Model**:

```python
from pydantic import BaseModel, Field, HttpUrl

class LLMConfig(BaseModel):
    """Configuration for LLM API connection."""

    endpoint: HttpUrl = Field(
        ...,
        description="LLM API endpoint URL (OpenAI or Ollama compatible)"
    )

    api_key: str = Field(
        ...,
        description="API key for authentication"
    )

    model: str = Field(
        ...,
        description="Model identifier (e.g., 'gpt-4-turbo-preview', 'llama2')"
    )

    num_ctx: int = Field(
        default=32768,
        ge=1024,
        description="Context window size (Ollama-specific, controls VRAM usage)"
    )

    class Config:
        frozen = True  # Immutable after loading

```

**Field Descriptions**:

- `endpoint`: Full URL to LLM service (e.g., `https://api.openai.com/v1` or `http://localhost:11434/v1`)
- `api_key`: Authentication key (stored securely in config file with mode 600)
- `model`: Model name to use for all requests
- `num_ctx`: Context window size (optional, Ollama-specific setting)

**Example JSON**:

```json
{
  "endpoint": "https://api.openai.com/v1",
  "api_key": "sk-proj-...",
  "model": "gpt-4-turbo-preview",
  "num_ctx": 32768
}

```

---

### LogseqConfig

Configuration for Logseq graph location.

**Purpose**: Store and validate path to Logseq graph directory.

**Pydantic Model**:

```python
from pydantic import BaseModel, Field, validator
from pathlib import Path

class LogseqConfig(BaseModel):
    """Configuration for Logseq graph location."""

    graph_path: str = Field(
        ...,
        description="Path to Logseq graph directory"
    )

    @validator('graph_path')
    def validate_graph_path(cls, v):
        """Validate graph path exists and is a directory."""
        path = Path(v).expanduser()
        if not path.exists():
            raise ValueError(
                f"Graph path does not exist: {path}\n"
                f"Please create the directory or update config.yaml"
            )
        if not path.is_dir():
            raise ValueError(
                f"Graph path is not a directory: {path}\n"
                f"Please provide a valid directory path"
            )
        return str(path)

    class Config:
        frozen = True  # Immutable after loading

```

**Field Descriptions**:

- `graph_path`: Absolute or tilde-expanded path to Logseq graph (validated lazily)

**Example JSON**:

```json
{
  "graph_path": "/home/user/Documents/logseq-graph"
}

```

---

### RAGConfig

Configuration for RAG semantic search behavior.

**Purpose**: Control how many candidate pages are retrieved during semantic search.

**Pydantic Model**:

```python
from pydantic import BaseModel, Field

class RAGConfig(BaseModel):
    """Configuration for RAG semantic search."""

    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of candidate pages to retrieve from vector search"
    )

    class Config:
        frozen = True  # Immutable after loading

```

**Field Descriptions**:

- `top_k`: How many top-ranked pages to retrieve (LLM filters these to relevant subset)

**Example JSON**:

```json
{
  "top_k": 10
}

```

---

### Config

Root configuration model combining all config sections.

**Purpose**: Top-level config loaded from `~/.config/logsqueak/config.yaml`.

**Pydantic Model**:

```python
from pydantic import BaseModel, Field
from pathlib import Path
import yaml
import os
import stat

class Config(BaseModel):
    """Root configuration for Logsqueak application."""

    llm: LLMConfig = Field(..., description="LLM API settings")
    logseq: LogseqConfig = Field(..., description="Logseq graph settings")
    rag: RAGConfig = Field(default_factory=RAGConfig, description="RAG search settings")

    @classmethod
    def load(cls, path: Path) -> "Config":
        """
        Load configuration from YAML file.

        Validates file permissions before loading.
        Raises PermissionError if file is group/world readable.
        """
        # Check file permissions (must be 600)
        mode = os.stat(path).st_mode
        if mode & (stat.S_IRWXG | stat.S_IRWXO):
            raise PermissionError(
                f"Config file has overly permissive permissions: {oct(mode)}\n"
                f"Run: chmod 600 {path}"
            )

        # Load and parse YAML
        with open(path) as f:
            data = yaml.safe_load(f)

        return cls(**data)

    class Config:
        frozen = True  # Immutable after loading

```

**Field Descriptions**:

- `llm`: LLM API configuration (required)
- `logseq`: Logseq graph configuration (required)
- `rag`: RAG search configuration (optional, uses defaults if not provided)

**Example YAML Configuration**:

```yaml
llm:
  endpoint: https://api.openai.com/v1
  api_key: sk-proj-...
  model: gpt-4-turbo-preview
  num_ctx: 32768  # Optional

logseq:
  graph_path: ~/Documents/logseq-graph

rag:
  top_k: 10  # Optional, defaults to 10

```

**Validation Rules**:

1. File must exist at `~/.config/logsqueak/config.yaml`
2. File permissions must be 600 (user read/write only)
3. YAML must be valid and match schema
4. Validation happens lazily (when config sections first accessed)
5. Validation failures exit immediately with clear error message

---

## Supporting Models

### FileTracker

Tracks file modification times for concurrent modification detection.

**Purpose**: Detect when Logseq files are modified externally during TUI session.

**Python Class** (Not Pydantic):

```python
from pathlib import Path
from typing import Dict

class FileTracker:
    """Track file modification times to detect external changes."""

    def __init__(self):
        self._mtimes: Dict[Path, float] = {}

    def record(self, path: Path) -> None:
        """Record current modification time."""
        self._mtimes[path] = path.stat().st_mtime

    def is_modified(self, path: Path) -> bool:
        """Check if file modified since last record."""
        if path not in self._mtimes:
            return True
        current_mtime = path.stat().st_mtime
        return current_mtime > self._mtimes[path]

    def refresh(self, path: Path) -> None:
        """Update recorded mtime after successful reload."""
        self._mtimes[path] = path.stat().st_mtime

```

**Usage Pattern**:

```python
tracker = FileTracker()

# On initial load
journal_path = Path("journal/2025-01-15.md")
outline = LogseqOutline.parse(journal_path.read_text())
tracker.record(journal_path)

# Before write operation
if tracker.is_modified(journal_path):
    # Reload file
    outline = LogseqOutline.parse(journal_path.read_text())
    # Re-validate operation
    if not validate_target_exists(outline, target_block_id):
        raise ValueError("Target block no longer exists")
    # Refresh tracker
    tracker.refresh(journal_path)

# Proceed with write

```

---

### TaskProgress

Helper model for displaying task progress in UI.

**Purpose**: Format background task progress for status widget display.

**Pydantic Model**:

```python
from pydantic import BaseModel, Field
from typing import Optional

class TaskProgress(BaseModel):
    """Formatted progress information for UI display."""

    label: str = Field(
        ...,
        description="Human-readable task description"
    )

    percentage: Optional[float] = Field(
        default=None,
        description="Progress as percentage (0.0-100.0)"
    )

    fraction: Optional[str] = Field(
        default=None,
        description="Progress as fraction string (e.g., '3/5')"
    )

    is_complete: bool = Field(
        default=False,
        description="Whether task is finished"
    )

    @classmethod
    def from_background_task(cls, task: BackgroundTask) -> "TaskProgress":
        """Create TaskProgress from BackgroundTask."""
        # Generate label
        labels = {
            "llm_classification": "Analyzing knowledge blocks",
            "page_indexing": "Building page index",
            "rag_search": "Finding relevant pages",
            "llm_rewording": "Refining knowledge content",
            "llm_decisions": "Processing knowledge blocks"
        }
        label = labels[task.task_type]

        # Format progress
        percentage = task.progress_percentage
        fraction = None
        if task.progress_current is not None and task.progress_total is not None:
            fraction = f"{task.progress_current}/{task.progress_total}"

        is_complete = task.status == "completed"

        return cls(
            label=label,
            percentage=percentage,
            fraction=fraction,
            is_complete=is_complete
        )

```

---

## Model Relationships

### Entity Relationship Diagram

```
┌─────────────┐
│ BlockState  │ (Phase 1: Selection)
│  block_id   │───┐
└─────────────┘   │
                  │ References same block_id
                  │
                  ▼
            ┌──────────────┐
            │EditedContent │ (Phase 2: Editing)
            │  block_id    │───┐
            └──────────────┘   │ References same block_id
                               │
                               ▼
                    ┌────────────────────┐
                    │IntegrationDecision │ (Phase 3: Integration)
                    │ knowledge_block_id │
                    │ target_page        │
                    │ refined_text       │ (Copied from EditedContent.current_content)
                    └────────────────────┘

┌──────────────────┐
│ BackgroundTask   │ (All phases)
│  task_type       │ Tracks async operations
│  status          │
└──────────────────┘

┌────────────┐
│   Config   │ (Startup)
│  llm       │───► LLMConfig
│  logseq    │───► LogseqConfig
│  rag       │───► RAGConfig
└────────────┘

┌──────────────┐
│ FileTracker  │ (Write operations)
│  _mtimes     │ Detects concurrent modifications
└──────────────┘

```

### Key Relationships

1. **BlockState → EditedContent**: Same `block_id` links selection state to edited content
2. **EditedContent → IntegrationDecision**: `knowledge_block_id` references `block_id`, `refined_text` copies `current_content`
3. **BackgroundTask**: Independent tracking for all async operations
4. **Config**: Loaded once at startup, accessed lazily throughout application
5. **FileTracker**: Shared instance tracks all file modification times

---

## State Transitions

### BlockState Transitions

```
┌─────────┐
│ pending │ (Initial state)
└─────────┘
    │
    ├─ LLM classifies as knowledge ────► classification="knowledge"
    │                                     source="llm"
    │                                     llm_classification="knowledge"
    │                                     llm_confidence=0.XX
    │                                     reason="..."
    │
    └─ User selects block ──────────────► classification="knowledge"
                                           source="user"
                                           confidence=1.0
                                           (llm_* fields may or may not be set)

```

**State Rules**:

- `classification` can transition from "pending" → "knowledge" only
- Once classified, user can toggle back to "pending" (clears selection)
- `llm_*` fields are immutable once set (LLM suggestion preserved)
- `source` and `confidence` update when user overrides

### BackgroundTask Transitions

```
┌─────────┐
│ running │ (Initial state)
└─────────┘
    │
    ├─ Task succeeds ──────► status="completed"
    │                        progress_percentage=100.0
    │
    └─ Task fails ─────────► status="failed"
                             error_message="..."

```

**State Rules**:

- `status` can only transition: "running" → "completed" OR "running" → "failed"
- `progress_percentage` increases monotonically (0.0 → 100.0)
- Failed tasks preserve partial `progress_*` values

### IntegrationDecision Transitions

```
┌─────────┐
│ pending │ (Initial state)
└─────────┘
    │
    ├─ User accepts ('y') ──────► write_status="completed" (if write succeeds)
    │                             Journal marked with processed:: property
    │
    └─ Write fails ─────────────► write_status="failed"
                                  error_message="Target block not found"

```

**State Rules**:

- `write_status` can transition: "pending" → "completed" OR "pending" → "failed"
- Multiple decisions for same `knowledge_block_id` can all be "completed" (multi-page integration)
- Failed decisions remain visible with error details

---

## Validation Rules

### Cross-Model Validation

1. **Block ID Consistency**: `BlockState.block_id` must match `EditedContent.block_id` must match `IntegrationDecision.knowledge_block_id` for the same knowledge block across all phases.

2. **Content Consistency**: `IntegrationDecision.refined_text` must be copied from `EditedContent.current_content` at decision generation time.

3. **Confidence Ranges**: All confidence scores must be in range [0.0, 1.0].

4. **Status Consistency**: BackgroundTask status transitions are one-way (no "completed" → "running").

5. **File Permissions**: Config file must have mode 600 (checked at load time).

### Phase-Specific Validation

**Phase 1 → Phase 2 Transition**:

- At least one BlockState must have `classification="knowledge"` and `source` (user or llm)

**Phase 2 → Phase 3 Transition**:

- BackgroundTask with `task_type="page_indexing"` must have `status="completed"`
- BackgroundTask with `task_type="rag_search"` must have `status="completed"`
- All EditedContent entries must have `current_content` (non-empty)

**Phase 3 Write Operations**:

- IntegrationDecision must have `write_status="pending"` to be writable
- Target file must pass FileTracker modification check before write
- Journal block must exist and be writable

---

## Notes

- All Pydantic models use `frozen=False` for mutable state and `frozen=True` for configuration
- Models serialize to/from JSON for logging and debugging
- FileTracker is not Pydantic (simple utility class with no validation needs)
- All datetime values use timestamps (float) for simplicity (no timezone handling required)
- Block IDs are strings (hybrid system: explicit `id::` UUID or content hash)

