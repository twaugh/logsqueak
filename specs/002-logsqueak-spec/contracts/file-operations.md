# File Operations Contract

**Date**: 2025-11-05

**Feature**: 002-logsqueak-spec

## Overview

This document defines the contract for all Logseq file operations in the Logsqueak Interactive TUI. All file operations use the `logseq-outline-parser` library and adhere to strict property order preservation and atomic write guarantees.

**Key Principles** (from constitution):

- **Property Order Preservation (NON-NEGOTIABLE)**: Property insertion order is sacred in Logseq
- **Non-Destructive Operations**: All operations traceable via `extracted-to::` markers
- **Atomic Writes**: Journal marked only when page write succeeds
- **Concurrent Modification Detection**: Check file mtimes before all writes

---

## Table of Contents

1. [Parser Operations](#parser-operations)
2. [Write Operations](#write-operations)
3. [Property Management](#property-management)
4. [Atomic Guarantees](#atomic-guarantees)
5. [Error Handling](#error-handling)

---

## Parser Operations

### Using logseq-outline-parser

All Logseq file I/O uses the `logseq-outline-parser` library for parsing and rendering.

#### Path Resolution with GraphPaths

```python
from logseq_outline.graph import GraphPaths
from pathlib import Path

# Initialize GraphPaths with graph root
graph_root = Path("~/Documents/logseq-graph").expanduser()
graph_paths = GraphPaths(graph_root)

# Get journal path
journal_path = graph_paths.get_journal_path("2025-11-05")
# Returns: ~/Documents/logseq-graph/journals/2025-11-05.md

# Get page path (handles hierarchical pages with '/' separator)
page_path = graph_paths.get_page_path("Python/Concurrency")
# Returns: ~/Documents/logseq-graph/pages/Python___Concurrency.md
```

**Contract**:
- `GraphPaths` handles conversion between logical names and filesystem paths
- Hierarchical pages use `/` separator in names, converted to `___` for filenames
- All file paths are absolute (resolved from graph root)

#### Read Operations

```python
from logseq_outline.parser import LogseqOutline

# Read journal entry
journal_path = graph_paths.journal_path("2025-11-05")
journal_text = journal_path.read_text()
journal_outline = LogseqOutline.parse(journal_text)

# Read page
page_path = graph_paths.page_path("Python/Concurrency")
page_text = page_path.read_text()
page_outline = LogseqOutline.parse(page_text)

```

**Contract**:

- Parser preserves exact structure and order
- Property order is preserved (NON-NEGOTIABLE)
- Indentation is auto-detected (default 2 spaces)
- Round-trip guarantee: `parse(text).render() == text` (except intentional changes)

#### Render Operations

```python
# Render back to markdown
output_text = journal_outline.render()

# Write to file
journal_path.write_text(output_text)

```

**Contract**:

- Renders with exact property order from source
- Uses detected indentation (2 spaces per level)
- Preserves continuation line formatting
- Preserves frontmatter (text before first bullet)

#### Block Finding

```python
from logseq_outline.parser import LogseqBlock

# Find block by ID (explicit or hash)
block = journal_outline.find_block_by_id("abc123-def456-...")

if block is None:
    raise ValueError("Block not found")

# Access block properties
block_id = block.block_id  # From id:: property or None
full_content = block.get_full_content()  # All lines as string
property_value = block.get_property("processed")  # Get property value

```

**Contract**:

- `find_block_by_id()` returns `LogseqBlock` if found, `None` otherwise
- `block.block_id` is explicit `id::` property value or `None`
- For content-hash based lookup, optionally pass `page_name` parameter

---

## Write Operations

### Deterministic UUID Generation

All integration operations use deterministic UUIDs to ensure idempotency. This allows retry operations to detect existing blocks reliably.

```python
import hashlib
import uuid

def generate_integration_id(
    knowledge_block_id: str,
    target_page: str,
    action: str,
    target_block_id: str | None = None
) -> str:
    """
    Generate deterministic UUID for integration block.

    Same inputs always produce the same UUID, enabling idempotent operations.

    Args:
        knowledge_block_id: ID of knowledge block being integrated
        target_page: Target page name
        action: Integration action type
        target_block_id: Target block ID (for add_under/replace actions)

    Returns:
        str: Deterministic UUID string
    """
    # Logsqueak namespace UUID (from utils/ids.py)
    namespace = uuid.UUID('32e497fc-abf0-4d71-8cff-e302eb3e2bb0')

    # Combine inputs to create reproducible identifier
    parts = [knowledge_block_id, target_page, action]
    if target_block_id:
        parts.append(target_block_id)

    # Generate UUID v5 (SHA-1 based, deterministic)
    name = ":".join(parts)
    return str(uuid.uuid5(namespace, name))

```

**Contract**:

- Same inputs → same UUID (deterministic)
- Enables retry detection (search by UUID finds existing block)
- Prevents duplicate blocks on journal write failure
- UUID v5 uses SHA-1 hash of namespace + name

### Integration Actions

Three types of write operations for integrating knowledge blocks:

#### 1. Add Section (New Top-Level Block)

```python
def write_add_section(
    page_outline: LogseqOutline,
    refined_text: str,
    knowledge_block_id: str,
    target_page: str
) -> str:
    """
    Add knowledge as new root-level block.

    Args:
        page_outline: Target page outline
        refined_text: Content to add
        knowledge_block_id: ID of knowledge block being integrated
        target_page: Target page name

    Returns:
        str: Deterministic UUID of newly created block (for provenance)
    """
    from logseq_outline.parser import LogseqBlock

    # Generate deterministic UUID
    new_block_id = generate_integration_id(
        knowledge_block_id=knowledge_block_id,
        target_page=target_page,
        action="add_section"
    )

    # Create new block with id:: property
    new_block = LogseqBlock(
        content=[
            refined_text,
            f"id:: {new_block_id}"
        ],
        indent_level=0,
        children=[]
    )

    # Add to end of page
    page_outline.blocks.append(new_block)

    return new_block_id

```

**Contract**:

- New block added at end of root-level blocks
- `id::` property uses deterministic UUID (same knowledge + target → same UUID)
- Content and property are separate lines in `content` list
- Idempotent: Retry with same inputs produces same UUID (detectable duplicate)

#### 2. Add Under (Child Block)

```python
def write_add_under(
    page_outline: LogseqOutline,
    target_block_id: str,
    refined_text: str,
    knowledge_block_id: str,
    target_page: str
) -> str:
    """
    Add knowledge as child of existing block.

    Args:
        page_outline: Target page outline
        target_block_id: ID of parent block
        refined_text: Content to add
        knowledge_block_id: ID of knowledge block being integrated
        target_page: Target page name

    Returns:
        str: Deterministic UUID of newly created block (for provenance)

    Raises:
        ValueError: If target block not found
    """
    # Generate deterministic UUID
    new_block_id = generate_integration_id(
        knowledge_block_id=knowledge_block_id,
        target_page=target_page,
        action="add_under",
        target_block_id=target_block_id
    )

    # Find target block
    target_block = page_outline.find_block_by_id(target_block_id)
    if target_block is None:
        raise ValueError(f"Target block not found: {target_block_id}")

    # Add as child (add_child handles indentation automatically)
    new_child = target_block.add_child(refined_text)

    # Add id:: property to new child
    new_child.set_property("id", new_block_id)

    return new_block_id

```

**Contract**:

- `add_child()` automatically calculates correct indentation
- New child added at end of existing children
- `id::` property uses deterministic UUID (includes target_block_id for uniqueness)
- Target block ID can be explicit `id::` or content hash
- Idempotent: Retry with same inputs produces same UUID (detectable duplicate)

#### 3. Replace (Update Existing Block)

```python
def write_replace(
    page_outline: LogseqOutline,
    target_block_id: str,
    refined_text: str,
    knowledge_block_id: str,
    target_page: str
) -> str:
    """
    Replace existing block content while preserving properties and children.

    Args:
        page_outline: Target page outline
        target_block_id: ID of block to replace
        refined_text: New content (may contain multiple lines)
        knowledge_block_id: ID of knowledge block being integrated
        target_page: Target page name

    Returns:
        str: UUID of block (existing or newly added)

    Raises:
        ValueError: If target block not found
    """
    # Find target block
    target_block = page_outline.find_block_by_id(target_block_id)
    if target_block is None:
        raise ValueError(f"Target block not found: {target_block_id}")

    # Separate properties from content lines
    property_lines = []
    for line in target_block.content:
        # Property pattern: key:: value (not starting with bullet)
        if "::" in line and not line.strip().startswith("-"):
            property_lines.append(line)

    # Replace all content lines, preserve all properties
    # refined_text might be multi-line, so split it
    new_content_lines = refined_text.split("\n") if "\n" in refined_text else [refined_text]
    target_block.content = new_content_lines + property_lines

    # Ensure block has id:: property (add deterministic UUID if missing)
    if target_block.block_id is None:
        new_block_id = generate_integration_id(
            knowledge_block_id=knowledge_block_id,
            target_page=target_page,
            action="replace",
            target_block_id=target_block_id
        )
        target_block.set_property("id", new_block_id)
        return new_block_id
    else:
        # If id:: already exists, it's preserved in property_lines above
        return target_block.block_id

```

**Contract**:

- All content lines (first line + continuation lines) are replaced
- All properties preserved in their original order
- Children blocks preserved unchanged
- If block lacks `id::`, add deterministic UUID (for provenance)
- Existing `id::` preserved if present (block already had identity)
- Multi-line `refined_text` is split and each line added to content
- Idempotent: Retry adds same deterministic UUID if missing

---

## Property Management

### Property Order Preservation (NON-NEGOTIABLE)

**CRITICAL**: Property insertion order MUST be preserved. Logseq relies on property order for some functionality.

#### Get Property

```python
# Get property value
processed_value = block.get_property("processed")

# Returns None if property doesn't exist
if processed_value is None:
    # Property not set
    pass

```

**Contract**:

- Returns property value as string or `None`
- Case-sensitive property key matching

#### Set Property

```python
# Set property (preserves order if exists, appends if new)
block.set_property("id", "abc123-def456-...")
block.set_property("processed", "[Page](((uuid)))")

```

**Contract**:

- If property exists: value updated in-place, order preserved
- If property new: appended to end of properties
- **NEVER reorders existing properties**

#### Add Provenance Property

```python
def add_provenance(
    journal_block: LogseqBlock,
    target_page: str,
    target_block_id: str
) -> None:
    """
    Add extracted-to:: property to journal block.

    Format: extracted-to:: [Page Name](((uuid))), [Other Page](((uuid2)))

    Args:
        journal_block: Journal block that was processed
        target_page: Name of target page (may contain hierarchy: "Projects/Logsqueak")
        target_block_id: UUID of integrated block
    """
    # Convert hierarchical page names (Projects___Logsqueak -> Projects/Logsqueak)
    display_name = target_page.replace("___", "/")
    provenance_link = f"[{display_name}]((({target_block_id})))"

    # Get existing extracted-to:: value or empty string
    existing = journal_block.get_property("processed") or ""

    # Append new link (comma-separated)
    if existing:
        new_value = f"{existing}, {provenance_link}"
    else:
        new_value = provenance_link

    # Update property (preserves order)
    journal_block.set_property("processed", new_value)

```

**Contract**:

- Multiple integrations append to same `extracted-to::` property (comma-separated)
- Link format: `[Display Name](((block-uuid)))` (markdown link with Logseq block reference)
- Display name uses `/` for hierarchical pages
- Property order preserved (existing `extracted-to::` stays in same position)

### Property Format Examples

**Single integration**:

```markdown

- Knowledge block content
  extracted-to:: [Python/Concurrency](((550e8400-e29b-41d4-a716-446655440000)))
  id:: abc123-def456-...

```

**Multiple integrations**:

```markdown

- Knowledge block content
  extracted-to:: [Python/Concurrency](((550e8400-e29b-41d4-a716-446655440000))), [Textual/Architecture](((6ba7b810-9dad-11d1-80b4-00c04fd430c8)))
  id:: abc123-def456-...

```

**Property order preservation**:

```markdown

- Knowledge block content
  tags:: #python #async
  extracted-to:: [Python/Concurrency](((uuid)))
  id:: abc123-def456-...
  author:: User Name

```

If new integration added, `extracted-to::` line updates but stays in same position:

```markdown

- Knowledge block content
  tags:: #python #async
  extracted-to:: [Python/Concurrency](((uuid))), [Textual/Workers](((uuid2)))
  id:: abc123-def456-...
  author:: User Name

```

---

## Atomic Guarantees

### Atomic Integration Operation

Integration requires **TWO writes** to be atomic:

1. Write knowledge block to target page
2. Add provenance marker to journal entry

**Contract**: Journal marked with `extracted-to::` only after successful page write.

```python
from pathlib import Path
import structlog
from logseq_outline.graph import GraphPaths

logger = structlog.get_logger()


def write_integration_atomic(
    decision: IntegrationDecision,
    journal_date: str,
    graph_paths: GraphPaths,
    file_monitor: FileMonitor
) -> None:
    """
    Atomically write integration to page and update journal provenance.

    This function ensures both operations succeed or none.

    Args:
        decision: Integration decision to execute
        journal_date: Journal date for provenance link (YYYY-MM-DD)
        graph_paths: GraphPaths instance for path resolution
        file_monitor: FileMonitor for concurrent modification detection

    Raises:
        FileModifiedError: If file modified since last read
        ValueError: If target block doesn't exist (validation failed)
        OSError: On file I/O errors
    """
    # Construct file paths using GraphPaths
    page_path = graph_paths.page_path(decision.target_page)
    journal_path = graph_paths.journal_path(journal_date)

    # Step 1: Check and reload page if modified
    if file_monitor.is_modified(page_path):
        logger.info("file_modified_reload", path=str(page_path))
        page_outline = LogseqOutline.parse(page_path.read_text())
        file_monitor.refresh(page_path)

        # Re-validate decision
        validate_decision(decision, page_outline)
    else:
        page_outline = LogseqOutline.parse(page_path.read_text())

    # Step 2: Check if block already exists (idempotency / retry detection)
    # Generate deterministic UUID to check for existing integration
    expected_block_id = generate_integration_id(
        knowledge_block_id=decision.knowledge_block_id,
        target_page=decision.target_page,
        action=decision.action,
        target_block_id=decision.target_block_id
    )

    existing_block = page_outline.find_block_by_id(expected_block_id)
    if existing_block is not None:
        logger.info(
            "block_already_exists",
            page=decision.target_page,
            block_id=expected_block_id,
            message="Skipping page write (idempotent retry)"
        )
        new_block_id = expected_block_id
        # Skip to journal update (Step 4)
    else:
        # Step 3: Apply integration to page outline
        new_block_id = apply_integration(decision, page_outline)

        # Step 4: Write page (FIRST WRITE)
        try:
            page_path.write_text(page_outline.render())
            file_monitor.refresh(page_path)
            logger.info(
                "page_write_success",
                page=decision.target_page,
                block_id=new_block_id
            )
        except Exception as e:
            logger.error(
                "page_write_failed",
                page=decision.target_page,
                error=str(e)
            )
            raise

    # Step 5: Check and reload journal if modified
    if file_monitor.is_modified(journal_path):
        logger.info("file_modified_reload", path=str(journal_path))
        journal_outline = LogseqOutline.parse(journal_path.read_text())
        file_monitor.refresh(journal_path)
    else:
        journal_outline = LogseqOutline.parse(journal_path.read_text())

    # Step 6: Add provenance to journal
    journal_block = journal_outline.find_block_by_id(decision.knowledge_block_id)
    if journal_block is None:
        raise ValueError(
            f"Knowledge block not found in journal: {decision.knowledge_block_id}"
        )

    add_provenance(journal_block, decision.target_page, new_block_id)

    # Step 7: Write journal (SECOND WRITE - only if page write succeeded)
    try:
        journal_path.write_text(journal_outline.render())
        file_monitor.refresh(journal_path)
        logger.info(
            "journal_provenance_added",
            journal_date=journal_date,
            knowledge_block_id=decision.knowledge_block_id,
            target_page=decision.target_page
        )
    except Exception as e:
        logger.error(
            "journal_write_failed",
            journal_date=journal_date,
            error=str(e)
        )
        raise


def validate_decision(
    decision: IntegrationDecision,
    page_outline: LogseqOutline
) -> None:
    """
    Validate that decision can still be applied.

    Args:
        decision: Integration decision to validate
        page_outline: Current page outline

    Raises:
        ValueError: If target block doesn't exist or structure invalid
    """
    if decision.action in ["add_under", "replace"]:
        if decision.target_block_id is None:
            raise ValueError(
                f"Action {decision.action} requires target_block_id"
            )

        block = page_outline.find_block_by_id(decision.target_block_id)
        if block is None:
            raise ValueError(
                f"Target block not found: {decision.target_block_id} "
                f"in page {decision.target_page}"
            )


def apply_integration(
    decision: IntegrationDecision,
    page_outline: LogseqOutline
) -> str:
    """
    Apply integration decision to page outline.

    Args:
        decision: Integration decision to apply
        page_outline: Page outline to modify (mutated in-place)

    Returns:
        str: Deterministic UUID of newly created/updated block (for provenance)

    Raises:
        ValueError: If action invalid or target block not found
    """
    if decision.action == "add_section":
        return write_add_section(
            page_outline,
            decision.refined_text,
            knowledge_block_id=decision.knowledge_block_id,
            target_page=decision.target_page
        )

    elif decision.action == "add_under":
        return write_add_under(
            page_outline,
            decision.target_block_id,
            decision.refined_text,
            knowledge_block_id=decision.knowledge_block_id,
            target_page=decision.target_page
        )

    elif decision.action == "replace":
        return write_replace(
            page_outline,
            decision.target_block_id,
            decision.refined_text,
            knowledge_block_id=decision.knowledge_block_id,
            target_page=decision.target_page
        )

    else:
        raise ValueError(f"Unknown action: {decision.action}")

```

**Contract**:

- Deterministic UUID generated first to check for existing integration (idempotency)
- If block already exists (UUID match), skip page write and proceed to journal update
- Page written BEFORE journal marked (for new integrations)
- If page write fails, journal NOT marked
- If journal write fails, page write already succeeded (partial state - recoverable via retry)
- File modification checks before both writes
- Reload and re-validate if files modified externally

### Partial State Recovery

If journal write fails after successful page write:

```markdown
# Partial state (page written, journal not marked)

Page: Python/Concurrency

- New knowledge block with id:: deterministic-uuid  ← Successfully written

Journal: 2025-11-05

- Original knowledge block
  id:: abc123  ← Missing extracted-to:: property (journal write failed)

```

**Recovery**: User can re-run integration. System will:

1. Generate same deterministic UUID (same knowledge_block_id + target_page + action)
2. Search page for block with that UUID
3. Detect block already exists (UUID match)
4. Skip page write (idempotent)
5. Add provenance marker to journal (completing the atomic operation)

**Key insight**: Deterministic UUIDs enable true idempotency. The same integration inputs always produce the same UUID, allowing reliable duplicate detection on retry.

---

## Error Handling

### File Modification Errors

```python
class FileModifiedError(Exception):
    """Raised when file modified during operation."""
    pass


# Detect modification
if file_monitor.is_modified(path):
    # Reload file
    outline = LogseqOutline.parse(path.read_text())
    file_monitor.refresh(path)

    # Re-validate operation
    try:
        validate_operation(outline)
    except ValueError as e:
        raise FileModifiedError(
            f"File modified and validation failed: {e}"
        ) from e

```

**Contract**:

- Always check modification before write
- Reload and re-validate if modified
- Raise descriptive error if validation fails after reload

### Validation Errors

```python
# Target block not found
if block is None:
    raise ValueError(
        f"Target block not found: {target_block_id}\n"
        f"Page: {page_name}\n"
        f"Possible reasons:\n"
        f"- Block was deleted externally\n"
        f"- Block ID changed (id:: property modified)\n"
        f"- Page structure changed significantly"
    )

# Required property missing
if decision.action == "add_under" and decision.target_block_id is None:
    raise ValueError(
        f"Action 'add_under' requires target_block_id, but it is None"
    )

```

**Contract**:

- Validation errors include context (page name, block ID, action)
- Errors suggest possible causes and remediation
- Validation happens BEFORE any file writes

### I/O Errors

```python
# File not found
if not page_path.exists():
    raise FileNotFoundError(
        f"Page not found: {page_path}\n"
        f"Create the page in Logseq before integrating knowledge."
    )

# Permission errors
try:
    page_path.write_text(output)
except PermissionError as e:
    raise PermissionError(
        f"Cannot write to page: {page_path}\n"
        f"Check file permissions and ensure file is not open in another application."
    ) from e

```

**Contract**:

- File I/O errors include full path and suggested remediation
- All errors preserve original exception chain (use `from e`)

---

## File Race Condition Mitigation

### Problem

The FileMonitor checks modification **before** write, but there's a timing gap between check and actual write where external modification could occur:

1. System: `is_modified()` → False (file unchanged)
2. User: Edits file in Logseq (between check and write)
3. System: Writes file → **overwrites user's changes**

### Solution: Atomic File Write Pattern

Use a **write-to-temp-then-rename** pattern to minimize the window of vulnerability and enable detection of concurrent modifications.

```python
import os
import tempfile
from pathlib import Path

def atomic_write(
    file_path: Path,
    content: str,
    file_monitor: FileMonitor
) -> None:
    """
    Write file atomically with concurrent modification detection.

    Uses temp file + rename pattern to minimize data loss risk.
    Checks modification time before final rename.

    Args:
        file_path: Target file path
        content: Content to write
        file_monitor: FileMonitor for concurrent modification detection

    Raises:
        FileModifiedError: If file modified during write operation
        OSError: On file I/O errors

    Implementation:
        1. Check if file modified (early detection)
        2. Write content to temp file in same directory
        3. Fsync temp file (ensure data on disk)
        4. Check modification time again (late detection)
        5. Rename temp to target (atomic operation on POSIX)
        6. Update FileMonitor with new mtime
    """
    from .exceptions import FileModifiedError

    # Step 1: Early modification check
    if file_monitor.is_modified(file_path):
        raise FileModifiedError(
            f"File modified externally before write: {file_path}\n"
            f"Reload file and retry operation."
        )

    # Step 2: Write to temp file in same directory (for atomic rename)
    temp_fd, temp_path = tempfile.mkstemp(
        dir=file_path.parent,
        prefix=f".{file_path.name}.",
        suffix=".tmp"
    )

    try:
        # Write content
        os.write(temp_fd, content.encode('utf-8'))

        # Step 3: Fsync to ensure data on disk
        os.fsync(temp_fd)
        os.close(temp_fd)

        # Step 4: Late modification check (narrow window)
        if file_monitor.is_modified(file_path):
            # File changed during our write - abort
            os.unlink(temp_path)
            raise FileModifiedError(
                f"File modified externally during write: {file_path}\n"
                f"Reload file and retry operation."
            )

        # Step 5: Atomic rename (POSIX guarantees atomicity)
        os.rename(temp_path, file_path)

        # Step 6: Update monitor with new mtime
        file_monitor.refresh(file_path)

    except Exception as e:
        # Cleanup temp file on any error
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass
        raise


# Exception class (add to src/logsqueak/services/exceptions.py):
class FileModifiedError(Exception):
    """Raised when file modified externally during write operation."""
    pass
```

**Contract**:
- Two modification checks: before temp write and before rename
- Temp file in same directory as target (ensures same filesystem for atomic rename)
- Fsync before rename (ensures data persisted)
- Temp file cleanup on errors
- FileMonitor updated only after successful rename

**Race Window Reduction**:
- **Before**: Entire time between check and write completion (~10-100ms)
- **After**: Only between late check and rename (~1-5ms)
- **Residual Risk**: Still possible but much less likely

**Platform Notes**:
- POSIX systems (Linux, macOS): `os.rename()` is atomic
- Windows: May need `os.replace()` for atomicity (Python 3.3+)
- Both work on same filesystem only (temp in same directory)

**Alternative Approaches Considered**:

1. **File Locking**: `fcntl.flock()` on POSIX, but Logseq doesn't use locks
2. **Compare-and-Swap**: Check mtime before/after, but still race-prone
3. **Exclusive Open**: `O_EXCL` flag, but prevents concurrent reads

**Recommendation**: Use atomic write pattern for POC. For production, consider:
- Advisory locks if Logseq adds lock support
- User education: "Don't edit files while TUI is running"
- Auto-reload with merge conflict detection (advanced)

---

## Testing Contracts

### Round-Trip Test

```python
def test_round_trip_preserves_structure():
    """Test that parse -> render produces identical output."""
    original = """- Block 1
  property1:: value1
  property2:: value2

  - Child block

- Block 2
  id:: abc123
"""

    outline = LogseqOutline.parse(original)
    rendered = outline.render()

    assert rendered == original

```

### Property Order Test

```python
def test_property_order_preservation():
    """Test that property order is preserved after modifications."""
    original = """- Block
  tags:: #python
  extracted-to:: [[Page1]]((uuid1))
  id:: abc123
  author:: User
"""

    outline = LogseqOutline.parse(original)
    block = outline.blocks[0]

    # Update existing property
    block.set_property("processed", "[[Page1]]((uuid1)), [[Page2]]((uuid2))")

    rendered = outline.render()

    # Verify property order unchanged
    lines = rendered.strip().split("\n")
    assert "tags::" in lines[1]
    assert "extracted-to::" in lines[2]
    assert "id::" in lines[3]
    assert "author::" in lines[4]

```

### Atomic Write Test

```python
def test_atomic_write_failure_recovery():
    """Test that page write + journal marking is atomic."""
    # Simulate journal write failure
    with patch("pathlib.Path.write_text", side_effect=[None, OSError]):
        with pytest.raises(OSError):
            write_integration_atomic(decision, journal_date, graph_path, file_monitor)

    # Verify page was written but journal was not
    page_outline = LogseqOutline.parse(page_path.read_text())
    journal_outline = LogseqOutline.parse(journal_path.read_text())

    # Page should have new block
    new_block, _ = page_outline.find_block_by_id(new_block_id)
    assert new_block is not None

    # Journal should NOT have extracted-to:: marker
    journal_block, _ = journal_outline.find_block_by_id(knowledge_block_id)
    assert journal_block.get_property("processed") is None

```

---

## Summary

File operations in Logsqueak adhere to strict contracts:

1. **Deterministic UUIDs (NEW)**: Integration blocks use UUID v5 (deterministic) for true idempotency
2. **Property order preservation (NON-NEGOTIABLE)**: Never reorder properties
3. **Atomic writes**: Page written before journal marked
4. **Idempotent operations**: Retry detection via deterministic UUIDs prevents duplicate blocks
5. **Concurrent modification detection**: Check mtimes before all writes
6. **Non-destructive operations**: All changes traceable via `extracted-to::` markers
7. **Validation before writes**: Ensure targets exist before modifying files
8. **Descriptive errors**: All errors include context and remediation suggestions

These contracts ensure data integrity, traceability, and reliable recovery from partial failures while respecting Logseq's property order requirements.

### Why Deterministic UUIDs?

The deterministic UUID approach solves a critical problem with retry/recovery:

**Problem**: Random UUIDs make retry impossible to detect
- First attempt generates `uuid-1`, writes page, journal write fails
- Retry generates `uuid-2` (different!), can't find existing block
- Result: Duplicate blocks on page

**Solution**: Deterministic UUIDs based on integration inputs
- Same knowledge_block_id + target_page + action → same UUID every time
- Retry generates identical UUID, finds existing block via `find_block_by_id()`
- Result: Skip page write, complete journal update only (true idempotency)

This enables **partial state recovery**: If journal write fails, user can safely retry the integration without creating duplicates.
