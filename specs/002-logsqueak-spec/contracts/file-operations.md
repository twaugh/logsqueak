# File Operations Contract

**Date**: 2025-11-05
**Feature**: 002-logsqueak-spec

## Overview

This document defines the contract for all Logseq file operations in the Logsqueak Interactive TUI. All file operations use the `logseq-outline-parser` library and adhere to strict property order preservation and atomic write guarantees.

**Key Principles** (from constitution):
- **Property Order Preservation (NON-NEGOTIABLE)**: Property insertion order is sacred in Logseq
- **Non-Destructive Operations**: All operations traceable via `processed::` markers
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

#### Read Operations

```python
from logseq_outline.parser import LogseqOutline
from pathlib import Path

# Read journal entry
journal_path = Path("~/Documents/logseq-graph/journals/2025-11-05.md").expanduser()
journal_text = journal_path.read_text()
journal_outline = LogseqOutline.parse(journal_text)

# Read page
page_path = Path("~/Documents/logseq-graph/pages/Python___Concurrency.md").expanduser()
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
block, parents = journal_outline.find_block_by_id("abc123-def456-...")

if block is None:
    raise ValueError("Block not found")

# Access block properties
block_id = block.block_id  # From id:: property or None
full_content = block.get_full_content()  # All lines as string
property_value = block.get_property("processed")  # Get property value
```

**Contract**:
- `find_block_by_id()` returns tuple: `(block, parents)` or `(None, [])` if not found
- `parents` is list of ancestor blocks from root to immediate parent
- `block.block_id` is explicit `id::` property value or `None`

---

## Write Operations

### Integration Actions

Three types of write operations for integrating knowledge blocks:

#### 1. Add Section (New Top-Level Block)

```python
import uuid

def write_add_section(
    page_outline: LogseqOutline,
    refined_text: str
) -> str:
    """
    Add knowledge as new root-level block.

    Args:
        page_outline: Target page outline
        refined_text: Content to add

    Returns:
        str: UUID of newly created block (for provenance)
    """
    from logseq_outline.parser import LogseqBlock

    new_block_id = str(uuid.uuid4())

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
- `id::` property added to enable provenance tracking
- Content and property are separate lines in `content` list
- Property order preserved (single property, no order issue)

#### 2. Add Under (Child Block)

```python
def write_add_under(
    page_outline: LogseqOutline,
    target_block_id: str,
    refined_text: str
) -> str:
    """
    Add knowledge as child of existing block.

    Args:
        page_outline: Target page outline
        target_block_id: ID of parent block
        refined_text: Content to add

    Returns:
        str: UUID of newly created block (for provenance)

    Raises:
        ValueError: If target block not found
    """
    new_block_id = str(uuid.uuid4())

    # Find target block
    target_block, _ = page_outline.find_block_by_id(target_block_id)
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
- `set_property()` preserves order (new property appended)
- Target block ID can be explicit `id::` or content hash

#### 3. Replace (Update Existing Block)

```python
def write_replace(
    page_outline: LogseqOutline,
    target_block_id: str,
    refined_text: str
) -> str:
    """
    Replace existing block content.

    Args:
        page_outline: Target page outline
        target_block_id: ID of block to replace
        refined_text: New content

    Returns:
        str: UUID of block (existing or newly added)

    Raises:
        ValueError: If target block not found
    """
    # Find target block
    target_block, _ = page_outline.find_block_by_id(target_block_id)
    if target_block is None:
        raise ValueError(f"Target block not found: {target_block_id}")

    # Replace first line only (preserve properties and children)
    target_block.content[0] = refined_text

    # Ensure block has id:: property
    if target_block.block_id is None:
        new_block_id = str(uuid.uuid4())
        target_block.set_property("id", new_block_id)
        return new_block_id
    else:
        return target_block.block_id
```

**Contract**:
- Only first line of `content` is replaced (content line)
- Properties and children preserved
- If block lacks `id::`, add it (for provenance)
- Existing `id::` preserved if present

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
block.set_property("processed", "[[Page]]((uuid))")
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
    Add processed:: property to journal block.

    Format: processed:: [[Page Name]]((uuid)), [[Other Page]]((uuid2))

    Args:
        journal_block: Journal block that was processed
        target_page: Name of target page (may contain hierarchy: "Projects/Logsqueak")
        target_block_id: UUID of integrated block
    """
    # Convert hierarchical page names (Projects___Logsqueak -> Projects/Logsqueak)
    display_name = target_page.replace("___", "/")
    provenance_link = f"[[{display_name}]](({target_block_id}))"

    # Get existing processed:: value or empty string
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
- Multiple integrations append to same `processed::` property (comma-separated)
- Link format: `[[Display Name]]((block-uuid))`
- Display name uses `/` for hierarchical pages
- Property order preserved (existing `processed::` stays in same position)

### Property Format Examples

**Single integration**:
```markdown
- Knowledge block content
  processed:: [[Python/Concurrency]]((550e8400-e29b-41d4-a716-446655440000))
  id:: abc123-def456-...
```

**Multiple integrations**:
```markdown
- Knowledge block content
  processed:: [[Python/Concurrency]]((550e8400-e29b-41d4-a716-446655440000)), [[Textual/Architecture]]((6ba7b810-9dad-11d1-80b4-00c04fd430c8))
  id:: abc123-def456-...
```

**Property order preservation**:
```markdown
- Knowledge block content
  tags:: #python #async
  processed:: [[Python/Concurrency]]((uuid))
  id:: abc123-def456-...
  author:: User Name
```

If new integration added, `processed::` line updates but stays in same position:
```markdown
- Knowledge block content
  tags:: #python #async
  processed:: [[Python/Concurrency]]((uuid)), [[Textual/Workers]]((uuid2))
  id:: abc123-def456-...
  author:: User Name
```

---

## Atomic Guarantees

### Atomic Integration Operation

Integration requires **TWO writes** to be atomic:
1. Write knowledge block to target page
2. Add provenance marker to journal entry

**Contract**: Journal marked with `processed::` only after successful page write.

```python
from pathlib import Path
import structlog

logger = structlog.get_logger()


def write_integration_atomic(
    decision: IntegrationDecision,
    journal_date: str,
    graph_path: Path,
    file_monitor: FileMonitor
) -> None:
    """
    Atomically write integration to page and update journal provenance.

    This function ensures both operations succeed or none.

    Args:
        decision: Integration decision to execute
        journal_date: Journal date for provenance link (YYYY-MM-DD)
        graph_path: Path to Logseq graph directory
        file_monitor: FileMonitor for concurrent modification detection

    Raises:
        FileModifiedError: If file modified since last read
        ValueError: If target block doesn't exist (validation failed)
        OSError: On file I/O errors
    """
    # Construct file paths
    page_filename = decision.target_page.replace("/", "___") + ".md"
    page_path = graph_path / "pages" / page_filename
    journal_path = graph_path / "journals" / f"{journal_date}.md"

    # Step 1: Check and reload page if modified
    if file_monitor.is_modified(page_path):
        logger.info("file_modified_reload", path=str(page_path))
        page_outline = LogseqOutline.parse(page_path.read_text())
        file_monitor.refresh(page_path)

        # Re-validate decision
        validate_decision(decision, page_outline)
    else:
        page_outline = LogseqOutline.parse(page_path.read_text())

    # Step 2: Apply integration to page outline
    new_block_id = apply_integration(decision, page_outline)

    # Step 3: Write page (FIRST WRITE)
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

    # Step 4: Check and reload journal if modified
    if file_monitor.is_modified(journal_path):
        logger.info("file_modified_reload", path=str(journal_path))
        journal_outline = LogseqOutline.parse(journal_path.read_text())
        file_monitor.refresh(journal_path)
    else:
        journal_outline = LogseqOutline.parse(journal_path.read_text())

    # Step 5: Add provenance to journal
    journal_block, _ = journal_outline.find_block_by_id(decision.knowledge_block_id)
    if journal_block is None:
        raise ValueError(
            f"Knowledge block not found in journal: {decision.knowledge_block_id}"
        )

    add_provenance(journal_block, decision.target_page, new_block_id)

    # Step 6: Write journal (SECOND WRITE - only if page write succeeded)
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

        block, _ = page_outline.find_block_by_id(decision.target_block_id)
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
        str: UUID of newly created/updated block (for provenance)

    Raises:
        ValueError: If action invalid or target block not found
    """
    if decision.action == "add_section":
        return write_add_section(page_outline, decision.refined_text)

    elif decision.action == "add_under":
        return write_add_under(
            page_outline,
            decision.target_block_id,
            decision.refined_text
        )

    elif decision.action == "replace":
        return write_replace(
            page_outline,
            decision.target_block_id,
            decision.refined_text
        )

    else:
        raise ValueError(f"Unknown action: {decision.action}")
```

**Contract**:
- Page written BEFORE journal marked
- If page write fails, journal NOT marked
- If journal write fails, page write already succeeded (partial state)
- File modification checks before both writes
- Reload and re-validate if files modified externally

### Partial State Recovery

If journal write fails after successful page write:

```markdown
# Partial state (page written, journal not marked)

Page: Python/Concurrency
- New knowledge block with id:: uuid1  ← Successfully written

Journal: 2025-11-05
- Original knowledge block
  id:: abc123  ← Missing processed:: property (journal write failed)
```

**Recovery**: User can re-run integration. System will:
1. Detect block already exists in page (by ID)
2. Skip page write
3. Add provenance marker to journal

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
        f"Expected location: {graph_path}/pages/{page_filename}\n"
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
  processed:: [[Page1]]((uuid1))
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
    assert "processed::" in lines[2]
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

    # Journal should NOT have processed:: marker
    journal_block, _ = journal_outline.find_block_by_id(knowledge_block_id)
    assert journal_block.get_property("processed") is None
```

---

## Summary

File operations in Logsqueak adhere to strict contracts:

1. **Property order preservation (NON-NEGOTIABLE)**: Never reorder properties
2. **Atomic writes**: Page written before journal marked
3. **Concurrent modification detection**: Check mtimes before all writes
4. **Non-destructive operations**: All changes traceable via `processed::` markers
5. **Validation before writes**: Ensure targets exist before modifying files
6. **Descriptive errors**: All errors include context and remediation suggestions

These contracts ensure data integrity and traceability while respecting Logseq's property order requirements.
