# CLI Command Contracts

**Type**: Command-Line Interface
**Tool**: Logsqueak
**Date**: 2025-10-22

## Overview

Since Logsqueak is a CLI tool (not a REST API), this document defines command contracts instead of HTTP endpoints. These contracts specify the expected behavior, inputs, outputs, and exit codes for each command.

## Base Command

```
logsqueak [OPTIONS] COMMAND [ARGS]

```

### Global Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--config` | Path | `~/.config/logsqueak/config.yaml` | Path to configuration file |
| `--verbose` | Flag | false | Enable verbose logging |
| `--help` | Flag | - | Show help message |
| `--version` | Flag | - | Show version |

---

## Commands

### 1. `extract` - Extract Knowledge from Journals

Extract lasting knowledge from journal entries and integrate into pages.

**Signature**:

```
logsqueak extract [OPTIONS] [DATE_OR_RANGE]

```

**Arguments**:

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `DATE_OR_RANGE` | String | No | Date(s) to process. Default: today |

**Date Format Options**:

- Single date: `2025-01-15` (ISO 8601)
- Date range: `2025-01-10..2025-01-15` (inclusive)
- Relative: `today`, `yesterday`, `last-week`

**Options**:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--dry-run` / `--apply` | Flag | `--dry-run` | Show preview (dry-run) or apply changes |
| `--model` | String | From config | Override LLM model |
| `--graph` | Path | From config | Override Logseq graph path |

**Examples**:

```bash
# Extract from today (dry-run mode)
logsqueak extract

# Extract from specific date
logsqueak extract 2025-01-15

# Extract from date range
logsqueak extract 2025-01-10..2025-01-15

# Extract and apply immediately (skip dry-run)
logsqueak extract --apply 2025-01-15

# Use different model
logsqueak extract --model gpt-4 yesterday

```

**Output** (dry-run mode):

```
Loading configuration from ~/.config/logsqueak/config.yaml
Building page index... (found 566 pages)
✓ Indexed 566 pages in 0.8s (561 cached, 5 embedded)

Processing journal: 2025-01-15

Extracting knowledge... (using model: gpt-4-turbo-preview)
✓ Found 3 knowledge blocks

Matching to target pages...
  1/3: Finding candidates... ✓ Matched to "Project X" (similarity: 0.87)
  2/3: Finding candidates... ✓ Matched to "Project X" (similarity: 0.85)
  3/3: Activity log detected, skipping

Preview of proposed changes:

1. "Project X deadline moved to May"
   → Target: Project X (similarity: 0.87)
   → Section: Projects > Active > Timeline
   → Action: Add child block
   → Content: "- Project X deadline moved to May [[2025-01-15]]"

2. "Main competitor is Product Y"
   → Target: Project X (similarity: 0.85)
   → Section: (page root)
   → Action: Add child block
   → Content: "- Main competitor is Product Y [[2025-01-15]]"

3. "Attended meeting"
   ⚠ SKIPPED: Activity log (not lasting knowledge)

Summary:
  - Knowledge blocks found: 3
  - Will integrate: 2
  - Skipped: 1

Apply these changes? [y/N/e]:

```

**Interactive Prompt Responses**:

- `y` or `yes` - Apply all proposed changes
- `n` or `no` - Cancel without applying (default)
- `e` or `edit` - Edit preview before applying (v2 feature)

**Output** (after applying):

```
Applying changes...
✓ Updated: pages/Project X.md (2 additions)

Complete! Processed 3 blocks, integrated 2, skipped 1.

```

**Exit Codes**:

| Code | Meaning |
|------|---------|
| 0 | Success (dry-run completed or changes applied) |
| 1 | General error (invalid arguments, config missing) |
| 2 | LLM API error (connection failed, invalid response) |
| 3 | File I/O error (cannot read journal, cannot write page) |
| 4 | Validation error (malformed journal, invalid config) |

**Error Handling**:

```bash
# Missing config file
$ logsqueak extract
Error: Configuration file not found at ~/.config/logsqueak/config.yaml
Create config file or set LOGSQUEAK_* environment variables.
Exit code: 1

# Invalid date format
$ logsqueak extract invalid-date
Error: Invalid date format: "invalid-date"
Expected: YYYY-MM-DD, "today", "yesterday", or "YYYY-MM-DD..YYYY-MM-DD"
Exit code: 1

# Journal file not found
$ logsqueak extract 2025-01-01
Warning: Journal file not found: journals/2025_01_01.md
Skipping date: 2025-01-01
Exit code: 0  # Warning, not error

# LLM API error
$ logsqueak extract 2025-01-15
Error: Failed to connect to LLM API at https://api.openai.com/v1
Details: Connection refused
Check endpoint configuration and network connectivity.
Exit code: 2

# Malformed journal (FR-018)
$ logsqueak extract 2025-01-15
Warning: Malformed markdown in journals/2025_01_15.md
Skipping extraction for this entry.
Exit code: 0  # Continue processing

# Entry too large (FR-019)
$ logsqueak extract 2025-01-15
Warning: Journal entry exceeds 2000 lines (2543 lines)
Processing first 2000 lines only.
[continues with extraction...]
Exit code: 0

```

---

### 2. Future Commands (Roadmap)

These commands are referenced in RESEARCH.md but deferred to future versions:

#### `summarize` - Summarize Historical Journals

```
logsqueak summarize [OPTIONS] [DATE_RANGE]

```

**Status**: Roadmap (not in v1)

**Purpose**: Create hierarchical summaries of journal entries (daily → weekly → monthly).

---

## LLM Interaction Contract

While not exposed to users, documenting the LLM API contract for internal consistency.

The extraction workflow uses LLM in two stages:
1. **Initial extraction**: Extract knowledge blocks from journal (no page targeting)
2. **Page selection**: Match knowledge to target pages using RAG candidates

### Stage 1: Knowledge Extraction Request

**Endpoint**: Configured LLM endpoint (e.g., `https://api.openai.com/v1/chat/completions`)

**Request Body**:

```json
{
  "model": "gpt-4-turbo-preview",
  "response_format": {"type": "json_object"},
  "messages": [
    {
      "role": "system",
      "content": "You are a knowledge extraction assistant. Analyze journal entries and identify lasting knowledge vs. temporary activity logs. Extract knowledge blocks with confidence scores. Do NOT attempt to assign target pages - that will be done separately."
    },
    {
      "role": "user",
      "content": "Journal content:\n\n- worked on Project X\n- Discovered Project X deadline slipping to May\n- attended meeting\n\nExtract knowledge blocks (content + confidence only)."
    }
  ]
}

```

**Expected Response**:

```json
{
  "blocks": [
    {
      "content": "Project X deadline slipping to May",
      "confidence": 0.92
    }
  ]
}

```

**Response Validation** (via Pydantic):

- `content`: non-empty string
- `confidence`: 0.0 ≤ x ≤ 1.0

---

### Stage 2: Page Selection Request (with RAG)

After extracting knowledge blocks, use RAG to find candidate pages, then LLM selects best match.

**Workflow**:

```python
# 1. Build page index at startup
page_index = PageIndex.build(graph_path)

# 2. For each extracted knowledge block
knowledge = KnowledgeBlock(content="Project X deadline slipping to May", confidence=0.92)

# 3. Find top-5 semantically similar pages
candidates = page_index.find_similar(knowledge.content, top_k=5)
# Returns: [
#   (TargetPage("Project X"), 0.87),
#   (TargetPage("Q1 Planning"), 0.72),
#   (TargetPage("Vendor Relations"), 0.65),
#   ...
# ]

# 4. Pass to LLM for final selection
```

**Request Body**:

```json
{
  "model": "gpt-4-turbo-preview",
  "response_format": {"type": "json_object"},
  "messages": [
    {
      "role": "system",
      "content": "You are a page selection assistant. Given a knowledge block and candidate pages (from semantic search), select the most appropriate target page and section. Consider both the semantic similarity score and logical fit."
    },
    {
      "role": "user",
      "content": "Knowledge: \"Project X deadline slipping to May\"\n\nCandidate pages (with similarity scores):\n1. Project X (score: 0.87)\n2. Q1 Planning (score: 0.72)\n3. Vendor Relations (score: 0.65)\n\nSelect target page and section."
    }
  ]
}

```

**Expected Response**:

```json
{
  "target_page": "Project X",
  "target_section": ["Projects", "Active", "Timeline"],
  "suggested_action": "add_child",
  "reasoning": "Highest similarity score and directly mentions 'Project X' - Timeline section is appropriate for deadline information"
}

```

**Response Validation** (via Pydantic):

- `target_page`: must be one of the candidate page names
- `target_section`: optional list of strings (hierarchical path, e.g., ["Projects", "Active", "Timeline"])
- `suggested_action`: "add_child" | "create_section"
- `reasoning`: optional explanation for debugging/logging

---

## File System Contracts

### Input: Journal Entry Files

**Location**: `{graph_path}/journals/{date}.md`

**Format**: Logseq outline markdown

**Example**: `~/logseq-graph/journals/2025_01_15.md`

```markdown
- worked on [[Project X]]
- Discovered [[Project X]] deadline slipping to May
  - Original deadline was March
- attended meeting

```

**Constraints** (FR-019):

- Maximum 2000 lines (warn and truncate if exceeded)
- Must be valid Logseq outline (bullets with indentation)

### Output: Modified Page Files

**Location**: `{graph_path}/pages/{page_name}.md`

**Example**: `~/logseq-graph/pages/Project X.md` (after integration)

```markdown
- ## Timeline
  - Kickoff: January 2025
  - MVP: March 2025 (original)
  - Project X deadline slipping to May [[2025-01-15]]
- ## Team
  - Alice (PM)

```

**Guarantees** (FR-002, FR-006, FR-008):

- Original content preserved (non-destructive)
- New content added as child bullets only
- Provenance link included (FR-003)
- Outline formatting maintained (FR-010)

### Configuration File

**Location**: `~/.config/logsqueak/config.yaml`

**Schema** (enforced by Pydantic):

```yaml
llm:
  endpoint: string (valid HTTP(S) URL)
  api_key: string (non-empty)
  model: string (non-empty)

logseq:
  graph_path: string (existing directory path)

```

**Validation**:

- `endpoint`: Must be valid URL
- `graph_path`: Must exist and be readable/writable directory
- File must be valid YAML syntax

---

## Testing Contracts

Each command must have corresponding test coverage:

### Unit Tests

Located in `tests/unit/`:

- `test_cli_parsing.py` - Argument parsing and validation
- `test_date_parsing.py` - Date format handling
- `test_exit_codes.py` - Error code correctness

### Integration Tests

Located in `tests/integration/`:

- `test_extract_workflow.py` - Full extract command flow
- `test_error_handling.py` - Error scenarios with correct exit codes

### Contract Test Examples

```python
def test_extract_with_valid_date():
    """Contract: extract accepts ISO date format"""
    result = run_cli(["extract", "2025-01-15"])
    assert result.exit_code == 0

def test_extract_dry_run_default():
    """Contract: dry-run is default mode"""
    result = run_cli(["extract", "2025-01-15"])
    assert "--dry-run" in result.output or "Apply changes?" in result.output
    # Should NOT modify files in dry-run

def test_extract_invalid_date_format():
    """Contract: invalid date returns exit code 1"""
    result = run_cli(["extract", "not-a-date"])
    assert result.exit_code == 1
    assert "Invalid date format" in result.output

def test_duplicate_detection_skip():
    """Contract: FR-017 duplicate detection skips integration"""
    # Setup: page already contains the knowledge
    result = run_cli(["extract", "2025-01-15"])
    assert "SKIPPED: Duplicate" in result.output

```

---

## Compliance Checklist

All commands must satisfy:

- ✅ **FR-002**: Dry-run mode is default
- ✅ **FR-011**: LLM errors exit gracefully (code 2)
- ✅ **FR-012**: Interactive y/n/e prompt in dry-run
- ✅ **FR-013**: Accept date or date range
- ✅ **FR-016**: Load config from XDG location
- ✅ **FR-018**: Malformed entries logged, skipped, continue
- ✅ **FR-019**: 2000-line limit enforced with warning
- ✅ **SC-001**: Clear progress feedback throughout
- ✅ **SC-005**: 100% graceful error handling

---

## Versioning

CLI contract version follows semantic versioning:

- **MAJOR**: Breaking changes to command syntax or behavior
- **MINOR**: New commands or options (backward compatible)
- **PATCH**: Bug fixes, no interface changes

**Current Version**: 0.1.0 (POC)

No backwards compatibility guarantees until v1.0.0 per Constitution Principle I.
