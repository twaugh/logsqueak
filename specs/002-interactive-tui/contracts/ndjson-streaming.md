# NDJSON Streaming Contract

**Version**: 1.0 | **Date**: 2025-10-29 | **Feature**: 002-interactive-tui

## Purpose

This document specifies the interface contract between the LLM client and TUI for NDJSON (Newline-Delimited JSON) streaming. It defines:

1. **LLM Client Interface** - Async methods for streaming NDJSON from LLM providers
2. **NDJSON Parser Contract** - Format, buffering, and error handling
3. **Message Schemas** - JSON structure for each response type
4. **Error Handling** - Recovery strategies for malformed data, network errors, incomplete streams
5. **Example Flows** - Complete end-to-end scenarios

## Background

The Interactive TUI requires real-time streaming updates to provide instant visual feedback as the LLM processes journal entries. NDJSON (one JSON object per line) enables:

- **Incremental parsing**: UI updates as each object arrives
- **Robust error handling**: Malformed lines don't break the entire stream
- **Simple buffering**: Only need to buffer up to the next newline
- **Predictable format**: Each line is a complete, self-contained JSON object

**Validation**: Live testing against Ollama confirmed NDJSON works reliably, with 6 objects parsed across 129 chunks.

## 1. LLM Client Interface

The `LLMClient` abstract base class will be extended with three async streaming methods:

### 1.1 Stream Knowledge Extraction (Phase 1)

```python
from typing import AsyncIterator
from abc import ABC, abstractmethod

class LLMClient(ABC):

    @abstractmethod
    async def stream_extract_ndjson(
        self,
        blocks: list[dict]
    ) -> AsyncIterator[dict]:
        """
        Stream knowledge extraction results as NDJSON.

        Phase 1 of the extraction pipeline: Classify each journal block
        as either "knowledge" (lasting information) or "activity" (temporary log).

        Args:
            blocks: List of journal blocks to classify. Each dict contains:
                - block_id: str - Hybrid ID (id:: property or content hash)
                - content: str - Block text content
                - hierarchy: str - Full hierarchical context (parent blocks)

        Yields:
            dict: One classification result per block:
                {
                    "block_id": "abc123",
                    "is_knowledge": true,
                    "confidence": 0.85
                }

        Raises:
            LLMError: Network error, API failure, or timeout

        Notes:
            - Each yielded dict is a complete, parseable JSON object
            - Objects arrive in arbitrary order (not necessarily input order)
            - Malformed lines are logged and skipped (parser handles gracefully)
            - Stream may end early on network errors (partial results preserved)
        """
        pass
```

**Expected Output Format** (NDJSON):

```json
{"block_id": "abc123", "is_knowledge": true, "confidence": 0.85}
{"block_id": "def456", "is_knowledge": false, "confidence": 0.92}
{"block_id": "ghi789", "is_knowledge": true, "confidence": 0.78}
```

**Error Handling**:

- Network timeout → Retry with exponential backoff (max 3 attempts)
- Malformed JSON line → Log warning, skip line, continue parsing
- HTTP 4xx/5xx → Raise `LLMAPIError` with status code

### 1.2 Stream Integration Decisions (Phase 3.1 - Decider)

```python
class LLMClient(ABC):

    @abstractmethod
    async def stream_decisions_ndjson(
        self,
        knowledge_block: dict,
        candidate_pages: list[dict]
    ) -> AsyncIterator[dict]:
        """
        Stream integration decisions for one knowledge block across candidate pages.

        Phase 3.1 (Decider): For each candidate page, decide what action to take
        (skip, add_section, add_under, replace).

        Args:
            knowledge_block: Dict with:
                - block_id: str - Source block ID
                - content: str - Full-context knowledge text
                - hierarchical_text: str - Hierarchical markdown representation

            candidate_pages: List of candidate pages from Phase 2 RAG search. Each dict:
                - page_name: str - Page name
                - similarity_score: float - Semantic match score (0.0-1.0)
                - chunks: list[dict] - Blocks within page with:
                    - target_id: str - Hybrid ID of block
                    - content: str - Block text
                    - title: str - Human-readable block identifier

        Yields:
            dict: One decision per candidate page:
                {
                    "page": "Software Architecture",
                    "action": "add_section",
                    "target_id": null,
                    "target_title": null,
                    "confidence": 0.88,
                    "reasoning": "This knowledge belongs as a new root-level section"
                }

                OR (for targeted actions):

                {
                    "page": "Microservices",
                    "action": "add_under",
                    "target_id": "block-uuid-123",
                    "target_title": "Benefits and Tradeoffs",
                    "confidence": 0.91,
                    "reasoning": "Fits naturally under existing 'Benefits' discussion"
                }

                OR (for skip):

                {
                    "page": "Docker",
                    "action": "skip",
                    "target_id": null,
                    "target_title": null,
                    "confidence": 0.95,
                    "reasoning": "Already covered in this page"
                }

        Raises:
            LLMError: Network error, API failure, or timeout

        Notes:
            - One decision per candidate page (order may not match input)
            - LLM may decide "skip" for all pages (valid outcome)
            - target_id and target_title are null for "skip" and "add_section"
            - target_id and target_title are required for "add_under" and "replace"
        """
        pass
```

**Expected Output Format** (NDJSON):

```json
{"page": "Software Architecture", "action": "add_section", "target_id": null, "target_title": null, "confidence": 0.88, "reasoning": "New root section"}
{"page": "Microservices", "action": "add_under", "target_id": "block-uuid-123", "target_title": "Benefits", "confidence": 0.91, "reasoning": "Fits under Benefits"}
{"page": "Docker", "action": "skip", "target_id": null, "target_title": null, "confidence": 0.95, "reasoning": "Already covered"}
```

**Action Mapping** (TUI-friendly names):

- `"skip"` → User sees "Skip (already covered or not relevant)"
- `"add_section"` → User sees "Add as new section"
- `"add_under"` → User sees "Add under [target_title]"
- `"replace"` → User sees "Replace [target_title]"

### 1.3 Stream Content Rewording (Phase 3.2 - Reworder)

```python
class LLMClient(ABC):

    @abstractmethod
    async def stream_rewording_ndjson(
        self,
        decisions: list[dict]
    ) -> AsyncIterator[dict]:
        """
        Stream reworded content for accepted integration decisions.

        Phase 3.2 (Reworder): Transform journal-specific knowledge into clean,
        evergreen content suitable for integration into permanent pages.

        Args:
            decisions: List of accepted decisions from Phase 3.1 (action != "skip"). Each dict:
                - knowledge_block_id: str - Source block ID
                - page: str - Target page name
                - action: str - Action type ("add_section", "add_under", "replace")
                - full_text: str - Full-context knowledge text
                - hierarchical_text: str - Hierarchical markdown representation

        Yields:
            dict: One reworded result per decision:
                {
                    "knowledge_block_id": "abc123",
                    "page": "Software Architecture",
                    "refined_text": "Bounded contexts matter more than service size"
                }

        IMPORTANT - refined_text format:
            - Should be PLAIN BLOCK CONTENT (no bullet markers, no indentation)
            - Will be used as the content of a single Logseq block
            - If the knowledge has child blocks, they should be flattened into continuation lines
            - Example: "Main point.\nSupporting detail.\nAdditional context."
            - NOT: "- Main point\n  - Supporting detail\n  - Additional context"

        Raises:
            LLMError: Network error, API failure, or timeout

        Notes:
            - Refined text removes journal-specific context (dates, "Today I learned", etc.)
            - Preserves important links: [[Page Links]] and ((block refs))
            - Maintains proper Logseq markdown structure (indented bullets)
            - Results may arrive in any order
            - Each result corresponds to exactly one input decision
        """
        pass
```

**Expected Output Format** (NDJSON):

```json
{"block_id": "abc123", "refined_text": "Bounded contexts matter more than service size.\nEach context should have clear ownership."}
{"block_id": "def456", "refined_text": "HNSW provides O(log n) nearest neighbor search.\nTrade-off between recall and speed via ef_construction parameter."}
```

**Note**: Each knowledge block is reworded only once, then the refined text is applied to all decisions that reference that block. The `page` field has been removed since rewording is independent of the target page.

**Refinement Rules**:

- Remove: "Today I learned", "Had discussion with X", temporal references
- Preserve: `[[Page Links]]`, `((block-refs))`, technical terminology
- Structure: Plain block content (no bullet markers or indentation)
  - Output will be used as the content of a single Logseq block
  - Multi-line content should use continuation lines (separated by `\n`)
  - Do NOT include hierarchical markdown structure with bullets

## 2. NDJSON Parser Contract

### 2.1 Parser Interface

```python
from typing import AsyncIterator
import json
import logging

async def parse_ndjson_stream(
    stream: AsyncIterator[str]
) -> AsyncIterator[dict]:
    """
    Parse NDJSON from chunked async stream.

    Implements line-buffering with incremental JSON parsing. Each complete line
    (ending with '\\n') is parsed as a standalone JSON object.

    Args:
        stream: Async iterator yielding string chunks from LLM provider.
                Chunks are token-by-token: '{"', 'foo', '": ', '"bar', '"}', '\\n', ...

    Yields:
        dict: Complete JSON objects as they become parseable (one per line)

    Error Handling:
        - Invalid JSON on a line → Log warning, skip line, continue
        - Incomplete line at stream end → Log warning, discard
        - Empty lines → Skip silently

    Performance:
        - Memory usage: O(longest line) - typically <1KB per line
        - Latency: Yields immediately upon line completion (no batching)
        - Throughput: Validated at ~21 chunks per object (128 chunks → 6 objects)

    Example:
        >>> async for obj in parse_ndjson_stream(llm_stream):
        ...     print(f"Received: {obj}")
        Received: {'block_id': 'abc123', 'is_knowledge': True, 'confidence': 0.85}
        Received: {'block_id': 'def456', 'is_knowledge': False, 'confidence': 0.92}
    """
    buffer = ""
    line_number = 0

    async for chunk in stream:
        buffer += chunk

        # Process all complete lines in buffer
        while '\n' in buffer:
            line_end = buffer.index('\n')
            complete_line = buffer[:line_end].strip()
            buffer = buffer[line_end + 1:]  # Remove processed line + newline
            line_number += 1

            if not complete_line:
                # Empty line, skip silently
                continue

            try:
                obj = json.loads(complete_line)

                # Validate expected type
                if not isinstance(obj, dict):
                    logging.warning(
                        f"Line {line_number}: Expected JSON object, got {type(obj).__name__}"
                    )
                    continue

                yield obj

            except json.JSONDecodeError as e:
                # Log error but continue processing
                logging.warning(
                    f"Line {line_number}: Malformed JSON: {complete_line[:100]}... "
                    f"Error at position {e.pos}: {e.msg}"
                )
                # Don't yield anything, just continue to next line

    # Handle any remaining content in buffer at stream end
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

### 2.2 Input Format

**Chunks from LLM provider** (token-by-token):

```python
# Ollama streaming response chunks:
["{\n", "\"block_id\": \"", "abc", "123", "\",\n", "\"is_knowledge\": ", "true", ",\n", "\"confidence\": 0.", "85", "\n}\n"]

# OpenAI-compatible streaming chunks (wrapped in SSE format):
["data: {\"choices\":[{\"delta\":{\"content\":\"{\"}}]}\n",
 "data: {\"choices\":[{\"delta\":{\"content\":\"block_id\"}}]}\n",
 "data: {\"choices\":[{\"delta\":{\"content\":\":\"abc123\"}}]}\n",
 ...]
```

**Parser must handle**:

- Chunks split mid-token: `"confi"` + `"dence"` → `"confidence"`
- Chunks split mid-line: `'{"foo": '` + `'"bar"}\n'` → Wait for `\n`, then parse
- Multiple lines per chunk: `'{"a":1}\n{"b":2}\n'` → Parse both objects

### 2.3 Output Format

**Complete JSON objects** (one per line):

```python
# Phase 1 extraction:
{"block_id": "abc123", "is_knowledge": true, "confidence": 0.85}
{"block_id": "def456", "is_knowledge": false, "confidence": 0.92}

# Phase 3.1 decisions:
{"page": "Software Architecture", "action": "add_section", "target_id": null, "target_title": null, "confidence": 0.88, "reasoning": "..."}
{"page": "Microservices", "action": "skip", "target_id": null, "target_title": null, "confidence": 0.95, "reasoning": "..."}

# Phase 3.2 rewording:
{"block_id": "abc123", "refined_text": "- Bounded contexts..."}
```

### 2.4 Performance Characteristics

**From validation testing** (`/tmp/test_ndjson_streaming_incremental.py`):

- **Input**: 129 chunks (token-by-token from Ollama)
- **Output**: 6 complete JSON objects
- **Average**: ~21 chunks per object
- **Latency**: Each object available immediately upon line completion
- **Memory**: Only buffers up to next newline (typically <1KB)

**Expected performance**:

- UI update latency: <50ms from object arrival to visual feedback
- No blocking: Parser yields immediately, UI updates asynchronously
- Scalability: Supports 100+ blocks per journal entry without buffering issues

## 3. Message Schemas

### 3.1 Phase 1: Extraction Result Schema

**Schema**:

```json
{
  "type": "object",
  "required": ["block_id", "is_knowledge", "confidence"],
  "properties": {
    "block_id": {
      "type": "string",
      "description": "Hybrid ID of journal block (id:: property or content hash)"
    },
    "is_knowledge": {
      "type": "boolean",
      "description": "True if lasting knowledge, false if temporary activity"
    },
    "confidence": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 1.0,
      "description": "LLM confidence score"
    }
  }
}
```

**Example**:

```json
{"block_id": "abc123", "is_knowledge": true, "confidence": 0.85}
```

**Validation Rules**:

- `block_id` must be non-empty string
- `is_knowledge` must be boolean (not string "true"/"false")
- `confidence` must be float in range [0.0, 1.0]

**Error Handling**:

- Missing required field → Log warning, skip object, continue parsing
- Invalid type (e.g., confidence="high") → Log warning, skip object
- Unknown fields (e.g., "extra_field") → Ignore, use valid fields

### 3.2 Phase 3.1: Decision Result Schema

**Schema**:

```json
{
  "type": "object",
  "required": ["page", "action", "confidence", "reasoning"],
  "properties": {
    "page": {
      "type": "string",
      "description": "Target page name"
    },
    "action": {
      "type": "string",
      "enum": ["skip", "add_section", "add_under", "replace"],
      "description": "Integration action"
    },
    "target_id": {
      "type": ["string", "null"],
      "description": "Hybrid ID of target block (null for skip/add_section)"
    },
    "target_title": {
      "type": ["string", "null"],
      "description": "Human-readable target block name (null for skip/add_section)"
    },
    "confidence": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 1.0,
      "description": "Decision confidence score"
    },
    "reasoning": {
      "type": "string",
      "description": "LLM explanation for the decision"
    }
  }
}
```

**Example (add_section)**:

```json
{
  "page": "Software Architecture",
  "action": "add_section",
  "target_id": null,
  "target_title": null,
  "confidence": 0.88,
  "reasoning": "This knowledge belongs as a new root-level section on the page"
}
```

**Example (add_under)**:

```json
{
  "page": "Microservices",
  "action": "add_under",
  "target_id": "block-uuid-123",
  "target_title": "Benefits and Tradeoffs",
  "confidence": 0.91,
  "reasoning": "This insight fits naturally under the existing 'Benefits' section"
}
```

**Validation Rules**:

- `action` must be one of: "skip", "add_section", "add_under", "replace"
- If `action` is "skip" or "add_section", `target_id` and `target_title` must be null
- If `action` is "add_under" or "replace", `target_id` and `target_title` must be non-null strings
- `confidence` must be in range [0.0, 1.0]
- `reasoning` should be non-empty (but parser won't fail if empty)

**Error Handling**:

- Invalid action value → Log warning, default to "skip"
- Missing target_id when required → Log warning, change action to "add_section"
- Invalid confidence range → Log warning, clamp to [0.0, 1.0]

### 3.3 Phase 3.2: Rewording Result Schema

**Schema**:

```json
{
  "type": "object",
  "required": ["block_id", "refined_text"],
  "properties": {
    "block_id": {
      "type": "string",
      "description": "Source block ID from Phase 1 (each unique block is reworded once)"
    },
    "refined_text": {
      "type": "string",
      "description": "Reworded content as plain block content (no bullet markers or indentation)"
    }
  }
}
```

**Note**: The `page` field has been removed. Each knowledge block is reworded only once, and the refined text is then applied to all decisions that reference that block ID. This avoids redundant rewording requests when the same knowledge is being integrated into multiple pages.

**Example**:

```json
{
  "block_id": "abc123",
  "refined_text": "Bounded contexts matter more than service size.\nEach context should have clear ownership.\nAvoid sharing databases across contexts."
}
```

**Validation Rules**:

- `block_id` must match a block from Phase 1
- `refined_text` should be plain block content (no bullet markers or indentation)
- Preserve `[[Page Links]]` and `((block-refs))` from original content
- The same refined text will be applied to all decisions that reference this block ID

**Error Handling**:

- Missing required field → Log warning, skip object
- Unmatched block_id or page → Log warning, skip object
- Empty refined_text → Log warning, skip object

## 4. Error Handling

### 4.1 Malformed JSON Lines

**Scenario**: LLM outputs invalid JSON on a line.

**Examples**:

```json
{"block_id": "abc123", is_knowledge: true}  ← Missing quotes around key
{"block_id": "def456", "is_knowledge": true,}  ← Trailing comma
{block_id: "ghi789"}  ← Unquoted key
```

**Handling**:

1. Parser catches `json.JSONDecodeError`
2. Logs warning with line number and first 100 chars
3. Continues parsing next line (doesn't raise exception)
4. UI shows partial results, user can retry failed blocks

**Logging Example**:

```
WARNING: Line 2: Malformed JSON: {"block_id": "def456", "is_knowledge": true,} Error at position 49: Expecting property name enclosed in double quotes
```

### 4.2 Network Errors

**Scenario**: Connection drops, timeout, or HTTP error mid-stream.

**Types**:

- `httpx.TimeoutException` - Request timeout (default: 30s)
- `httpx.RequestError` - Network unreachable, connection refused
- `httpx.HTTPStatusError` - 4xx/5xx response codes

**Handling**:

**Timeout and Network Errors** (retry with exponential backoff):

```python
async def stream_with_retry(
    endpoint: str,
    payload: dict,
    headers: dict,
    max_retries: int = 3,
    initial_backoff: float = 1.0
) -> AsyncIterator[str]:
    """Stream with automatic retry on transient errors"""

    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                async with client.stream("POST", endpoint, json=payload, headers=headers) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        yield line
                    return  # Success

        except (httpx.TimeoutException, httpx.RequestError) as e:
            logging.warning(f"Attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                backoff = initial_backoff * (2 ** attempt)
                logging.info(f"Retrying in {backoff:.1f}s...")
                await asyncio.sleep(backoff)
            else:
                raise LLMTimeoutError(f"Failed after {max_retries} attempts: {e}")
```

**HTTP Errors** (4xx/5xx - don't retry):

```python
except httpx.HTTPStatusError as e:
    logging.error(f"HTTP error {e.response.status_code}: {e.response.text}")
    raise LLMAPIError(
        f"LLM API error: {e.response.status_code}",
        status_code=e.response.status_code
    )
```

**UI Response**:

- Show error banner: "Network error. Retrying... (2/3)"
- On final failure: "Could not reach LLM server. Some blocks not classified."
- User can manually retry or proceed with partial results

### 4.3 Incomplete Streams

**Scenario**: Stream ends before all expected objects are received.

**Causes**:

- Network interruption mid-stream
- LLM stops generating (API timeout, rate limit)
- Incomplete final line (no trailing newline)

**Handling**:

```python
async def parse_ndjson_stream(stream: AsyncIterator[str]) -> AsyncIterator[dict]:
    buffer = ""

    async for chunk in stream:
        buffer += chunk
        # ... process complete lines

    # Handle incomplete final line
    if buffer.strip():
        try:
            obj = json.loads(buffer.strip())
            yield obj  # Final object without newline
        except json.JSONDecodeError:
            logging.warning(
                f"Discarding incomplete NDJSON at end of stream: {buffer[:100]}..."
            )
            # Don't yield anything
```

**UI Response**:

- Compare expected vs received objects
- Phase 1: If 10 blocks sent but only 8 classifications received, mark 2 as "pending"
- Show summary: "Classified 8/10 blocks. 2 blocks need retry."
- User can retry failed blocks or manually classify them

### 4.4 Retry Strategies

**Automatic Retries** (background, transparent to user):

- Network timeout: 3 retries with exponential backoff (1s, 2s, 4s)
- Connection refused: 3 retries
- Transient HTTP errors (502, 503, 504): 3 retries

**Manual Retries** (user-initiated):

- User presses `R` key to retry failed blocks
- TUI sends only failed blocks to LLM (not all blocks)
- Results merged with existing successful classifications

**Example Manual Retry Flow**:

1. Phase 1 completes with 2 failed blocks
2. User sees: "Classified 8/10 blocks. 2 pending. Press R to retry."
3. User presses `R`
4. TUI calls `stream_extract_ndjson(failed_blocks_only)`
5. Results update in real-time as they arrive
6. User can proceed once satisfied

## 5. Example Flows

### 5.1 Successful Extraction with 3 Blocks

**Setup**:

- Journal entry with 3 root blocks
- All blocks classified successfully
- NDJSON stream completes normally

**Flow**:

**1. TUI calls LLM client**:

```python
blocks = [
    {"block_id": "abc123", "content": "Read paper on vector databases", "hierarchy": "..."},
    {"block_id": "def456", "content": "Dentist appointment at 3pm", "hierarchy": "..."},
    {"block_id": "ghi789", "content": "ChromaDB uses HNSW for ANN search", "hierarchy": "..."},
]

async for result in llm_client.stream_extract_ndjson(blocks):
    # Process result...
```

**2. LLM responds with NDJSON stream**:

```
{"block_id": "abc123", "is_knowledge": true, "confidence": 0.92}
{"block_id": "def456", "is_knowledge": false, "confidence": 0.95}
{"block_id": "ghi789", "is_knowledge": true, "confidence": 0.88}
```

**3. Parser yields objects**:

```python
# Object 1 arrives
{"block_id": "abc123", "is_knowledge": True, "confidence": 0.92}

# Object 2 arrives
{"block_id": "def456", "is_knowledge": False, "confidence": 0.95}

# Object 3 arrives
{"block_id": "ghi789", "is_knowledge": True, "confidence": 0.88}
```

**4. TUI updates in real-time**:

```
Analyzing Journal: 2025-01-15
✓ Read paper on vector databases  92%
✗ Dentist appointment at 3pm  95%
✓ ChromaDB uses HNSW for ANN search  88%

Classified 3/3 blocks. Press Enter to continue.
```

**5. User proceeds to Phase 2**.

### 5.2 Decision Making with 2 Pages

**Setup**:

- 1 knowledge block: "Bounded contexts matter more than service size"
- 2 candidate pages from RAG: "Software Architecture", "Microservices"
- LLM decides: add_section to Architecture, skip Microservices

**Flow**:

**1. TUI calls LLM client**:

```python
knowledge_block = {
    "block_id": "abc123",
    "content": "Bounded contexts matter more than service size",
    "hierarchical_text": "- Read paper on DDD\n  - Bounded contexts matter more than service size"
}

candidate_pages = [
    {
        "page_name": "Software Architecture",
        "similarity_score": 0.87,
        "chunks": [
            {"target_id": "arch-001", "content": "Microservices architecture overview", "title": "Overview"},
            {"target_id": "arch-002", "content": "Service boundaries", "title": "Boundaries"},
        ]
    },
    {
        "page_name": "Microservices",
        "similarity_score": 0.82,
        "chunks": [
            {"target_id": "micro-001", "content": "Size doesn't matter", "title": "Common Misconceptions"},
        ]
    },
]

async for decision in llm_client.stream_decisions_ndjson(knowledge_block, candidate_pages):
    # Process decision...
```

**2. LLM responds with NDJSON stream**:

```
{"page": "Software Architecture", "action": "add_section", "target_id": null, "target_title": null, "confidence": 0.88, "reasoning": "Core architectural principle that deserves its own section"}
{"page": "Microservices", "action": "skip", "target_id": null, "target_title": null, "confidence": 0.93, "reasoning": "Already covered under 'Common Misconceptions' section"}
```

**3. Parser yields objects**:

```python
# Decision 1 arrives
{
    "page": "Software Architecture",
    "action": "add_section",
    "target_id": None,
    "target_title": None,
    "confidence": 0.88,
    "reasoning": "Core architectural principle that deserves its own section"
}

# Decision 2 arrives
{
    "page": "Microservices",
    "action": "skip",
    "target_id": None,
    "target_title": None,
    "confidence": 0.93,
    "reasoning": "Already covered under 'Common Misconceptions' section"
}
```

**4. TUI updates in real-time**:

```
Integration Decisions for: "Bounded contexts matter more than service size"

Software Architecture  87%
  ⊕ Add as new section  88%
    Reasoning: Core architectural principle that deserves its own section

Microservices  82%
  ⊘ Skip  93%
    Reasoning: Already covered under 'Common Misconceptions' section

Press Enter to continue to rewording.
```

**5. User proceeds to Phase 3.2 (Rewording)**.

### 5.3 Error Recovery Scenario

**Setup**:

- Journal entry with 5 blocks
- LLM successfully classifies 3 blocks
- Malformed JSON on line 4
- Network timeout before block 5 completes

**Flow**:

**1. TUI calls LLM client with 5 blocks**.

**2. LLM responds with partial NDJSON stream**:

```
{"block_id": "block-1", "is_knowledge": true, "confidence": 0.85}
{"block_id": "block-2", "is_knowledge": false, "confidence": 0.92}
{"block_id": "block-3", "is_knowledge": true, "confidence": 0.78}
{"block_id": "block-4", is_knowledge: true, "confidence": 0.88}  ← Malformed (missing quotes)
[stream times out here, block-5 never received]
```

**3. Parser processes**:

```python
# Line 1: Valid → Yield
{"block_id": "block-1", "is_knowledge": True, "confidence": 0.85}

# Line 2: Valid → Yield
{"block_id": "block-2", "is_knowledge": False, "confidence": 0.92}

# Line 3: Valid → Yield
{"block_id": "block-3", "is_knowledge": True, "confidence": 0.78}

# Line 4: Malformed → Log warning, skip
# WARNING: Line 4: Malformed JSON: {"block_id": "block-4", is_knowledge: true...

# Line 5: Never arrives (timeout)
# WARNING: Network timeout after 30s
```

**4. TUI shows partial results**:

```
Analyzing Journal: 2025-01-15
✓ Block 1 content...  85%
✗ Block 2 content...  92%
✓ Block 3 content...  78%
? Block 4 content...  [ERROR]
? Block 5 content...  [PENDING]

⚠ Classified 3/5 blocks. 1 error, 1 timeout. Press R to retry or Enter to continue.
```

**5. User presses R to retry**:

```python
# TUI retries only failed blocks
retry_blocks = [
    {"block_id": "block-4", "content": "...", "hierarchy": "..."},
    {"block_id": "block-5", "content": "...", "hierarchy": "..."},
]

async for result in llm_client.stream_extract_ndjson(retry_blocks):
    # Update state...
```

**6. LLM responds with valid NDJSON**:

```
{"block_id": "block-4", "is_knowledge": true, "confidence": 0.88}
{"block_id": "block-5", "is_knowledge": false, "confidence": 0.91}
```

**7. TUI updates**:

```
Analyzing Journal: 2025-01-15
✓ Block 1 content...  85%
✗ Block 2 content...  92%
✓ Block 3 content...  78%
✓ Block 4 content...  88%  [RETRY SUCCESS]
✗ Block 5 content...  91%  [RETRY SUCCESS]

Classified 5/5 blocks. Press Enter to continue.
```

**8. User proceeds to Phase 2**.

---

## 6. Implementation Checklist

- [ ] Implement `parse_ndjson_stream()` in `src/logsqueak/llm/streaming.py`
- [ ] Add `stream_extract_ndjson()` to `LLMClient` abstract class
- [ ] Add `stream_decisions_ndjson()` to `LLMClient` abstract class
- [ ] Add `stream_rewording_ndjson()` to `LLMClient` abstract class
- [ ] Implement HTTP streaming protocol handling in `OpenAICompatibleProvider`
- [ ] Add retry logic with exponential backoff
- [ ] Add schema validation for all message types
- [ ] Write unit tests for malformed JSON handling
- [ ] Write unit tests for incomplete stream handling
- [ ] Write integration test for full extraction flow
- [ ] Add error banner widget to TUI screens
- [ ] Implement manual retry mechanism (R key binding)
- [ ] Update prompt templates to request NDJSON output format
- [ ] Add logging for all error scenarios
- [ ] Document NDJSON format in CLAUDE.md

---

## 7. References

- **Plan**: `/home/twaugh/devel/logsqueak/specs/002-interactive-tui/plan.md` (Phase 1.2)
- **Research**: `/home/twaugh/devel/logsqueak/specs/002-interactive-tui/research.md` (Topic 1)
- **LLM Client**: `/home/twaugh/devel/logsqueak/src/logsqueak/llm/client.py`
- **Models**: `/home/twaugh/devel/logsqueak/src/logsqueak/models/knowledge.py`
- **Validation Test**: `/tmp/test_ndjson_streaming_incremental.py` (6 objects across 129 chunks)

---

**Contract Version**: 1.0 | **Status**: Ready for Implementation | **Last Updated**: 2025-10-29
