# LLM Streaming Contract

**Date**: 2025-11-05

**Feature**: 002-logsqueak-spec

## Overview

This document defines the NDJSON streaming formats for all LLM responses in the Logsqueak Interactive TUI. All LLM interactions use **NDJSON (Newline-Delimited JSON)** format for streaming incremental results.

**Decision Rationale** (from research.md):

- NDJSON is simpler than SSE (no `data:` prefix parsing)
- Each line is complete JSON object (easier to log and debug)
- Error isolation: bad JSON on one line doesn't corrupt stream
- Perfect fit with httpx.aiter_lines()
- Compatible with both OpenAI and Ollama APIs

---

## NDJSON Format Specification

### Structure

Each line in the response stream is:

1. A complete, valid JSON object
2. Terminated by a newline character (`\n`)
3. No special markers or prefixes required

### Parsing Pattern

```python
import httpx
import json

async def parse_ndjson_stream(url, payload):
    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream("POST", url, json=payload) as response:
            async for line in response.aiter_lines():
                if line.strip():  # Skip empty lines
                    try:
                        data = json.loads(line)
                        yield data
                    except json.JSONDecodeError as e:
                        logger.error("malformed_json_line", line=line, error=str(e))
                        continue  # Skip bad line, process remaining stream

```

### Error Handling

- **Malformed JSON**: Log error, skip line, continue with remaining stream
- **Network timeout**: Preserve partial results, trigger retry logic
- **Connection closed**: Treat as end-of-stream (no explicit marker needed)

---

## Phase 1: Knowledge Block Classification Stream

### Purpose
Stream incremental classification results as LLM analyzes journal blocks.

### Request Format

**Endpoint**: `{config.llm.endpoint}` (OpenAI-compatible)

**Headers**:

```
Authorization: Bearer {config.llm.api_key}
Content-Type: application/json

```

**Request Body**:

```json
{
  "model": "{config.llm.model}",
  "messages": [
    {
      "role": "system",
      "content": "You are an AI assistant that identifies lasting knowledge in journal entries. Classify each block as either 'knowledge' (lasting insight worth preserving) or 'activity' (temporal event/task). For each block classified as knowledge, respond with a single-line JSON object containing: block_id (string), is_knowledge (boolean, always true for returned items), confidence (float 0-1), reason (string explaining why this is knowledge). Only return blocks that are knowledge - omit activity blocks from output. Output one JSON object per line (NDJSON format)."
    },
    {
      "role": "user",
      "content": "Analyze these journal blocks:\n\n{formatted_blocks}"
    }
  ],
  "stream": true,
  "temperature": 0.3
}

```

**Formatted Blocks Example**:

```
Block ID: abc123
Content: - Learned that Python's asyncio.Queue is thread-safe
  This is important for background tasks in Textual

Block ID: def456
Content: - Had lunch with Sarah
  Discussed project timeline

Block ID: ghi789
Content: - Discovered that ChromaDB requires sentence-transformers
  It downloads a 90MB model on first run

```

### Response Format (NDJSON Stream)

Each line contains one classification result for blocks identified as knowledge:

```json
{"block_id":"abc123","is_knowledge":true,"confidence":0.92,"reason":"Contains a reusable insight about Python's asyncio.Queue being thread-safe, which is evergreen knowledge applicable to any async Python development."}

```

```json
{"block_id":"ghi789","is_knowledge":true,"confidence":0.85,"reason":"Documents an important technical detail about ChromaDB's dependencies that would be valuable for future reference when setting up projects."}

```

**Note**: Activity blocks (like `def456`) are NOT included in the output stream.

### Pydantic Model

```python
from pydantic import BaseModel, Field

class KnowledgeClassificationChunk(BaseModel):
    """Single classification result from NDJSON stream."""

    block_id: str = Field(
        ...,
        description="Block identifier from input"
    )

    is_knowledge: bool = Field(
        ...,
        description="Whether block contains knowledge (always true in output)"
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score (0.0-1.0)"
    )

    reason: str = Field(
        ...,
        description="Explanation for why this is knowledge"
    )

```

### Example NDJSON Stream

Complete stream for 3 journal blocks (2 knowledge, 1 activity):

```
{"block_id":"abc123","is_knowledge":true,"confidence":0.92,"reason":"Contains a reusable insight about Python's asyncio.Queue being thread-safe, which is evergreen knowledge applicable to any async Python development."}
{"block_id":"ghi789","is_knowledge":true,"confidence":0.85,"reason":"Documents an important technical detail about ChromaDB's dependencies that would be valuable for future reference when setting up projects."}

```

(Note: Block `def456` about lunch is omitted as it's activity, not knowledge)

### End-of-Stream

- Connection closes when LLM finishes
- No explicit `[DONE]` marker required
- Empty lines ignored during parsing

---

## Phase 2: Content Rewording Stream

### Purpose
Stream reworded versions of knowledge blocks that remove temporal context.

### Request Format

**Request Body**:

```json
{
  "model": "{config.llm.model}",
  "messages": [
    {
      "role": "system",
      "content": "You are an AI assistant that transforms journal-style content into evergreen knowledge. Remove temporal context (dates, 'today', 'yesterday'), convert first-person to third-person or neutral, and make the content timeless. Preserve all technical details and insights. For each block, respond with a single-line JSON object containing: block_id (string), reworded_content (string). Output one JSON object per line (NDJSON format)."
    },
    {
      "role": "user",
      "content": "Reword these knowledge blocks:\n\n{formatted_blocks}"
    }
  ],
  "stream": true,
  "temperature": 0.5
}

```

**Formatted Blocks Example**:

```
Block ID: abc123
Original Content: - Learned today that Python's asyncio.Queue is thread-safe
  This is important for background tasks in Textual
  I should remember this when designing concurrent systems

Block ID: ghi789
Original Content: - Yesterday discovered that ChromaDB requires sentence-transformers
  It downloads a 90MB model on first run

```

### Response Format (NDJSON Stream)

Each line contains one reworded result:

```json
{"block_id":"abc123","reworded_content":"Python's asyncio.Queue is thread-safe, making it suitable for communication between background tasks and the main thread in Textual applications. This property is important when designing concurrent systems."}

```

```json
{"block_id":"ghi789","reworded_content":"ChromaDB requires the sentence-transformers library as a dependency. The library downloads a 90MB model on first initialization."}

```

### Pydantic Model

```python
from pydantic import BaseModel, Field

class ContentRewordingChunk(BaseModel):
    """Single rewording result from NDJSON stream."""

    block_id: str = Field(
        ...,
        description="Block identifier from input"
    )

    reworded_content: str = Field(
        ...,
        min_length=1,
        description="Reworded content with temporal context removed"
    )

```

### Example NDJSON Stream

Complete stream for 2 knowledge blocks:

```
{"block_id":"abc123","reworded_content":"Python's asyncio.Queue is thread-safe, making it suitable for communication between background tasks and the main thread in Textual applications. This property is important when designing concurrent systems."}
{"block_id":"ghi789","reworded_content":"ChromaDB requires the sentence-transformers library as a dependency. The library downloads a 90MB model on first initialization."}

```

---

## Phase 3: Integration Decisions Stream

### Purpose
Stream integration decisions for where/how to integrate knowledge blocks into target pages.

### Request Format

**Request Body**:

```json
{
  "model": "{config.llm.model}",
  "messages": [
    {
      "role": "system",
      "content": "You are an AI assistant that decides where to integrate knowledge into a knowledge base. For each (knowledge block, candidate page) pair, decide the best integration action. Actions: 'add_section' (create new top-level section), 'add_under' (add as child under existing block), 'replace' (replace existing block content). Only return decisions for relevant integrations - omit irrelevant candidate pages. Return at most 2 decisions per (knowledge block, target page) pair. For each relevant integration, respond with a single-line JSON object containing: knowledge_block_id (string), target_page (string), action (string: add_section|add_under|replace), target_block_id (string or null), target_block_title (string or null), confidence (float 0-1), reasoning (string). Output one JSON object per line (NDJSON format)."
    },
    {
      "role": "user",
      "content": "Decide integration for these knowledge blocks and candidate pages:\n\n{formatted_knowledge_and_candidates}"
    }
  ],
  "stream": true,
  "temperature": 0.4
}

```

**Formatted Knowledge and Candidates Example**:

```
Knowledge Block ID: abc123
Content: Python's asyncio.Queue is thread-safe, making it suitable for communication between background tasks and the main thread in Textual applications.
Original Context: [[Python]] [[Textual]] [[Concurrency]]

Candidate Pages:

1. Page: Python/Concurrency
   Existing Structure:

   - Threading vs Asyncio
     - Threading uses locks for shared state
     - Asyncio uses coroutines

   - Common Patterns
     id:: pattern-section-123

2. Page: Textual/Architecture
   Existing Structure:

   - Widget Lifecycle
   - Message Passing
     id:: message-section-456

   - Worker Threads
     id:: worker-section-789

3. Page: Python/Data Structures
   Existing Structure:

   - Lists and Tuples
   - Dictionaries
   - Sets

```

### Response Format (NDJSON Stream)

Each line contains one integration decision (only for relevant pages):

```json
{"knowledge_block_id":"abc123","target_page":"Python/Concurrency","action":"add_under","target_block_id":"pattern-section-123","target_block_title":"Common Patterns","confidence":0.92,"reasoning":"This knowledge about asyncio.Queue's thread-safety is a common pattern in Python concurrency. Adding it under the 'Common Patterns' section provides clear context and makes it discoverable."}

```

```json
{"knowledge_block_id":"abc123","target_page":"Textual/Architecture","action":"add_under","target_block_id":"worker-section-789","target_block_title":"Worker Threads","confidence":0.78,"reasoning":"The thread-safety property is relevant for Textual's worker thread architecture. Adding it under 'Worker Threads' section connects it to the framework's design."}

```

**Note**: The irrelevant page `Python/Data Structures` is omitted from output.

### Pydantic Model

```python
from pydantic import BaseModel, Field
from typing import Literal, Optional

class IntegrationDecisionChunk(BaseModel):
    """Single integration decision from NDJSON stream."""

    knowledge_block_id: str = Field(
        ...,
        description="Block ID of the knowledge being integrated"
    )

    target_page: str = Field(
        ...,
        description="Target page name (hierarchical pages use '/' separator)"
    )

    action: Literal["add_section", "add_under", "replace"] = Field(
        ...,
        description="Type of integration action"
    )

    target_block_id: Optional[str] = Field(
        default=None,
        description="Target block ID for 'add_under' or 'replace' actions"
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

    reasoning: str = Field(
        ...,
        description="LLM's explanation for this integration decision"
    )

```

### Example NDJSON Stream

Complete stream for 1 knowledge block with 3 candidate pages (2 relevant, 1 irrelevant):

```
{"knowledge_block_id":"abc123","target_page":"Python/Concurrency","action":"add_under","target_block_id":"pattern-section-123","target_block_title":"Common Patterns","confidence":0.92,"reasoning":"This knowledge about asyncio.Queue's thread-safety is a common pattern in Python concurrency. Adding it under the 'Common Patterns' section provides clear context and makes it discoverable."}
{"knowledge_block_id":"abc123","target_page":"Textual/Architecture","action":"add_under","target_block_id":"worker-section-789","target_block_title":"Worker Threads","confidence":0.78,"reasoning":"The thread-safety property is relevant for Textual's worker thread architecture. Adding it under 'Worker Threads' section connects it to the framework's design."}

```

### Batching Behavior (Phase 3 Specific)

Per spec requirements (FR-032):

- System waits for **all decisions** for a given knowledge block to arrive before displaying
- LLM returns only active/relevant decisions (filters out irrelevant candidate pages)
- At most 2 decisions per (knowledge block, target page) pair

**Implementation Note**: Application must buffer all decisions with matching `knowledge_block_id` before showing them to user.

---

## Error Format

### Malformed JSON Line

If a line cannot be parsed as JSON:

```python
# Logged as error, skipped, processing continues
logger.error(
    "malformed_json_line",
    phase="classification",
    line='{"block_id":"abc123","is_knowledge":tru',  # Truncated
    error="Expecting value: line 1 column 37 (char 36)"
)

```

### HTTP Error Response

If the HTTP request fails (non-streaming):

```json
{
  "error": {
    "message": "Invalid API key",
    "type": "invalid_request_error"
  }
}

```

**Error Types**:

- `invalid_request_error`: 400/401 errors (bad request, auth failure)
- `rate_limit_error`: 429 error (too many requests)
- `api_error`: 500/503 errors (server-side failures)

---

## Timeout Configuration

All streaming requests use:

```python
timeout = httpx.Timeout(
    connect=10.0,  # Initial connection timeout
    read=60.0,     # Per-read timeout (important for streaming)
    write=10.0,    # Write timeout
    pool=10.0      # Pool timeout
)

```

**Rationale** (from research.md): 60s read timeout allows for slow streaming responses from resource-constrained local models (e.g., Ollama on CPU).

---

## Request Logging

All streaming requests must be logged for debugging.

### Request Start

```json
{
  "event": "llm_request_started",
  "timestamp": "2025-11-05T10:30:45.123Z",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "phase": "classification",
  "model": "gpt-4-turbo-preview",
  "endpoint": "https://api.openai.com/v1/chat/completions",
  "num_blocks": 15,
  "messages": [
    {
      "role": "system",
      "content": "You are an AI assistant..."
    }
  ]
}

```

### Streaming Chunk Received

```json
{
  "event": "llm_response_chunk",
  "timestamp": "2025-11-05T10:30:47.456Z",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "chunk_num": 3,
  "data": {
    "block_id": "abc123",
    "is_knowledge": true,
    "confidence": 0.92,
    "reason": "..."
  }
}

```

### Request Complete

```json
{
  "event": "llm_request_completed",
  "timestamp": "2025-11-05T10:31:02.789Z",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "total_chunks": 15,
  "duration_ms": 17666
}

```

### Request Failed

```json
{
  "event": "llm_request_failed",
  "timestamp": "2025-11-05T10:30:50.123Z",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "error_type": "httpx.ReadTimeout",
  "error_message": "Read timeout after 60.0 seconds",
  "chunks_received": 5
}

```

---

## Network Resilience

### Automatic Retry (FR-075 to FR-079)

**Retry once** after 2-second delay for:

- `httpx.ConnectError` (connection refused, network unreachable)
- `httpx.ConnectTimeout` (connection timeout)
- `httpx.ReadTimeout` (read timeout during streaming)

**Retry Pattern**:

```python
from asyncio import sleep

async def retry_request(func, max_retries=1, delay=2.0):
    """Retry async function once on transient errors."""
    try:
        return await func()
    except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout) as e:
        logger.warning("llm_request_retry", error=str(e), delay=delay)
        if max_retries > 0:
            await sleep(delay)
            return await func()
        raise

```

**After retry fails**: Prompt user with options (Retry, Skip, Cancel)

### Partial Results

- Preserve all successfully parsed chunks before network failure
- Allow user to continue with partial results where applicable
- Never discard partial state on network errors

---

## Summary

All LLM streaming in Logsqueak follows these principles:

1. **NDJSON format**: Each line is a complete JSON object
2. **Error isolation**: Bad JSON on one line doesn't corrupt stream
3. **Progressive results**: UI updates as each line arrives
4. **Filtered output**: LLM only returns relevant items (omits irrelevant results)
5. **Structured logging**: All requests and chunks logged to file
6. **Network resilience**: Automatic retry, partial result preservation
7. **Type-safe parsing**: Pydantic models validate each chunk

This contract ensures consistent, debuggable, and resilient LLM interactions across all phases of the TUI application.
