# LLM API Contracts

**Date**: 2025-11-04

**Feature**: 002-logsqueak-spec

## Overview

This document defines the expected request and response formats for LLM API interactions. The application supports both OpenAI-compatible APIs (OpenAI, Ollama with `/v1/chat/completions` endpoint) and Ollama's native API (`/api/chat`).

All LLM requests use **streaming responses** with Server-Sent Events (SSE) format for incremental results.

---

## Phase 1: Knowledge Block Classification

### Request

**Endpoint**: Configured via `config.llm.endpoint`

- OpenAI-compatible: `POST {endpoint}/chat/completions`
- Ollama native: `POST {endpoint}/api/chat`

**Headers**:

```
Authorization: Bearer {config.llm.api_key}
Content-Type: application/json

```

**Request Body** (OpenAI-compatible):

```json
{
  "model": "{config.llm.model}",
  "messages": [
    {
      "role": "system",
      "content": "You are an AI assistant that identifies lasting knowledge in journal entries. Classify each block as either 'knowledge' (lasting insight worth preserving) or 'activity' (temporal event/task). Respond with JSON array containing one object per block with fields: block_id (string), is_knowledge (boolean), confidence (float 0-1), reason (string)."
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

**Request Body** (Ollama native):

```json
{
  "model": "{config.llm.model}",
  "messages": [...same as above...],
  "stream": true,
  "options": {
    "temperature": 0.3,
    "num_ctx": "{config.llm.num_ctx}"
  }
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

```

---

### Response (Streaming)

**Format**: Server-Sent Events (SSE)

Each line contains either:

- `data: {json}` (OpenAI-compatible)
- `{json}` (Ollama native)

**Expected JSON per chunk**:

```json
{
  "block_id": "abc123",
  "is_knowledge": true,
  "confidence": 0.85,
  "reason": "This block contains a technical insight about Python's asyncio.Queue that is evergreen and worth preserving for future reference."
}

```

**Stream End**:

- OpenAI-compatible: `data: [DONE]`
- Ollama native: Connection closes

**Error Response** (non-streaming):

```json
{
  "error": {
    "message": "Invalid API key",
    "type": "invalid_request_error"
  }
}

```

---

## Phase 2: Content Rewording

### Request

**Endpoint**: Same as Phase 1

**Request Body** (OpenAI-compatible):

```json
{
  "model": "{config.llm.model}",
  "messages": [
    {
      "role": "system",
      "content": "You are an AI assistant that transforms journal-style content into evergreen knowledge. Remove temporal context (dates, 'today', 'yesterday'), convert first-person to third-person or neutral, and make the content timeless. Preserve technical details and insights. Respond with JSON array containing one object per block with fields: block_id (string), reworded_content (string)."
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

---

### Response (Streaming)

**Expected JSON per chunk**:

```json
{
  "block_id": "abc123",
  "reworded_content": "Python's asyncio.Queue is thread-safe, making it suitable for communication between background tasks and the main thread in Textual applications. This property is important when designing concurrent systems."
}

```

---

## Phase 3: Integration Decisions

### Request

**Endpoint**: Same as Phase 1

**Request Body** (OpenAI-compatible):

```json
{
  "model": "{config.llm.model}",
  "messages": [
    {
      "role": "system",
      "content": "You are an AI assistant that decides where to integrate knowledge into a knowledge base. For each (knowledge block, candidate page) pair, decide the best integration action: 'add_section' (create new section), 'add_under' (add under existing block), 'replace' (replace existing block), or 'skip' (not relevant for this page). Respond with JSON array containing one object per decision with fields: knowledge_block_id (string), target_page (string), action (string), target_block_id (string or null), target_block_title (string or null), confidence (float 0-1), reasoning (string), skip_reason (string or null)."
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

```

---

### Response (Streaming)

**Expected JSON per chunk**:

```json
{
  "knowledge_block_id": "abc123",
  "target_page": "Python/Concurrency",
  "action": "add_under",
  "target_block_id": "pattern-section-123",
  "target_block_title": "Common Patterns",
  "confidence": 0.92,
  "reasoning": "This knowledge about asyncio.Queue's thread-safety is a common pattern in Python concurrency. Adding it under the 'Common Patterns' section provides clear context and makes it discoverable.",
  "skip_reason": null
}

```

**Skip Example**:

```json
{
  "knowledge_block_id": "abc123",
  "target_page": "Textual/Architecture",
  "action": "skip",
  "target_block_id": null,
  "target_block_title": null,
  "confidence": 0.35,
  "reasoning": "While this knowledge mentions Textual, it's primarily about Python's asyncio.Queue, not Textual's architecture. Better suited for Python/Concurrency page.",
  "skip_reason": "Content is more Python-specific than Textual-specific"
}

```

---

## Error Handling

### Transient Errors (Auto-Retry)

These errors trigger automatic retry (once, 2s delay):

- `httpx.ConnectError` (connection refused, network unreachable)
- `httpx.ConnectTimeout` (connection timeout)
- `httpx.ReadTimeout` (read timeout during streaming)

### Permanent Errors (Show Error, Prompt User)

These errors show error message and prompt user for action:

- `401 Unauthorized` (invalid API key)
- `429 Too Many Requests` (rate limit exceeded)
- `500 Internal Server Error` (LLM service error)
- `503 Service Unavailable` (LLM service down)
- Malformed JSON in streaming response

**Error Display Format**:

```
Error: LLM request failed
Reason: Invalid API key (401 Unauthorized)
Suggested Action: Check llm.api_key in ~/.config/logsqueak/config.yaml

Options: [Retry] [Skip] [Cancel]

```

---

## Timeout Configuration

All requests use these timeout values (httpx.Timeout):

- `connect`: 10 seconds (initial connection)
- `read`: 60 seconds (per chunk during streaming)
- `write`: 10 seconds
- `pool`: 10 seconds

**Rationale**: 60s read timeout allows for slow streaming responses from resource-constrained local models (e.g., Ollama on CPU).

---

## Request Logging

All LLM requests must be logged for debugging. Log format:

```json
{
  "event": "llm_request_started",
  "timestamp": "2025-11-04T10:30:45.123Z",
  "request_id": "uuid-here",
  "phase": "classification",
  "model": "gpt-4-turbo-preview",
  "endpoint": "https://api.openai.com/v1/chat/completions",
  "num_blocks": 15,
  "messages": [...full messages array...]
}

```

Streaming chunks logged as:

```json
{
  "event": "llm_response_chunk",
  "timestamp": "2025-11-04T10:30:47.456Z",
  "request_id": "uuid-here",
  "chunk_num": 3,
  "data": {...parsed JSON...}
}

```

Request completion logged as:

```json
{
  "event": "llm_request_completed",
  "timestamp": "2025-11-04T10:31:02.789Z",
  "request_id": "uuid-here",
  "total_chunks": 15,
  "duration_ms": 17666
}

```

---

## Model Compatibility

### OpenAI API

- Endpoint: `https://api.openai.com/v1/chat/completions`
- Recommended models: `gpt-4-turbo-preview`, `gpt-3.5-turbo`
- Streaming format: SSE with `data:` prefix

### Ollama API

- Endpoint: `http://localhost:11434/v1/chat/completions` (OpenAI-compatible)
- Alternative: `http://localhost:11434/api/chat` (native)
- Recommended models: `llama2`, `mistral`, `mixtral`
- Streaming format: Newline-delimited JSON (native) or SSE (v1)
- Extra config: `num_ctx` controls context window (VRAM usage)

### Other OpenAI-Compatible Services

- Anthropic Claude (via proxy)
- Azure OpenAI
- Local models via text-generation-webui
- Any service implementing OpenAI chat completions API

---

## Summary

All LLM interactions follow these principles:

1. **Streaming-first**: All responses arrive incrementally
2. **JSON-structured**: LLM outputs parsed JSON (validated with Pydantic)
3. **Retryable**: Transient failures automatically retry once
4. **Logged**: All requests and responses logged for debugging
5. **Configurable**: Endpoint, model, and parameters from config file
