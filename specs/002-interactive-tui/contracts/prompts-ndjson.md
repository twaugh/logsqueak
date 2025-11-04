# NDJSON Prompt Templates

## Overview

This document defines the prompt structure for LLM NDJSON streaming responses. All prompts MUST instruct the LLM to output **newline-delimited JSON (NDJSON)** where each line is a complete, parseable JSON object.

## Phase 1: Knowledge Extraction

### Prompt Template

```
You are analyzing a Logseq journal entry to identify lasting knowledge versus temporary activity logs.

For each block in the journal entry, determine:
1. Is this block "knowledge" (lasting insights, learnings, decisions) or "activity" (temporary logs, todos, meetings)?
2. Your confidence level (0.0 to 1.0)

Output your analysis as NDJSON (newline-delimited JSON). Each line must be a complete JSON object with this structure:

{"block_id": "abc123", "is_knowledge": true, "confidence": 0.85}
{"block_id": "def456", "is_knowledge": false, "confidence": 0.92}

Rules:
- One JSON object per line
- Each object must be complete and parseable
- Output ONLY NDJSON, no other text
- Process blocks in order

Journal blocks:
{blocks_json}
```

### Expected Output Format

```
{"block_id": "hash_abc123", "is_knowledge": true, "confidence": 0.88}
{"block_id": "hash_def456", "is_knowledge": false, "confidence": 0.95}
{"block_id": "id_xyz789", "is_knowledge": true, "confidence": 0.72}
```

## Phase 3.1: Integration Decisions

### Prompt Template

```
You are deciding where to integrate a knowledge block into a target page.

Knowledge block:
{knowledge_block_text}

Target page: {page_name}
Page content:
{page_outline}

Determine the best action:
- "skip" - Don't integrate (duplicate, irrelevant, etc.)
- "add_section" - Add as new root-level section
- "add_under" - Add as child under specific block
- "replace" - Replace existing block content

Output as NDJSON with this structure:

{"page": "Page Name", "action": "add_section", "confidence": 0.88, "target_block_id": null, "skip_reason": null}
{"page": "Another Page", "action": "skip", "confidence": 0.95, "target_block_id": null, "skip_reason": "Duplicate content"}

Fields:
- page: Target page name
- action: One of [skip, add_section, add_under, replace]
- confidence: 0.0 to 1.0
- target_block_id: Block ID if action is add_under or replace, else null
- skip_reason: Explanation if action is skip, else null

Output ONLY NDJSON, one object per candidate page.
```

### Expected Output Format

```
{"page": "Software Architecture", "action": "add_section", "confidence": 0.85, "target_block_id": null, "skip_reason": null}
{"page": "Microservices", "action": "skip", "confidence": 0.92, "target_block_id": null, "skip_reason": "Already covered in existing page content"}
{"page": "Design Patterns", "action": "add_under", "confidence": 0.78, "target_block_id": "id_block789", "skip_reason": null}
```

## Phase 3.2: Content Rewording

### Prompt Template

```
You are rephrasing journal-based knowledge for integration into an evergreen knowledge base.

Original journal text:
{original_text}

Target page context:
{page_context}

Rephrase the content to:
1. Remove journal-specific context (dates, "today I learned", etc.)
2. Make it standalone and evergreen
3. Preserve important links and technical details
4. Keep it concise

Output as NDJSON with this structure:

{"refined_text": "The rephrased, evergreen version of the knowledge..."}

Output ONLY NDJSON, one JSON object per knowledge block.
```

### Expected Output Format

```
{"refined_text": "Event-driven architectures enable loose coupling between microservices by using message queues as intermediaries. Key benefits include improved scalability and fault isolation."}
```

## Error Handling

### Malformed JSON

If a line cannot be parsed as JSON:

1. Log error with line number and content
2. Skip that line
3. Continue processing remaining lines

### Incomplete Streams

If stream ends mid-line:

1. Log warning about incomplete final line
2. Discard incomplete line
3. Return all successfully parsed objects

### Empty Lines

Empty lines or whitespace-only lines should be skipped silently.

## Implementation Notes

- Use `llm/streaming.py:parse_ndjson_stream()` for parsing
- Buffer chunks by newline character (`\n`)
- Parse each complete line immediately upon arrival
- Provider-specific HTTP streaming handled in `llm/providers/openai_compat.py`
