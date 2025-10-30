"""Generic NDJSON stream parser for LLM responses.

This module provides utilities for parsing Newline-Delimited JSON (NDJSON) from
chunked async streams with proper buffering and error handling.
"""

import json
import logging
from typing import AsyncIterator

logger = logging.getLogger(__name__)


async def parse_ndjson_stream(stream: AsyncIterator[str]) -> AsyncIterator[dict]:
    """
    Parse NDJSON from chunked async stream.

    Input: Token-by-token chunks ('{"', 'foo', '": ', '"bar', '"}', '\\n', ...)
    Output: Complete JSON objects as they become parseable

    Error Handling:
    - Invalid JSON on a line → Log error, skip line, continue
    - Incomplete line at stream end → Log warning, discard
    - Empty lines → Skip silently

    Args:
        stream: Async iterator yielding string chunks from LLM response

    Yields:
        Parsed JSON objects (dicts) as complete lines arrive

    Example:
        ```python
        async def llm_stream():
            yield '{"block_id": '
            yield '"abc123", '
            yield '"is_knowledge": true}\\n'
            yield '{"block_id": "def456"'
            yield ', "is_knowledge": false}\\n'

        async for obj in parse_ndjson_stream(llm_stream()):
            print(obj)  # {"block_id": "abc123", "is_knowledge": true}
                        # {"block_id": "def456", "is_knowledge": false}
        ```
    """
    buffer = ""
    line_number = 0

    logger.debug("Starting NDJSON stream parsing")

    async for chunk in stream:
        buffer += chunk

        # Process all complete lines in buffer
        while "\n" in buffer:
            line_end = buffer.index("\n")
            complete_line = buffer[:line_end].strip()
            buffer = buffer[line_end + 1 :]  # Remove processed line + newline
            line_number += 1

            if not complete_line:
                # Empty line, skip silently
                continue

            try:
                obj = json.loads(complete_line)

                # Validate expected type
                if not isinstance(obj, dict):
                    logger.warning(
                        f"Line {line_number}: Expected JSON object, got {type(obj).__name__}"
                    )
                    continue

                logger.debug(f"Parsed NDJSON line {line_number}: {obj}")
                yield obj

            except json.JSONDecodeError as e:
                # Log error with context but continue processing
                logger.warning(
                    f"Line {line_number}: Malformed JSON: {complete_line[:100]}... "
                    f"Error at position {e.pos}: {e.msg}"
                )
                # Don't yield anything, just continue to next line

    logger.debug(f"NDJSON stream parsing complete ({line_number} lines processed)")

    # Handle any remaining content in buffer at stream end
    if buffer.strip():
        line_number += 1
        try:
            obj = json.loads(buffer.strip())
            if isinstance(obj, dict):
                logger.debug(f"Parsed final NDJSON line {line_number}: {obj}")
                yield obj
            else:
                logger.warning(
                    f"Line {line_number} (final): Expected JSON object, got {type(obj).__name__}"
                )
        except json.JSONDecodeError as e:
            logger.warning(
                f"Line {line_number} (incomplete): Could not parse final line: "
                f"{buffer[:100]}... Error: {e.msg}"
            )
