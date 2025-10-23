# Prompt Inspection

Logsqueak provides a comprehensive prompt inspection system to help you understand and debug LLM interactions. This feature logs all prompts sent to and responses received from the LLM provider.

## Usage

### Basic Inspection (stderr)

To inspect prompts and responses during extraction, use the `--inspect-prompts` flag:

```bash
logsqueak extract --inspect-prompts 2025-01-15
```

This will output all LLM interactions to stderr in a human-readable format.

### Saving to Log File

To save prompt logs to a file for later analysis:

```bash
logsqueak extract --inspect-prompts --prompt-log-file prompts.log 2025-01-15
```

The logs will be written to both stderr and the specified file.

## Output Format

### Request Logging

Each request is logged with:
- Interaction number (sequential)
- Stage identifier ("extraction" or "page_selection")
- Timestamp (ISO 8601 format)
- Model name
- Metadata (journal date, knowledge content preview, etc.)
- Full message history (system and user prompts)

Example:
```
================================================================================
[1] LLM REQUEST - extraction
Timestamp: 2025-10-23T15:30:45.123456
Model: gpt-4
Metadata: {
  "journal_date": "2025-01-15"
}

Messages:
--------------------------------------------------------------------------------
[Message 1] Role: system
You are a knowledge extraction assistant for a personal knowledge management system.

Your task is to identify pieces of information with lasting value from journal entries.
...
--------------------------------------------------------------------------------
[Message 2] Role: user
Journal entry from 2025-01-15:

- Worked on project X
- Had insight about Y
...
--------------------------------------------------------------------------------
```

### Response Logging

Each response is logged with:
- Interaction number (matching the request)
- Status (Success or ERROR)
- Raw API response (full JSON)
- Parsed content (structured data extracted from response)
- Error details (if applicable)

Example:
```
[1] LLM RESPONSE - extraction
Timestamp: 2025-10-23T15:30:47.654321
Status: Success

Raw Response:
--------------------------------------------------------------------------------
{
  "choices": [
    {
      "message": {
        "content": "{\"knowledge_blocks\": [{\"content\": \"Insight about Y\", \"confidence\": 0.85}]}"
      }
    }
  ]
}
--------------------------------------------------------------------------------

Parsed Content:
--------------------------------------------------------------------------------
{
  "knowledge_blocks": [
    {
      "content": "Insight about Y",
      "confidence": 0.85
    }
  ]
}
--------------------------------------------------------------------------------
================================================================================
```

### Session Summary

At the end of extraction, a summary is logged:

```
================================================================================
PROMPT INSPECTION SUMMARY
Total interactions: 5
Session ended: 2025-10-23T15:31:00.000000
Log file: prompts.log
================================================================================
```

## Use Cases

### Debugging LLM Behavior

Inspect prompts to understand why the LLM is making certain decisions:

```bash
logsqueak extract --inspect-prompts 2025-01-15 2>/tmp/prompts.txt
```

Then review `/tmp/prompts.txt` to see the exact prompts and responses.

### Prompt Engineering

Iterate on system prompts by inspecting what works and what doesn't:

1. Run extraction with prompt inspection
2. Review the logged prompts and responses
3. Modify the prompts in the provider code
4. Test again

### Auditing

Keep a permanent log of all LLM interactions for audit or reproducibility:

```bash
# Append to a dated log file
logsqueak extract \
  --inspect-prompts \
  --prompt-log-file ~/.cache/logsqueak/prompts-$(date +%Y-%m-%d).log \
  2025-01-15
```

### Performance Analysis

Analyze LLM response patterns and timing:

```bash
# Enable verbose mode and prompt inspection
logsqueak extract --verbose --inspect-prompts 2025-01-15 2>&1 | tee analysis.log
```

## Privacy Considerations

**Warning**: Prompt logs contain the full content of your journal entries and LLM responses.

- Do not share log files that contain sensitive information
- Be careful when storing logs in cloud-synced directories
- Consider encrypting log files if they contain private data

## Performance Impact

Prompt inspection has minimal performance overhead:
- Logging to stderr: ~1-2ms per interaction
- Logging to file: ~2-5ms per interaction

The impact is negligible compared to LLM API latency (typically 500-5000ms per request).

## Implementation Details

### Architecture

The prompt inspection system consists of:

1. **PromptLogger** (`src/logsqueak/llm/prompt_logger.py`)
   - Handles formatting and writing logs
   - Supports multiple output streams
   - Manages interaction counting

2. **Provider Integration** (`src/logsqueak/llm/providers/openai_compat.py`)
   - Optional `prompt_logger` parameter
   - Logs before and after each LLM call
   - Captures errors

3. **CLI Integration** (`src/logsqueak/cli/main.py`)
   - `--inspect-prompts` flag
   - `--prompt-log-file` option
   - Automatic summary at session end

### Extending to Other Providers

To add prompt inspection to a new LLM provider:

```python
from logsqueak.llm.prompt_logger import PromptLogger

class MyLLMProvider(LLMClient):
    def __init__(self, ..., prompt_logger: Optional[PromptLogger] = None):
        self.prompt_logger = prompt_logger

    def extract_knowledge(self, ...):
        messages = [...]

        # Log request
        if self.prompt_logger:
            self.prompt_logger.log_request(
                stage="extraction",
                messages=messages,
                model=self.model,
                metadata={"journal_date": journal_date.isoformat()},
            )

        try:
            response = self._make_request(messages)
            parsed = self._parse_response(response)

            # Log success
            if self.prompt_logger:
                self.prompt_logger.log_response(
                    stage="extraction",
                    response=response,
                    parsed_content=parsed,
                )

            return parsed

        except Exception as e:
            # Log error
            if self.prompt_logger:
                self.prompt_logger.log_response(
                    stage="extraction",
                    response={},
                    error=e,
                )
            raise
```

## Testing

The prompt inspection system is thoroughly tested:

- **Unit tests** (`tests/unit/test_prompt_logger.py`): 15 tests for PromptLogger
- **Integration tests** (`tests/unit/test_prompt_inspection_integration.py`): 7 tests for OpenAI provider integration

Run the tests:

```bash
pytest tests/unit/test_prompt_logger.py -v
pytest tests/unit/test_prompt_inspection_integration.py -v
```
