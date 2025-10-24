# Prompt Inspection

Logsqueak provides a comprehensive prompt inspection system to help you understand and debug LLM interactions. This feature logs all prompts sent to and responses received from the LLM provider.

**By default, all prompt logs are automatically saved** to timestamped files in `~/.cache/logsqueak/prompts/` for every extraction run.

## Usage

### Default Behavior

Every extraction run automatically logs prompts to a timestamped file:

```bash
logsqueak extract 2025-01-15
```

This creates a log file at `~/.cache/logsqueak/prompts/YYYYMMDD_HHMMSS.log`.

### Custom Log File

To specify a custom log file path:

```bash
logsqueak extract --prompt-log-file /path/to/custom/prompts.log 2025-01-15
```

The logs will be written to the specified file.

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
- Parsed content (structured data extracted from response)
- Error details (if applicable)

Example:
```
[1] LLM RESPONSE - extraction
Timestamp: 2025-10-23T15:30:47.654321
Status: Success

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

Inspect prompts to understand why the LLM is making certain decisions. After running extraction:

```bash
logsqueak extract 2025-01-15
```

Check the automatically created log file in `~/.cache/logsqueak/prompts/` to review the exact prompts and responses.

### Prompt Engineering

Iterate on system prompts by inspecting what works and what doesn't:

1. Run extraction with prompt inspection
2. Review the logged prompts and responses
3. Modify the prompts in the provider code
4. Test again

### Auditing

All LLM interactions are automatically logged for audit and reproducibility. Logs are stored in `~/.cache/logsqueak/prompts/` with timestamps.

To organize logs by date, use a custom file path:

```bash
logsqueak extract \
  --prompt-log-file ~/.cache/logsqueak/prompts/$(date +%Y-%m-%d).log \
  2025-01-15
```

### Performance Analysis

Analyze LLM response patterns and timing by reviewing the automatically generated prompt logs in `~/.cache/logsqueak/prompts/`:

```bash
# Enable verbose mode for additional context
logsqueak extract --verbose 2025-01-15
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
   - `--prompt-log-file` option (defaults to timestamped cache file)
   - Prompt logging always enabled
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
