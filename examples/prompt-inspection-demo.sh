#!/bin/bash
# Demo script showing prompt inspection feature

echo "=== Prompt Inspection Demo ==="
echo ""
echo "This demo shows how to inspect LLM prompts and responses in Logsqueak."
echo ""

# Example 1: Default behavior (automatic logging)
echo "Example 1: Default behavior - automatic prompt logging"
echo "Command: logsqueak extract 2025-01-15"
echo ""
echo "Output:"
echo "  - Prompts automatically logged to ~/.cache/logsqueak/prompts/YYYYMMDD_HHMMSS.log"
echo "  - All LLM requests with full prompts"
echo "  - All LLM responses with parsed content"
echo "  - Interaction numbering and timestamps"
echo ""

# Example 2: Custom log file
echo "Example 2: Specify custom log file path"
echo "Command: logsqueak extract --prompt-log-file prompts.log 2025-01-15"
echo ""
echo "This writes logs to the specified file:"
echo "  - prompts.log (custom location for analysis)"
echo ""

# Example 3: Organized by date
echo "Example 3: Organize logs by date"
echo "Command: logsqueak extract --prompt-log-file ~/.cache/logsqueak/prompts/\$(date +%Y-%m-%d).log 2025-01-15"
echo ""
echo "This creates dated log files for better organization."
echo ""

echo "=== What Gets Logged ==="
echo ""
echo "Request logging includes:"
echo "  - Stage (extraction or page_selection)"
echo "  - Model name"
echo "  - Metadata (journal date, etc.)"
echo "  - Full system and user prompts"
echo ""
echo "Response logging includes:"
echo "  - Status (Success or ERROR)"
echo "  - Raw API response (full JSON)"
echo "  - Parsed content (structured data)"
echo "  - Error details (if applicable)"
echo ""

echo "=== Privacy Warning ==="
echo ""
echo "⚠️  Prompt logs contain the full content of your journal entries!"
echo ""
echo "  - Do not share logs with sensitive information"
echo "  - Be careful with cloud-synced directories"
echo "  - Consider encrypting log files for private data"
echo ""

echo "=== Try it yourself ==="
echo ""
echo "Run: logsqueak extract --help"
echo ""
echo "Note: Prompt logging is ALWAYS enabled. Check ~/.cache/logsqueak/prompts/ for logs."
echo ""
