#!/bin/bash
# Demo script showing prompt inspection feature

echo "=== Prompt Inspection Demo ==="
echo ""
echo "This demo shows how to inspect LLM prompts and responses in Logsqueak."
echo ""

# Example 1: Basic inspection to stderr
echo "Example 1: Inspect prompts to stderr"
echo "Command: logsqueak extract --inspect-prompts 2025-01-15"
echo ""
echo "Output will show:"
echo "  - All LLM requests with full prompts"
echo "  - All LLM responses with parsed content"
echo "  - Interaction numbering"
echo "  - Timestamps"
echo ""

# Example 2: Save to file
echo "Example 2: Save prompt logs to file"
echo "Command: logsqueak extract --inspect-prompts --prompt-log-file prompts.log 2025-01-15"
echo ""
echo "This writes logs to both:"
echo "  - stderr (for immediate viewing)"
echo "  - prompts.log (for later analysis)"
echo ""

# Example 3: Redirect stderr to file
echo "Example 3: Only save to file (redirect stderr)"
echo "Command: logsqueak extract --inspect-prompts 2025-01-15 2>prompts.log"
echo ""
echo "This captures all prompt logs to prompts.log without cluttering stdout."
echo ""

# Example 4: Separate stdout and stderr
echo "Example 4: Separate normal output from prompts"
echo "Command: logsqueak extract --inspect-prompts 2025-01-15 2>prompts.log 1>output.txt"
echo ""
echo "  - output.txt: Normal extraction preview"
echo "  - prompts.log: LLM prompt inspection logs"
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
echo "Run: logsqueak extract --inspect-prompts --help"
echo ""
