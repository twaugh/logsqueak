"""Prompt and response logging for LLM inspection.

This module provides logging capabilities to inspect all prompts sent to
and responses received from LLM providers. Useful for debugging, auditing,
and understanding LLM behavior.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO


class PromptLogger:
    """Logger for LLM prompts and responses.

    Captures and formats all LLM interactions for inspection and debugging.
    """

    def __init__(
        self,
        output: Optional[TextIO] = None,
        log_file: Optional[Path] = None,
        include_timestamps: bool = True,
        pretty_print: bool = True,
    ):
        """Initialize prompt logger.

        Args:
            output: Output stream (default: stderr)
            log_file: Optional file path to write logs to
            include_timestamps: Include timestamps in output
            pretty_print: Pretty-print JSON content
        """
        self.output = output or sys.stderr
        self.log_file = log_file
        self.include_timestamps = include_timestamps
        self.pretty_print = pretty_print
        self._interaction_count = 0

        # Open log file if specified
        self._file_handle: Optional[TextIO] = None
        if log_file:
            self._file_handle = open(log_file, "a", encoding="utf-8")

    def __del__(self):
        """Clean up file handle."""
        if self._file_handle:
            self._file_handle.close()

    def log_request(
        self,
        stage: str,
        messages: List[Dict[str, str]],
        model: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log an LLM request.

        Args:
            stage: Stage identifier (e.g., "extraction", "page_selection")
            messages: Chat messages sent to LLM
            model: Model name
            metadata: Optional metadata (e.g., journal_date, knowledge_content)
        """
        self._interaction_count += 1

        output = []
        output.append("=" * 80)
        output.append(f"[{self._interaction_count}] LLM REQUEST - {stage}")

        if self.include_timestamps:
            output.append(f"Timestamp: {datetime.now().isoformat()}")

        output.append(f"Model: {model}")

        if metadata:
            output.append(f"Metadata: {json.dumps(metadata, indent=2 if self.pretty_print else None)}")

        output.append("")
        output.append("Messages:")
        output.append("-" * 80)

        for i, message in enumerate(messages, 1):
            role = message.get("role", "unknown")
            content = message.get("content", "")

            output.append(f"[Message {i}] Role: {role}")
            output.append(content)
            output.append("-" * 80)

        output.append("")

        self._write("\n".join(output))

    def log_response(
        self,
        stage: str,
        response: Dict[str, Any],
        parsed_content: Optional[Dict[str, Any]] = None,
        error: Optional[Exception] = None,
        raw_content: Optional[str] = None,
    ) -> None:
        """Log an LLM response.

        Args:
            stage: Stage identifier (e.g., "extraction", "page_selection")
            response: Raw API response (not logged, only used for error context)
            parsed_content: Parsed/structured content from response
            error: Exception if request failed
            raw_content: Raw content string (logged when error occurs for debugging)
        """
        output = []
        output.append(f"[{self._interaction_count}] LLM RESPONSE - {stage}")

        if self.include_timestamps:
            output.append(f"Timestamp: {datetime.now().isoformat()}")

        if error:
            output.append(f"ERROR: {type(error).__name__}: {error}")
        else:
            output.append("Status: Success")

        output.append("")

        # Log raw content when error occurs (for debugging malformed responses)
        if error and raw_content is not None:
            output.append("Raw LLM Content:")
            output.append("-" * 80)
            output.append(raw_content)
            output.append("-" * 80)
            output.append("")

        if parsed_content:
            output.append("Parsed Content:")
            output.append("-" * 80)
            if self.pretty_print:
                output.append(json.dumps(parsed_content, indent=2))
            else:
                output.append(json.dumps(parsed_content))
            output.append("-" * 80)

        output.append("=" * 80)
        output.append("")

        self._write("\n".join(output))

    def _write(self, text: str) -> None:
        """Write text to output stream(s).

        Args:
            text: Text to write
        """
        # Write to log file if specified, otherwise to output stream
        if self._file_handle:
            self._file_handle.write(text + "\n")
            self._file_handle.flush()
        else:
            self.output.write(text + "\n")
            self.output.flush()

    def log_summary(self) -> None:
        """Log summary of all interactions."""
        output = []
        output.append("=" * 80)
        output.append("PROMPT INSPECTION SUMMARY")
        output.append(f"Total interactions: {self._interaction_count}")

        if self.include_timestamps:
            output.append(f"Session ended: {datetime.now().isoformat()}")

        if self.log_file:
            output.append(f"Log file: {self.log_file}")

        output.append("=" * 80)

        self._write("\n".join(output))
