"""Tests for prompt logging functionality."""

import json
from io import StringIO
from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest

from logsqueak.llm.prompt_logger import PromptLogger


class TestPromptLogger:
    """Test cases for PromptLogger."""

    def test_init_default_output(self):
        """Should initialize with stderr as default output."""
        logger = PromptLogger()
        # Can't easily test stderr, but ensure it doesn't crash
        assert logger.output is not None

    def test_init_with_custom_output(self):
        """Should accept custom output stream."""
        output = StringIO()
        logger = PromptLogger(output=output)
        assert logger.output == output

    def test_init_with_log_file(self):
        """Should open log file for writing."""
        with NamedTemporaryFile(mode="w", delete=False, suffix=".log") as f:
            log_file = Path(f.name)

        try:
            logger = PromptLogger(log_file=log_file)
            assert logger.log_file == log_file
            assert logger._file_handle is not None

            # Clean up
            del logger  # Trigger __del__ to close file
        finally:
            if log_file.exists():
                log_file.unlink()

    def test_log_request_basic(self):
        """Should log basic request information."""
        output = StringIO()
        logger = PromptLogger(output=output, include_timestamps=False)

        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello, world!"},
        ]

        logger.log_request(
            stage="test_stage",
            messages=messages,
            model="gpt-4",
        )

        result = output.getvalue()

        # Check for key elements
        assert "[1] LLM REQUEST - test_stage" in result
        assert "Model: gpt-4" in result
        assert "Role: system" in result
        assert "You are a helpful assistant" in result
        assert "Role: user" in result
        assert "Hello, world!" in result

    def test_log_request_with_metadata(self):
        """Should log metadata when provided."""
        output = StringIO()
        logger = PromptLogger(output=output, include_timestamps=False)

        messages = [{"role": "user", "content": "Test"}]
        metadata = {"journal_date": "2025-01-15", "foo": "bar"}

        logger.log_request(
            stage="extraction",
            messages=messages,
            model="test-model",
            metadata=metadata,
        )

        result = output.getvalue()

        assert "Metadata:" in result
        assert "journal_date" in result
        assert "2025-01-15" in result

    def test_log_request_with_timestamps(self):
        """Should include timestamps when enabled."""
        output = StringIO()
        logger = PromptLogger(output=output, include_timestamps=True)

        logger.log_request(
            stage="test",
            messages=[{"role": "user", "content": "Test"}],
            model="test-model",
        )

        result = output.getvalue()
        assert "Timestamp:" in result

    def test_log_request_increments_counter(self):
        """Should increment interaction counter for each request."""
        output = StringIO()
        logger = PromptLogger(output=output, include_timestamps=False)

        logger.log_request(
            stage="test1",
            messages=[{"role": "user", "content": "Test 1"}],
            model="model",
        )

        logger.log_request(
            stage="test2",
            messages=[{"role": "user", "content": "Test 2"}],
            model="model",
        )

        result = output.getvalue()
        assert "[1] LLM REQUEST" in result
        assert "[2] LLM REQUEST" in result

    def test_log_response_success(self):
        """Should log successful response."""
        output = StringIO()
        logger = PromptLogger(output=output, include_timestamps=False)

        # Need to log request first to set interaction count
        logger.log_request(
            stage="test",
            messages=[{"role": "user", "content": "Test"}],
            model="model",
        )

        response = {
            "choices": [{"message": {"content": '{"result": "success"}'}}]
        }
        parsed = {"result": "success"}

        logger.log_response(
            stage="test",
            response=response,
            parsed_content=parsed,
        )

        result = output.getvalue()

        assert "[1] LLM RESPONSE - test" in result
        assert "Status: Success" in result
        assert "Parsed Content:" in result
        assert '"result": "success"' in result
        # Raw response should NOT be in output
        assert "Raw Response:" not in result
        assert '"choices"' not in result

    def test_log_response_with_error(self):
        """Should log error information."""
        output = StringIO()
        logger = PromptLogger(output=output, include_timestamps=False)

        logger.log_request(
            stage="test",
            messages=[{"role": "user", "content": "Test"}],
            model="model",
        )

        error = ValueError("Something went wrong")

        logger.log_response(
            stage="test",
            response={},
            error=error,
        )

        result = output.getvalue()

        assert "ERROR: ValueError: Something went wrong" in result

    def test_log_response_pretty_print(self):
        """Should pretty-print JSON when enabled."""
        output = StringIO()
        logger = PromptLogger(output=output, include_timestamps=False, pretty_print=True)

        logger.log_request(
            stage="test",
            messages=[{"role": "user", "content": "Test"}],
            model="model",
        )

        parsed = {"key1": "value1", "key2": {"nested": "value2"}}

        logger.log_response(
            stage="test",
            response={},
            parsed_content=parsed,
        )

        result = output.getvalue()

        # Pretty-printed JSON should have indentation
        assert '  "key1"' in result or '  "key2"' in result

    def test_log_response_compact_json(self):
        """Should use compact JSON when pretty_print=False."""
        output = StringIO()
        logger = PromptLogger(output=output, include_timestamps=False, pretty_print=False)

        logger.log_request(
            stage="test",
            messages=[{"role": "user", "content": "Test"}],
            model="model",
        )

        parsed = {"key": "value"}

        logger.log_response(
            stage="test",
            response={},
            parsed_content=parsed,
        )

        result = output.getvalue()

        # Compact JSON on single line
        assert '{"key": "value"}' in result or '{"key":"value"}' in result

    def test_log_summary(self):
        """Should log summary with interaction count."""
        output = StringIO()
        logger = PromptLogger(output=output, include_timestamps=False)

        # Make a few requests
        for i in range(3):
            logger.log_request(
                stage=f"test{i}",
                messages=[{"role": "user", "content": f"Test {i}"}],
                model="model",
            )

        logger.log_summary()

        result = output.getvalue()

        assert "PROMPT INSPECTION SUMMARY" in result
        assert "Total interactions: 3" in result

    def test_write_to_both_streams(self):
        """Should write to log file only when file is specified."""
        output = StringIO()

        with NamedTemporaryFile(mode="w", delete=False, suffix=".log") as f:
            log_file = Path(f.name)

        try:
            logger = PromptLogger(
                output=output,
                log_file=log_file,
                include_timestamps=False,
            )

            logger.log_request(
                stage="test",
                messages=[{"role": "user", "content": "Test message"}],
                model="test-model",
            )

            # Clean up to flush file
            del logger

            # Check output stream should be empty (file takes precedence)
            output_content = output.getvalue()
            assert "Test message" not in output_content

            # Check log file
            with open(log_file, "r") as f:
                file_content = f.read()
            assert "Test message" in file_content

        finally:
            if log_file.exists():
                log_file.unlink()

    def test_multiple_messages(self):
        """Should handle multiple messages in a request."""
        output = StringIO()
        logger = PromptLogger(output=output, include_timestamps=False)

        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "User message 1"},
            {"role": "assistant", "content": "Assistant response"},
            {"role": "user", "content": "User message 2"},
        ]

        logger.log_request(
            stage="test",
            messages=messages,
            model="model",
        )

        result = output.getvalue()

        assert "[Message 1] Role: system" in result
        assert "[Message 2] Role: user" in result
        assert "[Message 3] Role: assistant" in result
        assert "[Message 4] Role: user" in result
        assert "System prompt" in result
        assert "User message 1" in result
        assert "Assistant response" in result
        assert "User message 2" in result

    def test_separator_lines(self):
        """Should include separator lines for readability."""
        output = StringIO()
        logger = PromptLogger(output=output, include_timestamps=False)

        logger.log_request(
            stage="test",
            messages=[{"role": "user", "content": "Test"}],
            model="model",
        )

        result = output.getvalue()

        # Should have separator lines (80 characters of =)
        assert "=" * 80 in result
        assert "-" * 80 in result
