"""Error recovery and fault tolerance tests.

Tests the system's ability to handle and recover from various failure modes:
- Corrupted vector store
- LLM API failures
- File system errors
- Invalid data handling
"""

import tempfile
from pathlib import Path
from textwrap import dedent
from unittest.mock import Mock, patch

import pytest

from logsqueak.extraction.extractor import Extractor
from logsqueak.integration.executor import execute_write_list
from logsqueak.llm.client import LLMClient, LLMAPIError, LLMResponseError, LLMTimeoutError
from logseq_outline import LogseqOutline
from logsqueak.models.journal import JournalEntry
from logsqueak.models.knowledge import ActionType
from logsqueak.rag.indexer import IndexBuilder
from logsqueak.rag.manifest import CacheManifest
from logsqueak.rag.vector_store import ChromaDBStore


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_graph(tmp_path):
    """Create a temporary Logseq graph with sample pages."""
    graph_path = tmp_path / "graph"
    (graph_path / "pages").mkdir(parents=True)
    (graph_path / "journals").mkdir(parents=True)

    # Create a simple page
    page_file = graph_path / "pages" / "Test Page.md"
    page_file.write_text(dedent("""\
        - # Overview
          id:: test-overview
          - This is a test page
        """))

    # Create a simple journal
    journal_file = graph_path / "journals" / "2025_01_15.md"
    journal_file.write_text(dedent("""\
        - Test journal entry
          id:: journal-1
          - Some content
        """))

    return graph_path


# ============================================================================
# Vector Store Error Recovery Tests
# ============================================================================


class TestVectorStoreRecovery:
    """Test recovery from vector store corruption and failures."""

    def test_corrupted_vector_store_directory(self, temp_graph):
        """Test handling of corrupted vector store directory."""
        # Create a corrupted vector store directory (file instead of directory)
        chroma_path = temp_graph / ".chroma"
        chroma_path.write_text("corrupted")  # Should be a directory

        # Try to initialize - should handle gracefully
        with pytest.raises(Exception):  # ChromaDB will raise an error
            vector_store = ChromaDBStore(persist_directory=chroma_path)

    def test_missing_manifest_file(self, temp_graph):
        """Test handling of missing manifest file."""
        manifest_path = temp_graph / ".chroma" / "manifest.json"

        # Create manifest without file existing
        manifest = CacheManifest(manifest_path=manifest_path)

        # Should initialize with empty entries
        assert len(manifest.entries) == 0
        assert manifest.get_all_pages() == []

    def test_malformed_manifest_file(self, temp_graph):
        """Test handling of malformed manifest JSON."""
        manifest_path = temp_graph / ".chroma" / "manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        # Write invalid JSON
        manifest_path.write_text("{ invalid json }")

        # Should raise ValueError on load
        with pytest.raises(ValueError, match="Malformed manifest"):
            CacheManifest(manifest_path=manifest_path)

    def test_rebuild_after_corruption(self, temp_graph):
        """Test rebuilding index after corruption."""
        # Build initial index
        vector_store = ChromaDBStore(persist_directory=temp_graph / ".chroma")
        manifest = CacheManifest(manifest_path=temp_graph / ".chroma" / "manifest.json")
        builder = IndexBuilder(vector_store=vector_store, manifest=manifest)

        stats1 = builder.build_incremental(temp_graph)
        assert stats1["added"] > 0

        # Corrupt the manifest
        manifest.clear()
        manifest.save()

        # Rebuild should detect and re-add all pages
        vector_store2 = ChromaDBStore(persist_directory=temp_graph / ".chroma")
        manifest2 = CacheManifest(manifest_path=temp_graph / ".chroma" / "manifest.json")
        builder2 = IndexBuilder(vector_store=vector_store2, manifest=manifest2)

        stats2 = builder2.build_incremental(temp_graph)
        # Pages should be re-added since manifest was cleared
        assert stats2["added"] == stats1["added"]


# ============================================================================
# LLM Error Handling Tests
# ============================================================================


class MockFailingLLM(LLMClient):
    """Mock LLM that fails in various ways for testing."""

    def __init__(self, fail_mode="none", fail_count=0):
        """Initialize with failure mode.

        Args:
            fail_mode: Type of failure ("api", "timeout", "response", "none")
            fail_count: Number of times to fail before succeeding (0 = always fail)
        """
        self.fail_mode = fail_mode
        self.fail_count = fail_count
        self.call_count = 0

    def extract_knowledge(self, journal_content, journal_date, indent_str="  "):
        """Fail according to fail_mode."""
        self.call_count += 1

        if self.fail_count == 0 or self.call_count <= self.fail_count:
            if self.fail_mode == "api":
                raise LLMAPIError("API rate limit exceeded", status_code=429)
            elif self.fail_mode == "timeout":
                raise LLMTimeoutError("Request timeout")
            elif self.fail_mode == "response":
                raise LLMResponseError("Invalid JSON response")

        # Success case
        return []

    def decide_action(self, knowledge_text, candidate_chunks):
        """Fail according to fail_mode."""
        self.call_count += 1

        if self.fail_count == 0 or self.call_count <= self.fail_count:
            if self.fail_mode == "api":
                raise LLMAPIError("API error", status_code=500)
            elif self.fail_mode == "timeout":
                raise LLMTimeoutError("Timeout")
            elif self.fail_mode == "response":
                raise LLMResponseError("Bad response")

        # Success case - return ignore action
        from logsqueak.llm.client import DecisionResult
        return DecisionResult(
            action=ActionType.IGNORE_IRRELEVANT,
            page_name=None,
            target_id=None,
            reasoning="Mock decision"
        )

    def rephrase_content(self, knowledge_full_text):
        """Fail according to fail_mode."""
        self.call_count += 1

        if self.fail_count == 0 or self.call_count <= self.fail_count:
            if self.fail_mode == "api":
                raise LLMAPIError("API error", status_code=503)
            elif self.fail_mode == "timeout":
                raise LLMTimeoutError("Timeout")
            elif self.fail_mode == "response":
                raise LLMResponseError("Malformed response")

        # Success case
        from logsqueak.llm.client import RephrasedContent
        return RephrasedContent(content="Rephrased content")

    def select_target_page(self, knowledge_content, candidates, indent_str="  "):
        """Not used in new pipeline."""
        raise NotImplementedError()


class TestLLMErrorHandling:
    """Test handling of LLM failures."""

    def test_llm_api_error_propagates(self, temp_graph):
        """Test that LLM API errors propagate correctly."""
        mock_llm = MockFailingLLM(fail_mode="api")
        extractor = Extractor(llm_client=mock_llm)

        journal_file = temp_graph / "journals" / "2025_01_15.md"
        journal = JournalEntry.load(journal_file)

        # Should raise LLMAPIError
        with pytest.raises(LLMAPIError, match="API"):
            extractor.extract_knowledge(journal)

    def test_llm_timeout_error_propagates(self, temp_graph):
        """Test that LLM timeout errors propagate correctly."""
        mock_llm = MockFailingLLM(fail_mode="timeout")
        extractor = Extractor(llm_client=mock_llm)

        journal_file = temp_graph / "journals" / "2025_01_15.md"
        journal = JournalEntry.load(journal_file)

        # Should raise LLMTimeoutError
        with pytest.raises(LLMTimeoutError, match="timeout"):
            extractor.extract_knowledge(journal)

    def test_llm_response_error_propagates(self, temp_graph):
        """Test that LLM response errors propagate correctly."""
        mock_llm = MockFailingLLM(fail_mode="response")
        extractor = Extractor(llm_client=mock_llm)

        journal_file = temp_graph / "journals" / "2025_01_15.md"
        journal = JournalEntry.load(journal_file)

        # Should raise LLMResponseError
        with pytest.raises(LLMResponseError, match="response"):
            extractor.extract_knowledge(journal)

    def test_malformed_json_from_provider(self, temp_graph):
        """Test handling of malformed JSON from OpenAI-compatible provider."""
        from logsqueak.llm.providers.openai_compat import OpenAICompatibleProvider

        # Create provider with mocked httpx client
        provider = OpenAICompatibleProvider(
            endpoint="https://api.example.com",
            api_key="test-key",
            model="test-model"
        )

        # Mock response with invalid JSON
        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": "{ invalid json without closing brace"
                    }
                }
            ]
        }

        # Test _parse_json_response directly
        with pytest.raises(LLMResponseError, match="Failed to parse JSON"):
            provider._parse_json_response(mock_response)

    def test_missing_required_field_in_json(self, temp_graph):
        """Test handling of JSON missing required fields."""
        from logsqueak.llm.providers.openai_compat import OpenAICompatibleProvider

        provider = OpenAICompatibleProvider(
            endpoint="https://api.example.com",
            api_key="test-key",
            model="test-model"
        )

        # Valid JSON but missing required fields
        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": '{"wrong_field": "value"}'
                    }
                }
            ]
        }

        # Parse succeeds
        parsed = provider._parse_json_response(mock_response)
        assert "wrong_field" in parsed

        # But validation in extract_knowledge would catch missing fields
        # (tested separately in each method's validation logic)


# ============================================================================
# File System Error Tests
# ============================================================================


class TestFileSystemErrors:
    """Test handling of file system errors."""

    def test_missing_page_file(self, temp_graph):
        """Test handling of missing page file during write."""
        from logsqueak.extraction.extractor import WriteOperation

        # Create write operation for non-existent page
        write_op = WriteOperation(
            page_name="NonExistent Page",
            action=ActionType.APPEND_ROOT,
            target_id=None,
            new_content="Test content",
            original_id="test-id"
        )

        journal_path = temp_graph / "journals" / "2025_01_15.md"

        # Should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            execute_write_list(
                write_list=[write_op],
                processed_blocks_map={"test-id": [("NonExistent Page", None)]},
                graph_path=temp_graph,
                journal_path=journal_path,
            )

    def test_readonly_page_file(self, temp_graph):
        """Test handling of read-only page file."""
        # Create a page and make it read-only
        page_file = temp_graph / "pages" / "Test Page.md"
        page_file.chmod(0o444)  # Read-only

        from logsqueak.extraction.extractor import WriteOperation

        write_op = WriteOperation(
            page_name="Test Page",
            action=ActionType.APPEND_ROOT,
            target_id=None,
            new_content="Test content",
            original_id="test-id"
        )

        journal_path = temp_graph / "journals" / "2025_01_15.md"

        # Should raise PermissionError
        try:
            with pytest.raises(PermissionError):
                execute_write_list(
                    write_list=[write_op],
                    processed_blocks_map={"test-id": [("Test Page", None)]},
                    graph_path=temp_graph,
                    journal_path=journal_path,
                )
        finally:
            # Restore permissions for cleanup
            page_file.chmod(0o644)

    def test_invalid_target_block_id(self, temp_graph):
        """Test handling of invalid target block ID."""
        from logsqueak.extraction.extractor import WriteOperation

        # Create write operation with invalid target ID
        write_op = WriteOperation(
            page_name="Test Page",
            action=ActionType.APPEND_CHILD,
            target_id="non-existent-id",
            new_content="Test content",
            original_id="test-id"
        )

        journal_path = temp_graph / "journals" / "2025_01_15.md"

        # Should raise ValueError for missing target block
        with pytest.raises(ValueError, match="Target block not found"):
            execute_write_list(
                write_list=[write_op],
                processed_blocks_map={"test-id": [("Test Page", None)]},
                graph_path=temp_graph,
                journal_path=journal_path,
            )


# ============================================================================
# Atomic Operation Tests
# ============================================================================


class TestAtomicOperations:
    """Test atomic journal updates on write failures."""

    def test_journal_not_updated_on_page_write_failure(self, temp_graph):
        """Test that journal is not marked processed if page write fails."""
        from logsqueak.extraction.extractor import WriteOperation

        # Make page file read-only to cause write failure
        page_file = temp_graph / "pages" / "Test Page.md"
        page_file.chmod(0o444)

        write_op = WriteOperation(
            page_name="Test Page",
            action=ActionType.APPEND_ROOT,
            target_id=None,
            new_content="Test content",
            original_id="journal-1"
        )

        journal_path = temp_graph / "journals" / "2025_01_15.md"
        journal_before = journal_path.read_text()

        # Try to execute - should fail
        try:
            with pytest.raises(PermissionError):
                execute_write_list(
                    write_list=[write_op],
                    processed_blocks_map={"journal-1": [("Test Page", None)]},
                    graph_path=temp_graph,
                    journal_path=journal_path,
                )
        finally:
            page_file.chmod(0o644)

        # Journal should be unchanged
        journal_after = journal_path.read_text()
        assert journal_before == journal_after
        assert "processed::" not in journal_after

    def test_partial_success_marks_only_successful_pages(self, temp_graph):
        """Test that only successful page writes update the journal."""
        from logsqueak.extraction.extractor import WriteOperation

        # Create another page that will succeed
        success_page = temp_graph / "pages" / "Success Page.md"
        success_page.write_text("- # Success\n  id:: success-1\n")

        # Make one page read-only (will fail)
        fail_page = temp_graph / "pages" / "Fail Page.md"
        fail_page.write_text("- # Fail\n  id:: fail-1\n")
        fail_page.chmod(0o444)

        # Create operations for both pages
        write_ops = [
            WriteOperation(
                page_name="Success Page",
                action=ActionType.APPEND_CHILD,
                target_id="success-1",
                new_content="This should succeed",
                original_id="journal-1"
            ),
            WriteOperation(
                page_name="Fail Page",
                action=ActionType.APPEND_CHILD,
                target_id="fail-1",
                new_content="This should fail",
                original_id="journal-1"
            ),
        ]

        journal_path = temp_graph / "journals" / "2025_01_15.md"

        # Execute - should fail on second page
        try:
            with pytest.raises(PermissionError):
                execute_write_list(
                    write_list=write_ops,
                    processed_blocks_map={
                        "journal-1": [("Success Page", None), ("Fail Page", None)]
                    },
                    graph_path=temp_graph,
                    journal_path=journal_path,
                )
        finally:
            fail_page.chmod(0o644)

        # Journal should have been updated for Success Page only
        journal_content = journal_path.read_text()
        assert "processed::" in journal_content
        assert "Success Page" in journal_content


# ============================================================================
# Data Validation Tests
# ============================================================================


class TestDataValidation:
    """Test handling of invalid data."""

    def test_journal_exceeds_line_limit(self, temp_graph):
        """Test that oversized journals are truncated with warning."""
        # Create a journal with > 2000 lines
        journal_file = temp_graph / "journals" / "2025_01_16.md"
        large_content = "\n".join([f"- Line {i}" for i in range(2500)])
        journal_file.write_text(large_content)

        # Should load but truncate to 2000 lines
        journal = JournalEntry.load(journal_file)

        # Verify truncation occurred
        lines = journal.raw_content.splitlines()
        assert len(lines) == 2000, f"Expected 2000 lines, got {len(lines)}"

    def test_malformed_outline_markdown(self, temp_graph):
        """Test handling of malformed outline markdown."""
        # This should parse fine - the parser is lenient
        outline = LogseqOutline.parse("Not a valid outline")

        # Should return empty blocks
        assert len(outline.blocks) == 0

    def test_missing_journal_file(self, temp_graph):
        """Test handling of missing journal file."""
        missing_file = temp_graph / "journals" / "2025_12_31.md"

        # Should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            JournalEntry.load(missing_file)
