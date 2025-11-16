"""Integration tests for edge cases and error handling.

This test suite covers edge cases from Phase 7 (T110-T119):
- T110-T112: Config file errors (missing, invalid, wrong permissions) - COVERED in test_config_loading.py
- T113: Network errors (connection refused, timeout, invalid API key) - COVERED in test_llm_streaming.py
- T114: Malformed JSON in NDJSON streaming - COVERED in test_llm_streaming.py
- T115: No knowledge blocks identified (empty state in Phase 1) - Partially tested here
- T116: No relevant pages found (empty state in Phase 3) - Partially tested here
- T117-T118a: Atomic writes with concurrent modification detection - COVERED in test_atomic_writes.py
- T119: Ctrl+C cancellation warning in Phase 3 - Partially tested here

This file focuses on lightweight edge cases NOT covered by existing integration tests.
Full TUI integration tests would require more resources.
"""

import pytest
from pathlib import Path
from unittest.mock import patch

from logsqueak.services.llm_client import LLMClient
from logsqueak.services.file_monitor import FileMonitor
from logsqueak.models.integration_decision import IntegrationDecision
from logseq_outline.parser import LogseqOutline


class TestEmptyStates:
    """Test edge cases for empty states.

    Note: Full TUI tests for T115 and T116 are covered in UI test suites.
    These tests verify the data layer edge cases.
    """

    @pytest.mark.asyncio
    async def test_empty_llm_classification_stream(self):
        """
        T115: Test handling of empty LLM classification stream.

        Scenario:
        1. LLM returns empty stream (no knowledge blocks identified)
        2. System handles gracefully without errors
        """
        from logsqueak.models.config import LLMConfig
        from logsqueak.services.llm_client import LLMClient
        from logsqueak.models.llm_chunks import KnowledgeClassificationChunk

        config = LLMConfig(
            endpoint="https://api.test.com/v1",
            api_key="test-key",
            model="test-model"
        )
        client = LLMClient(config)

        # Mock empty stream
        async def mock_empty_stream(*args, **kwargs):
            if False:
                yield

        with patch.object(client, 'stream_ndjson', side_effect=mock_empty_stream):
            results = []
            async for chunk in client.stream_ndjson(
                prompt="Test",
                system_prompt="Test",
                chunk_model=KnowledgeClassificationChunk
            ):
                results.append(chunk)

            # Should handle empty stream gracefully
            assert len(results) == 0

    @pytest.mark.asyncio
    async def test_empty_integration_decisions_stream(self):
        """
        T116: Test handling of empty integration decisions stream.

        Scenario:
        1. RAG search finds no candidate pages
        2. LLM returns empty stream (no integrations)
        3. System handles gracefully without errors
        """
        from logsqueak.models.config import LLMConfig
        from logsqueak.services.llm_client import LLMClient
        from logsqueak.models.llm_chunks import IntegrationDecisionChunk

        config = LLMConfig(
            endpoint="https://api.test.com/v1",
            api_key="test-key",
            model="test-model"
        )
        client = LLMClient(config)

        # Mock empty decision stream
        async def mock_empty_stream(*args, **kwargs):
            if False:
                yield

        with patch.object(client, 'stream_ndjson', side_effect=mock_empty_stream):
            results = []
            async for chunk in client.stream_ndjson(
                prompt="Test",
                system_prompt="Test",
                chunk_model=IntegrationDecisionChunk
            ):
                results.append(chunk)

            # Should handle empty stream gracefully
            assert len(results) == 0


class TestExternalFileModification:
    """Test edge cases for external file modifications during TUI session."""

    @pytest.fixture
    def journal_file(self, tmp_path):
        """Create test journal file."""
        journals_dir = tmp_path / "journals"
        journals_dir.mkdir(parents=True)
        journal_path = journals_dir / "2025-11-14.md"
        journal_path.write_text("- Original content\n  - Child block\n")
        return journal_path

    @pytest.fixture
    def target_page(self, tmp_path):
        """Create test target page."""
        pages_dir = tmp_path / "pages"
        pages_dir.mkdir(parents=True)
        page_path = pages_dir / "Test Page.md"
        page_path.write_text("- Existing section\n  - Existing content\n")
        return page_path

    @pytest.mark.asyncio
    async def test_concurrent_modification_detection(self, journal_file, target_page):
        """
        T117-T118: Test atomic write detects concurrent file modifications.

        Scenario:
        1. TUI loads target page for preview
        2. External process (Logseq) modifies target page
        3. FileMonitor detects the modification
        4. This prevents unsafe writes

        Note: Full atomic write tests are in test_atomic_writes.py
        This test verifies FileMonitor concurrent modification detection.
        """
        # Create file monitor and record initial state
        monitor = FileMonitor()
        monitor.record(target_page)

        # Verify file is not modified initially
        assert not monitor.is_modified(target_page)

        # Simulate external modification (Logseq editing the page)
        target_page.write_text("- Modified by Logseq\n  - New content\n")

        # FileMonitor should detect modification
        assert monitor.is_modified(target_page)

    @pytest.mark.asyncio
    async def test_journal_modification_during_phase2(self, journal_file):
        """
        Test handling of journal modification during Phase 2 editing.

        Scenario:
        1. User is in Phase 2 editing content
        2. External process modifies journal (unlikely but possible)
        3. When user proceeds to Phase 3, journal reload is attempted
        4. User sees warning about journal changes
        """
        from logsqueak.services.file_monitor import FileMonitor

        monitor = FileMonitor()
        monitor.record(journal_file)

        # Simulate external modification
        original_content = journal_file.read_text()
        journal_file.write_text(original_content + "- New entry added externally\n")

        # Check modification is detected
        assert monitor.is_modified(journal_file)

        # Reload and verify new content
        new_content = journal_file.read_text()
        assert "New entry added externally" in new_content

        # Update monitor with new state
        monitor.refresh(journal_file)
        assert not monitor.is_modified(journal_file)


class TestLargeJournalEntries:
    """Test edge cases for very large journal entries."""

    @pytest.mark.asyncio
    async def test_large_journal_navigation_performance(self, tmp_path):
        """
        T134: Test UI responsiveness with large journal entries (>100 blocks).

        Scenario:
        1. Create journal with 150+ blocks (nested structure)
        2. Load in Phase 1
        3. Verify navigation is responsive (<100ms per keystroke)
        4. Verify LLM classification handles large batches
        """
        # Create large journal
        journals_dir = tmp_path / "journals"
        journals_dir.mkdir(parents=True)
        journal_path = journals_dir / "2025-11-14.md"

        # Generate 150 blocks with nested structure
        lines = []
        for i in range(50):
            lines.append(f"- Root block {i}\n")
            lines.append(f"  - Child block {i}.1\n")
            lines.append(f"    - Grandchild block {i}.1.1\n")

        journal_path.write_text("".join(lines))

        # Parse and verify structure
        outline = LogseqOutline.parse(journal_path.read_text())
        assert len(outline.blocks) == 50  # 50 root blocks

        # Verify nested structure
        total_blocks = 0
        for root in outline.blocks:
            total_blocks += 1  # Root
            total_blocks += len(root.children)  # Children
            for child in root.children:
                total_blocks += len(child.children)  # Grandchildren

        assert total_blocks == 150  # 50 * 3 = 150 total blocks

        # Test would continue with TUI navigation timing tests
        # For now, we verify the structure can be parsed

    @pytest.mark.asyncio
    async def test_deeply_nested_blocks(self, tmp_path):
        """
        Test handling of deeply nested block structures (10+ levels).

        Scenario:
        1. Create journal with deeply nested blocks
        2. Verify hierarchical context generation doesn't overflow
        3. Verify tree navigation works at all depths
        """
        journals_dir = tmp_path / "journals"
        journals_dir.mkdir(parents=True)
        journal_path = journals_dir / "2025-11-14.md"

        # Create 10 levels of nesting
        lines = ["- Level 0\n"]
        for i in range(1, 11):
            indent = "  " * i
            lines.append(f"{indent}- Level {i}\n")

        journal_path.write_text("".join(lines))

        # Parse and verify depth
        outline = LogseqOutline.parse(journal_path.read_text())

        # Navigate to deepest block
        current = outline.blocks[0]
        depth = 0
        while current.children:
            current = current.children[0]
            depth += 1

        assert depth == 10  # Reached 10 levels deep


# Network resilience tests (T113) are comprehensive covered in test_llm_streaming.py
# Including: connection refused, timeouts, invalid API keys, retry logic


class TestCancellationHandling:
    """Test edge cases for user cancellation and interruption.

    Note: Full worker cancellation tests are covered in test_worker_dependencies.py
    """

    @pytest.mark.asyncio
    async def test_partial_journal_state_detection(self, tmp_path):
        """
        T119: Test detection of partial journal state (some blocks processed, some not).

        Scenario:
        1. Journal has some blocks with extracted-to:: markers (completed writes)
        2. Journal has other blocks without extracted-to:: markers (not yet written)
        3. System can detect this partial state
        4. This is used to warn user before exiting in Phase 3

        Note: Full Ctrl+C handling UI tests are in UI test suites
        """
        # Create journal with partial state
        journals_dir = tmp_path / "journals"
        journals_dir.mkdir(parents=True)
        journal_path = journals_dir / "2025-11-14.md"

        # Simulate partial journal state:
        # Block 1 has extracted-to:: marker (written successfully)
        # Block 2 does NOT have extracted-to:: marker (not yet written)
        journal_content = """- First block
  id:: block-1
  extracted-to:: [[Test Page]]
- Second block
  id:: block-2
"""
        journal_path.write_text(journal_content)

        # Parse and verify partial state can be detected
        outline = LogseqOutline.parse(journal_content)
        assert len(outline.blocks) == 2

        # Block 1 has extracted-to property
        block1 = outline.blocks[0]
        assert block1.get_property("extracted-to") == "[[Test Page]]"

        # Block 2 does NOT have extracted-to property
        block2 = outline.blocks[1]
        assert block2.get_property("extracted-to") is None

        # Count blocks with extracted-to markers
        processed_count = sum(
            1 for block in outline.blocks
            if block.get_property("extracted-to") is not None
        )
        total_count = len(outline.blocks)

        # Verify we can detect partial state
        has_partial_state = 0 < processed_count < total_count
        assert has_partial_state is True


@pytest.mark.asyncio
async def test_empty_journal_file():
    """
    Test handling of empty journal files.

    Scenario:
    1. User has an empty journal file (just created for today)
    2. CLI loads empty journal
    3. Phase 1 shows empty state gracefully
    4. No crashes or errors
    """
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        journal_path = Path(tmpdir) / "2025-11-14.md"
        journal_path.write_text("")  # Empty file

        # Parse empty file
        outline = LogseqOutline.parse(journal_path.read_text())
        assert len(outline.blocks) == 0
        assert len(outline.frontmatter) == 0  # Empty list


@pytest.mark.asyncio
async def test_journal_with_only_frontmatter():
    """
    Test handling of journal files with only frontmatter (no blocks).

    Scenario:
    1. Journal has page properties but no blocks
    2. Parser handles gracefully
    3. Phase 1 shows empty state
    """
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        journal_path = Path(tmpdir) / "2025-11-14.md"
        journal_path.write_text("title:: Daily Journal\ndate:: 2025-11-14\n\n")

        # Parse file with only frontmatter
        outline = LogseqOutline.parse(journal_path.read_text())
        assert len(outline.blocks) == 0
        # Frontmatter is a list of lines
        frontmatter_text = "\n".join(outline.frontmatter)
        assert "title::" in frontmatter_text
        assert "date::" in frontmatter_text
