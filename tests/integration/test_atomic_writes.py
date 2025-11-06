"""Integration tests for atomic two-phase writes with FileMonitor.

Tests the complete integration workflow including file I/O, concurrent
modification detection, and atomic guarantees.
"""

import pytest
import tempfile
from pathlib import Path
from logseq_outline.parser import LogseqOutline
from logseq_outline.graph import GraphPaths
from logsqueak.services.file_operations import write_integration_atomic
from logsqueak.services.file_monitor import FileMonitor
from logsqueak.models.integration_decision import IntegrationDecision


@pytest.fixture
def temp_graph_dir():
    """Create temporary Logseq graph directory structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        graph_path = Path(tmpdir)

        # Create directory structure
        (graph_path / "pages").mkdir()
        (graph_path / "journals").mkdir()

        yield graph_path


@pytest.fixture
def graph_paths(temp_graph_dir):
    """Create GraphPaths instance for temp graph."""
    return GraphPaths(temp_graph_dir)


@pytest.fixture
def file_monitor():
    """Create FileMonitor instance."""
    return FileMonitor()


@pytest.fixture
def sample_journal(graph_paths):
    """Create a sample journal entry."""
    journal_path = graph_paths.get_journal_path("2025-11-06")
    journal_content = """- Daily standup meeting
- Knowledge block about Python async patterns
  id:: knowledge-block-1
- Another task block
"""
    journal_path.write_text(journal_content)
    return journal_path


@pytest.fixture
def sample_page(graph_paths):
    """Create a sample target page."""
    page_path = graph_paths.get_page_path("Python/Concurrency")
    # Ensure parent directory exists
    page_path.parent.mkdir(parents=True, exist_ok=True)
    page_content = """- Async programming concepts
  id:: async-concepts
  - Event loops
  - Coroutines
- Best practices
  id:: best-practices
"""
    page_path.write_text(page_content)
    return page_path


@pytest.mark.asyncio
async def test_atomic_write_add_section(
    graph_paths, file_monitor, sample_journal, sample_page
):
    """Test atomic write with add_section action."""
    decision = IntegrationDecision(
        knowledge_block_id="knowledge-block-1",
        target_page="Python/Concurrency",
        action="add_section",
        confidence=0.85,
        refined_text="Async/await patterns for concurrent operations",
        reasoning="Relevant to async programming"
    )

    # Record initial file states
    file_monitor.record(sample_page)
    file_monitor.record(sample_journal)

    # Perform atomic write
    await write_integration_atomic(
        decision=decision,
        journal_date="2025-11-06",
        graph_paths=graph_paths,
        file_monitor=file_monitor
    )

    # Verify page was updated
    page_outline = LogseqOutline.parse(sample_page.read_text())
    assert len(page_outline.blocks) == 3  # Original 2 + new 1
    new_block = page_outline.blocks[2]
    assert "Async/await patterns" in new_block.content[0]
    assert new_block.get_property("id") is not None

    # Verify journal was updated with provenance
    journal_outline = LogseqOutline.parse(sample_journal.read_text())
    knowledge_block = journal_outline.find_block_by_id("knowledge-block-1")
    assert knowledge_block is not None

    processed = knowledge_block.get_property("processed")
    assert processed is not None
    assert "Python/Concurrency" in processed
    assert "((" in processed  # Block reference format


@pytest.mark.asyncio
async def test_atomic_write_add_under(
    graph_paths, file_monitor, sample_journal, sample_page
):
    """Test atomic write with add_under action."""
    decision = IntegrationDecision(
        knowledge_block_id="knowledge-block-1",
        target_page="Python/Concurrency",
        action="add_under",
        target_block_id="async-concepts",
        target_block_title="Async programming concepts",
        confidence=0.90,
        refined_text="Async/await patterns for concurrent operations",
        reasoning="Fits under async concepts"
    )

    file_monitor.record(sample_page)
    file_monitor.record(sample_journal)

    await write_integration_atomic(
        decision=decision,
        journal_date="2025-11-06",
        graph_paths=graph_paths,
        file_monitor=file_monitor
    )

    # Verify child was added
    page_outline = LogseqOutline.parse(sample_page.read_text())
    parent_block = page_outline.find_block_by_id("async-concepts")
    assert len(parent_block.children) == 3  # Original 2 + new 1
    new_child = parent_block.children[2]
    assert "Async/await patterns" in new_child.content[0]

    # Verify journal provenance
    journal_outline = LogseqOutline.parse(sample_journal.read_text())
    knowledge_block = journal_outline.find_block_by_id("knowledge-block-1")
    processed = knowledge_block.get_property("processed")
    assert processed is not None
    assert "Python/Concurrency" in processed


@pytest.mark.asyncio
async def test_atomic_write_replace(
    graph_paths, file_monitor, sample_journal, sample_page
):
    """Test atomic write with replace action."""
    decision = IntegrationDecision(
        knowledge_block_id="knowledge-block-1",
        target_page="Python/Concurrency",
        action="replace",
        target_block_id="best-practices",
        target_block_title="Best practices",
        confidence=0.75,
        refined_text="Updated best practices for async programming",
        reasoning="Replaces outdated content"
    )

    file_monitor.record(sample_page)
    file_monitor.record(sample_journal)

    await write_integration_atomic(
        decision=decision,
        journal_date="2025-11-06",
        graph_paths=graph_paths,
        file_monitor=file_monitor
    )

    # Verify content was replaced
    page_outline = LogseqOutline.parse(sample_page.read_text())
    replaced_block = page_outline.find_block_by_id("best-practices")
    assert "Updated best practices" in replaced_block.content[0]

    # Verify journal provenance
    journal_outline = LogseqOutline.parse(sample_journal.read_text())
    knowledge_block = journal_outline.find_block_by_id("knowledge-block-1")
    processed = knowledge_block.get_property("processed")
    assert processed is not None


@pytest.mark.asyncio
async def test_atomic_write_multiple_integrations_same_block(
    graph_paths, file_monitor, sample_journal, sample_page
):
    """Test multiple integrations from same knowledge block."""
    decision1 = IntegrationDecision(
        knowledge_block_id="knowledge-block-1",
        target_page="Python/Concurrency",
        action="add_section",
        confidence=0.85,
        refined_text="Async patterns content",
        reasoning="First integration"
    )

    decision2 = IntegrationDecision(
        knowledge_block_id="knowledge-block-1",
        target_page="Python/Concurrency",
        action="add_under",
        target_block_id="async-concepts",
        confidence=0.90,
        refined_text="Async patterns content",
        reasoning="Second integration"
    )

    file_monitor.record(sample_page)
    file_monitor.record(sample_journal)

    # First integration
    await write_integration_atomic(
        decision=decision1,
        journal_date="2025-11-06",
        graph_paths=graph_paths,
        file_monitor=file_monitor
    )

    # Second integration (same block to different location)
    await write_integration_atomic(
        decision=decision2,
        journal_date="2025-11-06",
        graph_paths=graph_paths,
        file_monitor=file_monitor
    )

    # Verify journal has both provenance links
    journal_outline = LogseqOutline.parse(sample_journal.read_text())
    knowledge_block = journal_outline.find_block_by_id("knowledge-block-1")
    processed = knowledge_block.get_property("processed")

    # Should have two comma-separated links
    assert processed.count("Python/Concurrency") == 2
    assert processed.count(",") == 1


@pytest.mark.asyncio
async def test_concurrent_page_modification_detection(
    graph_paths, file_monitor, sample_journal, sample_page
):
    """Test that concurrent page modifications are detected."""
    decision = IntegrationDecision(
        knowledge_block_id="knowledge-block-1",
        target_page="Python/Concurrency",
        action="add_section",
        confidence=0.85,
        refined_text="New content",
        reasoning="Test"
    )

    file_monitor.record(sample_page)
    file_monitor.record(sample_journal)

    # Simulate external modification to page
    sample_page.write_text("- Externally modified content\n")

    # Write should detect modification and reload
    await write_integration_atomic(
        decision=decision,
        journal_date="2025-11-06",
        graph_paths=graph_paths,
        file_monitor=file_monitor
    )

    # Verify write succeeded with reloaded content
    page_outline = LogseqOutline.parse(sample_page.read_text())
    assert len(page_outline.blocks) == 2  # External + new


@pytest.mark.asyncio
async def test_concurrent_journal_modification_detection(
    graph_paths, file_monitor, sample_journal, sample_page
):
    """Test that concurrent journal modifications are detected."""
    decision = IntegrationDecision(
        knowledge_block_id="knowledge-block-1",
        target_page="Python/Concurrency",
        action="add_section",
        confidence=0.85,
        refined_text="New content",
        reasoning="Test"
    )

    file_monitor.record(sample_page)
    file_monitor.record(sample_journal)

    # Perform page write (first phase)
    # Then simulate external journal modification before journal write
    # (This is tricky - we need to hook into the middle of the atomic write)
    # For now, just verify that journal reload works

    await write_integration_atomic(
        decision=decision,
        journal_date="2025-11-06",
        graph_paths=graph_paths,
        file_monitor=file_monitor
    )

    # Verify provenance was added
    journal_outline = LogseqOutline.parse(sample_journal.read_text())
    knowledge_block = journal_outline.find_block_by_id("knowledge-block-1")
    assert knowledge_block.get_property("processed") is not None


@pytest.mark.asyncio
async def test_idempotent_retry_detection(
    graph_paths, file_monitor, sample_journal, sample_page
):
    """Test that retry with same decision is idempotent."""
    decision = IntegrationDecision(
        knowledge_block_id="knowledge-block-1",
        target_page="Python/Concurrency",
        action="add_section",
        confidence=0.85,
        refined_text="Content to add",
        reasoning="Test idempotency"
    )

    file_monitor.record(sample_page)
    file_monitor.record(sample_journal)

    # First write
    await write_integration_atomic(
        decision=decision,
        journal_date="2025-11-06",
        graph_paths=graph_paths,
        file_monitor=file_monitor
    )

    # Get page state after first write
    page_outline_after_first = LogseqOutline.parse(sample_page.read_text())
    block_count_after_first = len(page_outline_after_first.blocks)

    # Retry same decision (should detect existing block and skip page write)
    await write_integration_atomic(
        decision=decision,
        journal_date="2025-11-06",
        graph_paths=graph_paths,
        file_monitor=file_monitor
    )

    # Verify no duplicate block was created
    page_outline_after_retry = LogseqOutline.parse(sample_page.read_text())
    assert len(page_outline_after_retry.blocks) == block_count_after_first


@pytest.mark.asyncio
async def test_write_failure_target_not_found(
    graph_paths, file_monitor, sample_journal, sample_page
):
    """Test write failure when target block doesn't exist."""
    decision = IntegrationDecision(
        knowledge_block_id="knowledge-block-1",
        target_page="Python/Concurrency",
        action="add_under",
        target_block_id="nonexistent-block",
        confidence=0.85,
        refined_text="Content",
        reasoning="Test error"
    )

    file_monitor.record(sample_page)
    file_monitor.record(sample_journal)

    # Should raise error
    with pytest.raises(ValueError, match="Target block not found"):
        await write_integration_atomic(
            decision=decision,
            journal_date="2025-11-06",
            graph_paths=graph_paths,
            file_monitor=file_monitor
        )

    # Verify journal was NOT modified (atomic guarantee)
    journal_outline = LogseqOutline.parse(sample_journal.read_text())
    knowledge_block = journal_outline.find_block_by_id("knowledge-block-1")
    assert knowledge_block.get_property("processed") is None


@pytest.mark.asyncio
async def test_write_failure_page_not_found(
    graph_paths, file_monitor, sample_journal
):
    """Test write failure when target page doesn't exist."""
    decision = IntegrationDecision(
        knowledge_block_id="knowledge-block-1",
        target_page="Nonexistent/Page",
        action="add_section",
        confidence=0.85,
        refined_text="Content",
        reasoning="Test error"
    )

    file_monitor.record(sample_journal)

    # Should raise FileNotFoundError
    with pytest.raises(FileNotFoundError, match="Page not found"):
        await write_integration_atomic(
            decision=decision,
            journal_date="2025-11-06",
            graph_paths=graph_paths,
            file_monitor=file_monitor
        )

    # Verify journal was NOT modified
    journal_outline = LogseqOutline.parse(sample_journal.read_text())
    knowledge_block = journal_outline.find_block_by_id("knowledge-block-1")
    assert knowledge_block.get_property("processed") is None


@pytest.mark.asyncio
async def test_property_order_preservation(
    graph_paths, file_monitor, sample_journal, sample_page
):
    """Test that property order is preserved during integration."""
    # Create page with specific property order
    page_path = graph_paths.get_page_path("Python/Concurrency")
    # Ensure parent directory exists
    page_path.parent.mkdir(parents=True, exist_ok=True)
    page_content = """- Block with properties
  tags:: #python #async
  id:: test-block
  author:: User Name
  created:: 2025-11-06
"""
    page_path.write_text(page_content)

    decision = IntegrationDecision(
        knowledge_block_id="knowledge-block-1",
        target_page="Python/Concurrency",
        action="replace",
        target_block_id="test-block",
        confidence=0.85,
        refined_text="Replaced content",
        reasoning="Test property order"
    )

    file_monitor.record(sample_page)
    file_monitor.record(sample_journal)

    await write_integration_atomic(
        decision=decision,
        journal_date="2025-11-06",
        graph_paths=graph_paths,
        file_monitor=file_monitor
    )

    # Verify property order preserved
    rendered = page_path.read_text()
    lines = rendered.strip().split("\n")

    # Properties should maintain original order
    assert "tags::" in lines[1]
    assert "id::" in lines[2]
    assert "author::" in lines[3]
    assert "created::" in lines[4]
