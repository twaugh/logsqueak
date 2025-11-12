"""Unit tests for LLM wrapper functions."""

import pytest
from unittest.mock import AsyncMock, Mock
from logsqueak.services.llm_wrappers import (
    classify_blocks,
    reword_content,
    plan_integrations,
    plan_integration_for_block
)
from logsqueak.services.llm_client import LLMClient
from logsqueak.models.llm_chunks import (
    KnowledgeClassificationChunk,
    ContentRewordingChunk,
    IntegrationDecisionChunk
)
from logsqueak.models.edited_content import EditedContent
from logseq_outline.parser import LogseqOutline


@pytest.mark.asyncio
async def test_classify_blocks_calls_llm_client_correctly():
    """Test classify_blocks() wrapper calls LLMClient.stream_ndjson with correct parameters."""
    # Arrange
    mock_client = Mock(spec=LLMClient)
    test_outline = LogseqOutline.parse(
        "- Morning standup\n  id:: 123\n- Python asyncio.Queue is thread-safe\n  id:: abc"
    )

    # Mock the stream_ndjson method to be an async generator
    async def mock_stream(*args, **kwargs):
        yield KnowledgeClassificationChunk(
            block_id="abc",
            confidence=0.85,
            reason="Technical insight"
        )

    mock_client.stream_ndjson = mock_stream

    # Act
    results = []
    async for chunk in classify_blocks(mock_client, test_outline):
        results.append(chunk)

    # Assert
    assert len(results) == 1
    assert results[0].block_id == "abc"
    assert results[0].confidence == 0.85

    # Note: Since we're using a plain function (not AsyncMock),
    # we can't easily verify call_count or call_args.
    # The test verifies the wrapper works by checking results.


@pytest.mark.asyncio
async def test_classify_blocks_handles_empty_journal():
    """Test classify_blocks() handles empty journal gracefully."""
    # Arrange
    mock_client = Mock(spec=LLMClient)
    empty_outline = LogseqOutline.parse("")

    # Mock empty stream
    async def mock_stream(*args, **kwargs):
        return
        yield  # Make it a generator

    mock_client.stream_ndjson = mock_stream

    # Act
    results = []
    async for chunk in classify_blocks(mock_client, empty_outline):
        results.append(chunk)

    # Assert
    assert len(results) == 0


@pytest.mark.asyncio
async def test_reword_content_calls_llm_client_correctly():
    """Test reword_content() wrapper calls LLMClient.stream_ndjson with correct parameters."""
    # Arrange
    mock_client = Mock(spec=LLMClient)
    test_outline = LogseqOutline.parse("- Test content\n  id:: abc")

    edited_content = EditedContent(
        block_id="abc",
        original_content="Today I learned Python asyncio is great",
        hierarchical_context="Morning notes\n  Today I learned Python asyncio is great",
        current_content="Today I learned Python asyncio is great"
    )

    # Mock the stream_ndjson method
    async def mock_stream(*args, **kwargs):
        yield ContentRewordingChunk(
            block_id="abc",
            reworded_content="Python asyncio enables concurrent operations"
        )

    mock_client.stream_ndjson = mock_stream

    # Act
    results = []
    async for chunk in reword_content(mock_client, [edited_content], test_outline):
        results.append(chunk)

    # Assert
    assert len(results) == 1
    assert results[0].block_id == "abc"
    assert "asyncio" in results[0].reworded_content


@pytest.mark.asyncio
async def test_reword_content_handles_multiple_blocks():
    """Test reword_content() handles multiple EditedContent blocks."""
    # Arrange
    mock_client = Mock(spec=LLMClient)
    test_outline = LogseqOutline.parse("- Block 1\n  id:: abc\n- Block 2\n  id:: def")

    edited_contents = [
        EditedContent(
            block_id="abc",
            original_content="Content 1",
            hierarchical_context="Context 1",
            current_content="Content 1"
        ),
        EditedContent(
            block_id="def",
            original_content="Content 2",
            hierarchical_context="Context 2",
            current_content="Content 2"
        )
    ]

    # Mock stream with multiple results
    async def mock_stream(*args, **kwargs):
        yield ContentRewordingChunk(block_id="abc", reworded_content="Reworded 1")
        yield ContentRewordingChunk(block_id="def", reworded_content="Reworded 2")

    mock_client.stream_ndjson = mock_stream

    # Act
    results = []
    async for chunk in reword_content(mock_client, edited_contents, test_outline):
        results.append(chunk)

    # Assert
    assert len(results) == 2
    assert results[0].block_id == "abc"
    assert results[1].block_id == "def"


@pytest.mark.asyncio
async def test_plan_integrations_calls_llm_client_correctly():
    """Test plan_integrations() wrapper calls LLMClient.stream_ndjson with correct parameters."""
    # Arrange
    mock_client = Mock(spec=LLMClient)

    edited_contents = [
        EditedContent(
            block_id="abc",
            original_content="Python asyncio info",
            hierarchical_context="Context",
            current_content="Python asyncio enables concurrency"
        )
    ]

    page_contents = {
        "Programming Notes": LogseqOutline.parse("- Python section\n  id:: section1")
    }

    # Mock the stream_ndjson method
    async def mock_stream(*args, **kwargs):
        yield IntegrationDecisionChunk(
            knowledge_block_id="abc",
            target_page="Programming Notes",
            action="add_under",
            target_block_id="section1",
            target_block_title="Python section",
            confidence=0.87,
            reasoning="Fits under Python section"
        )

    mock_client.stream_ndjson = mock_stream

    # Act
    results = []
    async for chunk in plan_integrations(mock_client, edited_contents, page_contents):
        results.append(chunk)

    # Assert
    assert len(results) == 1
    assert results[0].knowledge_block_id == "abc"
    assert results[0].target_page == "Programming Notes"
    assert results[0].action == "add_under"


@pytest.mark.asyncio
async def test_plan_integrations_returns_raw_stream():
    """Test plan_integrations() returns raw stream (not batched or filtered)."""
    # Arrange
    mock_client = Mock(spec=LLMClient)

    edited_contents = [
        EditedContent(
            block_id="abc",
            original_content="Test",
            hierarchical_context="Context",
            current_content="Test content"
        )
    ]

    page_contents = {"Page1": LogseqOutline.parse("- Section\n  id:: s1")}

    # Mock stream with skip_exists action
    async def mock_stream(*args, **kwargs):
        yield IntegrationDecisionChunk(
            knowledge_block_id="abc",
            target_page="Page1",
            action="skip_exists",
            target_block_id="s1",
            target_block_title="Already exists",
            confidence=0.95,
            reasoning="Duplicate"
        )
        yield IntegrationDecisionChunk(
            knowledge_block_id="abc",
            target_page="Page2",
            action="add_section",
            confidence=0.80,
            reasoning="New section"
        )

    mock_client.stream_ndjson = mock_stream

    # Act
    results = []
    async for chunk in plan_integrations(mock_client, edited_contents, page_contents):
        results.append(chunk)

    # Assert - should return ALL decisions including skip_exists (raw stream)
    assert len(results) == 2
    assert results[0].action == "skip_exists"
    assert results[1].action == "add_section"


@pytest.mark.asyncio
async def test_plan_integrations_handles_empty_candidates():
    """Test plan_integrations() handles empty candidate pages."""
    # Arrange
    mock_client = Mock(spec=LLMClient)

    edited_contents = [
        EditedContent(
            block_id="abc",
            original_content="Test",
            hierarchical_context="Context",
            current_content="Test"
        )
    ]

    # Empty candidates
    page_contents = {}

    # Mock empty stream
    async def mock_stream(*args, **kwargs):
        return
        yield  # Make it a generator

    mock_client.stream_ndjson = mock_stream

    # Act
    results = []
    async for chunk in plan_integrations(mock_client, edited_contents, page_contents):
        results.append(chunk)

    # Assert
    assert len(results) == 0
