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
    # NOTE: The wrapper now uses short IDs (1, 2, 3...) in prompts and translates back
    # Block "123" will be mapped to "1", block "abc" will be mapped to "2"
    async def mock_stream(*args, **kwargs):
        yield KnowledgeClassificationChunk(
            block_id="2",  # Short ID for block "abc"
            reasoning="Python asyncio.Queue is thread-safe for concurrent access",
            confidence=0.85
        )

    mock_client.stream_ndjson = mock_stream

    # Act
    results = []
    async for chunk in classify_blocks(mock_client, test_outline):
        results.append(chunk)

    # Assert
    assert len(results) == 1
    assert results[0].block_id == "abc"  # Translated back to hybrid ID
    assert results[0].reasoning == "Python asyncio.Queue is thread-safe for concurrent access"
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
async def test_classify_blocks_excludes_frontmatter():
    """Test classify_blocks() excludes journal frontmatter from LLM prompt."""
    # Arrange
    mock_client = Mock(spec=LLMClient)

    # Journal with frontmatter (date, properties)
    journal_with_frontmatter = (
        "date:: 2025-01-15\n"
        "tags:: journal, daily\n"
        "\n"
        "- Morning standup meeting\n"
        "  id:: 123\n"
        "- Python asyncio.Queue is thread-safe\n"
        "  id:: abc"
    )
    test_outline = LogseqOutline.parse(journal_with_frontmatter)

    # Verify frontmatter was parsed
    assert len(test_outline.frontmatter) > 0
    assert test_outline.frontmatter[0] == "date:: 2025-01-15"

    # Capture the prompt sent to LLM
    captured_prompt = None
    async def mock_stream(*args, **kwargs):
        nonlocal captured_prompt
        captured_prompt = kwargs.get('prompt', args[0] if args else None)
        yield KnowledgeClassificationChunk(
            block_id="abc",
            reasoning="Python asyncio.Queue is thread-safe for concurrent access",
            confidence=0.85
        )

    mock_client.stream_ndjson = mock_stream

    # Act
    results = []
    async for chunk in classify_blocks(mock_client, test_outline):
        results.append(chunk)

    # Assert
    assert captured_prompt is not None
    # Frontmatter should NOT be in the prompt
    assert "date:: 2025-01-15" not in captured_prompt
    assert "tags:: journal, daily" not in captured_prompt
    # Block content SHOULD be in the prompt
    assert "Morning standup meeting" in captured_prompt
    assert "Python asyncio.Queue is thread-safe" in captured_prompt

    # Original outline should still have frontmatter preserved
    assert len(test_outline.frontmatter) > 0
    assert test_outline.frontmatter[0] == "date:: 2025-01-15"


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
    # NOTE: The wrapper uses short IDs - block "abc" will be mapped to "1"
    async def mock_stream(*args, **kwargs):
        yield ContentRewordingChunk(
            block_id="1",  # Short ID
            reworded_content="Python asyncio enables concurrent operations"
        )

    mock_client.stream_ndjson = mock_stream

    # Act
    results = []
    async for chunk in reword_content(mock_client, [edited_content], test_outline):
        results.append(chunk)

    # Assert
    assert len(results) == 1
    assert results[0].block_id == "abc"  # Translated back to hybrid ID
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
    # NOTE: Block "abc" → "1", block "def" → "2"
    async def mock_stream(*args, **kwargs):
        yield ContentRewordingChunk(block_id="1", reworded_content="Reworded 1")
        yield ContentRewordingChunk(block_id="2", reworded_content="Reworded 2")

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

    candidate_chunks = {
        "abc": [
            ("Programming Notes", "section1", "- Python section")
        ]
    }

    # Mock the stream_ndjson method
    # NOTE: LLM no longer returns knowledge_block_id (set by handler)
    # Only RAG candidate blocks get short IDs: "section1" → "1"
    async def mock_stream(*args, **kwargs):
        yield IntegrationDecisionChunk(
            target_page="Programming Notes",
            action="add_under",
            target_block_id="1",  # Short ID for "section1"
            target_block_title="Python section",
            confidence=0.87,
            reasoning="Fits under Python section"
        )

    mock_client.stream_ndjson = mock_stream

    # Act
    results = []
    async for chunk in plan_integrations(mock_client, edited_contents, page_contents, candidate_chunks):
        results.append(chunk)

    # Assert
    assert len(results) == 1
    assert results[0].knowledge_block_id == "abc"  # Translated back to hybrid ID
    assert results[0].target_block_id == "section1"  # Translated back to hybrid ID
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

    candidate_chunks = {
        "abc": [
            ("Page1", "s1", "- Section")
        ]
    }

    # Mock stream with skip_exists action
    # NOTE: LLM no longer returns knowledge_block_id
    # Only RAG candidate blocks get short IDs: "s1" → "1"
    async def mock_stream(*args, **kwargs):
        yield IntegrationDecisionChunk(
            target_page="Page1",
            action="skip_exists",
            target_block_id="1",  # Short ID for "s1"
            target_block_title="Already exists",
            confidence=0.95,
            reasoning="Duplicate"
        )
        yield IntegrationDecisionChunk(
            target_page="Page2",
            action="add_section",
            confidence=0.80,
            reasoning="New section"
        )

    mock_client.stream_ndjson = mock_stream

    # Act
    results = []
    async for chunk in plan_integrations(mock_client, edited_contents, page_contents, candidate_chunks):
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
    candidate_chunks = {}

    # Mock empty stream
    async def mock_stream(*args, **kwargs):
        return
        yield  # Make it a generator

    mock_client.stream_ndjson = mock_stream

    # Act
    results = []
    async for chunk in plan_integrations(mock_client, edited_contents, page_contents, candidate_chunks):
        results.append(chunk)

    # Assert
    assert len(results) == 0


@pytest.mark.asyncio
async def test_plan_integration_for_block_produces_correct_decisions():
    """Test plan_integration_for_block() for single block produces decisions with correct knowledge_block_id."""
    # Arrange
    mock_client = Mock(spec=LLMClient)

    edited_content = EditedContent(
        block_id="abc123",
        original_content="Python asyncio info",
        hierarchical_context="- Today's learning\n  - Python asyncio enables concurrency",
        current_content="Python asyncio enables concurrent operations"
    )

    candidate_chunks = [
        ("Programming Notes", "section1", "- Python section\n  - Concurrency basics")
    ]

    page_contents = {
        "Programming Notes": LogseqOutline.parse("- Python section\n  id:: section1\n  - Concurrency basics")
    }

    # Mock the stream_ndjson method
    # NOTE: LLM no longer returns knowledge_block_id
    # Only RAG candidate blocks get short IDs: "section1" → "1"
    async def mock_stream(*args, **kwargs):
        yield IntegrationDecisionChunk(
            target_page="Programming Notes",
            action="add_under",
            target_block_id="1",  # Short ID for "section1"
            target_block_title="Python section",
            confidence=0.87,
            reasoning="Fits under Python concurrency section"
        )

    mock_client.stream_ndjson = mock_stream

    # Act
    results = []
    async for chunk in plan_integration_for_block(mock_client, edited_content, candidate_chunks, page_contents):
        results.append(chunk)

    # Assert
    assert len(results) == 1
    assert results[0].knowledge_block_id == "abc123"
    assert results[0].target_page == "Programming Notes"
    assert results[0].action == "add_under"
    assert results[0].target_block_id == "section1"


@pytest.mark.asyncio
async def test_plan_integration_for_block_strips_id_from_knowledge_block():
    """Test that plan_integration_for_block() strips id:: from knowledge block context."""
    # Arrange
    mock_client = Mock(spec=LLMClient)

    # EditedContent with id:: property in hierarchical_context
    edited_content = EditedContent(
        block_id="abc123",
        original_content="Python type hints are powerful",
        hierarchical_context=(
            "- DONE Review [[Python]] docs\n"
            "  id:: parent-id-456\n"
            "  - Python type hints are powerful\n"
            "    id:: abc123"
        ),
        current_content="Python type hints are powerful for catching bugs"
    )

    candidate_chunks = [
        ("Python", "target1", "- Best practices\n  - Use type hints")
    ]

    page_contents = {
        "Python": LogseqOutline.parse("- Best practices\n  id:: target1")
    }

    # Track the prompt
    captured_prompt = None

    async def mock_stream(*args, **kwargs):
        nonlocal captured_prompt
        captured_prompt = kwargs.get('prompt', args[0] if args else None)
        yield IntegrationDecisionChunk(
            target_page="Python",
            action="add_under",
            target_block_id="1",  # Short ID for "target1"
            target_block_title="Best practices",
            confidence=0.95,
            reasoning="Related to type hints"
        )

    mock_client.stream_ndjson = mock_stream

    # Act
    results = []
    async for chunk in plan_integration_for_block(mock_client, edited_content, candidate_chunks, page_contents):
        results.append(chunk)

    # Assert - prompt should NOT contain id:: properties in knowledge block
    assert captured_prompt is not None
    assert "<knowledge_block>" in captured_prompt  # Singular tag (one block per call)

    # CRITICAL: id:: properties should be stripped from knowledge block content
    # The prompt should not show "id:: abc123" or "id:: parent-id-456"
    knowledge_block_section = captured_prompt.split("<knowledge_block>")[1].split("</knowledge_block>")[0]
    assert "id:: abc123" not in knowledge_block_section
    assert "id:: parent-id-456" not in knowledge_block_section

    # But the actual content should still be there
    assert "Python type hints are powerful" in knowledge_block_section
    assert "DONE Review [[Python]] docs" in knowledge_block_section


@pytest.mark.asyncio
async def test_plan_integration_for_block_formats_chunks_correctly():
    """Test plan_integration_for_block() formats RAG chunks correctly (hierarchical, id stripped)."""
    # Arrange
    mock_client = Mock(spec=LLMClient)

    edited_content = EditedContent(
        block_id="test456",
        original_content="Docker containers should be stateless",
        hierarchical_context="- DevOps notes\n  - Docker containers should be stateless",
        current_content="Docker containers should be stateless for easier scaling"
    )

    # Hierarchical chunks with nested context
    candidate_chunks = [
        ("DevOps", "parent1", "- Container best practices\n  - State management\n    - External volumes preferred"),
        ("DevOps", "parent2", "- Scalability patterns\n  - Stateless services enable horizontal scaling")
    ]

    page_contents = {
        "DevOps": LogseqOutline.parse(
            "title:: DevOps Guide\n"
            "- Container best practices\n"
            "  id:: parent1\n"
            "- Scalability patterns\n"
            "  id:: parent2"
        )
    }

    # Track the prompt that was sent to LLM
    captured_prompt = None

    async def mock_stream(*args, **kwargs):
        nonlocal captured_prompt
        captured_prompt = kwargs.get('prompt', args[0] if args else None)
        yield IntegrationDecisionChunk(
            knowledge_block_id="test456",
            target_page="DevOps",
            action="add_under",
            target_block_id="parent1",
            target_block_title="Container best practices",
            confidence=0.90,
            reasoning="Related to container state management"
        )

    mock_client.stream_ndjson = mock_stream

    # Act
    results = []
    async for chunk in plan_integration_for_block(mock_client, edited_content, candidate_chunks, page_contents):
        results.append(chunk)

    # Assert - verify hierarchical chunks are in prompt
    assert captured_prompt is not None
    assert "<page name=\"DevOps\">" in captured_prompt
    assert "Container best practices" in captured_prompt
    assert "Scalability patterns" in captured_prompt
    # Verify id:: properties are stripped from context
    assert "id:: parent1" not in captured_prompt
    assert "id:: parent2" not in captured_prompt
    # Verify page properties are included
    assert "title:: DevOps Guide" in captured_prompt


@pytest.mark.asyncio
async def test_plan_integration_for_block_handles_empty_rag_results():
    """Test plan_integration_for_block() handles empty RAG results gracefully."""
    # Arrange
    mock_client = Mock(spec=LLMClient)

    edited_content = EditedContent(
        block_id="xyz789",
        original_content="Novel insight with no existing pages",
        hierarchical_context="- New idea\n  - Novel insight with no existing pages",
        current_content="Novel insight with no related content"
    )

    # Empty candidate chunks
    candidate_chunks = []
    page_contents = {}

    # Mock stream that produces no decisions or add_section decision
    async def mock_stream(*args, **kwargs):
        # LLM might suggest creating a new page or return nothing
        return
        yield  # Make it a generator

    mock_client.stream_ndjson = mock_stream

    # Act
    results = []
    async for chunk in plan_integration_for_block(mock_client, edited_content, candidate_chunks, page_contents):
        results.append(chunk)

    # Assert - should handle gracefully (no crashes)
    assert len(results) == 0


@pytest.mark.asyncio
async def test_plan_integration_for_block_normalizes_page_level_chunks():
    """Test plan_integration_for_block() normalizes __PAGE__ targets to add_section."""
    # Arrange
    mock_client = Mock(spec=LLMClient)

    edited_content = EditedContent(
        block_id="abc123",
        original_content="[[diffused]] supports [[ACS]]",
        hierarchical_context="- [[diffused]] supports [[ACS]]",
        current_content="[[diffused]] can utilize [[ACS]]"
    )

    # Include page-level chunk (ends with ::__PAGE__)
    candidate_chunks = [
        ("diffused", "diffused::__PAGE__", "Page: diffused\nTitle: diffused\ntags:: system, kubernetes"),
        ("diffused", "block1", "- Deployment architecture\n  - Uses Kubernetes")
    ]

    page_contents = {
        "diffused": LogseqOutline.parse(
            "tags:: system, kubernetes\n\n"
            "- Deployment architecture\n"
            "  id:: block1\n"
            "  - Uses Kubernetes"
        )
    }

    # Mock LLM suggesting add_under action for page-level chunk
    # NOTE: "diffused::__PAGE__" will be mapped to short ID (e.g., "1")
    async def mock_stream(*args, **kwargs):
        # LLM suggests adding under page-level chunk
        yield IntegrationDecisionChunk(
            target_page="diffused",
            action="add_under",  # LLM can suggest any action
            target_block_id="1",  # Short ID for "diffused::__PAGE__"
            target_block_title="diffused",
            confidence=0.95,
            reasoning="The knowledge is about diffused system"
        )

    mock_client.stream_ndjson = mock_stream

    # Act
    results = []
    async for chunk in plan_integration_for_block(mock_client, edited_content, candidate_chunks, page_contents):
        results.append(chunk)

    # Assert - __PAGE__ target should be normalized to add_section
    assert len(results) == 1
    assert results[0].knowledge_block_id == "abc123"
    assert results[0].target_page == "diffused"
    assert results[0].action == "add_section"  # Normalized from add_under
    assert results[0].target_block_id is None  # Cleared (add_section has no target)
    assert results[0].confidence == 0.95


@pytest.mark.asyncio
async def test_plan_integration_for_block_preserves_regular_block_targets():
    """Test plan_integration_for_block() preserves regular block targets (not __PAGE__)."""
    # Arrange
    mock_client = Mock(spec=LLMClient)

    edited_content = EditedContent(
        block_id="abc123",
        original_content="Kubernetes scaling info",
        hierarchical_context="- Kubernetes scaling info",
        current_content="Kubernetes enables horizontal scaling"
    )

    # Regular block chunk (NOT a page-level chunk)
    candidate_chunks = [
        ("diffused", "block1", "- Deployment architecture\n  - Uses Kubernetes")
    ]

    page_contents = {
        "diffused": LogseqOutline.parse("- Deployment architecture\n  id:: block1")
    }

    # Mock LLM suggesting add_under action for regular block
    async def mock_stream(*args, **kwargs):
        yield IntegrationDecisionChunk(
            target_page="diffused",
            action="add_under",
            target_block_id="1",  # Short ID for "block1"
            target_block_title="Deployment architecture",
            confidence=0.90,
            reasoning="Related to deployment"
        )

    mock_client.stream_ndjson = mock_stream

    # Act
    results = []
    async for chunk in plan_integration_for_block(mock_client, edited_content, candidate_chunks, page_contents):
        results.append(chunk)

    # Assert - regular block should NOT be normalized
    assert len(results) == 1
    assert results[0].action == "add_under"  # Preserved
    assert results[0].target_block_id == "block1"  # Translated but not cleared
