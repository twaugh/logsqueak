"""Unit tests for data models."""

import pytest
from logsqueak.models.block_state import BlockState
from logsqueak.models.edited_content import EditedContent
from logsqueak.models.integration_decision import IntegrationDecision
from logsqueak.models.background_task import BackgroundTask
from logsqueak.models.llm_chunks import (
    KnowledgeClassificationChunk,
    ContentRewordingChunk,
    IntegrationDecisionChunk
)


class TestBlockState:
    """Test BlockState model."""

    def test_create_pending_block_state(self):
        """Test creating block state with pending classification."""
        state = BlockState(
            block_id="abc123",
            source="user"
        )

        assert state.block_id == "abc123"
        assert state.classification == "pending"
        assert state.source == "user"
        assert state.confidence is None
        assert state.llm_classification is None

    def test_create_llm_knowledge_block_state(self):
        """Test creating block state from LLM classification."""
        state = BlockState(
            block_id="def456",
            classification="knowledge",
            confidence=0.92,
            source="llm",
            llm_classification="knowledge",
            llm_confidence=0.92,
            reason="Contains reusable insight about Python async patterns."
        )

        assert state.classification == "knowledge"
        assert state.confidence == 0.92
        assert state.source == "llm"
        assert state.llm_classification == "knowledge"
        assert state.reason is not None

    def test_block_state_is_mutable(self):
        """Test that BlockState can be mutated (user can override)."""
        state = BlockState(
            block_id="abc123",
            source="llm",
            llm_classification="knowledge",
            llm_confidence=0.85
        )

        # User overrides LLM suggestion
        state.classification = "knowledge"
        state.confidence = 1.0
        state.source = "user"

        assert state.source == "user"
        assert state.confidence == 1.0
        # LLM suggestion preserved
        assert state.llm_classification == "knowledge"
        assert state.llm_confidence == 0.85

    def test_confidence_validates_range(self):
        """Test confidence score validates 0.0-1.0 range."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            BlockState(block_id="test", source="llm", confidence=1.5)


class TestEditedContent:
    """Test EditedContent model."""

    def test_create_edited_content(self):
        """Test creating edited content model."""
        content = EditedContent(
            block_id="abc123",
            original_content="Today I learned about asyncio.",
            hierarchical_context="- Python learning notes\n  - Concurrency patterns\n    - Today I learned about asyncio.",
            current_content="Today I learned about asyncio."
        )

        assert content.block_id == "abc123"
        assert content.original_content == content.current_content
        assert content.reworded_content is None
        assert content.rewording_complete is False

    def test_edited_content_with_llm_rewording(self):
        """Test edited content with LLM reworded version."""
        content = EditedContent(
            block_id="abc123",
            original_content="Today I learned about asyncio.",
            hierarchical_context="- Python learning notes\n  - Concurrency patterns\n    - Today I learned about asyncio.",
            reworded_content="asyncio enables concurrent programming in Python.",
            current_content="asyncio enables concurrent programming in Python.",
            rewording_complete=True
        )

        assert content.reworded_content != content.original_content
        assert content.current_content == content.reworded_content
        assert content.rewording_complete is True

    def test_edited_content_is_mutable(self):
        """Test that EditedContent can be mutated during editing."""
        content = EditedContent(
            block_id="abc123",
            original_content="Original text",
            hierarchical_context="- Journal entry\n  - Notes section\n    - Original text",
            current_content="Original text"
        )

        # User edits content
        content.current_content = "Manually edited text"

        assert content.current_content == "Manually edited text"
        assert content.original_content == "Original text"


class TestIntegrationDecision:
    """Test IntegrationDecision model."""

    def test_create_integration_decision(self):
        """Test creating integration decision."""
        edited_content = EditedContent(
            block_id="abc123",
            original_content="asyncio enables concurrent programming.",
            hierarchical_context="- asyncio enables concurrent programming.",
            current_content="asyncio enables concurrent programming."
        )

        decision = IntegrationDecision(
            knowledge_block_id="abc123",
            target_page="Programming Notes/Python",
            action="add_under",
            target_block_id="section-async",
            target_block_title="Async Patterns",
            confidence=0.87,
            edited_content=edited_content,
            reasoning="Fits well under Async Patterns section."
        )

        assert decision.knowledge_block_id == "abc123"
        assert decision.target_page == "Programming Notes/Python"
        assert decision.action == "add_under"
        assert decision.write_status == "pending"
        assert decision.error_message is None
        assert decision.refined_text == "asyncio enables concurrent programming."

    def test_decision_write_status_transitions(self):
        """Test decision write status can transition."""
        edited_content = EditedContent(
            block_id="abc123",
            original_content="Test content",
            hierarchical_context="- Test content",
            current_content="Test content"
        )

        decision = IntegrationDecision(
            knowledge_block_id="abc123",
            target_page="Test Page",
            action="add_section",
            confidence=0.9,
            edited_content=edited_content,
            reasoning="Test reasoning"
        )

        assert decision.write_status == "pending"

        # Accept and write successfully
        decision.write_status = "completed"
        assert decision.write_status == "completed"

    def test_decision_write_failure(self):
        """Test decision write failure tracking."""
        edited_content = EditedContent(
            block_id="abc123",
            original_content="Test content",
            hierarchical_context="- Test content",
            current_content="Test content"
        )

        decision = IntegrationDecision(
            knowledge_block_id="abc123",
            target_page="Test Page",
            action="replace",
            target_block_id="missing-block",
            confidence=0.9,
            edited_content=edited_content,
            reasoning="Test reasoning"
        )

        # Write fails
        decision.write_status = "failed"
        decision.error_message = "Target block not found"

        assert decision.write_status == "failed"
        assert decision.error_message == "Target block not found"


class TestBackgroundTask:
    """Test BackgroundTask model."""

    def test_create_running_task(self):
        """Test creating running background task."""
        task = BackgroundTask(
            task_type="llm_classification"
        )

        assert task.task_type == "llm_classification"
        assert task.status == "running"
        assert task.progress_percentage is None
        assert task.error_message is None

    def test_task_with_percentage_progress(self):
        """Test task with percentage-based progress."""
        task = BackgroundTask(
            task_type="page_indexing",
            status="running",
            progress_percentage=45.5
        )

        assert task.progress_percentage == 45.5
        assert task.progress_current is None
        assert task.progress_total is None

    def test_task_with_count_progress(self):
        """Test task with count-based progress."""
        task = BackgroundTask(
            task_type="llm_rewording",
            status="running",
            progress_current=3,
            progress_total=5
        )

        assert task.progress_current == 3
        assert task.progress_total == 5
        assert task.progress_percentage is None

    def test_task_completion(self):
        """Test task completion."""
        task = BackgroundTask(task_type="rag_search")

        task.status = "completed"
        task.progress_percentage = 100.0

        assert task.status == "completed"

    def test_task_failure(self):
        """Test task failure."""
        task = BackgroundTask(task_type="llm_decisions")

        task.status = "failed"
        task.error_message = "Connection timeout"

        assert task.status == "failed"
        assert task.error_message == "Connection timeout"


class TestLLMChunks:
    """Test LLM NDJSON chunk models."""

    def test_knowledge_classification_chunk(self):
        """Test parsing knowledge classification chunk."""
        chunk = KnowledgeClassificationChunk(
            block_id="abc123",
            confidence=0.92,
            reason="Contains reusable insight."
        )

        assert chunk.type == "classification"
        assert chunk.block_id == "abc123"
        assert chunk.confidence == 0.92

    def test_content_rewording_chunk(self):
        """Test parsing content rewording chunk."""
        chunk = ContentRewordingChunk(
            block_id="abc123",
            reworded_content="asyncio enables concurrent programming."
        )

        assert chunk.type == "rewording"
        assert chunk.block_id == "abc123"
        assert chunk.reworded_content is not None

    def test_integration_decision_chunk(self):
        """Test parsing integration decision chunk."""
        chunk = IntegrationDecisionChunk(
            knowledge_block_id="abc123",
            target_page="Programming/Python",
            action="add_under",
            target_block_id="async-section",
            target_block_title="Async Patterns",
            confidence=0.85,
            reasoning="Fits under async patterns section."
        )

        assert chunk.type == "decision"
        assert chunk.knowledge_block_id == "abc123"
        assert chunk.target_page == "Programming/Python"
        assert chunk.action == "add_under"
