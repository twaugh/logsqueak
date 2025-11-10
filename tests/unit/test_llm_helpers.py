"""Unit tests for LLM helper functions (decision batching, filtering)."""

import pytest
import pytest_asyncio
from logsqueak.models.integration_decision import IntegrationDecision
from logsqueak.services.llm_helpers import (
    batch_decisions_by_block,
    filter_skip_exists_blocks,
)


@pytest_asyncio.fixture
async def sample_decisions():
    """Sample integration decisions for testing."""
    return [
        IntegrationDecision(
            knowledge_block_id="block-A",
            target_page="Page1",
            action="add_section",
            confidence=0.9,
            refined_text="Content A1",
            reasoning="Reason A1",
        ),
        IntegrationDecision(
            knowledge_block_id="block-A",
            target_page="Page2",
            action="add_under",
            target_block_id="parent-1",
            confidence=0.85,
            refined_text="Content A2",
            reasoning="Reason A2",
        ),
        IntegrationDecision(
            knowledge_block_id="block-B",
            target_page="Page1",
            action="add_section",
            confidence=0.8,
            refined_text="Content B1",
            reasoning="Reason B1",
        ),
        IntegrationDecision(
            knowledge_block_id="block-C",
            target_page="Page3",
            action="skip_exists",
            target_block_id="existing-1",
            confidence=0.95,
            refined_text="Content C1",
            reasoning="Already recorded",
        ),
    ]


# ============================================================================
# T095a: Tests for batch_decisions_by_block()
# ============================================================================


class TestBatchDecisionsByBlock:
    """Test batch_decisions_by_block() helper function."""

    @pytest.mark.asyncio
    async def test_batch_consecutive_blocks(self, sample_decisions):
        """Should batch consecutive decisions by knowledge_block_id."""
        async def decision_stream():
            for decision in sample_decisions:
                yield decision

        batches = []
        async for batch in batch_decisions_by_block(decision_stream()):
            batches.append(batch)

        # Should produce 3 batches: block-A (2 decisions), block-B (1), block-C (1)
        assert len(batches) == 3
        assert len(batches[0]) == 2  # block-A
        assert len(batches[1]) == 1  # block-B
        assert len(batches[2]) == 1  # block-C

    @pytest.mark.asyncio
    async def test_batch_preserves_decision_data(self, sample_decisions):
        """Should preserve all decision data in batches."""
        async def decision_stream():
            for decision in sample_decisions:
                yield decision

        batches = []
        async for batch in batch_decisions_by_block(decision_stream()):
            batches.append(batch)

        # Verify first batch contains block-A decisions
        assert batches[0][0].knowledge_block_id == "block-A"
        assert batches[0][0].target_page == "Page1"
        assert batches[0][1].knowledge_block_id == "block-A"
        assert batches[0][1].target_page == "Page2"

    @pytest.mark.asyncio
    async def test_batch_single_decision_per_block(self):
        """Should handle single decision per block correctly."""
        decisions = [
            IntegrationDecision(
                knowledge_block_id="block-1",
                target_page="Page1",
                action="add_section",
                confidence=0.9,
                refined_text="Content 1",
                reasoning="Reason 1",
            ),
            IntegrationDecision(
                knowledge_block_id="block-2",
                target_page="Page2",
                action="add_section",
                confidence=0.8,
                refined_text="Content 2",
                reasoning="Reason 2",
            ),
        ]

        async def decision_stream():
            for decision in decisions:
                yield decision

        batches = []
        async for batch in batch_decisions_by_block(decision_stream()):
            batches.append(batch)

        # Should produce 2 batches, each with 1 decision
        assert len(batches) == 2
        assert len(batches[0]) == 1
        assert len(batches[1]) == 1

    @pytest.mark.asyncio
    async def test_batch_empty_stream(self):
        """Should handle empty decision stream without errors."""
        async def decision_stream():
            return
            yield  # Make this a generator

        batches = []
        async for batch in batch_decisions_by_block(decision_stream()):
            batches.append(batch)

        assert len(batches) == 0

    @pytest.mark.asyncio
    async def test_batch_maintains_order(self):
        """Should maintain order of decisions within batches."""
        decisions = [
            IntegrationDecision(
                knowledge_block_id="block-A",
                target_page="Page1",
                action="add_section",
                confidence=0.9,
                refined_text="First",
                reasoning="Reason 1",
            ),
            IntegrationDecision(
                knowledge_block_id="block-A",
                target_page="Page2",
                action="add_section",
                confidence=0.8,
                refined_text="Second",
                reasoning="Reason 2",
            ),
            IntegrationDecision(
                knowledge_block_id="block-A",
                target_page="Page3",
                action="add_section",
                confidence=0.7,
                refined_text="Third",
                reasoning="Reason 3",
            ),
        ]

        async def decision_stream():
            for decision in decisions:
                yield decision

        batches = []
        async for batch in batch_decisions_by_block(decision_stream()):
            batches.append(batch)

        assert len(batches) == 1
        assert batches[0][0].refined_text == "First"
        assert batches[0][1].refined_text == "Second"
        assert batches[0][2].refined_text == "Third"


# ============================================================================
# T095b: Tests for filter_skip_exists_blocks()
# ============================================================================


class TestFilterSkipExistsBlocks:
    """Test filter_skip_exists_blocks() helper function."""

    @pytest.mark.asyncio
    async def test_filter_blocks_with_skip_exists(self, sample_decisions):
        """Should filter out entire blocks that have ANY skip_exists decision."""
        async def decision_stream():
            for decision in sample_decisions:
                yield decision

        filtered_stream = filter_skip_exists_blocks(decision_stream())

        filtered = []
        async for decision in filtered_stream:
            filtered.append(decision)

        # block-C has skip_exists, should be filtered out
        # block-A and block-B should remain
        assert len(filtered) == 3  # 2 from block-A, 1 from block-B
        assert all(d.knowledge_block_id != "block-C" for d in filtered)
        assert filtered_stream.skipped_count == 1  # Only block-C was skipped

    @pytest.mark.asyncio
    async def test_filter_preserves_non_skip_blocks(self):
        """Should preserve blocks without skip_exists decisions."""
        decisions = [
            IntegrationDecision(
                knowledge_block_id="block-A",
                target_page="Page1",
                action="add_section",
                confidence=0.9,
                refined_text="Content A1",
                reasoning="Reason A1",
            ),
            IntegrationDecision(
                knowledge_block_id="block-A",
                target_page="Page2",
                action="add_under",
                target_block_id="parent-1",
                confidence=0.85,
                refined_text="Content A2",
                reasoning="Reason A2",
            ),
        ]

        async def decision_stream():
            for decision in decisions:
                yield decision

        filtered_stream = filter_skip_exists_blocks(decision_stream())

        filtered = []
        async for decision in filtered_stream:
            filtered.append(decision)

        # All decisions should be preserved
        assert len(filtered) == 2
        assert filtered_stream.skipped_count == 0

    @pytest.mark.asyncio
    async def test_filter_counts_skipped_blocks(self):
        """Should correctly count number of skipped blocks."""
        decisions = [
            IntegrationDecision(
                knowledge_block_id="block-A",
                target_page="Page1",
                action="skip_exists",
                confidence=0.9,
                refined_text="Content A",
                reasoning="Already exists",
            ),
            IntegrationDecision(
                knowledge_block_id="block-B",
                target_page="Page2",
                action="add_section",
                confidence=0.85,
                refined_text="Content B",
                reasoning="New content",
            ),
            IntegrationDecision(
                knowledge_block_id="block-C",
                target_page="Page3",
                action="skip_exists",
                target_block_id="existing-1",
                confidence=0.95,
                refined_text="Content C",
                reasoning="Duplicate",
            ),
        ]

        async def decision_stream():
            for decision in decisions:
                yield decision

        filtered_stream = filter_skip_exists_blocks(decision_stream())

        filtered = []
        async for decision in filtered_stream:
            filtered.append(decision)

        # block-A and block-C should be filtered out
        assert len(filtered) == 1
        assert filtered[0].knowledge_block_id == "block-B"
        assert filtered_stream.skipped_count == 2  # block-A and block-C

    @pytest.mark.asyncio
    async def test_filter_entire_block_when_any_decision_is_skip(self):
        """Should filter entire block if ANY decision has skip_exists action."""
        decisions = [
            IntegrationDecision(
                knowledge_block_id="block-A",
                target_page="Page1",
                action="add_section",
                confidence=0.9,
                refined_text="Content A1",
                reasoning="New section",
            ),
            IntegrationDecision(
                knowledge_block_id="block-A",
                target_page="Page2",
                action="skip_exists",  # This should filter entire block-A
                target_block_id="existing-1",
                confidence=0.95,
                refined_text="Content A2",
                reasoning="Already exists",
            ),
            IntegrationDecision(
                knowledge_block_id="block-B",
                target_page="Page3",
                action="add_section",
                confidence=0.85,
                refined_text="Content B",
                reasoning="New content",
            ),
        ]

        async def decision_stream():
            for decision in decisions:
                yield decision

        filtered_stream = filter_skip_exists_blocks(decision_stream())

        filtered = []
        async for decision in filtered_stream:
            filtered.append(decision)

        # Entire block-A should be filtered (both decisions)
        assert len(filtered) == 1
        assert filtered[0].knowledge_block_id == "block-B"
        assert filtered_stream.skipped_count == 1  # Only block-A (as a whole) was skipped

    @pytest.mark.asyncio
    async def test_filter_empty_stream(self):
        """Should handle empty stream without errors."""
        async def decision_stream():
            return
            yield  # Make this a generator

        filtered_stream = filter_skip_exists_blocks(decision_stream())

        filtered = []
        async for decision in filtered_stream:
            filtered.append(decision)

        assert len(filtered) == 0
        assert filtered_stream.skipped_count == 0
