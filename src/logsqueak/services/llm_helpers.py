"""Helper functions for LLM decision processing and batching."""

from typing import AsyncIterator, Tuple
from logsqueak.models.integration_decision import IntegrationDecision


async def batch_decisions_by_block(
    decision_stream: AsyncIterator[IntegrationDecision]
) -> AsyncIterator[list[IntegrationDecision]]:
    """
    Batch consecutive decisions by knowledge_block_id.

    Takes a stream of IntegrationDecision objects and groups consecutive
    decisions with the same knowledge_block_id into batches.

    Args:
        decision_stream: Async iterator yielding IntegrationDecision objects

    Yields:
        Lists of IntegrationDecision objects grouped by knowledge_block_id

    Example:
        Input stream: [A1, A2, B1, C1, C2, C3]
        Output batches: [[A1, A2], [B1], [C1, C2, C3]]
    """
    current_batch: list[IntegrationDecision] = []
    current_block_id: str | None = None

    async for decision in decision_stream:
        # If this is a new block, yield the current batch and start a new one
        if current_block_id is not None and decision.knowledge_block_id != current_block_id:
            if current_batch:
                yield current_batch
            current_batch = []

        # Add decision to current batch
        current_batch.append(decision)
        current_block_id = decision.knowledge_block_id

    # Yield final batch if not empty
    if current_batch:
        yield current_batch


class FilteredStreamWithCount:
    """
    Wrapper that filters decisions and tracks skipped count.

    This class provides both an async iterator interface and a skipped_count
    property that can be accessed after iteration completes.
    """

    def __init__(self, stream: AsyncIterator[IntegrationDecision]):
        self.stream = stream
        self.skipped_count = 0
        self._skip_block_ids: set[str] = set()
        self._current_block_id: str | None = None
        self._current_batch: list[IntegrationDecision] = []

    def __aiter__(self):
        """Async iterator protocol."""
        return self._iterate()

    async def _iterate(self):
        """Async iterator implementation."""
        async for decision in self.stream:
            # Check if we've moved to a new block
            if (
                self._current_block_id is not None
                and decision.knowledge_block_id != self._current_block_id
            ):
                # Process completed batch - yield decisions immediately
                async for yielded_decision in self._process_batch():
                    yield yielded_decision

            # Add to current batch
            self._current_batch.append(decision)
            self._current_block_id = decision.knowledge_block_id

        # Process final batch
        if self._current_batch:
            async for yielded_decision in self._process_batch():
                yield yielded_decision

    async def _process_batch(self):
        """Process a completed batch of decisions for one block.

        Yields decisions from the batch if the block should not be skipped.
        """
        # Check if any decision in batch has skip_exists
        has_skip = any(d.action == "skip_exists" for d in self._current_batch)

        if has_skip:
            # Skip entire block
            self.skipped_count += 1
            self._skip_block_ids.add(self._current_block_id)
        else:
            # Yield all decisions from this block immediately
            for decision in self._current_batch:
                yield decision

        # Clear batch
        self._current_batch = []


def filter_skip_exists_blocks(
    decision_stream: AsyncIterator[IntegrationDecision]
) -> FilteredStreamWithCount:
    """
    Filter out entire knowledge blocks that have ANY skip_exists decision.

    This function wraps the decision stream and filters out all decisions
    for blocks where at least one decision has action="skip_exists".
    It also tracks the count of skipped blocks.

    Args:
        decision_stream: Async iterator yielding IntegrationDecision objects

    Returns:
        FilteredStreamWithCount object that provides:
        - Async iterator protocol for filtered decisions
        - skipped_count property (accessible after iteration)

    Example:
        Input: [A1(add), A2(skip_exists), B1(add)]
        Output: FilteredStreamWithCount where iteration yields [B1] and .skipped_count == 1

    Usage:
        filtered_stream = filter_skip_exists_blocks(decision_stream())
        filtered = []
        async for decision in filtered_stream:
            filtered.append(decision)
        print(f"Skipped {filtered_stream.skipped_count} blocks")
    """
    return FilteredStreamWithCount(decision_stream)
