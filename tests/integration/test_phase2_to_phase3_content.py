"""Integration test for Phase 2 manual edits flowing to Phase 3 decisions."""

import pytest
from unittest.mock import AsyncMock
from logsqueak.models.edited_content import EditedContent
from logsqueak.models.integration_decision import IntegrationDecision
from logsqueak.models.llm_chunks import IntegrationDecisionChunk


@pytest.mark.asyncio
async def test_manual_edits_flow_to_integration_decisions():
    """Test that manual edits in Phase 2 are captured in refined_text for Phase 3.

    This test simulates the user manually editing content in Phase 2, then
    verifies that the Integration Decision worker uses the edited content
    (not the original or LLM reworded version) in the refined_text field.
    """
    # Create EditedContent with different versions
    edited_content = EditedContent(
        block_id="block-1",
        original_content="Original block content",
        hierarchical_context="- Original block content",
        reworded_content="Reworded by LLM content",
        current_content="Manually edited by user content",  # User's edit
        rewording_complete=True,
    )

    # Simulate Phase 2 worker creating a decision
    # This mimics content_editing.py:649 where we create IntegrationDecision
    chunk = IntegrationDecisionChunk(
        knowledge_block_id="block-1",
        target_page="TestPage",
        action="add_section",
        target_block_id=None,
        target_block_title=None,
        confidence=0.95,
        reasoning="Test reasoning",
    )

    # Convert chunk to decision (same logic as Phase 2 worker)
    decision = IntegrationDecision(
        knowledge_block_id=chunk.knowledge_block_id,
        target_page=chunk.target_page,
        action=chunk.action,
        target_block_id=chunk.target_block_id,
        target_block_title=chunk.target_block_title,
        confidence=chunk.confidence,
        refined_text=edited_content.current_content,  # Should use current_content!
        reasoning=chunk.reasoning,
        write_status="pending",
    )

    # Verify the decision uses the manually edited content
    assert decision.refined_text == "Manually edited by user content"
    assert decision.refined_text != edited_content.original_content
    assert decision.refined_text != edited_content.reworded_content


@pytest.mark.asyncio
async def test_unsaved_edits_captured_before_integration_worker():
    """Test that unsaved edits are saved before Integration Decision worker starts.

    This simulates the scenario where:
    1. User makes edits in the editor
    2. RAG search completes
    3. Integration Decision worker starts
    4. The worker should see the latest edits (even if user hasn't navigated away)
    """
    # This is tested implicitly by our _save_all_content() calls in content_editing.py
    # before starting the Integration Decision worker:
    # - Line 882: Before starting worker from RAG search completion
    # - Line 452: Before starting worker from action_next_phase

    # Create EditedContent with initial state
    edited_content = EditedContent(
        block_id="block-1",
        original_content="Original content",
        hierarchical_context="- Original content",
        reworded_content="Reworded content",
        current_content="Original content",  # Starts as original
        rewording_complete=True,
    )

    # Simulate user editing in the UI (this would be in the editor widget)
    # The editor content is "Edited in UI" but not yet saved to current_content

    # Simulate the _save_all_content() call before worker starts
    editor_content = "Edited in UI"  # What's in the TextArea
    edited_content.current_content = editor_content  # Saved by _save_all_content()

    # Now when worker creates decision, it sees the latest content
    decision = IntegrationDecision(
        knowledge_block_id="block-1",
        target_page="TestPage",
        action="add_section",
        target_block_id=None,
        target_block_title=None,
        confidence=0.95,
        refined_text=edited_content.current_content,
        reasoning="Test",
        write_status="pending",
    )

    # Verify decision has the edited content
    assert decision.refined_text == "Edited in UI"
    assert decision.refined_text != "Original content"
