"""Tests for decision list tooltips showing LLM reasoning."""

import pytest
import pytest_asyncio
from pathlib import Path
from textual.app import App
from logsqueak.tui.widgets.decision_list import DecisionList, DecisionItem
from logsqueak.models.integration_decision import IntegrationDecision
from logsqueak.models.edited_content import EditedContent


@pytest.fixture
def sample_edited_content():
    """Sample EditedContent for testing."""
    return EditedContent(
        block_id="test-block-1",
        original_content="Test knowledge block",
        hierarchical_context="- Test knowledge block",
        reworded_content="Refined test content",
        current_content="Refined test content",
        rewording_complete=True
    )


@pytest.fixture
def sample_decision(sample_edited_content):
    """Sample integration decision with reasoning."""
    return IntegrationDecision(
        knowledge_block_id="test-block-1",
        target_page="Python/Testing",
        action="add_section",
        target_block_id=None,
        target_block_title=None,
        confidence=0.85,
        edited_content=sample_edited_content,
        reasoning="This content is highly relevant to Python testing best practices. "
                  "It provides concrete examples and practical advice that would complement "
                  "the existing Testing section.",
        write_status="pending"
    )


def test_decision_item_has_tooltip_with_reasoning(sample_decision):
    """Test that DecisionItem widgets have tooltips showing LLM reasoning."""
    item = DecisionItem(
        decision=sample_decision,
        is_current=True,
        graph_path=Path("/test/graph")
    )

    # Verify tooltip is set to the decision's reasoning
    assert item.tooltip == sample_decision.reasoning
    assert "highly relevant to Python testing" in item.tooltip
    assert "practical advice" in item.tooltip


@pytest.mark.asyncio
async def test_decision_list_creates_items_with_tooltips(sample_decision, sample_edited_content):
    """Test that DecisionList creates DecisionItem widgets with tooltips."""
    # Create another decision with different reasoning
    decision2 = IntegrationDecision(
        knowledge_block_id="test-block-1",
        target_page="Python/Debugging",
        action="add_under",
        target_block_id="debug-block-1",
        target_block_title="Debugging Tools",
        confidence=0.75,
        edited_content=sample_edited_content,
        reasoning="This fits well under the Debugging Tools section because it discusses "
                  "specific debugging techniques.",
        write_status="pending"
    )

    class TestApp(App):
        """Test app for decision list."""

        def compose(self):
            decision_list = DecisionList(graph_path=Path("/test/graph"))
            yield decision_list

    app = TestApp()
    async with app.run_test() as pilot:
        decision_list = app.query_one(DecisionList)
        decision_list.load_decisions([sample_decision, decision2], current_index=0)
        await pilot.pause()

        # Get the mounted DecisionItem widgets
        children = list(decision_list.children)

        # Should have 2 DecisionItem children
        assert len(children) == 2
        assert all(isinstance(child, DecisionItem) for child in children)

        # Verify tooltips are set correctly
        assert children[0].tooltip == sample_decision.reasoning
        assert children[1].tooltip == decision2.reasoning


@pytest.mark.asyncio
async def test_tooltip_updates_when_decisions_change(sample_decision, sample_edited_content):
    """Test that tooltips update when decision list is reloaded."""
    class TestApp(App):
        """Test app for decision list."""

        def compose(self):
            decision_list = DecisionList(graph_path=Path("/test/graph"))
            yield decision_list

    app = TestApp()
    async with app.run_test() as pilot:
        decision_list = app.query_one(DecisionList)

        # Load initial decision
        decision_list.load_decisions([sample_decision], current_index=0)
        await pilot.pause()
        children = list(decision_list.children)
        assert len(children) == 1
        assert children[0].tooltip == sample_decision.reasoning

        # Create new decision with different reasoning
        new_decision = IntegrationDecision(
            knowledge_block_id="test-block-2",
            target_page="Python/Architecture",
            action="replace",
            target_block_id="arch-block-1",
            target_block_title="Design Patterns",
            confidence=0.90,
            edited_content=sample_edited_content,
            reasoning="This should replace the outdated Design Patterns section with "
                      "modern best practices.",
            write_status="pending"
        )

        # Reload with new decision
        decision_list.load_decisions([new_decision], current_index=0)
        await pilot.pause()
        children = list(decision_list.children)
        assert len(children) == 1
        assert children[0].tooltip == new_decision.reasoning
        assert "modern best practices" in children[0].tooltip
