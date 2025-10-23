"""Unit tests for ExtractionPreview model."""

from datetime import date

import pytest

from logsqueak.models.knowledge import ActionType, KnowledgeBlock
from logsqueak.models.preview import ActionStatus, ExtractionPreview, ProposedAction


class TestProposedAction:
    """Tests for ProposedAction model."""

    def test_create_ready_action(self):
        """Test creating a READY action."""
        kb = KnowledgeBlock(
            content="Test content",
            source_date=date(2025, 1, 15),
            confidence=0.9,
            target_page="Test Page",
            target_section=["Timeline"],
            suggested_action=ActionType.ADD_CHILD,
        )

        action = ProposedAction(knowledge=kb, status=ActionStatus.READY)

        assert action.knowledge == kb
        assert action.status == ActionStatus.READY
        assert action.reason is None

    def test_create_skipped_action(self):
        """Test creating a SKIPPED action with reason."""
        kb = KnowledgeBlock(
            content="Test content",
            source_date=date(2025, 1, 15),
            confidence=0.9,
            target_page="Test Page",
            target_section=None,
            suggested_action=ActionType.ADD_CHILD,
        )

        action = ProposedAction(
            knowledge=kb, status=ActionStatus.SKIPPED, reason="Duplicate content"
        )

        assert action.status == ActionStatus.SKIPPED
        assert action.reason == "Duplicate content"

    def test_describe_ready_action(self):
        """Test describing a READY action."""
        kb = KnowledgeBlock(
            content="Deadline moved to May",
            source_date=date(2025, 1, 15),
            confidence=0.9,
            target_page="Project X",
            target_section=["Timeline"],
            suggested_action=ActionType.ADD_CHILD,
        )

        action = ProposedAction(knowledge=kb, status=ActionStatus.READY)
        description = action.describe()

        assert "Deadline moved to May" in description
        assert "Project X" in description
        assert "Timeline" in description
        assert "add_child" in description
        assert "SKIPPED" not in description

    def test_describe_skipped_action(self):
        """Test describing a SKIPPED action."""
        kb = KnowledgeBlock(
            content="Test content",
            source_date=date(2025, 1, 15),
            confidence=0.9,
            target_page="Page",
            target_section=None,
            suggested_action=ActionType.ADD_CHILD,
        )

        action = ProposedAction(
            knowledge=kb, status=ActionStatus.SKIPPED, reason="Duplicate"
        )
        description = action.describe()

        assert "SKIPPED" in description
        assert "Duplicate" in description

    def test_describe_warning_action(self):
        """Test describing a WARNING action."""
        kb = KnowledgeBlock(
            content="Test content",
            source_date=date(2025, 1, 15),
            confidence=0.9,
            target_page="Page",
            target_section=None,
            suggested_action=ActionType.ADD_CHILD,
        )

        action = ProposedAction(
            knowledge=kb, status=ActionStatus.WARNING, reason="Low confidence"
        )
        description = action.describe()

        assert "WARNING" in description
        assert "Low confidence" in description

    def test_describe_truncates_long_content(self):
        """Test that long content is truncated in description."""
        long_content = "A" * 100  # 100 characters
        kb = KnowledgeBlock(
            content=long_content,
            source_date=date(2025, 1, 15),
            confidence=0.9,
            target_page="Page",
            target_section=None,
            suggested_action=ActionType.ADD_CHILD,
        )

        action = ProposedAction(knowledge=kb, status=ActionStatus.READY)
        description = action.describe()

        # Should truncate to 60 chars and add "..."
        assert '..."' in description
        assert len(long_content) > 60  # Original is longer


class TestExtractionPreview:
    """Tests for ExtractionPreview model."""

    def test_create_preview(self):
        """Test creating an extraction preview."""
        kb = KnowledgeBlock(
            content="Test",
            source_date=date(2025, 1, 15),
            confidence=0.9,
            target_page="Page",
            target_section=None,
            suggested_action=ActionType.ADD_CHILD,
        )

        action = ProposedAction(knowledge=kb, status=ActionStatus.READY)

        preview = ExtractionPreview(
            journal_date=date(2025, 1, 15),
            knowledge_blocks=[kb],
            proposed_actions=[action],
            warnings=[],
        )

        assert preview.journal_date == date(2025, 1, 15)
        assert len(preview.knowledge_blocks) == 1
        assert len(preview.proposed_actions) == 1
        assert preview.warnings == []

    def test_display_single_block(self):
        """Test displaying preview with single knowledge block."""
        kb = KnowledgeBlock(
            content="Deadline moved to May",
            source_date=date(2025, 1, 15),
            confidence=0.9,
            target_page="Project X",
            target_section=["Timeline"],
            suggested_action=ActionType.ADD_CHILD,
        )

        action = ProposedAction(knowledge=kb, status=ActionStatus.READY)

        preview = ExtractionPreview(
            journal_date=date(2025, 1, 15),
            knowledge_blocks=[kb],
            proposed_actions=[action],
            warnings=[],
        )

        output = preview.display()

        assert "Found 1 knowledge blocks" in output
        assert "journals/2025_01_15.md" in output
        assert "Deadline moved to May" in output
        assert "Project X" in output
        assert "Timeline" in output
        assert "Will integrate: 1" in output
        assert "Skipped: 0" in output

    def test_display_multiple_blocks(self):
        """Test displaying preview with multiple blocks."""
        kb1 = KnowledgeBlock(
            content="Block 1",
            source_date=date(2025, 1, 15),
            confidence=0.9,
            target_page="Page A",
            target_section=None,
            suggested_action=ActionType.ADD_CHILD,
        )

        kb2 = KnowledgeBlock(
            content="Block 2",
            source_date=date(2025, 1, 15),
            confidence=0.8,
            target_page="Page B",
            target_section=None,
            suggested_action=ActionType.ADD_CHILD,
        )

        action1 = ProposedAction(knowledge=kb1, status=ActionStatus.READY)
        action2 = ProposedAction(knowledge=kb2, status=ActionStatus.READY)

        preview = ExtractionPreview(
            journal_date=date(2025, 1, 15),
            knowledge_blocks=[kb1, kb2],
            proposed_actions=[action1, action2],
            warnings=[],
        )

        output = preview.display()

        assert "Found 2 knowledge blocks" in output
        assert "1." in output
        assert "2." in output
        assert "Block 1" in output
        assert "Block 2" in output

    def test_display_with_warnings(self):
        """Test displaying preview with warnings."""
        kb = KnowledgeBlock(
            content="Test",
            source_date=date(2025, 1, 15),
            confidence=0.9,
            target_page="Page",
            target_section=None,
            suggested_action=ActionType.ADD_CHILD,
        )

        action = ProposedAction(knowledge=kb, status=ActionStatus.READY)

        preview = ExtractionPreview(
            journal_date=date(2025, 1, 15),
            knowledge_blocks=[kb],
            proposed_actions=[action],
            warnings=["Warning 1", "Warning 2"],
        )

        output = preview.display()

        assert "Warnings:" in output
        assert "Warning 1" in output
        assert "Warning 2" in output

    def test_display_with_skipped_actions(self):
        """Test displaying preview with skipped actions."""
        kb1 = KnowledgeBlock(
            content="Block 1",
            source_date=date(2025, 1, 15),
            confidence=0.9,
            target_page="Page",
            target_section=None,
            suggested_action=ActionType.ADD_CHILD,
        )

        kb2 = KnowledgeBlock(
            content="Block 2",
            source_date=date(2025, 1, 15),
            confidence=0.8,
            target_page="Page",
            target_section=None,
            suggested_action=ActionType.ADD_CHILD,
        )

        action1 = ProposedAction(knowledge=kb1, status=ActionStatus.READY)
        action2 = ProposedAction(
            knowledge=kb2, status=ActionStatus.SKIPPED, reason="Duplicate"
        )

        preview = ExtractionPreview(
            journal_date=date(2025, 1, 15),
            knowledge_blocks=[kb1, kb2],
            proposed_actions=[action1, action2],
            warnings=[],
        )

        output = preview.display()

        assert "Will integrate: 1" in output
        assert "Skipped: 1" in output
        assert "SKIPPED" in output
        assert "Duplicate" in output

    def test_summary_counts(self):
        """Test that summary counts are correct."""
        kb1 = KnowledgeBlock(
            content="Block 1",
            source_date=date(2025, 1, 15),
            confidence=0.9,
            target_page="Page",
            target_section=None,
            suggested_action=ActionType.ADD_CHILD,
        )

        kb2 = KnowledgeBlock(
            content="Block 2",
            source_date=date(2025, 1, 15),
            confidence=0.8,
            target_page="Page",
            target_section=None,
            suggested_action=ActionType.ADD_CHILD,
        )

        kb3 = KnowledgeBlock(
            content="Block 3",
            source_date=date(2025, 1, 15),
            confidence=0.7,
            target_page="Page",
            target_section=None,
            suggested_action=ActionType.ADD_CHILD,
        )

        actions = [
            ProposedAction(knowledge=kb1, status=ActionStatus.READY),
            ProposedAction(knowledge=kb2, status=ActionStatus.SKIPPED, reason="Dup"),
            ProposedAction(knowledge=kb3, status=ActionStatus.WARNING, reason="Low"),
        ]

        preview = ExtractionPreview(
            journal_date=date(2025, 1, 15),
            knowledge_blocks=[kb1, kb2, kb3],
            proposed_actions=actions,
            warnings=[],
        )

        output = preview.display()

        assert "Knowledge blocks found: 3" in output
        assert "Will integrate: 1" in output  # Only READY
        assert "Skipped: 1" in output  # Only SKIPPED
