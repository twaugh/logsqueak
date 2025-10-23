"""Unit tests for KnowledgeBlock model."""

from datetime import date

import pytest

from logsqueak.models.knowledge import ActionType, KnowledgeBlock


class TestKnowledgeBlock:
    """Tests for KnowledgeBlock model."""

    def test_create_knowledge_block(self):
        """Test creating a knowledge block."""
        kb = KnowledgeBlock(
            content="Project deadline moved to May",
            source_date=date(2025, 1, 15),
            confidence=0.92,
            target_page="Project X",
            target_section=["Timeline"],
            suggested_action=ActionType.ADD_CHILD,
        )

        assert kb.content == "Project deadline moved to May"
        assert kb.source_date == date(2025, 1, 15)
        assert kb.confidence == 0.92
        assert kb.target_page == "Project X"
        assert kb.target_section == ["Timeline"]
        assert kb.suggested_action == ActionType.ADD_CHILD

    def test_content_hash(self):
        """Test content hash generation for duplicate detection (FR-017)."""
        kb1 = KnowledgeBlock(
            content="Same content",
            source_date=date(2025, 1, 15),
            confidence=0.9,
            target_page="Page A",
            target_section=None,
            suggested_action=ActionType.ADD_CHILD,
        )

        kb2 = KnowledgeBlock(
            content="Same content",
            source_date=date(2025, 1, 16),  # Different date
            confidence=0.8,  # Different confidence
            target_page="Page B",  # Different page
            target_section=None,
            suggested_action=ActionType.ADD_CHILD,
        )

        # Same content should produce same hash
        assert kb1.content_hash() == kb2.content_hash()

    def test_content_hash_different_content(self):
        """Test that different content produces different hash."""
        kb1 = KnowledgeBlock(
            content="Content A",
            source_date=date(2025, 1, 15),
            confidence=0.9,
            target_page="Page",
            target_section=None,
            suggested_action=ActionType.ADD_CHILD,
        )

        kb2 = KnowledgeBlock(
            content="Content B",
            source_date=date(2025, 1, 15),
            confidence=0.9,
            target_page="Page",
            target_section=None,
            suggested_action=ActionType.ADD_CHILD,
        )

        assert kb1.content_hash() != kb2.content_hash()

    def test_content_hash_length(self):
        """Test that hash is 8 characters."""
        kb = KnowledgeBlock(
            content="Test content",
            source_date=date(2025, 1, 15),
            confidence=0.9,
            target_page="Page",
            target_section=None,
            suggested_action=ActionType.ADD_CHILD,
        )

        assert len(kb.content_hash()) == 8

    def test_provenance_link(self):
        """Test FR-003: Provenance link generation."""
        kb = KnowledgeBlock(
            content="Test content",
            source_date=date(2025, 1, 15),
            confidence=0.9,
            target_page="Page",
            target_section=None,
            suggested_action=ActionType.ADD_CHILD,
        )

        assert kb.provenance_link() == "[[2025-01-15]]"

    def test_provenance_link_different_date(self):
        """Test provenance link with different date."""
        kb = KnowledgeBlock(
            content="Test content",
            source_date=date(2024, 12, 31),
            confidence=0.9,
            target_page="Page",
            target_section=None,
            suggested_action=ActionType.ADD_CHILD,
        )

        assert kb.provenance_link() == "[[2024-12-31]]"

    def test_with_provenance(self):
        """Test content with provenance link appended."""
        kb = KnowledgeBlock(
            content="Deadline moved to May",
            source_date=date(2025, 1, 15),
            confidence=0.9,
            target_page="Page",
            target_section=None,
            suggested_action=ActionType.ADD_CHILD,
        )

        assert kb.with_provenance() == "Deadline moved to May (see [[2025-01-15]] entry)"

    def test_section_path_with_single_section(self):
        """Test section path formatting with single section."""
        kb = KnowledgeBlock(
            content="Test",
            source_date=date(2025, 1, 15),
            confidence=0.9,
            target_page="Page",
            target_section=["Timeline"],
            suggested_action=ActionType.ADD_CHILD,
        )

        assert kb.section_path() == "Timeline"

    def test_section_path_with_nested_sections(self):
        """Test section path formatting with nested sections."""
        kb = KnowledgeBlock(
            content="Test",
            source_date=date(2025, 1, 15),
            confidence=0.9,
            target_page="Page",
            target_section=["Projects", "Active", "Timeline"],
            suggested_action=ActionType.ADD_CHILD,
        )

        assert kb.section_path() == "Projects > Active > Timeline"

    def test_section_path_with_no_section(self):
        """Test section path formatting when no section specified."""
        kb = KnowledgeBlock(
            content="Test",
            source_date=date(2025, 1, 15),
            confidence=0.9,
            target_page="Page",
            target_section=None,
            suggested_action=ActionType.ADD_CHILD,
        )

        assert kb.section_path() == "(page root)"

    def test_section_path_with_empty_list(self):
        """Test section path with empty list."""
        kb = KnowledgeBlock(
            content="Test",
            source_date=date(2025, 1, 15),
            confidence=0.9,
            target_page="Page",
            target_section=[],
            suggested_action=ActionType.ADD_CHILD,
        )

        assert kb.section_path() == "(page root)"


class TestActionType:
    """Tests for ActionType enum."""

    def test_action_types(self):
        """Test that action types have correct values."""
        assert ActionType.ADD_CHILD.value == "add_child"
        assert ActionType.CREATE_SECTION.value == "create_section"
