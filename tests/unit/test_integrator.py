"""Unit tests for integration orchestrator.

Tests coordination of knowledge integration into pages,
including grouping, loading, and result reporting.
"""

from datetime import date
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from logsqueak.integration.integrator import Integrator, IntegrationResult
from logsqueak.models.knowledge import ActionType, KnowledgeBlock
from logsqueak.models.page import PageIndex, TargetPage
from logsqueak.models.preview import ActionStatus, ProposedAction


@pytest.fixture
def sample_knowledge_block():
    """Create sample knowledge block for testing."""
    return KnowledgeBlock(
        content="Use PostgreSQL for database",
        source_date=date(2025, 1, 15),
        confidence=0.9,
        target_page="Project Architecture",
        target_section=["Tech Stack"],
        suggested_action=ActionType.ADD_CHILD,
    )


@pytest.fixture
def sample_proposed_action(sample_knowledge_block):
    """Create sample proposed action."""
    return ProposedAction(
        knowledge=sample_knowledge_block,
        status=ActionStatus.READY,
        similarity_score=0.85,
    )


class TestIntegrationResult:
    """Test IntegrationResult data structure."""

    def test_create_success_result(self):
        """Test creating successful integration result."""
        result = IntegrationResult(
            success=True,
            actions_applied=5,
            actions_skipped=0,
            modified_pages=["Page 1", "Page 2"],
        )

        assert result.success is True
        assert result.actions_applied == 5
        assert result.actions_skipped == 0
        assert result.errors == []
        assert result.modified_pages == ["Page 1", "Page 2"]

    def test_create_failure_result(self):
        """Test creating failed integration result."""
        result = IntegrationResult(
            success=False,
            actions_applied=2,
            actions_skipped=3,
            errors=["Page not found", "Write failed"],
        )

        assert result.success is False
        assert len(result.errors) == 2
        assert "Page not found" in result.errors

    def test_default_values(self):
        """Test default values for optional fields."""
        result = IntegrationResult(success=True)

        assert result.actions_applied == 0
        assert result.actions_skipped == 0
        assert result.errors == []
        assert result.modified_pages == []


class TestIntegrator:
    """Test Integrator orchestration logic."""

    def test_initialization(self, tmp_path):
        """Test integrator initialization."""
        graph_path = tmp_path
        integrator = Integrator(graph_path)

        assert integrator.graph_path == graph_path
        assert integrator.page_index is None

    def test_initialization_with_page_index(self, tmp_path):
        """Test integrator with PageIndex."""
        graph_path = tmp_path
        mock_index = Mock(spec=PageIndex)

        integrator = Integrator(graph_path, page_index=mock_index)

        assert integrator.page_index is mock_index

    def test_group_by_page_single_page(self, sample_proposed_action):
        """Test grouping actions by page with single page."""
        integrator = Integrator(Path("/test"))

        grouped = integrator._group_by_page([sample_proposed_action])

        assert len(grouped) == 1
        assert "Project Architecture" in grouped
        assert len(grouped["Project Architecture"]) == 1

    def test_group_by_page_multiple_pages(self):
        """Test grouping actions by page with multiple pages."""
        integrator = Integrator(Path("/test"))

        actions = [
            ProposedAction(
                knowledge=KnowledgeBlock(
                    content="Content 1",
                    source_date=date(2025, 1, 15),
                    confidence=0.9,
                    target_page="Page A",
                    target_section=None,
                    suggested_action=ActionType.ADD_CHILD,
                ),
                status=ActionStatus.READY,
            ),
            ProposedAction(
                knowledge=KnowledgeBlock(
                    content="Content 2",
                    source_date=date(2025, 1, 15),
                    confidence=0.8,
                    target_page="Page B",
                    target_section=None,
                    suggested_action=ActionType.ADD_CHILD,
                ),
                status=ActionStatus.READY,
            ),
            ProposedAction(
                knowledge=KnowledgeBlock(
                    content="Content 3",
                    source_date=date(2025, 1, 15),
                    confidence=0.85,
                    target_page="Page A",
                    target_section=None,
                    suggested_action=ActionType.ADD_CHILD,
                ),
                status=ActionStatus.READY,
            ),
        ]

        grouped = integrator._group_by_page(actions)

        assert len(grouped) == 2
        assert len(grouped["Page A"]) == 2
        assert len(grouped["Page B"]) == 1

    def test_load_page_not_found(self, tmp_path):
        """Test loading non-existent page."""
        graph_path = tmp_path
        integrator = Integrator(graph_path)

        result = integrator._load_page("Nonexistent Page")

        assert result is None

    def test_load_page_exists(self, tmp_path):
        """Test loading existing page."""
        graph_path = tmp_path
        pages_dir = graph_path / "pages"
        pages_dir.mkdir()

        page_file = pages_dir / "Test Page.md"
        page_file.write_text("- Test content")

        integrator = Integrator(graph_path)

        result = integrator._load_page("Test Page")

        assert result is not None
        assert result.name == "Test Page"

    def test_integrate_dry_run(self, tmp_path, sample_proposed_action):
        """Test integration in dry-run mode."""
        graph_path = tmp_path
        pages_dir = graph_path / "pages"
        pages_dir.mkdir()

        # Create target page
        page_file = pages_dir / "Project Architecture.md"
        page_file.write_text("- ## Tech Stack\n  - Placeholder")

        integrator = Integrator(graph_path)

        # Dry run modifies pages in memory but doesn't write to disk
        result = integrator.integrate([sample_proposed_action], dry_run=True)

        # Should successfully apply action in memory
        assert result.actions_applied == 1
        assert result.actions_skipped == 0
        # But file shouldn't be written in dry-run mode
        assert not result.modified_pages

    def test_integrate_skips_non_ready_actions(self, tmp_path):
        """Test that non-READY actions are skipped."""
        graph_path = tmp_path
        pages_dir = graph_path / "pages"
        pages_dir.mkdir()

        skipped_action = ProposedAction(
            knowledge=KnowledgeBlock(
                content="Skipped content",
                source_date=date(2025, 1, 15),
                confidence=0.9,
                target_page="Test Page",
                target_section=None,
                suggested_action=ActionType.ADD_CHILD,
            ),
            status=ActionStatus.SKIPPED,
            reason="Duplicate",
        )

        page_file = pages_dir / "Test Page.md"
        page_file.write_text("- Content")

        integrator = Integrator(graph_path)

        # Should handle skipped actions without calling unimplemented methods
        result = integrator.integrate([skipped_action], dry_run=True)

        assert result.actions_skipped == 1
        assert result.actions_applied == 0

    def test_integrate_handles_missing_page(self, tmp_path, sample_proposed_action):
        """Test integration when target page doesn't exist."""
        graph_path = tmp_path
        integrator = Integrator(graph_path)

        result = integrator.integrate([sample_proposed_action], dry_run=True)

        assert result.success is False
        assert result.actions_skipped == 1
        assert len(result.errors) > 0
        assert "not found" in result.errors[0].lower()
