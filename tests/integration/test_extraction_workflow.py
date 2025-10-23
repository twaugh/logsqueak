"""Integration tests for end-to-end extraction workflow.

Tests the complete pipeline:
1. RAG index building
2. Stage 1: Extract knowledge from journal
3. Classification (knowledge vs activity)
4. Stage 2: Match knowledge to pages using RAG
5. Duplicate detection
6. Preview generation

Uses mocked LLM but real models and data structures.
"""

from datetime import date
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from logsqueak.extraction.classifier import classify_extractions
from logsqueak.extraction.extractor import Extractor, create_knowledge_block
from logsqueak.llm.client import ExtractionResult, PageCandidate, PageSelectionResult
from logsqueak.logseq.parser import LogseqBlock, LogseqOutline
from logsqueak.models.journal import JournalEntry
from logsqueak.models.knowledge import ActionType
from logsqueak.models.page import PageIndex, TargetPage
from logsqueak.models.preview import ActionStatus, ExtractionPreview, ProposedAction


@pytest.fixture
def sample_journal_entry():
    """Create a sample journal entry for testing."""
    source_text = """- Had standup meeting with team
- Decided to use PostgreSQL for persistence layer
- Reviewed competitor features
- Key insight: Users prefer simple UI over feature-rich"""

    blocks = [
        LogseqBlock(content="- Had standup meeting with team", indent_level=0, properties={}),
        LogseqBlock(
            content="- Decided to use PostgreSQL for persistence layer",
            indent_level=0,
            properties={},
        ),
        LogseqBlock(content="- Reviewed competitor features", indent_level=0, properties={}),
        LogseqBlock(
            content="- Key insight: Users prefer simple UI over feature-rich",
            indent_level=0,
            properties={},
        ),
    ]

    return JournalEntry(
        date=date(2025, 1, 15),
        file_path=Path("/test/journals/2025_01_15.md"),
        raw_content=source_text,
        outline=LogseqOutline(blocks=blocks, source_text=source_text),
        line_count=4,
    )


@pytest.fixture
def mock_llm_client():
    """Mock LLM client with realistic responses."""
    client = Mock()

    # Stage 1: Extract knowledge (returns 3 knowledge + 1 activity)
    client.extract_knowledge.return_value = [
        ExtractionResult(content="Decided to use PostgreSQL for persistence layer", confidence=0.9),
        ExtractionResult(content="Had standup meeting with team", confidence=0.2),  # Activity
        ExtractionResult(
            content="Key insight: Users prefer simple UI over feature-rich", confidence=0.85
        ),
        ExtractionResult(content="Reviewed competitor features", confidence=0.3),  # Activity
    ]

    # Stage 2: Page selection (called per knowledge block)
    def mock_select_target_page(knowledge_content, candidates, indent_str="  "):
        if "PostgreSQL" in knowledge_content:
            return PageSelectionResult(
                target_page="Project Architecture",
                target_section=["Database"],
                suggested_action=ActionType.ADD_CHILD,
                reasoning="Database technology decision",
            )
        elif "simple UI" in knowledge_content:
            return PageSelectionResult(
                target_page="Product Strategy",
                target_section=["User Feedback"],
                suggested_action=ActionType.ADD_CHILD,
                reasoning="User preference insight",
            )
        else:
            return PageSelectionResult(
                target_page="General Notes",
                target_section=None,
                suggested_action=ActionType.ADD_CHILD,
                reasoning="General knowledge",
            )

    client.select_target_page.side_effect = mock_select_target_page

    return client


@pytest.fixture
def mock_page_index(tmp_path):
    """Mock PageIndex with sample pages."""
    # Create mock pages
    page1_source = "- ## Database\n  - Using MongoDB currently\n- ## API\n  - REST endpoints"
    page1 = TargetPage(
        name="Project Architecture",
        file_path=tmp_path / "pages" / "Project_Architecture.md",
        outline=LogseqOutline(
            blocks=[
                LogseqBlock(content="- ## Database", indent_level=0, properties={}),
                LogseqBlock(content="  - Using MongoDB currently", indent_level=1, properties={}),
            ],
            source_text=page1_source,
        ),
        organizational_convention="heading_bullets",
    )

    page2_source = "- ## User Feedback\n  - Survey results\n- ## Roadmap\n  - Q1 2025 features"
    page2 = TargetPage(
        name="Product Strategy",
        file_path=tmp_path / "pages" / "Product_Strategy.md",
        outline=LogseqOutline(
            blocks=[
                LogseqBlock(content="- ## User Feedback", indent_level=0, properties={}),
                LogseqBlock(content="  - Survey results", indent_level=1, properties={}),
            ],
            source_text=page2_source,
        ),
        organizational_convention="heading_bullets",
    )

    # Mock PageIndex.find_similar to return these pages
    mock_index = Mock(spec=PageIndex)

    def mock_find_similar(content, top_k=5):
        if "PostgreSQL" in content:
            return [(page1, 0.87), (page2, 0.42)]
        elif "simple UI" in content:
            return [(page2, 0.83), (page1, 0.39)]
        else:
            return [(page1, 0.5), (page2, 0.48)]

    mock_index.find_similar.side_effect = mock_find_similar

    return mock_index


# End-to-End Workflow Tests


def test_full_extraction_workflow(sample_journal_entry, mock_llm_client, mock_page_index):
    """Test complete workflow from journal to preview."""
    # Setup
    extractor = Extractor(mock_llm_client)

    # Stage 1: Extract knowledge blocks
    extractions = extractor.extract_knowledge(sample_journal_entry)

    assert len(extractions) == 4
    assert mock_llm_client.extract_knowledge.called

    # Classify into knowledge vs activity
    knowledge_blocks, activity_logs = classify_extractions(extractions)

    assert len(knowledge_blocks) == 2  # 0.9 and 0.85 confidence
    assert len(activity_logs) == 2  # 0.2 and 0.3 confidence

    # Stage 2: Match each knowledge block to target page
    proposed_actions = []
    for extraction in knowledge_blocks:
        selection = extractor.select_target_page(extraction.content, mock_page_index)

        # Create knowledge block
        kb = create_knowledge_block(
            extraction,
            sample_journal_entry.date,
            selection.target_page,
            selection.target_section,
            selection.suggested_action,
        )

        # Get similarity score
        similar_pages = mock_page_index.find_similar(extraction.content, top_k=1)
        similarity_score = similar_pages[0][1] if similar_pages else 0.0

        proposed_actions.append(
            ProposedAction(
                knowledge=kb,
                status=ActionStatus.READY,
                similarity_score=similarity_score,
            )
        )

    assert len(proposed_actions) == 2

    # Verify first action (PostgreSQL)
    assert proposed_actions[0].knowledge.target_page == "Project Architecture"
    assert proposed_actions[0].knowledge.target_section == ["Database"]
    assert proposed_actions[0].similarity_score == 0.87

    # Verify second action (UI insight)
    assert proposed_actions[1].knowledge.target_page == "Product Strategy"
    assert proposed_actions[1].knowledge.target_section == ["User Feedback"]
    assert proposed_actions[1].similarity_score == 0.83

    # Generate preview
    preview = ExtractionPreview(
        journal_date=sample_journal_entry.date,
        knowledge_blocks=[action.knowledge for action in proposed_actions],
        proposed_actions=proposed_actions,
        warnings=[],
    )

    output = preview.display()

    assert "Found 2 knowledge blocks" in output
    assert "PostgreSQL" in output
    assert "simple UI" in output
    assert "Project Architecture" in output
    assert "Product Strategy" in output


def test_workflow_with_duplicate_detection(
    sample_journal_entry, mock_llm_client, mock_page_index, tmp_path
):
    """Test workflow with duplicate detection."""
    # Create a target page with existing content
    existing_content = "- ## Database\n  - Decided to use PostgreSQL for persistence layer"
    page = TargetPage(
        name="Project Architecture",
        file_path=tmp_path / "Project_Architecture.md",
        outline=LogseqOutline(
            blocks=[
                LogseqBlock(content="- ## Database", indent_level=0, properties={}),
                LogseqBlock(
                    content="  - Decided to use PostgreSQL for persistence layer",
                    indent_level=1,
                    properties={},
                ),
            ],
            source_text=existing_content,
        ),
        organizational_convention="heading_bullets",
    )

    extractor = Extractor(mock_llm_client)
    extractions = extractor.extract_knowledge(sample_journal_entry)
    knowledge_blocks, _ = classify_extractions(extractions)

    # Check for duplicate
    postgresql_extraction = knowledge_blocks[0]
    is_dup = extractor.is_duplicate(postgresql_extraction.content, page)

    assert is_dup is True  # Content already exists in page


def test_workflow_with_no_knowledge_found(mock_llm_client, mock_page_index):
    """Test workflow when journal contains only activity logs."""
    # Journal with only activities
    source_text = "- Had meeting\n- Sent emails\n- Updated ticket"
    journal = JournalEntry(
        date=date(2025, 1, 15),
        file_path=Path("/test/journals/2025_01_15.md"),
        raw_content=source_text,
        outline=LogseqOutline(
            blocks=[
                LogseqBlock(content="- Had meeting", indent_level=0, properties={}),
                LogseqBlock(content="- Sent emails", indent_level=0, properties={}),
                LogseqBlock(content="- Updated ticket", indent_level=0, properties={}),
            ],
            source_text=source_text,
        ),
        line_count=3,
    )

    # Mock LLM returns low confidence for all
    mock_llm_client.extract_knowledge.return_value = [
        ExtractionResult(content="Had meeting", confidence=0.1),
        ExtractionResult(content="Sent emails", confidence=0.2),
        ExtractionResult(content="Updated ticket", confidence=0.3),
    ]

    extractor = Extractor(mock_llm_client)
    extractions = extractor.extract_knowledge(journal)
    knowledge_blocks, activity_logs = classify_extractions(extractions)

    assert len(knowledge_blocks) == 0
    assert len(activity_logs) == 3


def test_workflow_with_missing_target_page(
    sample_journal_entry, mock_llm_client, mock_page_index, tmp_path
):
    """Test workflow when target page doesn't exist."""
    extractor = Extractor(mock_llm_client)
    extractions = extractor.extract_knowledge(sample_journal_entry)
    knowledge_blocks, _ = classify_extractions(extractions)

    # Try to load non-existent page
    with patch("logsqueak.models.page.TargetPage.load") as mock_load:
        mock_load.return_value = None  # Page doesn't exist

        selection = extractor.select_target_page(knowledge_blocks[0].content, mock_page_index)

        # Create action with SKIPPED status
        kb = create_knowledge_block(
            knowledge_blocks[0],
            sample_journal_entry.date,
            selection.target_page,
            selection.target_section,
            selection.suggested_action,
        )

        action = ProposedAction(
            knowledge=kb,
            status=ActionStatus.SKIPPED,
            reason=f"Target page '{selection.target_page}' does not exist",
            similarity_score=0.87,
        )

        assert action.status == ActionStatus.SKIPPED
        assert "does not exist" in action.reason


def test_workflow_classification_boundary_cases(mock_llm_client):
    """Test classification at confidence threshold boundaries."""
    # Mock extractions at exact threshold (0.5)
    mock_llm_client.extract_knowledge.return_value = [
        ExtractionResult(content="Exactly at threshold", confidence=0.5),
        ExtractionResult(content="Just below", confidence=0.49),
        ExtractionResult(content="Just above", confidence=0.51),
    ]

    journal = JournalEntry(
        date=date(2025, 1, 15),
        file_path=Path("/test/journals/2025_01_15.md"),
        raw_content="test",
        outline=LogseqOutline(blocks=[], source_text="test"),
        line_count=1,
    )

    extractor = Extractor(mock_llm_client)
    extractions = extractor.extract_knowledge(journal)
    knowledge_blocks, activity_logs = classify_extractions(extractions)

    assert len(knowledge_blocks) == 2  # 0.5 and 0.51
    assert len(activity_logs) == 1  # 0.49


def test_workflow_with_multiple_sections(mock_page_index):
    """Test workflow with nested section paths."""
    # Create fresh mock client without side_effect
    fresh_mock_client = Mock()
    fresh_mock_client.extract_knowledge.return_value = [
        ExtractionResult(content="Deep nested knowledge", confidence=0.9),
    ]

    fresh_mock_client.select_target_page.return_value = PageSelectionResult(
        target_page="Complex Page",
        target_section=["Level 1", "Level 2", "Level 3"],
        suggested_action=ActionType.ADD_CHILD,
        reasoning="Nested section",
    )

    journal = JournalEntry(
        date=date(2025, 1, 15),
        file_path=Path("/test/journals/2025_01_15.md"),
        raw_content="test",
        outline=LogseqOutline(blocks=[], source_text="test"),
        line_count=1,
    )

    extractor = Extractor(fresh_mock_client)
    extractions = extractor.extract_knowledge(journal)
    knowledge_blocks, _ = classify_extractions(extractions)

    selection = extractor.select_target_page(knowledge_blocks[0].content, mock_page_index)

    assert selection.target_section == ["Level 1", "Level 2", "Level 3"]
    assert len(selection.target_section) == 3


def test_workflow_with_create_section_action(mock_page_index):
    """Test workflow when LLM suggests creating a new section."""
    # Create fresh mock client without side_effect
    fresh_mock_client = Mock()
    fresh_mock_client.extract_knowledge.return_value = [
        ExtractionResult(content="New competitor identified: CompanyX", confidence=0.88),
    ]

    fresh_mock_client.select_target_page.return_value = PageSelectionResult(
        target_page="Market Analysis",
        target_section=["Competitors"],
        suggested_action=ActionType.CREATE_SECTION,
        reasoning="Section doesn't exist yet",
    )

    journal = JournalEntry(
        date=date(2025, 1, 15),
        file_path=Path("/test/journals/2025_01_15.md"),
        raw_content="test",
        outline=LogseqOutline(blocks=[], source_text="test"),
        line_count=1,
    )

    extractor = Extractor(fresh_mock_client)
    extractions = extractor.extract_knowledge(journal)
    knowledge_blocks, _ = classify_extractions(extractions)

    selection = extractor.select_target_page(knowledge_blocks[0].content, mock_page_index)

    kb = create_knowledge_block(
        knowledge_blocks[0],
        journal.date,
        selection.target_page,
        selection.target_section,
        selection.suggested_action,
    )

    assert kb.suggested_action == ActionType.CREATE_SECTION
    assert kb.target_section == ["Competitors"]


def test_workflow_error_propagation(mock_llm_client):
    """Test that LLM errors propagate correctly through workflow."""
    mock_llm_client.extract_knowledge.side_effect = Exception("LLM API error")

    journal = JournalEntry(
        date=date(2025, 1, 15),
        file_path=Path("/test/journals/2025_01_15.md"),
        raw_content="test",
        outline=LogseqOutline(blocks=[], source_text="test"),
        line_count=1,
    )

    extractor = Extractor(mock_llm_client)

    with pytest.raises(Exception, match="LLM API error"):
        extractor.extract_knowledge(journal)


def test_workflow_preview_with_warnings(
    sample_journal_entry, mock_llm_client, mock_page_index
):
    """Test workflow generates preview with warnings."""
    extractor = Extractor(mock_llm_client)
    extractions = extractor.extract_knowledge(sample_journal_entry)
    knowledge_blocks, _ = classify_extractions(extractions)

    warnings = [
        "Target page 'Old Page' does not exist",
        "Low similarity score for some matches",
    ]

    # Create preview with warnings
    proposed_actions = []
    for extraction in knowledge_blocks:
        selection = extractor.select_target_page(extraction.content, mock_page_index)
        kb = create_knowledge_block(
            extraction,
            sample_journal_entry.date,
            selection.target_page,
            selection.target_section,
            selection.suggested_action,
        )
        proposed_actions.append(ProposedAction(knowledge=kb, status=ActionStatus.READY))

    preview = ExtractionPreview(
        journal_date=sample_journal_entry.date,
        knowledge_blocks=[action.knowledge for action in proposed_actions],
        proposed_actions=proposed_actions,
        warnings=warnings,
    )

    output = preview.display()

    assert "Warnings:" in output
    assert "Old Page" in output
    assert "Low similarity" in output


def test_workflow_mixed_action_statuses(sample_journal_entry, mock_llm_client, mock_page_index):
    """Test workflow with mix of READY, SKIPPED, and WARNING actions."""
    extractor = Extractor(mock_llm_client)
    extractions = extractor.extract_knowledge(sample_journal_entry)
    knowledge_blocks, _ = classify_extractions(extractions)

    # Create mixed actions
    selection1 = extractor.select_target_page(knowledge_blocks[0].content, mock_page_index)
    kb1 = create_knowledge_block(
        knowledge_blocks[0],
        sample_journal_entry.date,
        selection1.target_page,
        selection1.target_section,
        selection1.suggested_action,
    )

    selection2 = extractor.select_target_page(knowledge_blocks[1].content, mock_page_index)
    kb2 = create_knowledge_block(
        knowledge_blocks[1],
        sample_journal_entry.date,
        selection2.target_page,
        selection2.target_section,
        selection2.suggested_action,
    )

    actions = [
        ProposedAction(knowledge=kb1, status=ActionStatus.READY, similarity_score=0.87),
        ProposedAction(
            knowledge=kb2,
            status=ActionStatus.SKIPPED,
            reason="Duplicate",
            similarity_score=0.83,
        ),
    ]

    preview = ExtractionPreview(
        journal_date=sample_journal_entry.date,
        knowledge_blocks=[kb1, kb2],
        proposed_actions=actions,
        warnings=[],
    )

    output = preview.display()

    assert "Will integrate: 1" in output
    assert "Skipped: 1" in output
    assert "SKIPPED" in output
    assert "Duplicate" in output


def test_workflow_empty_journal(mock_llm_client, mock_page_index):
    """Test workflow with empty journal entry."""
    empty_journal = JournalEntry(
        date=date(2025, 1, 15),
        file_path=Path("/test/journals/2025_01_15.md"),
        raw_content="",
        outline=LogseqOutline(blocks=[], source_text=""),
        line_count=0,
    )

    mock_llm_client.extract_knowledge.return_value = []

    extractor = Extractor(mock_llm_client)
    extractions = extractor.extract_knowledge(empty_journal)

    assert len(extractions) == 0


def test_workflow_rag_candidate_selection(sample_journal_entry, mock_llm_client, mock_page_index):
    """Test that RAG finds top-5 candidates and LLM selects from them."""
    extractor = Extractor(mock_llm_client)
    extractions = extractor.extract_knowledge(sample_journal_entry)
    knowledge_blocks, _ = classify_extractions(extractions)

    # Select target page
    selection = extractor.select_target_page(knowledge_blocks[0].content, mock_page_index)

    # Verify RAG was called for candidates
    mock_page_index.find_similar.assert_called()

    # Verify LLM select_target_page was called
    mock_llm_client.select_target_page.assert_called()

    # Verify selection has expected structure
    assert hasattr(selection, "target_page")
    assert hasattr(selection, "target_section")
    assert hasattr(selection, "suggested_action")
    assert hasattr(selection, "reasoning")


def test_workflow_provenance_links(sample_journal_entry, mock_llm_client, mock_page_index):
    """Test that all knowledge blocks have correct provenance."""
    extractor = Extractor(mock_llm_client)
    extractions = extractor.extract_knowledge(sample_journal_entry)
    knowledge_blocks, _ = classify_extractions(extractions)

    for extraction in knowledge_blocks:
        selection = extractor.select_target_page(extraction.content, mock_page_index)
        kb = create_knowledge_block(
            extraction,
            sample_journal_entry.date,
            selection.target_page,
            selection.target_section,
            selection.suggested_action,
        )

        # Verify provenance
        assert kb.source_date == date(2025, 1, 15)
        provenance_link = kb.provenance_link()
        assert "[[2025-01-15]]" in provenance_link


def test_workflow_confidence_preservation(sample_journal_entry, mock_llm_client, mock_page_index):
    """Test that confidence scores are preserved through workflow."""
    extractor = Extractor(mock_llm_client)
    extractions = extractor.extract_knowledge(sample_journal_entry)

    # Original confidences from mock: 0.9, 0.2, 0.85, 0.3
    assert extractions[0].confidence == 0.9
    assert extractions[2].confidence == 0.85

    knowledge_blocks, _ = classify_extractions(extractions)

    # After classification, knowledge blocks should preserve confidence
    assert knowledge_blocks[0].confidence == 0.9
    assert knowledge_blocks[1].confidence == 0.85

    # Create full knowledge blocks
    selection = extractor.select_target_page(knowledge_blocks[0].content, mock_page_index)
    kb = create_knowledge_block(
        knowledge_blocks[0],
        sample_journal_entry.date,
        selection.target_page,
        selection.target_section,
        selection.suggested_action,
    )

    # Confidence should still be preserved in final knowledge block
    assert kb.confidence == 0.9
