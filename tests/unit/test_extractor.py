"""Unit tests for knowledge extraction orchestrator.

Tests cover:
- Stage 1: Extract knowledge blocks from journal
- Stage 2: Page selection using RAG candidates
- Duplicate detection
- Helper functions (create_knowledge_block)
"""

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest

from logsqueak.extraction.extractor import Extractor, create_knowledge_block
from logsqueak.llm.client import ExtractionResult, PageCandidate, PageSelectionResult
from logsqueak.logseq.parser import LogseqBlock, LogseqOutline
from logsqueak.models.journal import JournalEntry
from logsqueak.models.knowledge import ActionType, KnowledgeBlock
from logsqueak.models.page import PageIndex, TargetPage


@pytest.fixture
def mock_llm_client():
    """Mock LLM client for testing."""
    return Mock()


@pytest.fixture
def extractor(mock_llm_client):
    """Extractor instance with mocked LLM client."""
    return Extractor(mock_llm_client)


@pytest.fixture
def sample_journal():
    """Sample journal entry for testing."""
    source_text = "- Had meeting with stakeholders\n- Key decision: Use PostgreSQL for persistence"
    outline = LogseqOutline(
        blocks=[
            LogseqBlock(
                content="- Had meeting with stakeholders",
                indent_level=0,
                properties={},
            ),
            LogseqBlock(
                content="- Key decision: Use PostgreSQL for persistence",
                indent_level=0,
                properties={},
            ),
        ],
        source_text=source_text,
    )

    return JournalEntry(
        date=date(2025, 1, 15),
        file_path=Path("/test/journals/2025_01_15.md"),
        raw_content=source_text,
        outline=outline,
        line_count=2,
    )


@pytest.fixture
def mock_page_index():
    """Mock PageIndex for testing."""
    index = Mock(spec=PageIndex)

    # Create mock target pages
    mock_page1 = Mock(spec=TargetPage)
    mock_page1.name = "Project Architecture"
    mock_page1.outline = Mock(spec=LogseqOutline)
    mock_page1.outline.render.return_value = "- ## Tech Stack\n  - Using Python\n- ## Database\n  - TBD"
    mock_page1.outline.frontmatter = []  # No frontmatter
    mock_page1.outline.blocks = []  # Mock blocks (not used in preview generation anymore)
    mock_page1.outline.indent_str = "  "

    mock_page2 = Mock(spec=TargetPage)
    mock_page2.name = "Engineering Decisions"
    mock_page2.outline = Mock(spec=LogseqOutline)
    mock_page2.outline.render.return_value = "- ## 2025 Q1\n  - Various decisions"
    mock_page2.outline.frontmatter = []  # No frontmatter
    mock_page2.outline.blocks = []  # Mock blocks
    mock_page2.outline.indent_str = "  "

    # find_similar returns (page, similarity_score) tuples
    index.find_similar.return_value = [
        (mock_page1, 0.85),
        (mock_page2, 0.72),
    ]

    return index


# Stage 1: Extract Knowledge Tests


def test_extract_knowledge_calls_llm(extractor, mock_llm_client, sample_journal):
    """Test that extract_knowledge calls LLM with correct parameters."""
    mock_llm_client.extract_knowledge.return_value = [
        ExtractionResult(
            content="Use PostgreSQL for persistence",
            confidence=0.9,
        )
    ]

    results = extractor.extract_knowledge(sample_journal)

    # Verify LLM was called with journal content, date, and indent string
    mock_llm_client.extract_knowledge.assert_called_once_with(
        journal_content=sample_journal.raw_content,
        journal_date=sample_journal.date,
        indent_str=sample_journal.outline.indent_str,
    )

    assert len(results) == 1
    assert results[0].content == "Use PostgreSQL for persistence"
    assert results[0].confidence == 0.9


def test_extract_knowledge_returns_multiple_blocks(extractor, mock_llm_client, sample_journal):
    """Test extraction of multiple knowledge blocks."""
    mock_llm_client.extract_knowledge.return_value = [
        ExtractionResult(content="Knowledge 1", confidence=0.9),
        ExtractionResult(content="Knowledge 2", confidence=0.85),
        ExtractionResult(content="Knowledge 3", confidence=0.7),
    ]

    results = extractor.extract_knowledge(sample_journal)

    assert len(results) == 3
    assert results[0].content == "Knowledge 1"
    assert results[1].content == "Knowledge 2"
    assert results[2].content == "Knowledge 3"


def test_extract_knowledge_empty_journal(extractor, mock_llm_client):
    """Test extraction from empty journal returns empty list."""
    empty_journal = JournalEntry(
        date=date(2025, 1, 15),
        file_path=Path("/test/journals/2025_01_15.md"),
        raw_content="",
        outline=LogseqOutline(blocks=[], source_text=""),
        line_count=0,
    )

    mock_llm_client.extract_knowledge.return_value = []

    results = extractor.extract_knowledge(empty_journal)

    assert results == []


def test_extract_knowledge_propagates_llm_error(extractor, mock_llm_client, sample_journal):
    """Test that LLM errors are propagated."""
    mock_llm_client.extract_knowledge.side_effect = Exception("LLM API error")

    with pytest.raises(Exception, match="LLM API error"):
        extractor.extract_knowledge(sample_journal)


# Stage 2: Page Selection Tests


def test_select_target_page_uses_rag_candidates(extractor, mock_llm_client, mock_page_index):
    """Test that page selection uses RAG to find top-5 candidates."""
    knowledge_content = "Use PostgreSQL for persistence"

    mock_llm_client.select_target_page.return_value = PageSelectionResult(
        target_page="Project Architecture",
        target_section=["Database"],
        suggested_action=ActionType.ADD_CHILD,
        reasoning="Database technology decision",
    )

    result = extractor.select_target_page(knowledge_content, mock_page_index)

    # Verify RAG search was called
    mock_page_index.find_similar.assert_called_once_with(knowledge_content, top_k=5)

    # Verify LLM was called with candidates
    mock_llm_client.select_target_page.assert_called_once()
    call_args = mock_llm_client.select_target_page.call_args

    assert call_args.kwargs["knowledge_content"] == knowledge_content
    candidates = call_args.kwargs["candidates"]
    assert len(candidates) == 2
    assert candidates[0].page_name == "Project Architecture"
    assert candidates[0].similarity_score == 0.85
    assert candidates[1].page_name == "Engineering Decisions"
    assert candidates[1].similarity_score == 0.72


def test_select_target_page_includes_preview(extractor, mock_llm_client, mock_page_index):
    """Test that candidates include page content preview."""
    knowledge_content = "Test knowledge"

    mock_llm_client.select_target_page.return_value = PageSelectionResult(
        target_page="Project Architecture",
        target_section=None,
        suggested_action=ActionType.ADD_CHILD,
        reasoning="General note",
    )

    extractor.select_target_page(knowledge_content, mock_page_index)

    # Check that preview is included (first 200 chars)
    call_args = mock_llm_client.select_target_page.call_args
    candidates = call_args.kwargs["candidates"]

    assert len(candidates[0].preview) <= 200
    assert "Tech Stack" in candidates[0].preview


def test_select_target_page_with_section_path(extractor, mock_llm_client, mock_page_index):
    """Test page selection with specific section path."""
    knowledge_content = "Use PostgreSQL"

    mock_llm_client.select_target_page.return_value = PageSelectionResult(
        target_page="Project Architecture",
        target_section=["Tech Stack", "Database"],
        suggested_action=ActionType.ADD_CHILD,
        reasoning="Database technology",
    )

    result = extractor.select_target_page(knowledge_content, mock_page_index)

    assert result.target_page == "Project Architecture"
    assert result.target_section == ["Tech Stack", "Database"]
    assert result.suggested_action == ActionType.ADD_CHILD


def test_select_target_page_no_section(extractor, mock_llm_client, mock_page_index):
    """Test page selection without specific section (page-level)."""
    knowledge_content = "General project note"

    mock_llm_client.select_target_page.return_value = PageSelectionResult(
        target_page="Project Notes",
        target_section=None,
        suggested_action=ActionType.ADD_CHILD,
        reasoning="General note",
    )

    result = extractor.select_target_page(knowledge_content, mock_page_index)

    assert result.target_page == "Project Notes"
    assert result.target_section is None


def test_select_target_page_create_section_action(extractor, mock_llm_client, mock_page_index):
    """Test page selection with CREATE_SECTION action."""
    knowledge_content = "New competitor identified"

    mock_llm_client.select_target_page.return_value = PageSelectionResult(
        target_page="Project X",
        target_section=["Competitors"],
        suggested_action=ActionType.CREATE_SECTION,
        reasoning="New section needed",
    )

    result = extractor.select_target_page(knowledge_content, mock_page_index)

    assert result.suggested_action == ActionType.CREATE_SECTION


def test_select_target_page_propagates_llm_error(extractor, mock_llm_client, mock_page_index):
    """Test that LLM errors in page selection are propagated."""
    mock_llm_client.select_target_page.side_effect = Exception("LLM selection error")

    with pytest.raises(Exception, match="LLM selection error"):
        extractor.select_target_page("Test content", mock_page_index)


def test_select_target_page_with_frontmatter_in_preview(extractor, mock_llm_client):
    """Test that page previews with frontmatter are formatted with XML tags."""
    index = Mock(spec=PageIndex)

    # Create a page with frontmatter
    mock_page = Mock(spec=TargetPage)
    mock_page.name = "syft"
    mock_outline = Mock(spec=LogseqOutline)
    mock_outline.frontmatter = ["type:: software", "upstream-url:: https://github.com/anchore/syft", "area:: [[SBOM]]"]
    mock_outline.indent_str = "  "
    # Create actual blocks to test rendering
    block1 = LogseqBlock(content="## Overview", indent_level=0)
    block2 = LogseqBlock(content="SBOM generation tool", indent_level=1)
    block1.children = [block2]
    mock_outline.blocks = [block1]
    mock_page.outline = mock_outline

    index.find_similar.return_value = [(mock_page, 0.85)]

    mock_llm_client.select_target_page.return_value = PageSelectionResult(
        target_page="syft",
        target_section=["Overview"],  # Should select a bullet block, not frontmatter
        suggested_action=ActionType.ADD_CHILD,
        reasoning="Related to SBOM tools",
    )

    extractor.select_target_page("SBOM analysis", index)

    # Verify the preview passed to LLM contains frontmatter in XML tags
    call_args = mock_llm_client.select_target_page.call_args
    candidates = call_args.kwargs["candidates"]

    preview = candidates[0].preview
    # Preview should contain frontmatter wrapped in tags
    assert "<frontmatter>" in preview
    assert "</frontmatter>" in preview
    assert "type:: software" in preview
    assert "area:: [[SBOM]]" in preview
    # Preview should also contain the bullet blocks
    assert "- ## Overview" in preview
    assert "  - SBOM generation tool" in preview


def test_select_target_page_passes_indent_str(extractor, mock_llm_client, mock_page_index):
    """Test that select_target_page passes indent_str to LLM."""
    knowledge_content = "Test knowledge"
    custom_indent = "\t"  # Tab-based indentation

    mock_llm_client.select_target_page.return_value = PageSelectionResult(
        target_page="Test Page",
        target_section=None,
        suggested_action=ActionType.ADD_CHILD,
        reasoning="Test",
    )

    extractor.select_target_page(knowledge_content, mock_page_index, indent_str=custom_indent)

    # Verify indent_str was passed to LLM
    call_args = mock_llm_client.select_target_page.call_args
    assert call_args.kwargs["indent_str"] == custom_indent


# Duplicate Detection Tests


def test_is_duplicate_delegates_to_target_page(extractor):
    """Test that duplicate detection delegates to TargetPage."""
    mock_page = Mock(spec=TargetPage)
    mock_page.has_duplicate.return_value = True

    knowledge_content = "Duplicate content"

    result = extractor.is_duplicate(knowledge_content, mock_page)

    mock_page.has_duplicate.assert_called_once_with(knowledge_content)
    assert result is True


def test_is_duplicate_returns_false_for_new_content(extractor):
    """Test duplicate detection returns False for new content."""
    mock_page = Mock(spec=TargetPage)
    mock_page.has_duplicate.return_value = False

    knowledge_content = "New unique content"

    result = extractor.is_duplicate(knowledge_content, mock_page)

    assert result is False


# Helper Function Tests


def test_create_knowledge_block_combines_results():
    """Test create_knowledge_block combines extraction and selection results."""
    extraction = ExtractionResult(
        content="Use PostgreSQL for persistence",
        confidence=0.9,
    )

    source_date = date(2025, 1, 15)
    target_page = "Project Architecture"
    target_section = ["Database"]
    suggested_action = ActionType.ADD_CHILD

    block = create_knowledge_block(
        extraction,
        source_date,
        target_page,
        target_section,
        suggested_action,
    )

    assert isinstance(block, KnowledgeBlock)
    assert block.content == "Use PostgreSQL for persistence"
    assert block.confidence == 0.9
    assert block.source_date == source_date
    assert block.target_page == "Project Architecture"
    assert block.target_section == ["Database"]
    assert block.suggested_action == ActionType.ADD_CHILD


def test_create_knowledge_block_no_section():
    """Test create_knowledge_block with no section path."""
    extraction = ExtractionResult(
        content="General note",
        confidence=0.8,
    )

    block = create_knowledge_block(
        extraction,
        date(2025, 1, 15),
        "Project Notes",
        None,
        ActionType.ADD_CHILD,
    )

    assert block.target_section is None


def test_create_knowledge_block_create_section_action():
    """Test create_knowledge_block with CREATE_SECTION action."""
    extraction = ExtractionResult(
        content="Competitor info",
        confidence=0.85,
    )

    block = create_knowledge_block(
        extraction,
        date(2025, 1, 15),
        "Project X",
        ["Competitors"],
        ActionType.CREATE_SECTION,
    )

    assert block.suggested_action == ActionType.CREATE_SECTION


def test_create_knowledge_block_low_confidence():
    """Test create_knowledge_block preserves low confidence scores."""
    extraction = ExtractionResult(
        content="Uncertain information",
        confidence=0.5,
    )

    block = create_knowledge_block(
        extraction,
        date(2025, 1, 15),
        "Notes",
        None,
        ActionType.ADD_CHILD,
    )

    assert block.confidence == 0.5
