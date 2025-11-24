"""Unit tests for LLM helper functions."""

import pytest
from logsqueak.services.llm_helpers import format_chunks_for_llm
from logsqueak.utils.llm_id_mapper import LLMIDMapper
from logseq_outline.parser import LogseqOutline


def test_format_chunks_for_llm_strips_page_level_frontmatter():
    """Test format_chunks_for_llm() strips frontmatter from page-level chunks."""
    # Arrange
    id_mapper = LLMIDMapper()
    page_name = "diffused"
    page_chunk_id = f"{page_name}::__PAGE__"
    regular_block_id = "block123"

    # Add IDs to mapper
    id_mapper.add(page_chunk_id)
    id_mapper.add(regular_block_id)

    # Page-level chunk with frontmatter in context
    page_chunk_context = "Page: diffused\nTitle: Diffused System\ntags:: system, kubernetes"
    # Regular block chunk (already cleaned during indexing)
    regular_block_context = "- Deployment architecture\n  - Uses Kubernetes"

    chunks = [
        (page_name, page_chunk_id, page_chunk_context),
        (page_name, regular_block_id, regular_block_context),
    ]

    page_contents = {
        page_name: LogseqOutline.parse(
            "tags:: system, kubernetes\n\n"
            "- Deployment architecture\n"
            "  id:: block123\n"
            "  - Uses Kubernetes"
        )
    }

    # Act
    xml = format_chunks_for_llm(chunks, page_contents, id_mapper)

    # Assert - frontmatter should appear only in <properties>, not in page-level block
    assert "<properties>" in xml
    assert "tags:: system, kubernetes" in xml.split("<properties>")[1].split("</properties>")[0]

    # Page-level block should have "Page:" and "Title:" but NOT frontmatter
    page_block_section = xml.split(f'<block id="1">')[1].split("</block>")[0]
    assert "Page: diffused" in page_block_section
    assert "Title: Diffused System" in page_block_section
    assert "tags::" not in page_block_section  # Frontmatter stripped from block

    # Regular block should be unchanged
    regular_block_section = xml.split(f'<block id="2">')[1].split("</block>")[0]
    assert "Deployment architecture" in regular_block_section


def test_format_chunks_for_llm_handles_regular_blocks_normally():
    """Test format_chunks_for_llm() doesn't strip frontmatter from regular blocks."""
    # Arrange
    id_mapper = LLMIDMapper()
    page_name = "Python"
    block_id = "block456"

    id_mapper.add(block_id)

    # Regular block (not a page-level chunk)
    # These are already cleaned during indexing, but test that we don't break them
    chunks = [
        (page_name, block_id, "- Type hints are essential\n  - Improve code quality"),
    ]

    page_contents = {
        page_name: LogseqOutline.parse(
            "type:: language\n\n"
            "- Type hints are essential\n"
            "  id:: block456\n"
            "  - Improve code quality"
        )
    }

    # Act
    xml = format_chunks_for_llm(chunks, page_contents, id_mapper)

    # Assert - regular block content is preserved as-is
    assert "Type hints are essential" in xml
    assert "Improve code quality" in xml


def test_format_chunks_for_llm_handles_multiple_pages():
    """Test format_chunks_for_llm() handles multiple pages with page-level chunks."""
    # Arrange
    id_mapper = LLMIDMapper()

    page1_id = "Page1::__PAGE__"
    page2_id = "Page2::__PAGE__"

    id_mapper.add(page1_id)
    id_mapper.add(page2_id)

    chunks = [
        ("Page1", page1_id, "Page: Page1\ntags:: web"),
        ("Page2", page2_id, "Page: Page2\ntype:: language"),
    ]

    page_contents = {
        "Page1": LogseqOutline.parse("tags:: web\n\n"),
        "Page2": LogseqOutline.parse("type:: language\n\n"),
    }

    # Act
    xml = format_chunks_for_llm(chunks, page_contents, id_mapper)

    # Assert - each page should have properties section and cleaned blocks
    assert xml.count("<page name=") == 2
    assert xml.count("<properties>") == 2

    # Page1 block should not have frontmatter
    page1_section = xml.split('<page name="Page1">')[1].split("</page>")[0]
    page1_block = page1_section.split('<block id="1">')[1].split("</block>")[0]
    assert "Page: Page1" in page1_block
    assert "tags::" not in page1_block  # Stripped from block

    # But properties should have frontmatter
    page1_properties = page1_section.split("<properties>")[1].split("</properties>")[0]
    assert "tags:: web" in page1_properties
