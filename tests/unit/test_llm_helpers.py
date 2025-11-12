"""Unit tests for LLM helper functions (filtering, formatting)."""

import pytest
from logsqueak.models.integration_decision import IntegrationDecision
from logsqueak.services.llm_helpers import (
    filter_skip_exists_blocks,
    format_chunks_for_llm,
)
from logseq_outline.parser import LogseqOutline


@pytest.fixture
def sample_decisions():
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
# T095a: Removed batch_decisions_by_block() tests (function removed in T108q)
# Batching is no longer needed since per-block LLM calls naturally group decisions.
# ============================================================================


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


# ============================================================================
# T108k: Tests for format_chunks_for_llm()
# ============================================================================


class TestFormatChunksForLLM:
    """Test format_chunks_for_llm() helper function."""

    def test_groups_chunks_by_page(self):
        """Should group chunks by page correctly."""
        # Chunks from different pages (with block IDs)
        chunks = [
            ("Page1", "block-1", "- Content 1"),
            ("Page2", "block-2", "- Content 2"),
            ("Page1", "block-3", "- Content 3"),
        ]

        page_contents = {
            "Page1": LogseqOutline(blocks=[], source_text="", frontmatter=[]),
            "Page2": LogseqOutline(blocks=[], source_text="", frontmatter=[]),
        }

        xml = format_chunks_for_llm(chunks, page_contents)

        # Should have both pages
        assert '<page name="Page1">' in xml
        assert '<page name="Page2">' in xml

        # Page1 should have 2 blocks
        page1_section = xml[xml.find('<page name="Page1">'):xml.find('</page>', xml.find('Page1'))]
        assert page1_section.count('<block') == 2

        # Page2 should have 1 block
        page2_section = xml[xml.find('<page name="Page2">'):xml.find('</page>', xml.find('Page2'))]
        assert page2_section.count('<block') == 1

    def test_strips_id_properties_from_content(self):
        """Should strip id:: properties from block content."""
        # Context string with id:: property
        context = "- Content line\n  id:: 12345\n  Another line"

        chunks = [("Page1", "block-1", context)]
        page_contents = {"Page1": LogseqOutline(blocks=[], source_text="", frontmatter=[])}

        xml = format_chunks_for_llm(chunks, page_contents)

        # id:: should NOT be in block content
        assert "id:: 12345" not in xml

        # Other content should remain
        assert "Content line" in xml
        assert "Another line" in xml

    def test_includes_page_properties_if_present(self):
        """Should include page properties in <properties> section."""
        chunks = [("Page1", "block-1", "- Content")]
        page_contents = {
            "Page1": LogseqOutline(
                blocks=[],
                source_text="",
                frontmatter=["area:: [[Ideas]]", "type:: Best Practice"]
            )
        }

        xml = format_chunks_for_llm(chunks, page_contents)

        # Should have properties section
        assert "<properties>" in xml
        assert "area:: [[Ideas]]" in xml
        assert "type:: Best Practice" in xml
        assert "</properties>" in xml

    def test_hierarchical_context_with_parents(self):
        """Should include hierarchical context with parent blocks."""
        # Hierarchical context string (already formatted by ChromaDB)
        context = "- Parent content\n  - Child content"

        chunks = [("Page1", "block-1", context)]
        page_contents = {"Page1": LogseqOutline(blocks=[], source_text="", frontmatter=[])}

        xml = format_chunks_for_llm(chunks, page_contents)

        # Should contain hierarchical context
        assert "Parent content" in xml
        assert "Child content" in xml

        # Should have proper bullet formatting
        assert "- Parent content" in xml
        assert "- Child content" in xml

    def test_xml_escaping_in_page_names(self):
        """Should escape XML special characters in page names."""
        chunks = [("Page<>&\"'", "block-1", "- Content")]
        page_contents = {"Page<>&\"'": LogseqOutline(blocks=[], source_text="", frontmatter=[])}

        xml = format_chunks_for_llm(chunks, page_contents)

        # Should escape special characters
        assert "Page&lt;&gt;&amp;&quot;&apos;" in xml
        assert "Page<>&\"'" not in xml  # Original should not appear

    def test_empty_chunks(self):
        """Should handle empty chunk list without errors."""
        chunks = []
        page_contents = {}

        xml = format_chunks_for_llm(chunks, page_contents)

        # Should produce minimal XML
        assert "<pages>" in xml
        assert "</pages>" in xml
        # Check that there are no individual page tags (not counting the "s" in pages)
        assert '<page name=' not in xml  # No individual page tags

    def test_page_without_properties(self):
        """Should omit <properties> section if page has no frontmatter."""
        chunks = [("Page1", "block-1", "- Content")]
        page_contents = {"Page1": LogseqOutline(blocks=[], source_text="", frontmatter=[])}

        xml = format_chunks_for_llm(chunks, page_contents)

        # Should NOT have properties section
        assert "<properties>" not in xml
        assert "</properties>" not in xml

    def test_multiple_blocks_same_page(self):
        """Should handle multiple blocks from same page."""
        chunks = [
            ("Page1", "block-1", "- Content 1"),
            ("Page1", "block-2", "- Content 2"),
            ("Page1", "block-3", "- Content 3"),
        ]
        page_contents = {"Page1": LogseqOutline(blocks=[], source_text="", frontmatter=[])}

        xml = format_chunks_for_llm(chunks, page_contents)

        # Should have all blocks
        assert xml.count("<block") == 3
        assert 'id="block-1"' in xml
        assert 'id="block-2"' in xml
        assert 'id="block-3"' in xml
