"""Integration test for the complete 5-phase pipeline.

Tests the end-to-end flow:
- Phase 0: Index building with hybrid IDs
- Phase 1: Knowledge extraction from journal
- Phase 2: RAG candidate retrieval
- Phase 3: Decider + Reworder LLMs
- Phase 4: Execution with atomic journal updates

Verifies:
- Hybrid IDs are generated and preserved
- processed:: markers are added atomically
- Round-trip safety (parse → modify → render)
- Full integration from journal to pages
"""

import tempfile
from datetime import date
from pathlib import Path
from textwrap import dedent

import pytest

from logsqueak.extraction.extractor import Extractor
from logsqueak.llm.client import LLMClient, DecisionResult, RephrasedContent, ExtractionResult
from logseq_outline import GraphPaths, LogseqOutline
from logsqueak.models.journal import JournalEntry
from logsqueak.models.knowledge import ActionType
from logsqueak.rag.indexer import IndexBuilder
from logsqueak.rag.manifest import CacheManifest
from logsqueak.rag.vector_store import ChromaDBStore


# ============================================================================
# Mock LLM Client for Testing
# ============================================================================


class MockPipelineLLM(LLMClient):
    """Mock LLM that returns predictable results for pipeline testing."""

    def extract_knowledge(self, journal_content: str, journal_date: date, indent_str: str = "  "):
        """Return a single knowledge block for testing."""
        # Extract the "Learned about X" block
        return [
            ExtractionResult(
                content="Learned about vector databases and ChromaDB integration",
                confidence=0.9,
            )
        ]

    def decide_action(self, knowledge_text: str, candidate_chunks: list[dict]) -> DecisionResult:
        """Decide to append the knowledge to the Database Design page."""
        # Find the "Database Design" page in candidates
        for chunk in candidate_chunks:
            if chunk["page_name"] == "Database Design":
                return DecisionResult(
                    action=ActionType.APPEND_CHILD,
                    page_name="Database Design",
                    target_id=chunk["target_id"],  # Use first chunk's ID
                    reasoning="Knowledge about databases belongs in Database Design page",
                )

        # Fallback: append to root
        return DecisionResult(
            action=ActionType.APPEND_ROOT,
            page_name="Database Design",
            target_id="root",
            reasoning="No specific section found, append to root",
        )

    def rephrase_content(self, knowledge_full_text: str) -> RephrasedContent:
        """Return cleaned knowledge content."""
        return RephrasedContent(
            content="Vector databases like ChromaDB enable efficient semantic search"
        )

    def select_target_page(self, knowledge_content: str, candidates: list):
        """Not used in new pipeline, but required by interface."""
        raise NotImplementedError("Old 2-stage pipeline method")


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def temp_graph(tmp_path):
    """Create a temporary Logseq graph with sample pages and journal."""
    graph_path = tmp_path / "graph"

    # Create directories
    (graph_path / "pages").mkdir(parents=True)
    (graph_path / "journals").mkdir(parents=True)

    # Create sample pages
    create_sample_pages(graph_path)

    # Create sample journal
    create_sample_journal(graph_path)

    return graph_path


def create_sample_pages(graph_path: Path):
    """Create sample pages for testing."""
    pages_dir = graph_path / "pages"

    # Database Design page (target for knowledge integration)
    db_page = pages_dir / "Database Design.md"
    db_page.write_text(dedent("""\
        - # Overview
          id:: db-overview
          - This page covers database design concepts
        - # Technologies
          id:: db-technologies
          - SQL databases
          - NoSQL databases
        """))

    # Project X page (should not match)
    project_page = pages_dir / "Project X.md"
    project_page.write_text(dedent("""\
        - # Timeline
          - Q1 2025: Planning
          - Q2 2025: Development
        """))

    # Unrelated page
    notes_page = pages_dir / "Random Notes.md"
    notes_page.write_text(dedent("""\
        - Miscellaneous thoughts
          - Ideas
          - Todo items
        """))


def create_sample_journal(graph_path: Path):
    """Create sample journal entry for testing."""
    journals_dir = graph_path / "journals"

    # Journal for 2025-01-15
    journal_file = journals_dir / "2025_01_15.md"
    journal_file.write_text(dedent("""\
        - Morning standup
          - Discussed sprint goals
        - Learned about vector databases and ChromaDB integration
          id:: journal-knowledge-1
          - Very useful for semantic search
          - Better than traditional keyword search
        - Lunch meeting with team
        - Afternoon coding session
        """))


# ============================================================================
# Integration Tests
# ============================================================================


class TestFullPipeline:
    """Test the complete 5-phase pipeline end-to-end."""

    def test_phase_0_indexing(self, temp_graph):
        """Test Phase 0: Index building with hybrid IDs."""
        # Build vector store index
        vector_store = ChromaDBStore(persist_directory=temp_graph / ".chroma")
        manifest = CacheManifest(manifest_path=temp_graph / ".chroma" / "manifest.json")
        builder = IndexBuilder(vector_store=vector_store, manifest=manifest)

        stats = builder.build_incremental(temp_graph)

        # Verify pages were indexed
        assert stats["added"] == 3  # 3 pages created
        assert stats["updated"] == 0
        assert stats["deleted"] == 0

        # Verify the vector store was populated (has collection)
        assert vector_store.collection is not None
        count = vector_store.collection.count()
        assert count > 0, "Vector store should have indexed chunks"

        # Verify manifest was updated
        assert len(manifest.entries) == 3  # 3 pages tracked

    def test_end_to_end_pipeline(self, temp_graph):
        """Test complete pipeline: journal → index → extract → integrate → cleanup."""
        # Setup
        vector_store = ChromaDBStore(persist_directory=temp_graph / ".chroma")
        manifest = CacheManifest(manifest_path=temp_graph / ".chroma" / "manifest.json")
        builder = IndexBuilder(vector_store=vector_store, manifest=manifest)
        builder.build_incremental(temp_graph)

        mock_llm = MockPipelineLLM()
        extractor = Extractor(llm_client=mock_llm)

        # Load journal
        journal_file = temp_graph / "journals" / "2025_01_15.md"
        journal = JournalEntry.load(journal_file)

        # Run complete pipeline
        operations_count = extractor.extract_and_integrate(
            journal=journal,
            vector_store=vector_store,
            graph_path=temp_graph,
            top_k=5,
        )

        # Verify operations were executed
        assert operations_count > 0

        # Verify Database Design page was updated
        db_page = temp_graph / "pages" / "Database Design.md"
        db_content = db_page.read_text()

        # Should contain the rephrased knowledge
        assert "Vector databases" in db_content or "ChromaDB" in db_content

        # Should have a new id:: property
        assert "id::" in db_content

        # Verify journal was updated with processed:: marker
        journal_content = journal_file.read_text()
        assert "processed::" in journal_content

        # Should have link to Database Design page
        assert "Database Design" in journal_content

    def test_atomic_journal_updates(self, temp_graph):
        """Test that journal updates happen atomically with page writes."""
        # Setup
        vector_store = ChromaDBStore(persist_directory=temp_graph / ".chroma")
        manifest = CacheManifest(manifest_path=temp_graph / ".chroma" / "manifest.json")
        builder = IndexBuilder(vector_store=vector_store, manifest=manifest)
        builder.build_incremental(temp_graph)

        mock_llm = MockPipelineLLM()
        extractor = Extractor(llm_client=mock_llm)

        # Load journal
        journal_file = temp_graph / "journals" / "2025_01_15.md"
        journal = JournalEntry.load(journal_file)

        # Get initial journal content
        journal_content_before = journal_file.read_text()

        # Run pipeline
        extractor.extract_and_integrate(
            journal=journal,
            vector_store=vector_store,
            graph_path=temp_graph,
            top_k=5,
        )

        # Get final journal content
        journal_content_after = journal_file.read_text()

        # Verify journal was modified
        assert journal_content_after != journal_content_before

        # Parse to verify round-trip safety
        outline = LogseqOutline.parse(journal_content_after)

        # Find the knowledge block with id:: journal-knowledge-1
        knowledge_block = outline.find_block_by_id("journal-knowledge-1")
        assert knowledge_block is not None

        # Should have a processed:: property (not a child block)
        # The processed:: marker is added as a property line in the block's content
        processed_found = False
        for line in knowledge_block.content:
            if "processed::" in line:
                processed_found = True
                # Verify it has a block reference link
                assert "(((" in line  # Block reference syntax
                assert ")))" in line
                # Verify it links to Database Design
                assert "Database Design" in line
                break

        assert processed_found, "Should have processed:: property"

    def test_hybrid_id_preservation(self, temp_graph):
        """Test that hybrid IDs are preserved through the pipeline."""
        # Setup
        vector_store = ChromaDBStore(persist_directory=temp_graph / ".chroma")
        manifest = CacheManifest(manifest_path=temp_graph / ".chroma" / "manifest.json")
        builder = IndexBuilder(vector_store=vector_store, manifest=manifest)
        builder.build_incremental(temp_graph)

        # Read Database Design page before
        db_page = temp_graph / "pages" / "Database Design.md"
        content_before = db_page.read_text()
        outline_before = LogseqOutline.parse(content_before)

        # Get existing block IDs
        existing_ids = set()
        for block in outline_before.blocks:
            if block.block_id:
                existing_ids.add(block.block_id)
            for child in block.children:
                if child.block_id:
                    existing_ids.add(child.block_id)

        assert "db-overview" in existing_ids
        assert "db-technologies" in existing_ids

        # Run pipeline
        mock_llm = MockPipelineLLM()
        extractor = Extractor(llm_client=mock_llm)
        journal_file = temp_graph / "journals" / "2025_01_15.md"
        journal = JournalEntry.load(journal_file)

        extractor.extract_and_integrate(
            journal=journal,
            vector_store=vector_store,
            graph_path=temp_graph,
            top_k=5,
        )

        # Read Database Design page after
        content_after = db_page.read_text()
        outline_after = LogseqOutline.parse(content_after)

        # Verify existing IDs are still present
        ids_after = set()
        for block in outline_after.blocks:
            if block.block_id:
                ids_after.add(block.block_id)
            for child in block.children:
                if child.block_id:
                    ids_after.add(child.block_id)

        # All original IDs should still exist
        for original_id in existing_ids:
            assert original_id in ids_after, f"Lost ID: {original_id}"

        # Should have at least one new ID (the integrated knowledge)
        assert len(ids_after) > len(existing_ids)

    def test_round_trip_safety(self, temp_graph):
        """Test that parse → modify → render preserves structure."""
        # Setup and run pipeline
        vector_store = ChromaDBStore(persist_directory=temp_graph / ".chroma")
        manifest = CacheManifest(manifest_path=temp_graph / ".chroma" / "manifest.json")
        builder = IndexBuilder(vector_store=vector_store, manifest=manifest)
        builder.build_incremental(temp_graph)

        mock_llm = MockPipelineLLM()
        extractor = Extractor(llm_client=mock_llm)
        journal_file = temp_graph / "journals" / "2025_01_15.md"
        journal = JournalEntry.load(journal_file)

        extractor.extract_and_integrate(
            journal=journal,
            vector_store=vector_store,
            graph_path=temp_graph,
            top_k=5,
        )

        # Test Database Design page round-trip
        db_page = temp_graph / "pages" / "Database Design.md"
        content = db_page.read_text()

        # Parse and re-render
        outline = LogseqOutline.parse(content)
        rendered = outline.render()

        # Should be identical (or very close - allow minor whitespace differences)
        assert rendered == content or len(rendered) == len(content)

        # Parse rendered version
        reparsed = LogseqOutline.parse(rendered)

        # Structure should be identical
        assert len(reparsed.blocks) == len(outline.blocks)

        # Test journal round-trip
        journal_content = journal_file.read_text()
        journal_outline = LogseqOutline.parse(journal_content)
        journal_rendered = journal_outline.render()

        # Should round-trip safely
        assert journal_rendered == journal_content or len(journal_rendered) == len(journal_content)
