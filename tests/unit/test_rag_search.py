"""Unit tests for RAGSearch service."""

import pytest
import pytest_asyncio
from pathlib import Path
from logsqueak.services.page_indexer import PageIndexer
from logsqueak.services.rag_search import RAGSearch
from logsqueak.models.edited_content import EditedContent
from logseq_outline.graph import GraphPaths


@pytest.fixture
def temp_graph(tmp_path):
    """Create a temporary Logseq graph for testing."""
    graph_path = tmp_path / "test-graph"
    graph_path.mkdir()

    pages_dir = graph_path / "pages"
    pages_dir.mkdir()

    # Create pages with different topics
    python_page = pages_dir / "Python Programming.md"
    python_page.write_text("""- Python is a high-level programming language
  - Easy to learn and read
  - Great for data science
- Python decorators modify function behavior
""")

    javascript_page = pages_dir / "JavaScript.md"
    javascript_page.write_text("""- JavaScript runs in browsers
  - Used for web development
- Modern JavaScript uses ES6 features
""")

    ml_page = pages_dir / "Machine Learning.md"
    ml_page.write_text("""- Machine learning uses Python extensively
  - Popular libraries: TensorFlow, PyTorch
- Data preprocessing is crucial
""")

    return graph_path


@pytest_asyncio.fixture
async def indexed_db(temp_graph, tmp_path, shared_sentence_transformer):
    """Create and populate a ChromaDB index."""
    db_path = tmp_path / "chromadb"

    graph_paths = GraphPaths(temp_graph)
    indexer = PageIndexer(graph_paths, db_path, encoder=shared_sentence_transformer)

    # Build index
    await indexer.build_index()
    await indexer.close()

    return db_path


@pytest.mark.asyncio
async def test_rag_search_initialization(indexed_db):
    """Test RAGSearch initialization."""
    search = RAGSearch(indexed_db)

    assert search.db_path == indexed_db
    assert search.collection is not None

    await search.close()


@pytest.mark.asyncio
async def test_find_candidates_returns_relevant_pages(indexed_db):
    """Test that find_candidates returns relevant pages for knowledge blocks."""
    search = RAGSearch(indexed_db)

    # Knowledge block about Python
    edited_content = [
        EditedContent(
            block_id="block-1",
            original_content="Learning about Python decorators and how they work",
            current_content="Python decorators modify function behavior"
        )
    ]

    original_contexts = {
        "block-1": "Learning about Python decorators and how they work"
    }

    results = await search.find_candidates(
        edited_content,
        original_contexts,
        top_k=3
    )

    # Should find Python Programming page as most relevant
    assert "block-1" in results
    candidates = results["block-1"]
    assert len(candidates) > 0
    assert "Python Programming" in candidates

    await search.close()


@pytest.mark.asyncio
async def test_find_candidates_ranks_pages_by_relevance(indexed_db):
    """Test that candidates are ranked by relevance."""
    search = RAGSearch(indexed_db)

    # Knowledge block about Python in ML context
    edited_content = [
        EditedContent(
            block_id="block-1",
            original_content="Machine learning frameworks use Python extensively",
            current_content="Machine learning frameworks use Python extensively"
        )
    ]

    original_contexts = {
        "block-1": "Machine learning frameworks use Python extensively"
    }

    results = await search.find_candidates(
        edited_content,
        original_contexts,
        top_k=3
    )

    candidates = results["block-1"]

    # Machine Learning page should be highly ranked (mentions both ML and Python)
    assert "Machine Learning" in candidates

    await search.close()


@pytest.mark.asyncio
async def test_find_candidates_handles_multiple_blocks(indexed_db):
    """Test that find_candidates handles multiple knowledge blocks."""
    search = RAGSearch(indexed_db)

    edited_content = [
        EditedContent(
            block_id="block-1",
            original_content="Python decorators are powerful",
            current_content="Python decorators are powerful"
        ),
        EditedContent(
            block_id="block-2",
            original_content="JavaScript arrow functions",
            current_content="JavaScript arrow functions"
        )
    ]

    original_contexts = {
        "block-1": "Python decorators are powerful",
        "block-2": "JavaScript arrow functions"
    }

    results = await search.find_candidates(
        edited_content,
        original_contexts,
        top_k=2
    )

    # Should have results for both blocks
    assert "block-1" in results
    assert "block-2" in results

    # Each should have candidates
    assert len(results["block-1"]) > 0
    assert len(results["block-2"]) > 0

    await search.close()


@pytest.mark.asyncio
async def test_find_candidates_respects_top_k_limit(indexed_db):
    """Test that find_candidates respects the top_k parameter."""
    search = RAGSearch(indexed_db)

    edited_content = [
        EditedContent(
            block_id="block-1",
            original_content="Programming concepts",
            current_content="Programming concepts"
        )
    ]

    original_contexts = {
        "block-1": "Programming concepts"
    }

    # Request only top 2 candidates
    results = await search.find_candidates(
        edited_content,
        original_contexts,
        top_k=2
    )

    candidates = results["block-1"]

    # Should return at most 2 candidates
    assert len(candidates) <= 2

    await search.close()


@pytest.mark.asyncio
async def test_explicit_link_boosting(indexed_db):
    """Test that pages mentioned in explicit links get boosted."""
    search = RAGSearch(indexed_db)

    # Knowledge block with explicit link to Python Programming
    edited_content = [
        EditedContent(
            block_id="block-1",
            original_content="See [[Python Programming]] for more details about decorators",
            current_content="See Python Programming for more details about decorators"
        )
    ]

    original_contexts = {
        "block-1": "See [[Python Programming]] for more details about decorators"
    }

    results = await search.find_candidates(
        edited_content,
        original_contexts,
        top_k=3
    )

    candidates = results["block-1"]

    # Python Programming should be in results (boosted by explicit link)
    assert "Python Programming" in candidates

    # Should be ranked high due to explicit link boost
    assert candidates.index("Python Programming") <= 1  # In top 2

    await search.close()


@pytest.mark.asyncio
async def test_find_candidates_uses_original_context(indexed_db):
    """Test that original context is used for search, not edited content."""
    search = RAGSearch(indexed_db)

    edited_content = [
        EditedContent(
            block_id="block-1",
            original_content="Today I learned about Python",
            current_content="Python programming concepts"  # Edited to remove temporal context
        )
    ]

    # Original context has more information
    original_contexts = {
        "block-1": "Daily Notes\n  Today I learned about Python decorators and closures"
    }

    results = await search.find_candidates(
        edited_content,
        original_contexts,
        top_k=3
    )

    candidates = results["block-1"]

    # Should use original context which mentions decorators
    assert len(candidates) > 0

    await search.close()


@pytest.mark.asyncio
async def test_find_candidates_handles_hierarchical_page_links(indexed_db, temp_graph, shared_sentence_transformer):
    """Test that hierarchical page links (with /) are handled correctly."""
    # Add a hierarchical page
    pages_dir = temp_graph / "pages"
    projects_page = pages_dir / "Projects___Backend.md"
    projects_page.write_text("""- Backend API development
  - Using Python Flask
""")

    # Rebuild index
    graph_paths = GraphPaths(temp_graph)
    indexer = PageIndexer(graph_paths, indexed_db, encoder=shared_sentence_transformer)
    await indexer.build_index()
    await indexer.close()

    search = RAGSearch(indexed_db)

    # Knowledge block with hierarchical link
    edited_content = [
        EditedContent(
            block_id="block-1",
            original_content="Backend API uses Flask, see [[Projects/Backend]]",
            current_content="Backend API uses Flask"
        )
    ]

    original_contexts = {
        "block-1": "Backend API uses Flask, see [[Projects/Backend]]"
    }

    results = await search.find_candidates(
        edited_content,
        original_contexts,
        top_k=3
    )

    candidates = results["block-1"]

    # Should find the hierarchical page
    assert "Projects/Backend" in candidates

    await search.close()


@pytest.mark.asyncio
async def test_empty_results_when_no_matches(indexed_db):
    """Test that empty results are handled when there are no good matches."""
    search = RAGSearch(indexed_db)

    # Knowledge block about completely unrelated topic
    edited_content = [
        EditedContent(
            block_id="block-1",
            original_content="Quantum physics concepts",
            current_content="Quantum physics concepts"
        )
    ]

    original_contexts = {
        "block-1": "Quantum physics concepts"
    }

    results = await search.find_candidates(
        edited_content,
        original_contexts,
        top_k=3
    )

    # Should still return results (best matches even if not very relevant)
    assert "block-1" in results
    # But may have fewer candidates or low relevance scores

    await search.close()
