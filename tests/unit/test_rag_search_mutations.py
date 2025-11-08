"""Unit tests for RAGSearch service (mutation tests with function-scoped fixtures)."""

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
            hierarchical_context="- Development notes\n  - Web frameworks\n    - Backend API uses Flask, see [[Projects/Backend]]",
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
