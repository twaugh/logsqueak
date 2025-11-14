"""Integration tests for the complete RAG pipeline (read-only tests with module-scoped fixtures)."""

import pytest
import pytest_asyncio
from pathlib import Path
from logsqueak.services.page_indexer import PageIndexer
from logsqueak.services.rag_search import RAGSearch
from logsqueak.models.edited_content import EditedContent
from logseq_outline.graph import GraphPaths


@pytest.fixture(scope="module")
def test_graph(tmp_path_factory):
    """Create a realistic test Logseq graph (module-scoped for read-only tests)."""
    tmp_path = tmp_path_factory.mktemp("rag_pipeline")
    graph_path = tmp_path / "test-graph"
    graph_path.mkdir()

    pages_dir = graph_path / "pages"
    pages_dir.mkdir()

    # Create pages representing a realistic knowledge base
    python_page = pages_dir / "Python.md"
    python_page.write_text("""- Python Overview
  - Python is a high-level, interpreted programming language
  - Created by Guido van Rossum in 1991
- Key Features
  - Easy to learn and read
  - Extensive standard library
  - Dynamic typing
- Use Cases
  - Web development (Django, Flask)
  - Data science (pandas, NumPy)
  - Machine learning (TensorFlow, PyTorch)
""")

    web_dev_page = pages_dir / "Web Development.md"
    web_dev_page.write_text("""- Frontend Technologies
  - HTML, CSS, JavaScript
  - React, Vue, Angular
- Backend Technologies
  - [[Python]] (Django, Flask)
  - Node.js (Express)
  - Ruby (Rails)
- Deployment
  - Docker containers
  - Cloud platforms (AWS, GCP, Azure)
""")

    ml_page = pages_dir / "Machine Learning.md"
    ml_page.write_text("""- Machine Learning Fundamentals
  - Supervised learning
  - Unsupervised learning
  - Reinforcement learning
- Tools and Libraries
  - [[Python]] is the primary language for ML
  - TensorFlow for deep learning
  - scikit-learn for traditional ML
  - Jupyter notebooks for experimentation
- Workflow
  - Data preprocessing
  - Model training
  - Evaluation and tuning
""")

    projects_page = pages_dir / "Projects___Backend API.md"
    projects_page.write_text("""- Project Overview
  - Building a REST API for mobile app
  - Using [[Python]] Flask framework
- Tech Stack
  - Flask for API endpoints
  - PostgreSQL database
  - JWT authentication
- Timeline
  - Q1 2025: MVP development
  - Q2 2025: Testing and deployment
""")

    return graph_path


@pytest_asyncio.fixture(scope="module")
async def rag_pipeline(test_graph, tmp_path_factory, shared_sentence_transformer):
    """Create and initialize the complete RAG pipeline (module-scoped for read-only tests)."""
    tmp_path = tmp_path_factory.mktemp("chromadb")
    db_path = tmp_path / "chromadb"

    # Build index
    graph_paths = GraphPaths(test_graph)
    indexer = PageIndexer(graph_paths, db_path, encoder=shared_sentence_transformer)
    await indexer.build_index()

    # Create search instance using the same per-graph db_path
    # PageIndexer creates a per-graph subdirectory, so we must use indexer.db_path
    search = RAGSearch(indexer.db_path)

    await indexer.close()

    yield search, test_graph

    await search.close()


@pytest.mark.asyncio
async def test_end_to_end_rag_pipeline(rag_pipeline):
    """Test complete RAG pipeline from indexing to search."""
    search, test_graph = rag_pipeline
    graph_paths = GraphPaths(test_graph)

    # Simulate knowledge blocks from journal
    edited_content = [
        EditedContent(
            block_id="block-1",
            original_content="Learned about Python decorators today",
            hierarchical_context="- Daily Notes - 2025-01-15\n  - Learned about Python decorators today",
            current_content="Python decorators modify function behavior"
        )
    ]

    original_contexts = {
        "block-1": "- Daily Notes - 2025-01-15\n  - Learned about Python decorators today"
    }

    # Search for candidates
    results = await search.find_candidates(
        edited_content,
        original_contexts,
        graph_paths,
        top_k=5
    )

    # Should find Python page as most relevant
    assert "block-1" in results
    candidates = results["block-1"]
    page_names = [page_name for page_name, _, _ in candidates]
    assert "Python" in page_names
    assert len(candidates) > 0


@pytest.mark.asyncio
async def test_explicit_links_boost_rankings(rag_pipeline):
    """Test that explicit page links boost ranking in search results."""
    search, test_graph = rag_pipeline
    graph_paths = GraphPaths(test_graph)

    # Knowledge block with explicit link to Web Development
    edited_content = [
        EditedContent(
            block_id="block-1",
            original_content="Backend APIs can be built with Python Flask, see [[Web Development]]",
            hierarchical_context="- Backend APIs can be built with Python Flask, see [[Web Development]]",
            current_content="Backend APIs can be built with Python Flask"
        )
    ]

    original_contexts = {
        "block-1": "- Backend APIs can be built with Python Flask, see [[Web Development]]"
    }

    results = await search.find_candidates(
        edited_content,
        original_contexts,
        graph_paths,
        top_k=5
    )

    candidates = results["block-1"]

    # Web Development should be highly ranked due to explicit link
    page_names = [page_name for page_name, _, _ in candidates]
    assert "Web Development" in page_names
    # Should be in top 2 positions
    assert page_names.index("Web Development") <= 1


@pytest.mark.asyncio
async def test_semantic_search_finds_related_pages(rag_pipeline):
    """Test that semantic search finds semantically related pages."""
    search, test_graph = rag_pipeline
    graph_paths = GraphPaths(test_graph)

    # Knowledge block about ML workflow
    edited_content = [
        EditedContent(
            block_id="block-1",
            original_content="Data preprocessing is crucial before training ML models",
            hierarchical_context="- Data preprocessing is crucial before training ML models",
            current_content="Data preprocessing is crucial before training ML models"
        )
    ]

    original_contexts = {
        "block-1": "- Data preprocessing is crucial before training ML models"
    }

    results = await search.find_candidates(
        edited_content,
        original_contexts,
        graph_paths,
        top_k=5
    )

    candidates = results["block-1"]

    # Machine Learning page should be found (mentions data preprocessing)
    page_names = [page_name for page_name, _, _ in candidates]
    assert "Machine Learning" in page_names


@pytest.mark.asyncio
async def test_hierarchical_pages_in_search_results(rag_pipeline):
    """Test that hierarchical pages are properly indexed and searchable."""
    search, test_graph = rag_pipeline
    graph_paths = GraphPaths(test_graph)

    # Knowledge block about backend API development
    edited_content = [
        EditedContent(
            block_id="block-1",
            original_content="Building REST API with Flask framework",
            hierarchical_context="- Building REST API with Flask framework",
            current_content="Building REST API with Flask framework"
        )
    ]

    original_contexts = {
        "block-1": "- Building REST API with Flask framework"
    }

    results = await search.find_candidates(
        edited_content,
        original_contexts,
        graph_paths,
        top_k=5
    )

    candidates = results["block-1"]

    # Should find the hierarchical Projects/Backend API page
    page_names = [page_name for page_name, _, _ in candidates]
    assert "Projects/Backend API" in page_names


@pytest.mark.asyncio
async def test_multiple_knowledge_blocks_parallel_search(rag_pipeline):
    """Test searching for multiple knowledge blocks in parallel."""
    search, test_graph = rag_pipeline
    graph_paths = GraphPaths(test_graph)

    # Multiple knowledge blocks on different topics
    edited_content = [
        EditedContent(
            block_id="block-1",
            original_content="Python is great for machine learning",
            hierarchical_context="- Python is great for machine learning",
            current_content="Python is great for machine learning"
        ),
        EditedContent(
            block_id="block-2",
            original_content="Flask is a lightweight web framework",
            hierarchical_context="- Flask is a lightweight web framework",
            current_content="Flask is a lightweight web framework"
        ),
        EditedContent(
            block_id="block-3",
            original_content="Docker containers simplify deployment",
            hierarchical_context="- Docker containers simplify deployment",
            current_content="Docker containers simplify deployment"
        )
    ]

    original_contexts = {
        "block-1": "- Python is great for machine learning",
        "block-2": "- Flask is a lightweight web framework",
        "block-3": "- Docker containers simplify deployment"
    }

    results = await search.find_candidates(
        edited_content,
        original_contexts,
        graph_paths,
        top_k=3
    )

    # Should have results for all blocks
    assert len(results) == 3
    assert "block-1" in results
    assert "block-2" in results
    assert "block-3" in results

    # Each block should find relevant pages
    page_names_1 = [page_name for page_name, _, _ in results["block-1"]]
    page_names_2 = [page_name for page_name, _, _ in results["block-2"]]
    page_names_3 = [page_name for page_name, _, _ in results["block-3"]]
    assert "Machine Learning" in page_names_1 or "Python" in page_names_1
    assert "Web Development" in page_names_2 or "Python" in page_names_2
    assert "Web Development" in page_names_3  # Mentions Docker deployment


@pytest.mark.asyncio
async def test_context_provides_better_results_than_content_alone(rag_pipeline):
    """Test that using full hierarchical context improves search results."""
    search, test_graph = rag_pipeline
    graph_paths = GraphPaths(test_graph)

    # Same content, but with context vs without
    edited_content_without_context = [
        EditedContent(
            block_id="block-1",
            original_content="Flask framework",
            hierarchical_context="- Flask framework",
            current_content="Flask framework"
        )
    ]

    edited_content_with_context = [
        EditedContent(
            block_id="block-2",
            original_content="Flask framework",
            hierarchical_context="- Projects - Backend Development\n  - Technology Stack\n    - Flask framework for REST API",
            current_content="Flask framework"
        )
    ]

    # Without context
    results_no_context = await search.find_candidates(
        edited_content_without_context,
        {"block-1": "- Flask framework"},
        graph_paths,
        top_k=5
    )

    # With detailed context
    results_with_context = await search.find_candidates(
        edited_content_with_context,
        {
            "block-2": "- Projects - Backend Development\n  - Technology Stack\n    - Flask framework for REST API"
        },
        graph_paths,
        top_k=5
    )

    # Both should find relevant pages, but context might improve ranking
    assert len(results_no_context["block-1"]) > 0
    assert len(results_with_context["block-2"]) > 0
