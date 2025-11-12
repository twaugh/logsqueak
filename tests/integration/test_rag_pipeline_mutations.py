"""Integration tests for the complete RAG pipeline (mutation tests with function-scoped fixtures)."""

import pytest
import pytest_asyncio
from pathlib import Path
from logsqueak.services.page_indexer import PageIndexer
from logsqueak.services.rag_search import RAGSearch
from logsqueak.models.edited_content import EditedContent
from logseq_outline.graph import GraphPaths


@pytest.fixture
def test_graph(tmp_path):
    """Create a realistic test Logseq graph."""
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


@pytest.mark.asyncio
async def test_incremental_index_update_affects_search(test_graph, tmp_path, shared_sentence_transformer):
    """Test that updating the index affects search results."""
    db_path = tmp_path / "chromadb"

    # Initial index build
    graph_paths = GraphPaths(test_graph)
    indexer = PageIndexer(graph_paths, db_path, encoder=shared_sentence_transformer)
    await indexer.build_index()
    await indexer.close()

    # Initial search
    search = RAGSearch(db_path)

    edited_content = [
        EditedContent(
            block_id="block-1",
            original_content="Quantum computing concepts",
            hierarchical_context="- Quantum computing concepts",
            current_content="Quantum computing concepts"
        )
    ]

    original_contexts = {
        "block-1": "- Quantum computing concepts"
    }

    initial_results = await search.find_candidates(
        edited_content,
        original_contexts,
        graph_paths,
        top_k=3
    )

    await search.close()

    # Add a new page about quantum computing
    pages_dir = test_graph / "pages"
    quantum_page = pages_dir / "Quantum Computing.md"
    quantum_page.write_text("""- Quantum Computing Basics
  - Qubits and superposition
  - Quantum entanglement
- Applications
  - Cryptography
  - Drug discovery
""")

    # Rebuild index
    indexer = PageIndexer(graph_paths, db_path, encoder=shared_sentence_transformer)
    await indexer.build_index()
    await indexer.close()

    # Search again
    search = RAGSearch(db_path)
    updated_results = await search.find_candidates(
        edited_content,
        original_contexts,
        graph_paths,
        top_k=3
    )

    # Should now find the Quantum Computing page
    assert "Quantum Computing" in updated_results["block-1"]

    await search.close()
