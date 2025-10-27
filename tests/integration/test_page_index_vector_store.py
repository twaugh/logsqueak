"""Integration tests for PageIndex with VectorStore backend (M2.6)."""

from pathlib import Path
from textwrap import dedent
from unittest.mock import Mock

import pytest

from logsqueak.models.page import PageIndex


@pytest.fixture
def mock_embedding_model():
    """Create a mock embedding model for testing."""
    mock_model = Mock()
    # Return 384-dimensional embeddings (all-MiniLM-L6-v2 dimension)
    def encode_mock(texts, convert_to_numpy=False):
        import numpy as np
        # Return simple embeddings based on text length
        if isinstance(texts, str):
            return np.array([float(len(texts)) / 100.0] * 384)
        return np.array([[float(len(t)) / 100.0] * 384 for t in texts])

    mock_model.encode = encode_mock
    return mock_model


class TestPageIndexVectorStore:
    """Tests for PageIndex.build_with_vector_store()."""

    def test_build_with_vector_store_empty_graph(self, tmp_path, mock_embedding_model):
        """Test building index for empty graph."""
        graph_path = tmp_path / "graph"
        graph_path.mkdir()
        pages_dir = graph_path / "pages"
        pages_dir.mkdir()

        # Build index with mock model
        vector_store_path = tmp_path / "chroma"
        index = PageIndex.build_with_vector_store(
            graph_path, vector_store_path, embedding_model=mock_embedding_model
        )

        assert len(index.pages) == 0
        assert hasattr(index, "_vector_store")
        assert hasattr(index, "_graph_path")

        index._vector_store.close()

    def test_build_with_vector_store_with_pages(self, tmp_path, mock_embedding_model):
        """Test building index with multiple pages."""
        graph_path = tmp_path / "graph"
        pages_dir = graph_path / "pages"
        pages_dir.mkdir(parents=True)

        # Create test pages
        (pages_dir / "Page A.md").write_text("- Block 1\n- Block 2", encoding="utf-8")
        (pages_dir / "Page B.md").write_text("- Content B", encoding="utf-8")

        # Build index with mock model
        vector_store_path = tmp_path / "chroma"
        index = PageIndex.build_with_vector_store(
            graph_path, vector_store_path, embedding_model=mock_embedding_model
        )

        assert len(index.pages) == 2
        page_names = {p.name for p in index.pages}
        assert page_names == {"Page A", "Page B"}

        index._vector_store.close()

    def test_find_similar_with_vector_store(self, tmp_path, mock_embedding_model):
        """Test find_similar() uses VectorStore when available."""
        graph_path = tmp_path / "graph"
        pages_dir = graph_path / "pages"
        pages_dir.mkdir(parents=True)

        # Create test pages with distinctive content
        (pages_dir / "Python.md").write_text(
            "- Python is a programming language\n- Used for data science",
            encoding="utf-8",
        )
        (pages_dir / "Cooking.md").write_text(
            "- Recipes and cooking tips\n- Baking bread",
            encoding="utf-8",
        )

        # Build index with mock model
        vector_store_path = tmp_path / "chroma"
        index = PageIndex.build_with_vector_store(
            graph_path, vector_store_path, embedding_model=mock_embedding_model
        )

        # Query for similar pages
        query = "programming and software development"
        results = index.find_similar(query, top_k=2)

        # Should return results (exact ordering depends on mock embeddings)
        assert len(results) <= 2
        assert all(isinstance(page_name, str) or hasattr(page_name, "name") for page_name, _ in results)
        assert all(isinstance(score, float) for _, score in results)

        index._vector_store.close()

    def test_incremental_updates(self, tmp_path, mock_embedding_model):
        """Test that incremental indexing detects updates."""
        graph_path = tmp_path / "graph"
        pages_dir = graph_path / "pages"
        pages_dir.mkdir(parents=True)

        # Create initial page
        page_file = pages_dir / "Test.md"
        page_file.write_text("- Initial content", encoding="utf-8")

        # First build
        vector_store_path = tmp_path / "chroma"
        index1 = PageIndex.build_with_vector_store(
            graph_path, vector_store_path, embedding_model=mock_embedding_model
        )
        assert len(index1.pages) == 1
        index1._vector_store.close()

        # Modify page
        import time
        time.sleep(0.01)  # Ensure mtime changes
        page_file.write_text("- Initial content\n- Updated content", encoding="utf-8")

        # Second build (should detect update)
        index2 = PageIndex.build_with_vector_store(
            graph_path, vector_store_path, embedding_model=mock_embedding_model
        )
        assert len(index2.pages) == 1
        index2._vector_store.close()

    def test_default_cache_path(self, tmp_path, mock_embedding_model):
        """Test that default cache path is used when not specified."""
        graph_path = tmp_path / "graph"
        pages_dir = graph_path / "pages"
        pages_dir.mkdir(parents=True)

        # Create test page
        (pages_dir / "Test.md").write_text("- Content", encoding="utf-8")

        # Build without specifying vector_store_path
        index = PageIndex.build_with_vector_store(
            graph_path, embedding_model=mock_embedding_model
        )

        # Should use default path
        assert index.cache_dir == Path.home() / ".cache" / "logsqueak" / "chroma"

        index._vector_store.close()

    def test_page_aggregation(self, tmp_path, mock_embedding_model):
        """Test that block-level results are aggregated by page."""
        graph_path = tmp_path / "graph"
        pages_dir = graph_path / "pages"
        pages_dir.mkdir(parents=True)

        # Create page with multiple blocks
        (pages_dir / "Multi.md").write_text(
            dedent("""\
                - Block 1: Python programming
                - Block 2: Data science
                - Block 3: Machine learning"""),
            encoding="utf-8",
        )

        # Build index with mock model
        vector_store_path = tmp_path / "chroma"
        index = PageIndex.build_with_vector_store(
            graph_path, vector_store_path, embedding_model=mock_embedding_model
        )

        # Query
        results = index.find_similar("programming", top_k=5)

        # Should return page (not individual blocks)
        assert len(results) == 1
        page, score = results[0]
        assert page.name == "Multi"
        assert 0.0 <= score <= 1.0

        index._vector_store.close()
