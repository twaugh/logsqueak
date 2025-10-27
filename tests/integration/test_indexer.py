"""Integration tests for incremental index builder."""

from pathlib import Path
from textwrap import dedent
from unittest.mock import Mock

import pytest

from logsqueak.rag.indexer import IndexBuilder
from logsqueak.rag.manifest import CacheManifest
from logsqueak.rag.vector_store import ChromaDBStore


@pytest.fixture
def mock_embedding_model():
    """Create a mock embedding model for testing."""
    mock_model = Mock()
    # Return 384-dimensional embeddings (all-MiniLM-L6-v2 dimension)
    def encode_mock(texts, convert_to_numpy=False):
        import numpy as np
        # Return simple embeddings based on text length
        return np.array([[float(len(t)) / 100.0] * 384 for t in texts])

    mock_model.encode = encode_mock
    return mock_model


class TestIndexBuilderIncremental:
    """Tests for incremental index building."""

    def test_build_incremental_empty_graph(self, tmp_path, mock_embedding_model):
        """Test building index for empty graph."""
        graph_path = tmp_path / "graph"
        graph_path.mkdir()
        pages_dir = graph_path / "pages"
        pages_dir.mkdir()

        # Setup
        manifest_path = tmp_path / "manifest.json"
        manifest = CacheManifest(manifest_path)
        vector_store = ChromaDBStore(tmp_path / "chroma")

        builder = IndexBuilder(vector_store, manifest, mock_embedding_model)

        # Build
        stats = builder.build_incremental(graph_path)

        assert stats == {'added': 0, 'updated': 0, 'deleted': 0, 'unchanged': 0}

        vector_store.close()

    def test_build_incremental_new_page(self, tmp_path, mock_embedding_model):
        """Test adding a new page."""
        graph_path = tmp_path / "graph"
        pages_dir = graph_path / "pages"
        pages_dir.mkdir(parents=True)

        # Create a page
        page_file = pages_dir / "Test Page.md"
        page_file.write_text("- Block 1\n- Block 2", encoding="utf-8")

        # Setup
        manifest_path = tmp_path / "manifest.json"
        manifest = CacheManifest(manifest_path)
        vector_store = ChromaDBStore(tmp_path / "chroma")

        builder = IndexBuilder(vector_store, manifest, mock_embedding_model)

        # Build
        stats = builder.build_incremental(graph_path)

        assert stats['added'] == 1
        assert stats['updated'] == 0
        assert stats['deleted'] == 0

        # Verify manifest updated
        assert manifest.has_page("Test Page")

        vector_store.close()

    def test_build_incremental_unchanged_page(self, tmp_path, mock_embedding_model):
        """Test that unchanged pages are not reprocessed."""
        graph_path = tmp_path / "graph"
        pages_dir = graph_path / "pages"
        pages_dir.mkdir(parents=True)

        # Create a page
        page_file = pages_dir / "Test Page.md"
        page_file.write_text("- Block 1", encoding="utf-8")

        # Setup
        manifest_path = tmp_path / "manifest.json"
        manifest = CacheManifest(manifest_path)
        vector_store = ChromaDBStore(tmp_path / "chroma")

        builder = IndexBuilder(vector_store, manifest, mock_embedding_model)

        # First build (addition)
        stats1 = builder.build_incremental(graph_path)
        assert stats1['added'] == 1

        # Second build (unchanged)
        stats2 = builder.build_incremental(graph_path)
        assert stats2['added'] == 0
        assert stats2['unchanged'] == 1

        vector_store.close()

    def test_build_incremental_updated_page(self, tmp_path, mock_embedding_model):
        """Test updating an existing page."""
        graph_path = tmp_path / "graph"
        pages_dir = graph_path / "pages"
        pages_dir.mkdir(parents=True)

        # Create a page
        page_file = pages_dir / "Test Page.md"
        page_file.write_text("- Block 1", encoding="utf-8")

        # Setup
        manifest_path = tmp_path / "manifest.json"
        manifest = CacheManifest(manifest_path)
        vector_store = ChromaDBStore(tmp_path / "chroma")

        builder = IndexBuilder(vector_store, manifest, mock_embedding_model)

        # First build
        builder.build_incremental(graph_path)

        # Modify page
        import time
        time.sleep(0.01)  # Ensure mtime changes
        page_file.write_text("- Block 1\n- Block 2", encoding="utf-8")

        # Second build (should detect update)
        stats = builder.build_incremental(graph_path)

        assert stats['updated'] == 1
        assert stats['added'] == 0

        vector_store.close()

    def test_build_incremental_deleted_page(self, tmp_path, mock_embedding_model):
        """Test deleting a page."""
        graph_path = tmp_path / "graph"
        pages_dir = graph_path / "pages"
        pages_dir.mkdir(parents=True)

        # Create a page
        page_file = pages_dir / "Test Page.md"
        page_file.write_text("- Block 1", encoding="utf-8")

        # Setup
        manifest_path = tmp_path / "manifest.json"
        manifest = CacheManifest(manifest_path)
        vector_store = ChromaDBStore(tmp_path / "chroma")

        builder = IndexBuilder(vector_store, manifest, mock_embedding_model)

        # First build
        builder.build_incremental(graph_path)

        # Delete page
        page_file.unlink()

        # Second build (should detect deletion)
        stats = builder.build_incremental(graph_path)

        assert stats['deleted'] == 1

        # Verify removed from manifest
        assert not manifest.has_page("Test Page")

        vector_store.close()

    def test_build_incremental_multiple_changes(self, tmp_path, mock_embedding_model):
        """Test handling multiple types of changes at once."""
        graph_path = tmp_path / "graph"
        pages_dir = graph_path / "pages"
        pages_dir.mkdir(parents=True)

        # Create initial pages
        (pages_dir / "Page A.md").write_text("- Content A", encoding="utf-8")
        (pages_dir / "Page B.md").write_text("- Content B", encoding="utf-8")

        # Setup
        manifest_path = tmp_path / "manifest.json"
        manifest = CacheManifest(manifest_path)
        vector_store = ChromaDBStore(tmp_path / "chroma")

        builder = IndexBuilder(vector_store, manifest, mock_embedding_model)

        # First build
        stats1 = builder.build_incremental(graph_path)
        assert stats1['added'] == 2

        # Make changes:
        # - Delete Page A
        # - Update Page B
        # - Add Page C
        import time
        time.sleep(0.01)
        (pages_dir / "Page A.md").unlink()
        (pages_dir / "Page B.md").write_text("- Content B updated", encoding="utf-8")
        (pages_dir / "Page C.md").write_text("- Content C", encoding="utf-8")

        # Second build
        stats2 = builder.build_incremental(graph_path)

        assert stats2['deleted'] == 1  # Page A
        assert stats2['updated'] == 1  # Page B
        assert stats2['added'] == 1    # Page C

        vector_store.close()

    def test_build_incremental_empty_page(self, tmp_path, mock_embedding_model):
        """Test handling empty page."""
        graph_path = tmp_path / "graph"
        pages_dir = graph_path / "pages"
        pages_dir.mkdir(parents=True)

        # Create empty page
        page_file = pages_dir / "Empty.md"
        page_file.write_text("", encoding="utf-8")

        # Setup
        manifest_path = tmp_path / "manifest.json"
        manifest = CacheManifest(manifest_path)
        vector_store = ChromaDBStore(tmp_path / "chroma")

        builder = IndexBuilder(vector_store, manifest, mock_embedding_model)

        # Build
        stats = builder.build_incremental(graph_path)

        assert stats['added'] == 1

        # Manifest should be updated even for empty page
        assert manifest.has_page("Empty")

        vector_store.close()

    def test_build_incremental_page_with_nested_blocks(self, tmp_path, mock_embedding_model):
        """Test indexing page with nested block structure."""
        graph_path = tmp_path / "graph"
        pages_dir = graph_path / "pages"
        pages_dir.mkdir(parents=True)

        # Create page with nested blocks
        page_file = pages_dir / "Nested.md"
        content = dedent(
            """\
            - Parent
              - Child 1
              - Child 2
                - Grandchild"""
        )
        page_file.write_text(content, encoding="utf-8")

        # Setup
        manifest_path = tmp_path / "manifest.json"
        manifest = CacheManifest(manifest_path)
        vector_store = ChromaDBStore(tmp_path / "chroma")

        builder = IndexBuilder(vector_store, manifest, mock_embedding_model)

        # Build
        stats = builder.build_incremental(graph_path)

        assert stats['added'] == 1

        # Query vector store to verify chunks were added
        # Use a dummy query embedding
        query_embedding = [0.0] * 384  # all-MiniLM-L6-v2 dimension
        result_ids, _, _ = vector_store.query(query_embedding, n_results=10)

        # Should have 4 chunks (Parent, Child 1, Child 2, Grandchild)
        assert len(result_ids) == 4

        vector_store.close()
