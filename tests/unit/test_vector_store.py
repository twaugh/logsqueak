"""Unit tests for VectorStore abstraction and ChromaDB implementation."""

from pathlib import Path

import pytest

from logsqueak.rag.vector_store import ChromaDBStore


class TestChromaDBStore:
    """Tests for ChromaDB vector store implementation."""

    def test_initialization(self, tmp_path):
        """Test ChromaDB store initialization."""
        persist_dir = tmp_path / "chroma"

        store = ChromaDBStore(persist_directory=persist_dir)

        assert store.persist_directory == persist_dir
        assert store.collection_name == "logsqueak_blocks"
        assert persist_dir.exists()

        store.close()

    def test_custom_collection_name(self, tmp_path):
        """Test initialization with custom collection name."""
        persist_dir = tmp_path / "chroma"

        store = ChromaDBStore(
            persist_directory=persist_dir,
            collection_name="test_collection"
        )

        assert store.collection_name == "test_collection"

        store.close()

    def test_add_embeddings(self, tmp_path):
        """Test adding embeddings to store."""
        persist_dir = tmp_path / "chroma"
        store = ChromaDBStore(persist_directory=persist_dir)

        ids = ["id-1", "id-2"]
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        metadatas = [
            {"page_name": "Page A", "block_content": "Content A"},
            {"page_name": "Page B", "block_content": "Content B"},
        ]
        documents = ["- Content A", "- Content B"]

        # Should not raise
        store.add(ids, embeddings, metadatas, documents)

        store.close()

    def test_add_empty_list(self, tmp_path):
        """Test adding empty list does nothing."""
        persist_dir = tmp_path / "chroma"
        store = ChromaDBStore(persist_directory=persist_dir)

        # Should not raise
        store.add([], [], [], [])

        store.close()

    def test_query_embeddings(self, tmp_path):
        """Test querying for similar embeddings."""
        persist_dir = tmp_path / "chroma"
        store = ChromaDBStore(persist_directory=persist_dir)

        # Add some embeddings
        ids = ["id-1", "id-2", "id-3"]
        embeddings = [
            [1.0, 0.0, 0.0],  # Similar to query
            [0.9, 0.1, 0.0],  # Very similar to query
            [0.0, 1.0, 0.0],  # Different from query
        ]
        metadatas = [
            {"page_name": "Page A"},
            {"page_name": "Page B"},
            {"page_name": "Page C"},
        ]
        documents = ["Content A", "Content B", "Content C"]

        store.add(ids, embeddings, metadatas, documents)

        # Query with embedding similar to id-1 and id-2
        query_embedding = [0.95, 0.05, 0.0]
        result_ids, distances, result_metadatas = store.query(
            query_embedding, n_results=2
        )

        assert len(result_ids) == 2
        assert len(distances) == 2
        assert len(result_metadatas) == 2

        # Should return id-2 first (most similar), then id-1
        assert "id-2" in result_ids
        assert "id-1" in result_ids

        store.close()

    def test_query_with_metadata_filter(self, tmp_path):
        """Test querying with metadata filter."""
        persist_dir = tmp_path / "chroma"
        store = ChromaDBStore(persist_directory=persist_dir)

        # Add embeddings with different page names
        ids = ["id-1", "id-2", "id-3"]
        embeddings = [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.8, 0.2, 0.0],
        ]
        metadatas = [
            {"page_name": "Page A"},
            {"page_name": "Page B"},
            {"page_name": "Page A"},  # Same page as id-1
        ]
        documents = ["Content 1", "Content 2", "Content 3"]

        store.add(ids, embeddings, metadatas, documents)

        # Query filtering for Page A only
        query_embedding = [1.0, 0.0, 0.0]
        result_ids, distances, result_metadatas = store.query(
            query_embedding,
            n_results=5,
            filter_metadata={"page_name": "Page A"}
        )

        # Should only return results from Page A
        assert len(result_ids) == 2
        assert "id-1" in result_ids
        assert "id-3" in result_ids
        assert "id-2" not in result_ids

        store.close()

    def test_delete_embeddings(self, tmp_path):
        """Test deleting embeddings from store."""
        persist_dir = tmp_path / "chroma"
        store = ChromaDBStore(persist_directory=persist_dir)

        # Add embeddings
        ids = ["id-1", "id-2", "id-3"]
        embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        metadatas = [{"page": "A"}, {"page": "B"}, {"page": "C"}]
        documents = ["Doc 1", "Doc 2", "Doc 3"]

        store.add(ids, embeddings, metadatas, documents)

        # Delete one embedding
        store.delete(["id-2"])

        # Query should not return deleted embedding
        query_embedding = [0.3, 0.4]
        result_ids, _, _ = store.query(query_embedding, n_results=10)

        assert "id-2" not in result_ids
        assert "id-1" in result_ids
        assert "id-3" in result_ids

        store.close()

    def test_delete_empty_list(self, tmp_path):
        """Test deleting empty list does nothing."""
        persist_dir = tmp_path / "chroma"
        store = ChromaDBStore(persist_directory=persist_dir)

        # Should not raise
        store.delete([])

        store.close()

    def test_persistence_across_instances(self, tmp_path):
        """Test that data persists across store instances."""
        persist_dir = tmp_path / "chroma"

        # Create first instance and add data
        store1 = ChromaDBStore(persist_directory=persist_dir)
        ids = ["persistent-id"]
        embeddings = [[0.7, 0.3]]
        metadatas = [{"test": "persistence"}]
        documents = ["Persistent doc"]

        store1.add(ids, embeddings, metadatas, documents)
        store1.close()

        # Create second instance (same directory)
        store2 = ChromaDBStore(persist_directory=persist_dir)

        # Query should find the data from first instance
        query_embedding = [0.7, 0.3]
        result_ids, _, result_metadatas = store2.query(query_embedding, n_results=1)

        assert len(result_ids) == 1
        assert result_ids[0] == "persistent-id"
        assert result_metadatas[0]["test"] == "persistence"

        store2.close()

    def test_query_returns_empty_on_no_results(self, tmp_path):
        """Test query returns empty lists when no results."""
        persist_dir = tmp_path / "chroma"
        store = ChromaDBStore(persist_directory=persist_dir)

        # Query empty store
        query_embedding = [1.0, 0.0]
        result_ids, distances, metadatas = store.query(query_embedding, n_results=5)

        assert result_ids == []
        assert distances == []
        assert metadatas == []

        store.close()
