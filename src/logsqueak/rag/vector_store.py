"""Vector store abstraction for persistent embeddings storage.

This module provides an interface for storing and querying block-level embeddings
using ChromaDB as the backend. It replaces the session-based pkl cache approach
with persistent storage and efficient similarity search.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple

import chromadb
from chromadb.config import Settings


class VectorStore(ABC):
    """Abstract interface for vector storage and retrieval."""

    @abstractmethod
    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[dict],
        documents: List[str],
    ) -> None:
        """Add embeddings to the store.

        Args:
            ids: Unique identifiers for each embedding (hybrid IDs)
            embeddings: Vector embeddings
            metadatas: Metadata dicts (e.g., page_name, block_content)
            documents: Full-context text for each block
        """
        pass

    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """Delete embeddings by ID.

        Args:
            ids: Hybrid IDs to delete
        """
        pass

    @abstractmethod
    def query(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        filter_metadata: Optional[dict] = None,
    ) -> Tuple[List[str], List[float], List[dict]]:
        """Query for similar embeddings.

        Args:
            query_embedding: Query vector
            n_results: Number of results to return
            filter_metadata: Optional metadata filter (e.g., {"page_name": "Project X"})

        Returns:
            Tuple of (ids, distances, metadatas)
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the vector store and release resources."""
        pass


class ChromaDBStore(VectorStore):
    """ChromaDB implementation of VectorStore.

    Stores embeddings in a persistent ChromaDB collection with SQLite backend.
    """

    def __init__(
        self,
        persist_directory: Path,
        collection_name: str = "logsqueak_blocks",
    ):
        """Initialize ChromaDB store.

        Args:
            persist_directory: Directory for ChromaDB persistence
            collection_name: Name of the collection to use
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # Create persist directory if needed
        persist_directory.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB persistent client
        self.client = chromadb.PersistentClient(
            path=str(persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )

    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[dict],
        documents: List[str],
    ) -> None:
        """Add embeddings to ChromaDB collection.

        Args:
            ids: Unique hybrid IDs for each block
            embeddings: Vector embeddings
            metadatas: Metadata (page_name, block_content, etc.)
            documents: Full-context text
        """
        if not ids:
            return

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )

    def delete(self, ids: List[str]) -> None:
        """Delete embeddings from ChromaDB.

        Args:
            ids: Hybrid IDs to delete
        """
        if not ids:
            return

        self.collection.delete(ids=ids)

    def query(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        filter_metadata: Optional[dict] = None,
    ) -> Tuple[List[str], List[float], List[dict]]:
        """Query ChromaDB for similar embeddings.

        Args:
            query_embedding: Query vector
            n_results: Number of results to return
            filter_metadata: Optional metadata filter

        Returns:
            Tuple of (ids, distances, metadatas)
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_metadata,
        )

        # ChromaDB returns nested lists (batch queries)
        # We only query one embedding, so extract first element
        ids = results["ids"][0] if results["ids"] else []
        distances = results["distances"][0] if results["distances"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []

        return ids, distances, metadatas

    def close(self) -> None:
        """Close ChromaDB client.

        Note: ChromaDB client doesn't require explicit closing,
        but this method is provided for interface completeness.
        """
        # ChromaDB client handles cleanup automatically
        pass
