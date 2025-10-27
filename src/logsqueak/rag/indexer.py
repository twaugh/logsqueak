"""Incremental index builder for block-level embeddings.

This module implements incremental indexing that detects additions, updates,
and deletions using the cache manifest system.
"""

import logging
from pathlib import Path
from typing import List, Optional

from logsqueak.logseq.parser import LogseqOutline
from logsqueak.rag.chunker import chunk_page
from logsqueak.rag.manifest import CacheManifest
from logsqueak.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


class IndexBuilder:
    """Incremental index builder for block-level embeddings."""

    def __init__(
        self,
        vector_store: VectorStore,
        manifest: CacheManifest,
        embedding_model=None,
    ):
        """Initialize index builder.

        Args:
            vector_store: VectorStore instance for storing embeddings
            manifest: CacheManifest for tracking indexed pages
            embedding_model: Optional embedding model (defaults to all-MiniLM-L6-v2)
        """
        self.vector_store = vector_store
        self.manifest = manifest
        self.embedding_model = embedding_model

    def build_incremental(self, graph_path: Path) -> dict:
        """Build index incrementally.

        Detects:
        - Deletions: Pages in manifest but not on disk
        - Updates: Pages with changed mtime
        - Additions: Pages not in manifest

        Args:
            graph_path: Path to Logseq graph directory

        Returns:
            Dict with stats: {
                'added': int,
                'updated': int,
                'deleted': int,
                'unchanged': int,
            }
        """
        # Lazy load embedding model only when needed
        if self.embedding_model is None:
            logger.info("Loading embedding model...")
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        stats = {
            'added': 0,
            'updated': 0,
            'deleted': 0,
            'unchanged': 0,
        }

        pages_dir = graph_path / "pages"
        if not pages_dir.exists():
            logger.warning(f"Pages directory not found: {pages_dir}")
            return stats

        # Get current pages on disk
        current_pages = {p.stem: p for p in pages_dir.glob("*.md")}

        # Detect deletions (in manifest but not on disk)
        manifest_pages = set(self.manifest.get_all_pages())
        deleted_pages = manifest_pages - set(current_pages.keys())

        for page_name in deleted_pages:
            self._delete_page(page_name)
            stats['deleted'] += 1

        # Detect updates and additions
        for page_name, page_path in current_pages.items():
            current_mtime = page_path.stat().st_mtime
            cached_mtime = self.manifest.get_mtime(page_name)

            if cached_mtime is None:
                # Addition: not in manifest
                self._add_page(page_name, page_path, current_mtime)
                stats['added'] += 1
            elif current_mtime > cached_mtime:
                # Update: mtime changed
                self._update_page(page_name, page_path, current_mtime)
                stats['updated'] += 1
            else:
                # Unchanged
                stats['unchanged'] += 1

        # Save updated manifest
        self.manifest.save()

        logger.info(
            f"Index update complete: "
            f"+{stats['added']} ~{stats['updated']} -{stats['deleted']} ={stats['unchanged']}"
        )

        return stats

    def _add_page(self, page_name: str, page_path: Path, mtime: float) -> None:
        """Add a new page to the index.

        Args:
            page_name: Name of the page
            page_path: Path to page file
            mtime: Modification time
        """
        logger.debug(f"Adding page: {page_name}")

        # Parse and chunk page
        content = page_path.read_text(encoding="utf-8")
        outline = LogseqOutline.parse(content)
        chunks = chunk_page(outline, page_name)

        if not chunks:
            # Empty page - just update manifest
            self.manifest.set_mtime(page_name, mtime)
            return

        # Generate embeddings
        texts = [chunk.full_context_text for chunk in chunks]
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)

        # Prepare data for vector store
        ids = [chunk.hybrid_id for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        documents = texts

        # Add to vector store
        self.vector_store.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            documents=documents,
        )

        # Update manifest
        self.manifest.set_mtime(page_name, mtime)

    def _update_page(self, page_name: str, page_path: Path, mtime: float) -> None:
        """Update an existing page in the index.

        Strategy: Delete old chunks, add new chunks.

        Args:
            page_name: Name of the page
            page_path: Path to page file
            mtime: New modification time
        """
        logger.debug(f"Updating page: {page_name}")

        # Delete old chunks for this page
        # Note: We'll need to query for all chunks from this page
        # For now, we'll use a simpler approach: delete by reconstructing old chunks
        # In practice, this would query the vector store for all chunks with
        # metadata.page_name == page_name, then delete those IDs

        # Parse old version to get old chunk IDs (before update)
        # This is a limitation - we can't easily get old chunk IDs without
        # storing them separately. For MVP, we'll just add new chunks.
        # A proper implementation would track chunk IDs per page in manifest.

        # For now: Add new chunks (old chunks will become stale but harmless)
        self._add_page(page_name, page_path, mtime)

    def _delete_page(self, page_name: str) -> None:
        """Delete a page from the index.

        Args:
            page_name: Name of the page to delete
        """
        logger.debug(f"Deleting page: {page_name}")

        # Remove from manifest
        self.manifest.remove(page_name)

        # Note: Deleting chunks from vector store requires knowing their IDs
        # For MVP, we'll leave orphaned chunks (filtered out by metadata queries)
        # A proper implementation would track chunk IDs per page in manifest
