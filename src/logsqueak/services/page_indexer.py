"""PageIndexer service for building ChromaDB vector index.

NOTE: SentenceTransformer is imported lazily to avoid startup delays.
The encoder is initialized on first use (build_index), not at __init__.
"""

from pathlib import Path
from typing import Optional, Callable, Any
import hashlib
import sqlite3
import chromadb
from logseq_outline.parser import LogseqOutline
from logseq_outline.context import generate_chunks
from logseq_outline.graph import GraphPaths
import structlog

logger = structlog.get_logger()

# Index schema version - increment when making breaking changes to index structure
# Version history:
# - 1 (implicit): Initial implementation (no version tracking)
# - 2: Added deleted page cleanup, version tracking
INDEX_SCHEMA_VERSION = 2


def generate_graph_db_name(graph_path: Path) -> str:
    """
    Generate unique database directory name for a graph.

    Uses pattern: (basename)-(16-digit-hash)

    Example:
        /home/user/Documents/my-graph -> my-graph-a1b2c3d4e5f6a7b8

    Args:
        graph_path: Path to Logseq graph directory

    Returns:
        Database directory name string safe for filesystem
    """
    # Get basename (already filesystem-safe since it's from a valid Path)
    basename = graph_path.name

    # Generate 16-digit hash from full absolute path
    abs_path = str(graph_path.resolve())
    hash_obj = hashlib.sha256(abs_path.encode('utf-8'))
    hash_hex = hash_obj.hexdigest()[:16]  # First 16 characters

    return f"{basename}-{hash_hex}"


class PageIndexer:
    """Builds and maintains ChromaDB vector index for Logseq pages.

    The SentenceTransformer encoder is loaded lazily on first use to avoid
    blocking the UI during application startup.
    """

    def __init__(
        self,
        graph_paths: GraphPaths,
        db_path: Optional[Path] = None,
        embedding_model: str = "all-mpnet-base-v2",
        encoder: Optional[Any] = None
    ):
        """
        Initialize page indexer.

        Args:
            graph_paths: GraphPaths instance for path resolution
            db_path: Optional base path for ChromaDB storage. If None, uses ~/.cache/logsqueak/chromadb.
                     A per-graph subdirectory will be created under this path.
            embedding_model: SentenceTransformer model name (loaded lazily if encoder not provided)
            encoder: Optional pre-loaded SentenceTransformer encoder (for testing/performance)
        """
        self.graph_paths = graph_paths
        self.embedding_model = embedding_model
        self._encoder: Optional[Any] = encoder  # Pre-loaded or lazy-loaded SentenceTransformer

        # Compute per-graph database path
        if db_path is None:
            db_path = Path.home() / ".cache" / "logsqueak" / "chromadb"

        graph_db_name = generate_graph_db_name(graph_paths.graph_path)
        self.db_path = db_path / graph_db_name
        self.db_path.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB with per-graph directory
        self.chroma_client = chromadb.PersistentClient(path=str(self.db_path))

        # Check if collection exists and verify schema version
        existing_collections = [c.name for c in self.chroma_client.list_collections()]
        needs_rebuild = False

        if "logsqueak_blocks" in existing_collections:
            # Collection exists - check schema version
            collection = self.chroma_client.get_collection(name="logsqueak_blocks")
            stored_version = collection.metadata.get("schema_version")

            if stored_version != INDEX_SCHEMA_VERSION:
                logger.warning(
                    "index_schema_mismatch",
                    stored_version=stored_version,
                    current_version=INDEX_SCHEMA_VERSION,
                    action="will_rebuild"
                )
                # Delete old collection to force full rebuild
                self.chroma_client.delete_collection(name="logsqueak_blocks")
                needs_rebuild = True

        # Create collection with current schema version
        self.collection = self.chroma_client.get_or_create_collection(
            name="logsqueak_blocks",
            metadata={
                "hnsw:space": "cosine",
                "schema_version": INDEX_SCHEMA_VERSION
            }
        )

        logger.info(
            "page_indexer_initialized",
            graph_path=str(graph_paths.graph_path),
            db_path=str(self.db_path),
            schema_version=INDEX_SCHEMA_VERSION,
            needs_rebuild=needs_rebuild
        )

    @property
    def encoder(self) -> Any:
        """Lazy-load the SentenceTransformer encoder on first access."""
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            logger.info("loading_embedding_model", model=self.embedding_model)
            self._encoder = SentenceTransformer(self.embedding_model)
        return self._encoder

    async def build_index(
        self,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> None:
        """
        Build or update vector index for all pages in graph.

        Uses incremental indexing: only re-indexes modified pages.
        Uses batch embedding with progress reporting: chunks are encoded in batches of 256.

        Args:
            progress_callback: Optional callback(current, total) for progress updates.
                Phase 1 (parsing): Called with (current_page, total_pages)
                Model loading: Called with (-1, total_pages) to signal model is loading
                Phase 4 (encoding): Called with (chunks_encoded, total_chunks) after each batch

        Raises:
            ValueError: If graph pages directory doesn't exist or contains no pages
            OSError: On file I/O errors
        """
        pages_dir = self.graph_paths.pages_dir
        if not pages_dir.exists():
            raise ValueError(f"Pages directory not found: {pages_dir}")

        # Get all .md files
        page_files = list(pages_dir.glob("*.md"))
        if not page_files:
            raise ValueError(f"No pages found in {pages_dir}")

        logger.info(
            "page_indexing_started",
            graph_path=str(self.graph_paths.graph_path),
            total_pages=len(page_files)
        )

        # OPTIMIZATION: Batch-load all page metadata and mtimes upfront
        indexed_pages = self._get_all_indexed_pages()

        # Batch-load all file mtimes to avoid repeated stat() calls
        file_mtimes = {
            page_file.stem.replace("___", "/"): page_file.stat().st_mtime
            for page_file in page_files
        }

        # Phase 1: Collect all chunks that need indexing
        all_chunks = {}  # dict[chunk_id] -> chunk_data (for deduplication)
        pages_to_index = []  # List of (page_name, outline, mtime) tuples

        for idx, page_file in enumerate(page_files, 1):
            page_name = page_file.stem.replace("___", "/")
            current_mtime = file_mtimes[page_name]

            # Check if page needs re-indexing using cached metadata
            if page_name in indexed_pages:
                stored_mtime = indexed_pages[page_name]
                # Use tolerance-based comparison to handle floating point precision issues
                # ChromaDB may store floats with slightly different precision than Python
                mtime_tolerance = 1e-6  # 1 microsecond tolerance
                if abs(stored_mtime - current_mtime) < mtime_tolerance:
                    logger.debug("page_index_skip", page_name=page_name, reason="not_modified")
                    if progress_callback:
                        progress_callback(idx, len(page_files))
                    continue
                else:
                    # Page modified - log mtime difference for debugging
                    logger.debug(
                        "page_mtime_changed",
                        page_name=page_name,
                        stored_mtime=stored_mtime,
                        current_mtime=current_mtime,
                        diff=current_mtime - stored_mtime
                    )
            else:
                # Page not in index yet
                logger.debug("page_not_indexed", page_name=page_name)

            # Parse page
            outline = LogseqOutline.parse(page_file.read_text())
            pages_to_index.append((page_name, outline, current_mtime))

            if progress_callback:
                progress_callback(idx, len(page_files))

        # Phase 2: Delete chunks for pages that no longer exist in the filesystem
        # Find pages in index that are not in current file list
        deleted_pages = set(indexed_pages.keys()) - set(file_mtimes.keys())
        if deleted_pages:
            logger.info("deleting_chunks_for_removed_pages", count=len(deleted_pages))
            for page_name in deleted_pages:
                try:
                    self.collection.delete(where={"page_name": page_name})
                    logger.debug("deleted_chunks_for_removed_page", page_name=page_name)
                except Exception as e:
                    logger.warning("failed_to_delete_chunks_for_removed_page", page_name=page_name, error=str(e))

        # Phase 3: Delete old chunks for modified pages
        # When a page is re-indexed, its content may have changed, leading to new content hashes
        # (new block IDs). We must delete the old chunks to prevent stale data in the index.
        if pages_to_index:
            pages_to_reindex = {page_name for page_name, _, _ in pages_to_index if page_name in indexed_pages}
            if pages_to_reindex:
                logger.info("deleting_old_chunks", pages_to_reindex=len(pages_to_reindex))
                for page_name in pages_to_reindex:
                    # Delete all chunks for this page
                    # ChromaDB requires a where clause, so we filter by page_name metadata
                    try:
                        self.collection.delete(where={"page_name": page_name})
                        logger.debug("deleted_old_chunks", page_name=page_name)
                    except Exception as e:
                        logger.warning("failed_to_delete_old_chunks", page_name=page_name, error=str(e))

        # Phase 4: Generate chunks for all pages to index
        if pages_to_index:
            logger.info("generating_chunks", pages_to_index=len(pages_to_index))
            for page_name, outline, mtime in pages_to_index:
                page_chunks = self._prepare_page_chunks(page_name, outline, mtime)
                all_chunks.update(page_chunks)  # Merge with deduplication
                logger.debug("page_chunks_prepared", page_name=page_name, chunk_count=len(page_chunks))

            # Phase 5: Batch encode all chunks
            logger.info("batch_encoding", total_chunks=len(all_chunks))
            chunk_ids = list(all_chunks.keys())
            documents = [chunk["document"] for chunk in all_chunks.values()]

            # Signal model loading phase (before first encoder access)
            if progress_callback:
                # Use negative progress to signal model loading phase
                progress_callback(-1, len(page_files))

            # Access encoder (triggers lazy loading on first use)
            encoder = self.encoder

            # Encode in batches with progress reporting
            # Encoding is the slowest phase (85-95% of total time), so we report incremental progress
            import numpy as np

            batch_size = 256  # Balance between progress granularity and overhead
            embeddings_list = []

            for batch_start in range(0, len(documents), batch_size):
                batch_end = min(batch_start + batch_size, len(documents))
                batch_docs = documents[batch_start:batch_end]

                # Encode batch
                batch_embeddings = encoder.encode(
                    batch_docs,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
                embeddings_list.append(batch_embeddings)

                # Report progress (chunks encoded so far, total chunks)
                if progress_callback:
                    progress_callback(batch_end, len(documents))

            # Concatenate all batch embeddings
            embeddings = np.vstack(embeddings_list) if embeddings_list else np.array([])

            # Phase 6: Prepare data for bulk upsert
            metadatas = [chunk["metadata"] for chunk in all_chunks.values()]

            # Phase 7: Bulk upsert to ChromaDB
            logger.info("bulk_upserting", total_chunks=len(chunk_ids))
            self.collection.upsert(
                documents=documents,
                embeddings=embeddings.tolist(),
                ids=chunk_ids,
                metadatas=metadatas
            )

            # Phase 8: Compact database to reclaim fragmented space
            # Bulk upserts can leave ~70% wasted space due to SQLite fragmentation
            self._vacuum_database()

            logger.info("page_indexing_completed", total_pages=len(page_files), chunks_indexed=len(all_chunks))
        else:
            logger.info("page_indexing_completed", total_pages=len(page_files), chunks_indexed=0, reason="all_up_to_date")

    def _get_all_indexed_pages(self) -> dict[str, float]:
        """
        Get all indexed pages with their modification times.

        Returns a dict mapping page_name -> mtime for all pages in the collection.
        This enables efficient batch checking instead of querying per page.

        Returns:
            Dict of {page_name: mtime} for all indexed pages
        """
        # Get all documents in collection
        results = self.collection.get(include=["metadatas"])

        # Build dict of page_name -> mtime
        # (Multiple blocks per page all have same mtime, so we can use any)
        indexed_pages = {}
        for metadata in results["metadatas"]:
            page_name = metadata["page_name"]
            mtime = metadata.get("mtime")
            if mtime is not None:
                indexed_pages[page_name] = mtime

        return indexed_pages

    def _prepare_page_chunks(
        self,
        page_name: str,
        outline: LogseqOutline,
        mtime: float
    ) -> dict[str, dict]:
        """Prepare chunks for a page without embedding or upserting.

        Creates a page-level chunk and block-level chunks for every page.
        Returns a dict mapping chunk_id -> chunk_data (without embeddings).

        Args:
            page_name: Name of the page (e.g., "Konflux/Java builds")
            outline: Parsed LogseqOutline
            mtime: File modification time

        Returns:
            Dict of {chunk_id: {"document": str, "metadata": dict}}
        """
        # Use dict to deduplicate by ID (keeps last occurrence)
        chunks_by_id = {}

        # ALWAYS create page-level chunk (enables empty page indexing and name search)
        page_title = self._extract_page_title(outline)
        page_context = self._create_page_context(page_name, page_title, outline)

        page_chunk_id = f"{page_name}::__PAGE__"
        chunks_by_id[page_chunk_id] = {
            "document": page_context,
            "metadata": {
                "page_name": page_name,
                "block_id": "__PAGE__",
                "mtime": mtime,
                "page_title": page_title  # Store title:: for display
            }
        }

        # Generate block-level chunks with full context and hybrid IDs
        chunks = generate_chunks(outline, page_name)

        for block, full_context, hybrid_id, parents in chunks:
            # Store chunk (overwrites if duplicate ID)
            full_id = f"{page_name}::{hybrid_id}"
            chunks_by_id[full_id] = {
                "document": full_context,
                "metadata": {
                    "page_name": page_name,
                    "block_id": hybrid_id,
                    "mtime": mtime,
                    "page_title": page_title  # Include in block metadata too
                }
            }

        return chunks_by_id

    def _extract_page_title(self, outline: LogseqOutline) -> Optional[str]:
        """Extract title:: property from page frontmatter.

        Returns:
            Page title from title:: property, or None if not present
        """
        # Parse frontmatter for title:: property
        if not outline.frontmatter:
            return None

        for line in outline.frontmatter:
            line = line.strip()
            if line.startswith('title::'):
                # Extract value after 'title::'
                title = line[7:].strip()  # Skip 'title::'
                return title if title else None

        return None

    def _create_page_context(
        self,
        page_name: str,
        page_title: Optional[str],
        outline: LogseqOutline
    ) -> str:
        """Create searchable context for page-level chunk.

        Includes page name, title:: property, and frontmatter for semantic matching.

        Args:
            page_name: Page name from filename (e.g., "Konflux/Java builds")
            page_title: Value from title:: property if present
            outline: Parsed LogseqOutline

        Returns:
            Context string for embedding
        """
        parts = []

        # Always include page name
        parts.append(f"Page: {page_name}")

        # Include title:: if different from page_name
        if page_title and page_title != page_name:
            parts.append(f"Title: {page_title}")

        # Include frontmatter properties for additional context
        if outline.frontmatter:
            parts.append("\n".join(outline.frontmatter))

        return "\n".join(parts)

    def _vacuum_database(self) -> None:
        """Compact the SQLite database to reclaim fragmented space.

        Bulk upsert operations can create significant SQLite fragmentation
        (up to 70% wasted space). Running VACUUM after bulk operations
        reclaims this space and reduces database file size.

        This is a blocking operation but typically completes in <1 second
        for typical graph sizes.
        """
        sqlite_path = self.db_path / "chroma.sqlite3"
        if not sqlite_path.exists():
            logger.warning("vacuum_skipped", reason="sqlite_not_found", path=str(sqlite_path))
            return

        try:
            logger.debug("vacuum_started", db_path=str(sqlite_path))
            conn = sqlite3.connect(str(sqlite_path))
            conn.execute("VACUUM")
            conn.close()
            logger.debug("vacuum_completed", db_path=str(sqlite_path))
        except Exception as e:
            # VACUUM is an optimization - don't fail if it errors
            logger.warning("vacuum_failed", error=str(e), db_path=str(sqlite_path))

    async def close(self) -> None:
        """Close ChromaDB client."""
        # ChromaDB client closes automatically
        pass
