"""PageIndexer service for building ChromaDB vector index.

NOTE: SentenceTransformer is imported lazily to avoid startup delays.
The encoder is initialized on first use (build_index), not at __init__.
"""

from pathlib import Path
from typing import Optional, Callable, Any
import hashlib
import sqlite3
import chromadb
from logseq_outline.parser import LogseqOutline, LogseqBlock
from logseq_outline.context import generate_chunks
from logseq_outline.graph import GraphPaths
import structlog

logger = structlog.get_logger()


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
        self.collection = self.chroma_client.get_or_create_collection(
            name="logsqueak_blocks",
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(
            "page_indexer_initialized",
            graph_path=str(graph_paths.graph_path),
            db_path=str(self.db_path)
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
        Uses batch embedding for performance: all chunks are collected and embedded together.

        Args:
            progress_callback: Optional callback(current, total) for progress updates.
                Called during Phase 1 (page scanning/parsing) with (current_page, total_pages).
                Called at start of Phase 3 (batch encoding) with (total_pages, total_pages) to signal 100%.
                Note: Batch encoding itself doesn't provide incremental updates since it's a single operation.

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

        # Phase 2: Generate chunks for all pages to index
        if pages_to_index:
            logger.info("generating_chunks", pages_to_index=len(pages_to_index))
            for page_name, outline, mtime in pages_to_index:
                page_chunks = self._prepare_page_chunks(page_name, outline, mtime)
                all_chunks.update(page_chunks)  # Merge with deduplication
                logger.debug("page_chunks_prepared", page_name=page_name, chunk_count=len(page_chunks))

            # Phase 3: Batch encode all chunks
            logger.info("batch_encoding", total_chunks=len(all_chunks))
            chunk_ids = list(all_chunks.keys())
            documents = [chunk["document"] for chunk in all_chunks.values()]

            # Batch encode all documents at once
            # Note: SentenceTransformer's show_progress_bar doesn't integrate with our progress_callback,
            # but we can at least signal that encoding is happening
            if progress_callback:
                # Signal start of encoding phase (use total_pages + 1 to show we're in a new phase)
                progress_callback(len(page_files), len(page_files))

            embeddings = self.encoder.encode(documents, convert_to_numpy=True, show_progress_bar=False)

            # Phase 4: Prepare data for bulk upsert
            metadatas = [chunk["metadata"] for chunk in all_chunks.values()]

            # Phase 5: Bulk upsert to ChromaDB
            logger.info("bulk_upserting", total_chunks=len(chunk_ids))
            self.collection.upsert(
                documents=documents,
                embeddings=embeddings.tolist(),
                ids=chunk_ids,
                metadatas=metadatas
            )

            # Phase 6: Compact database to reclaim fragmented space
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

        for block, full_context, hybrid_id in chunks:
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
