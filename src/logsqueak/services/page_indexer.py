"""PageIndexer service for building ChromaDB vector index.

NOTE: SentenceTransformer is imported lazily to avoid startup delays.
The encoder is initialized on first use (build_index), not at __init__.
"""

from pathlib import Path
from typing import Optional, Callable, Any
import hashlib
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

        Args:
            progress_callback: Optional callback(current, total) for progress updates

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

        for idx, page_file in enumerate(page_files, 1):
            page_name = page_file.stem.replace("___", "/")

            # Check if page needs re-indexing
            if self._is_page_indexed(page_name, page_file):
                logger.debug("page_index_skip", page_name=page_name, reason="not_modified")
                if progress_callback:
                    progress_callback(idx, len(page_files))
                continue

            # Parse page
            outline = LogseqOutline.parse(page_file.read_text())

            # Index all blocks
            self._index_page_blocks(page_name, outline, page_file.stat().st_mtime)

            if progress_callback:
                progress_callback(idx, len(page_files))

            logger.debug("page_indexed", page_name=page_name, block_count=len(outline.blocks))

        logger.info("page_indexing_completed", total_pages=len(page_files))

    def _is_page_indexed(self, page_name: str, page_file: Path) -> bool:
        """
        Check if page is already indexed and unmodified.

        Uses mtime equality check to detect changes. This handles
        Logseq graphs in git where files may revert to earlier content.
        """
        # Query collection for page metadata
        results = self.collection.get(
            where={"page_name": page_name},
            limit=1
        )

        if not results["ids"]:
            return False

        # Check modification time (use == to check if unchanged)
        stored_mtime = results["metadatas"][0].get("mtime")
        current_mtime = page_file.stat().st_mtime

        return stored_mtime is not None and current_mtime == stored_mtime

    def _index_page_blocks(
        self,
        page_name: str,
        outline: LogseqOutline,
        mtime: float
    ) -> None:
        """Index all blocks from a page using semantic chunks."""
        # Use dict to deduplicate by ID (keeps last occurrence)
        chunks_by_id = {}

        # Generate chunks with full context and hybrid IDs
        chunks = generate_chunks(outline, page_name)

        for block, full_context, hybrid_id in chunks:
            # Generate embedding
            embedding = self.encoder.encode(full_context, convert_to_numpy=True)

            # Store chunk (overwrites if duplicate ID)
            full_id = f"{page_name}::{hybrid_id}"
            chunks_by_id[full_id] = {
                "document": full_context,
                "embedding": embedding.tolist(),
                "metadata": {
                    "page_name": page_name,
                    "block_id": hybrid_id,
                    "mtime": mtime
                }
            }

        # Convert to lists for ChromaDB
        if chunks_by_id:
            ids = list(chunks_by_id.keys())
            documents = [chunk["document"] for chunk in chunks_by_id.values()]
            embeddings = [chunk["embedding"] for chunk in chunks_by_id.values()]
            metadatas = [chunk["metadata"] for chunk in chunks_by_id.values()]

            self.collection.upsert(
                documents=documents,
                embeddings=embeddings,
                ids=ids,
                metadatas=metadatas
            )

    async def close(self) -> None:
        """Close ChromaDB client."""
        # ChromaDB client closes automatically
        pass
