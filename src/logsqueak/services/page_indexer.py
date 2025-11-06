"""PageIndexer service for building ChromaDB vector index.

NOTE: SentenceTransformer is imported lazily to avoid startup delays.
The encoder is initialized on first use (build_index), not at __init__.
"""

from pathlib import Path
from typing import Optional, Callable, Any
import chromadb
from logseq_outline.parser import LogseqOutline, LogseqBlock
from logseq_outline.context import generate_chunks
from logseq_outline.graph import GraphPaths
import structlog

logger = structlog.get_logger()


class PageIndexer:
    """Builds and maintains ChromaDB vector index for Logseq pages.

    The SentenceTransformer encoder is loaded lazily on first use to avoid
    blocking the UI during application startup.
    """

    def __init__(
        self,
        graph_paths: GraphPaths,
        db_path: Path,
        embedding_model: str = "all-MiniLM-L6-v2",
        encoder: Optional[Any] = None
    ):
        """
        Initialize page indexer.

        Args:
            graph_paths: GraphPaths instance for path resolution
            db_path: Path to ChromaDB persistent storage
            embedding_model: SentenceTransformer model name (loaded lazily if encoder not provided)
            encoder: Optional pre-loaded SentenceTransformer encoder (for testing/performance)
        """
        self.graph_paths = graph_paths
        self.db_path = db_path
        self.embedding_model = embedding_model
        self._encoder: Optional[Any] = encoder  # Pre-loaded or lazy-loaded SentenceTransformer

        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=str(db_path))
        self.collection = self.chroma_client.get_or_create_collection(
            name="logsqueak_blocks",
            metadata={"hnsw:space": "cosine"}
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
        documents = []
        embeddings = []
        ids = []
        metadatas = []

        # Generate chunks with full context and hybrid IDs
        chunks = generate_chunks(outline, page_name)

        for block, full_context, hybrid_id in chunks:
            # Generate embedding
            embedding = self.encoder.encode(full_context, convert_to_numpy=True)

            # Store chunk
            documents.append(full_context)
            embeddings.append(embedding.tolist())
            ids.append(f"{page_name}::{hybrid_id}")
            metadatas.append({
                "page_name": page_name,
                "block_id": hybrid_id,
                "mtime": mtime
            })

        # Add to collection (upsert)
        if documents:
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
