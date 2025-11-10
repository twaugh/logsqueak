"""RAGSearch service for semantic search using ChromaDB.

NOTE: SentenceTransformer is imported lazily to avoid startup delays.
The encoder is initialized on first use (find_candidates), not at __init__.
"""

from pathlib import Path
from typing import Optional, Any
import re
from collections import defaultdict
import chromadb
from logsqueak.models.edited_content import EditedContent
from logseq_outline.parser import LogseqOutline
from logseq_outline.graph import GraphPaths
import structlog

logger = structlog.get_logger()


class RAGSearch:
    """Semantic search for candidate pages using ChromaDB.

    The SentenceTransformer encoder is loaded lazily on first use to avoid
    blocking the UI during application startup.
    """

    def __init__(
        self,
        db_path: Path,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize RAG search.

        Args:
            db_path: Path to ChromaDB persistent storage
            embedding_model: SentenceTransformer model name (must match indexer, loaded lazily)
        """
        self.db_path = db_path
        self.embedding_model = embedding_model
        self._encoder: Optional[Any] = None  # Lazy-loaded SentenceTransformer

        # Load ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=str(db_path))
        self.collection = self.chroma_client.get_collection("logsqueak_blocks")

    @property
    def encoder(self) -> Any:
        """Lazy-load the SentenceTransformer encoder on first access."""
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            logger.info("loading_embedding_model", model=self.embedding_model)
            self._encoder = SentenceTransformer(self.embedding_model)
        return self._encoder

    async def find_candidates(
        self,
        edited_content: list[EditedContent],
        original_contexts: dict[str, str],  # block_id -> hierarchical context
        top_k: int = 10
    ) -> dict[str, list[str]]:
        """
        Find candidate pages for each knowledge block.

        Args:
            edited_content: List of EditedContent to find candidates for
            original_contexts: Mapping of block_id to original hierarchical context
            top_k: Number of candidate pages per block

        Returns:
            dict mapping block_id to list of candidate page names (ranked by relevance)

        Raises:
            ValueError: If ChromaDB collection doesn't exist
        """
        results = {}

        for ec in edited_content:
            context = original_contexts.get(ec.block_id, ec.original_content)

            # Generate embedding
            embedding = self.encoder.encode(context, convert_to_numpy=True)

            # Query ChromaDB
            query_results = self.collection.query(
                query_embeddings=[embedding.tolist()],
                n_results=top_k * 3  # Over-fetch to group by page
            )

            # Extract and rank pages
            candidate_pages = self._rank_pages(
                query_results,
                context,
                top_k
            )

            results[ec.block_id] = candidate_pages

            logger.debug(
                "rag_search_candidates",
                block_id=ec.block_id,
                num_candidates=len(candidate_pages)
            )

        return results

    def _rank_pages(
        self,
        query_results: dict,
        context: str,
        top_k: int
    ) -> list[str]:
        """
        Rank pages by total similarity score.

        Also boost pages mentioned in explicit links (e.g., [[Page Name]]).
        """
        # Extract explicit page links from context
        explicit_links = set(re.findall(r'\[\[([^\]]+)\]\]', context))

        # Group results by page
        page_scores = defaultdict(float)
        for metadata, distance in zip(
            query_results["metadatas"][0],
            query_results["distances"][0]
        ):
            page_name = metadata["page_name"]
            similarity = 1.0 - distance  # Convert distance to similarity

            # Boost pages mentioned in explicit links
            if page_name in explicit_links or page_name.replace("/", "___") in explicit_links:
                similarity *= 1.5

            page_scores[page_name] += similarity

        # Sort by total score
        ranked_pages = sorted(
            page_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [page for page, score in ranked_pages[:top_k]]

    async def load_page_contents(
        self,
        candidate_pages: dict[str, list[str]],
        graph_paths: GraphPaths
    ) -> dict[str, LogseqOutline]:
        """
        Load and parse page contents for all candidate pages.

        This is the critical step between RAG search completion and LLM decision
        generation. It loads the actual page files from disk and parses them
        into LogseqOutline objects.

        Args:
            candidate_pages: Mapping of block_id to candidate page names (from find_candidates)
            graph_paths: GraphPaths instance for resolving page paths

        Returns:
            dict mapping page_name to parsed LogseqOutline

        Raises:
            FileNotFoundError: If a candidate page file doesn't exist
            ValueError: If page parsing fails

        Usage:
            # After RAG search completes:
            candidate_pages = await rag_search.find_candidates(...)

            # Load page contents:
            page_contents = await rag_search.load_page_contents(
                candidate_pages,
                graph_paths
            )

            # Then call LLM decisions:
            async for decision in plan_integrations(..., page_contents=page_contents)
        """
        page_contents = {}

        # Collect unique page names across all blocks
        unique_pages = set()
        for page_names in candidate_pages.values():
            unique_pages.update(page_names)

        # Load each unique page once
        for page_name in unique_pages:
            page_path = graph_paths.get_page_path(page_name)

            if not page_path.exists():
                logger.warning(
                    "candidate_page_not_found",
                    page_name=page_name,
                    path=str(page_path)
                )
                continue  # Skip missing pages (RAG might have stale index)

            try:
                page_text = page_path.read_text()
                outline = LogseqOutline.parse(page_text)
                page_contents[page_name] = outline

                logger.debug(
                    "page_loaded",
                    page_name=page_name,
                    num_blocks=len(outline.blocks)
                )
            except Exception as e:
                logger.error(
                    "page_parse_failed",
                    page_name=page_name,
                    error=str(e)
                )
                # Don't fail entire operation - skip this page
                continue

        return page_contents

    async def close(self) -> None:
        """Close ChromaDB client."""
        # ChromaDB client closes automatically
        pass
