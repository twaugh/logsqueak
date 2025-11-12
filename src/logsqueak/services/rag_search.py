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
from logseq_outline.parser import LogseqOutline, LogseqBlock
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

        # Load ChromaDB (create collection if it doesn't exist)
        self.chroma_client = chromadb.PersistentClient(path=str(db_path))
        try:
            self.collection = self.chroma_client.get_collection("logsqueak_blocks")
        except Exception:
            # Collection doesn't exist yet - create it
            logger.info("creating_chromadb_collection", name="logsqueak_blocks")
            self.collection = self.chroma_client.create_collection("logsqueak_blocks")

    def has_indexed_data(self) -> bool:
        """Check if the ChromaDB collection exists and has data.

        Returns:
            True if collection has indexed blocks, False otherwise
        """
        try:
            count = self.collection.count()
            return count > 0
        except Exception as e:
            logger.warning("collection_count_check_failed", error=str(e))
            return False

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
        original_contexts: dict[str, str],
        graph_paths: GraphPaths,
        top_k: int = 10
    ) -> dict[str, list[tuple[str, LogseqBlock, list[LogseqBlock]]]]:
        """
        Find candidate chunks (block + parents) for each knowledge block.

        Returns hierarchical chunks ready for format_chunks_for_llm().

        Args:
            edited_content: List of EditedContent to find candidates for
            original_contexts: Mapping of block_id to original hierarchical context
            graph_paths: GraphPaths instance for resolving page paths
            top_k: Number of candidate chunks per block

        Returns:
            dict mapping block_id to list of (page_name, block, parents) tuples
            Empty lists if no indexed data exists

        Raises:
            ValueError: If ChromaDB collection doesn't exist
        """
        # Early return if no indexed data - but encoder already preloaded in background
        if not self.has_indexed_data():
            logger.info(
                "rag_search_skipped_no_data",
                message="ChromaDB collection is empty, returning empty results"
            )
            return {ec.block_id: [] for ec in edited_content}

        results = {}

        for ec in edited_content:
            context = original_contexts.get(ec.block_id, ec.original_content)

            # Log the full search query
            logger.info(
                "rag_search_query",
                block_id=ec.block_id,
                query=context,
                query_length=len(context)
            )

            # Generate embedding
            embedding = self.encoder.encode(context, convert_to_numpy=True)

            # Query ChromaDB
            query_results = self.collection.query(
                query_embeddings=[embedding.tolist()],
                n_results=top_k * 3  # Over-fetch for diversity
            )

            # Extract chunks with hierarchy
            chunks = await self._extract_chunks_from_results(
                query_results,
                context,
                graph_paths,
                top_k
            )

            results[ec.block_id] = chunks

            logger.info(
                "rag_search_chunks_found",
                block_id=ec.block_id,
                num_chunks=len(chunks),
                unique_pages=len(set(page_name for page_name, _, _ in chunks))
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

        # Log if explicit links were found
        if explicit_links:
            logger.debug(
                "rag_search_explicit_links",
                links=list(explicit_links),
                link_count=len(explicit_links)
            )

        # Group results by page
        page_scores = defaultdict(float)
        for metadata, distance in zip(
            query_results["metadatas"][0],
            query_results["distances"][0]
        ):
            page_name = metadata["page_name"]
            similarity = 1.0 - distance  # Convert distance to similarity

            # Boost pages mentioned in explicit links
            boost_applied = False
            if page_name in explicit_links or page_name.replace("/", "___") in explicit_links:
                similarity *= 1.5
                boost_applied = True

            page_scores[page_name] += similarity

        # Sort by total score
        ranked_pages = sorted(
            page_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Log all candidates with scores
        top_results = ranked_pages[:top_k]
        if top_results:
            logger.debug(
                "rag_search_ranked_results",
                ranked_candidates=[
                    {"page": page, "score": round(score, 3)}
                    for page, score in top_results  # Show all results with scores
                ]
            )

        return [page for page, score in top_results]

    async def _extract_chunks_from_results(
        self,
        query_results: dict,
        context: str,
        graph_paths: GraphPaths,
        top_k: int
    ) -> list[tuple[str, LogseqBlock, list[LogseqBlock]]]:
        """
        Extract hierarchical chunks (block + parents) from ChromaDB results.

        Args:
            query_results: Raw results from ChromaDB query
            context: Original search context (for explicit link boosting)
            graph_paths: GraphPaths for resolving page paths
            top_k: Number of chunks to return

        Returns:
            List of (page_name, block, parents) tuples
        """
        from logseq_outline.parser import LogseqBlock

        # Extract explicit page links for boosting
        explicit_links = set(re.findall(r'\[\[([^\]]+)\]\]', context))

        # Score each chunk
        chunk_scores = []
        for metadata, distance in zip(
            query_results["metadatas"][0],
            query_results["distances"][0]
        ):
            page_name = metadata["page_name"]
            block_id = metadata["block_id"]
            similarity = 1.0 - distance

            # Boost chunks from explicitly linked pages
            if page_name in explicit_links or page_name.replace("/", "___") in explicit_links:
                similarity *= 1.5

            chunk_scores.append((page_name, block_id, similarity))

        # Sort by score
        chunk_scores.sort(key=lambda x: x[2], reverse=True)

        # Load blocks and build chunks
        chunks = []
        loaded_pages = {}  # Cache loaded pages

        for page_name, block_id, score in chunk_scores[:top_k]:
            # Load page if not cached
            if page_name not in loaded_pages:
                page_path = graph_paths.get_page_path(page_name)
                if not page_path.exists():
                    logger.warning(
                        "chunk_page_not_found",
                        page_name=page_name,
                        block_id=block_id
                    )
                    continue

                try:
                    page_text = page_path.read_text()
                    outline = LogseqOutline.parse(page_text)
                    loaded_pages[page_name] = outline
                except Exception as e:
                    logger.error(
                        "chunk_page_parse_failed",
                        page_name=page_name,
                        error=str(e)
                    )
                    continue

            outline = loaded_pages[page_name]

            # Find block by hybrid ID using generate_chunks
            from logseq_outline.context import generate_chunks
            found_block = None
            found_parents = []

            for chunk_block, full_context, chunk_id in generate_chunks(outline, page_name):
                if chunk_id == block_id:
                    # Found the target block, now need to build parents list
                    # We can find parents by walking up from this block
                    found_block = chunk_block

                    # Build parents list by traversing from root to this block
                    def find_path_to_block(blocks, target, current_path=[]):
                        for block in blocks:
                            if block == target:
                                return current_path
                            result = find_path_to_block(block.children, target, current_path + [block])
                            if result is not None:
                                return result
                        return None

                    parents_path = find_path_to_block(outline.blocks, chunk_block)
                    if parents_path:
                        found_parents = parents_path
                    break

            if not found_block:
                logger.warning(
                    "chunk_block_not_found",
                    page_name=page_name,
                    block_id=block_id
                )
                continue

            chunks.append((page_name, found_block, found_parents))

        logger.debug(
            "chunks_extracted",
            total_results=len(chunk_scores),
            chunks_loaded=len(chunks),
            unique_pages=len(loaded_pages)
        )

        return chunks

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
