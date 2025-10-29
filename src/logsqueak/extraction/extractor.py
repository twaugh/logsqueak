"""Knowledge extraction orchestrator.

This module implements the multi-stage extraction process:
1. Phase 1: Extract knowledge blocks from journal (exact text + AST context)
2. Phase 2: Candidate retrieval (semantic + hinted search)
3. Phase 3: Decider + Reworder (LLM selects action and generates clean content)
4. Phase 4: Execution + Cleanup (apply changes and mark journal processed)
"""

import logging
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from logsqueak.llm.client import LLMClient
from logsqueak.logseq.parser import LogseqBlock
from logsqueak.models.block_locator import BlockLocator
from logsqueak.models.journal import JournalEntry
from logsqueak.models.knowledge import ActionType, KnowledgePackage
from logsqueak.models.page import TargetPage
from logsqueak.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class CandidateChunk:
    """A candidate chunk for knowledge integration (Phase 2 output).

    Represents a specific block in a target page that could receive
    the knowledge. Used by Phase 3 (Decider) to select the best match.

    Attributes:
        page_name: Name of the target page
        target_id: Hybrid ID of the target block
        content: Content of the target block (for LLM context)
        similarity_score: Semantic similarity score (0.0-1.0)
    """

    page_name: str
    target_id: str
    content: str
    similarity_score: float


@dataclass
class WriteOperation:
    """A pending write operation (Phase 3 output).

    Represents a decision by the Decider LLM to integrate knowledge
    into a specific location. Collected into a Write List for Phase 4.

    Attributes:
        page_name: Target page name
        action: What to do (UPDATE, APPEND_CHILD, APPEND_ROOT)
        target_id: Hybrid ID of target block (None for APPEND_ROOT)
        new_content: Rephrased content ready for integration
        original_id: Hybrid ID of source journal block (for cleanup tracking)
        block_locator: Logical position to re-find the journal block (from Phase 1)
    """

    page_name: str
    action: ActionType
    target_id: Optional[str]
    new_content: str
    original_id: str  # For tracking back to journal block
    block_locator: "BlockLocator"  # From Phase 1


class Extractor:
    """Orchestrates knowledge extraction from journal entries.

    Uses LLM to identify lasting knowledge (vs. activity logs) and
    extract knowledge blocks with confidence scores.
    """

    def __init__(self, llm_client: LLMClient, use_fuzzy_matching: bool = True):
        """Initialize the extractor.

        Args:
            llm_client: LLM client for knowledge extraction
            use_fuzzy_matching: Enable fuzzy block matching with embeddings (default: True)
        """
        self.llm_client = llm_client
        self.use_fuzzy_matching = use_fuzzy_matching
        self._embedding_model = None  # Lazy-loaded

    def extract_knowledge(
        self, journal: JournalEntry
    ) -> List[KnowledgePackage]:
        """Extract knowledge blocks from journal entry.

        This is Phase 1 of the multi-stage extraction pipeline.
        The LLM identifies exact block text containing lasting knowledge,
        then code adds parent context via AST walk.

        The returned KnowledgePackages contain:
        - original_id: Hybrid ID of the source journal block
        - exact_text: Raw block text as identified by LLM
        - full_text: Contextualized text with parent blocks
        - confidence: LLM confidence score (0.0-1.0)

        Args:
            journal: Journal entry to extract from

        Returns:
            List of KnowledgePackage objects with full context

        Raises:
            LLMError: If LLM request fails
        """
        # Call LLM to extract exact knowledge block text
        llm_results = self.llm_client.extract_knowledge(
            journal_content=journal.raw_content,
            journal_date=journal.date,
            indent_str=journal.outline.indent_str,
        )

        # Convert ExtractionResults to KnowledgePackages with AST walk
        packages = []
        for result in llm_results:
            exact_text = result.content  # LLM returns (possibly approximate) block text

            # Try exact match first
            block = self._find_block_by_content(journal.outline.blocks, exact_text)

            # Fall back to fuzzy matching if enabled and exact match fails
            if block is None and self.use_fuzzy_matching:
                logger.debug(f"Exact match failed, trying fuzzy matching for: {exact_text[:50]}...")
                block, similarity = self._find_block_by_fuzzy_match(
                    journal.outline.blocks, exact_text
                )
                if block is not None:
                    logger.info(
                        f"Fuzzy match found (similarity: {similarity:.3f}): "
                        f"{block.content[0][:50]}..."
                    )

            if block is None:
                logger.warning(f"Could not find block in AST for text: {exact_text[:50]}...")
                continue

            # Find parent chain for hybrid ID generation
            parents = self._find_parent_chain(block, journal.outline.blocks)

            # Generate original_id using hybrid ID system
            original_id = block.get_hybrid_id(parents=parents, indent_str=journal.outline.indent_str)

            # Build full_text with parent context (flattened for RAG)
            full_text = self._build_full_context(block, journal.outline.blocks)

            # Build hierarchical_text with parent context (hierarchical for reworder)
            hierarchical_text = self._build_hierarchical_context(
                block, journal.outline.blocks, indent_str=journal.outline.indent_str
            )

            # Create block locator for reliable re-finding in Phase 4
            block_locator = BlockLocator.from_block(block, parents, root_blocks=journal.outline.blocks)

            packages.append(
                KnowledgePackage(
                    original_id=original_id,
                    exact_text=exact_text,
                    full_text=full_text,
                    hierarchical_text=hierarchical_text,
                    confidence=result.confidence,
                    block_locator=block_locator,  # Store locator for Phase 4 cleanup
                )
            )

        return packages

    def find_candidate_chunks(
        self,
        knowledge: KnowledgePackage,
        vector_store: VectorStore,
        graph_path: Path,
        top_k: int = 20,
    ) -> List[CandidateChunk]:
        """Find candidate chunks for knowledge integration (Phase 2).

        Uses both semantic search and hinted search to find relevant
        blocks in the knowledge base:
        - Semantic: Query vector store for Top-K similar chunks
        - Hinted: Parse ((block-id)) references from knowledge text and fetch those blocks

        Args:
            knowledge: Knowledge package to find candidates for
            vector_store: Vector store for semantic search
            graph_path: Path to Logseq graph (for loading hinted blocks)
            top_k: Number of semantic search results to retrieve (default: 20)

        Returns:
            List of CandidateChunk objects, deduplicated and sorted by similarity

        Example:
            knowledge.full_text = "See ((65f3a8e0-1234)): Added API documentation"
            Returns chunks from:
            - Semantic: Top-20 similar blocks across all pages
            - Hinted: The specific block with id 65f3a8e0-1234 (high score: 0.95)
        """
        # 1. Semantic search: Query vector store for Top-K similar chunks
        semantic_chunks = self._semantic_search(knowledge, vector_store, top_k)

        # 2. Hinted search: Parse ((block-id)) references and fetch those blocks
        hinted_chunks = self._hinted_search(knowledge, graph_path)

        # 3. Aggregate and deduplicate (prefer semantic scores)
        all_chunks = self._aggregate_chunks(semantic_chunks, hinted_chunks)

        return all_chunks

    def _semantic_search(
        self, knowledge: KnowledgePackage, vector_store: VectorStore, top_k: int
    ) -> List[CandidateChunk]:
        """Perform semantic search for similar chunks.

        Args:
            knowledge: Knowledge package to search for
            vector_store: Vector store to query
            top_k: Number of results to return

        Returns:
            List of CandidateChunk objects from semantic search
        """
        # Embed the knowledge full_text
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        query_embedding = model.encode(knowledge.full_text, convert_to_numpy=True)

        # Query vector store
        ids, distances, metadatas = vector_store.query(
            query_embedding=query_embedding.tolist(), n_results=top_k
        )

        # Convert to CandidateChunks
        chunks = []
        for i, block_id in enumerate(ids):
            metadata = metadatas[i]
            similarity = 1.0 - distances[i]  # Convert distance to similarity

            chunks.append(
                CandidateChunk(
                    page_name=metadata["page_name"],
                    target_id=block_id,
                    content=metadata.get("block_content", ""),
                    similarity_score=similarity,
                )
            )

        return chunks

    def _hinted_search(
        self, knowledge: KnowledgePackage, graph_path: Path
    ) -> List[CandidateChunk]:
        """Perform hinted search by parsing block references.

        Extracts ((block-id)) references from the knowledge text and
        fetches those specific blocks. Supports both standalone ((uuid))
        and markdown link [text](((uuid))) syntax.

        Args:
            knowledge: Knowledge package to extract hints from
            graph_path: Path to Logseq graph

        Returns:
            List of CandidateChunk objects from hinted block references
        """
        # Parse ((block-id)) references from knowledge full_text
        # Matches both standalone ((uuid)) and markdown links [text](((uuid)))
        block_refs = re.findall(r'\(\(\(([^\)]+)\)\)\)|\(\(([^\)]+)\)\)', knowledge.full_text)
        # Flatten the tuple results (regex groups) and filter out empty strings
        block_refs = [ref for group in block_refs for ref in group if ref]

        if not block_refs:
            return []

        chunks = []
        for block_id in set(block_refs):  # Deduplicate
            # Find the block across all pages
            # We need to search through pages to find which one contains this block
            # For now, we'll search through all pages (could be optimized with an index)
            block_info = self._find_block_by_id_in_graph(block_id, graph_path)

            if not block_info:
                logger.debug(f"Hinted block reference not found: {block_id}")
                continue

            page_name, block = block_info

            chunks.append(
                CandidateChunk(
                    page_name=page_name,
                    target_id=block_id,
                    content=block.get_full_content(normalize_whitespace=True),
                    similarity_score=0.95,  # Block refs get very high score (explicit reference)
                )
            )

        return chunks

    def _find_block_by_id_in_graph(
        self, block_id: str, graph_path: Path
    ) -> Optional[Tuple[str, LogseqBlock]]:
        """Find a block by its ID across all pages in the graph.

        Args:
            block_id: Hybrid ID or explicit id:: to find
            graph_path: Path to Logseq graph

        Returns:
            Tuple of (page_name, block) if found, None otherwise
        """
        from logsqueak.logseq.graph import GraphPaths

        graph_paths = GraphPaths(graph_path)

        # Search through all pages
        for page_file in graph_paths.pages_dir.glob("*.md"):
            page_name = page_file.stem
            page = TargetPage.load(graph_path, page_name)
            if not page:
                continue

            # Use the find_block_by_id method from LogseqOutline
            block = page.outline.find_block_by_id(block_id)
            if block:
                return (page_name, block)

        return None

    def _iter_blocks(self, blocks: List[LogseqBlock]):
        """Recursively iterate over all blocks in a tree.

        Args:
            blocks: List of blocks to iterate

        Yields:
            Each block in the tree (depth-first)
        """
        for block in blocks:
            yield block
            if block.children:
                yield from self._iter_blocks(block.children)

    def _aggregate_chunks(
        self, semantic_chunks: List[CandidateChunk], hinted_chunks: List[CandidateChunk]
    ) -> List[CandidateChunk]:
        """Aggregate and deduplicate chunks from semantic and hinted search.

        Combines results, deduplicates by target_id, and prefers semantic
        scores over hinted scores.

        Args:
            semantic_chunks: Results from semantic search
            hinted_chunks: Results from hinted search

        Returns:
            Deduplicated list sorted by similarity score (descending)
        """
        # Build a map of target_id -> CandidateChunk (prefer semantic scores)
        chunk_map = {}

        # Add hinted chunks first (lower priority)
        for chunk in hinted_chunks:
            chunk_map[chunk.target_id] = chunk

        # Add semantic chunks (higher priority, overwrites hints)
        for chunk in semantic_chunks:
            chunk_map[chunk.target_id] = chunk

        # Sort by similarity score descending
        chunks = sorted(chunk_map.values(), key=lambda c: c.similarity_score, reverse=True)

        return chunks

    def process_knowledge_packages(
        self,
        packages: List[KnowledgePackage],
        vector_store: VectorStore,
        graph_path: Path,
        top_k: int = 20,
    ) -> Tuple[List[WriteOperation], dict]:
        """Process knowledge packages through Phase 3 (Decider + Reworder).

        For each knowledge package, retrieves candidate chunks and evaluates
        them with the Decider LLM. For non-IGNORE actions, calls the Reworder
        to generate clean content. Builds a Write List for Phase 4 execution.

        Args:
            packages: List of KnowledgePackage from Phase 1
            vector_store: VectorStore for candidate retrieval
            graph_path: Path to Logseq graph
            top_k: Number of candidate chunks to retrieve (default: 20)

        Returns:
            Tuple of (write_list, processed_blocks_map)
            - write_list: List of WriteOperation for Phase 4
            - processed_blocks_map: Dict[original_id -> List[(page_name, new_id)]]
              (new_id is None until Phase 4 execution)
        """
        write_list: List[WriteOperation] = []
        processed_blocks_map: dict = {}

        logger.info(f"Processing {len(packages)} knowledge packages through Phase 3")

        for package in packages:
            logger.debug(f"Processing package: {package.exact_text[:50]}...")

            # Phase 2: Get candidate chunks
            candidates = self.find_candidate_chunks(
                knowledge=package,
                vector_store=vector_store,
                graph_path=graph_path,
                top_k=top_k,
            )

            if not candidates:
                logger.warning(f"No candidates found for: {package.exact_text[:50]}...")
                continue

            logger.debug(f"Found {len(candidates)} candidates for knowledge package")

            # Track whether we found a match for this knowledge
            found_match = False

            # Phase 3.1: Evaluate each candidate with Decider LLM
            for candidate in candidates:
                # Convert to dict format expected by decide_action
                candidate_dict = {
                    "page_name": candidate.page_name,
                    "target_id": candidate.target_id,
                    "content": candidate.content,
                    "similarity_score": candidate.similarity_score,
                }

                logger.debug(
                    f"Calling Decider for candidate: {candidate.page_name} "
                    f"(score: {candidate.similarity_score:.3f})"
                )

                decision = self.llm_client.decide_action(
                    knowledge_text=package.hierarchical_text,
                    candidate_chunks=[candidate_dict],
                )

                logger.debug(f"Decider action: {decision.action.value}, reasoning: {decision.reasoning[:100]}...")

                # Handle IGNORE actions - skip to next candidate
                if decision.action in (ActionType.IGNORE_ALREADY_PRESENT, ActionType.IGNORE_IRRELEVANT):
                    logger.debug(f"Ignoring candidate: {decision.action.value}")
                    continue

                # Phase 3.2: Call Reworder for UPDATE/APPEND actions
                logger.debug("Calling Reworder to generate clean content")
                rephrased = self.llm_client.rephrase_content(package.hierarchical_text)

                # Determine page_name and target_id based on action
                if decision.action == ActionType.APPEND_ROOT:
                    page_name = decision.page_name
                    target_id = None
                else:
                    # UPDATE or APPEND_CHILD - use target_id to determine page
                    page_name = candidate.page_name
                    target_id = decision.target_id

                # Add to Write List
                write_op = WriteOperation(
                    page_name=page_name,
                    action=decision.action,
                    target_id=target_id,
                    new_content=rephrased.content,
                    original_id=package.original_id,
                    block_locator=package.block_locator,  # Propagate from Phase 1
                )
                write_list.append(write_op)

                logger.info(
                    f"Added write operation: {decision.action.value} to {page_name}"
                )

                # Track in processed blocks map (new_id will be added in Phase 4)
                if package.original_id not in processed_blocks_map:
                    processed_blocks_map[package.original_id] = []
                processed_blocks_map[package.original_id].append((page_name, None))

                found_match = True
                break  # Found a match, stop evaluating candidates for this knowledge

            if not found_match:
                logger.warning(
                    f"No suitable target found for knowledge: {package.exact_text[:50]}..."
                )

        logger.info(
            f"Phase 3 complete: {len(write_list)} write operations generated"
        )
        return write_list, processed_blocks_map

    def extract_and_integrate(
        self,
        journal: "JournalEntry",
        vector_store: VectorStore,
        graph_path: Path,
        top_k: int = 20,
    ) -> int:
        """End-to-end pipeline: Extract knowledge and integrate into pages.

        Runs the complete multi-stage pipeline:
        - Phase 1: Extract knowledge packages from journal
        - Phase 2: Find candidate chunks (in Phase 3 loop)
        - Phase 3: Decider + Reworder to build Write List
        - Phase 4: Execute write operations
        - Phase 4.5: Add processed:: markers to journal

        Args:
            journal: JournalEntry to process
            vector_store: VectorStore for semantic search
            graph_path: Path to Logseq graph
            top_k: Number of candidate chunks to retrieve (default: 20)

        Returns:
            Number of write operations executed

        Raises:
            LLMError: If LLM calls fail
            IOError: If file operations fail
        """
        from logsqueak.integration.executor import execute_write_list

        logger.info(f"Starting end-to-end pipeline for journal: {journal.date}")

        # Phase 1: Extract knowledge packages
        logger.info("Phase 1: Extracting knowledge from journal")
        packages = self.extract_knowledge(journal)

        if not packages:
            logger.info("No knowledge found in journal")
            return 0

        logger.info(f"Extracted {len(packages)} knowledge packages")

        # Phase 2 & 3: Process packages (candidate retrieval + Decider + Reworder)
        logger.info("Phase 2 & 3: Processing knowledge packages")
        write_list, processed_blocks_map = self.process_knowledge_packages(
            packages=packages,
            vector_store=vector_store,
            graph_path=graph_path,
            top_k=top_k,
        )

        if not write_list:
            logger.info("No write operations generated")
            return 0

        # Phase 4: Execute write operations with atomic journal updates
        logger.info("Phase 4: Executing write operations with atomic journal updates")
        journal_path = graph_path / "journals" / f"{journal.date.strftime('%Y_%m_%d')}.md"
        updated_map = execute_write_list(
            write_list=write_list,
            processed_blocks_map=processed_blocks_map,
            graph_path=graph_path,
            journal_path=journal_path,
        )

        logger.info(f"Pipeline complete: {len(write_list)} operations executed")
        return len(write_list)

    def _find_block_by_content(
        self, blocks: List[LogseqBlock], exact_text: str
    ) -> Optional[LogseqBlock]:
        """Find a block in the AST by exact content match.

        Recursively searches through blocks and their children to find
        a block whose content matches the exact_text.

        Args:
            blocks: List of LogseqBlock to search
            exact_text: Exact block content to find

        Returns:
            Matching LogseqBlock or None if not found
        """
        for block in blocks:
            # Strip whitespace for comparison (use first line)
            if block.content[0].strip() == exact_text.strip():
                return block

            # Recursively search children
            if block.children:
                found = self._find_block_by_content(block.children, exact_text)
                if found:
                    return found

        return None

    def _find_block_by_fuzzy_match(
        self, blocks: List[LogseqBlock], approximate_text: str, threshold: float = 0.7
    ) -> Tuple[Optional[LogseqBlock], float]:
        """Find a block in the AST using semantic similarity (fuzzy matching).

        When LLMs (especially smaller models) return approximate text instead of
        exact copies, this method uses embedding-based similarity to find the
        most likely matching block.

        Args:
            blocks: List of LogseqBlock to search
            approximate_text: Approximate block content from LLM
            threshold: Minimum similarity score (0.0-1.0) to accept match (default: 0.7)

        Returns:
            Tuple of (matching_block, similarity_score) or (None, 0.0) if no match above threshold

        Example:
            LLM returns: "The service will be read-only"
            Actual block: "This means that the service will be read-only"
            Similarity: 0.92 → Match!
        """
        # Lazy-load embedding model
        if self._embedding_model is None:
            logger.info("Loading embedding model for fuzzy matching...")
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Flatten all blocks into a list with their content
        all_blocks = []

        def collect_blocks(block_list: List[LogseqBlock]) -> None:
            for block in block_list:
                all_blocks.append(block)
                if block.children:
                    collect_blocks(block.children)

        collect_blocks(blocks)

        if not all_blocks:
            return None, 0.0

        # Extract content from all blocks (use full content, normalized)
        block_contents = [block.get_full_content(normalize_whitespace=True) for block in all_blocks]

        # Encode all blocks and the query text
        logger.debug(f"Encoding {len(all_blocks)} blocks for fuzzy matching...")
        block_embeddings = self._embedding_model.encode(block_contents, convert_to_numpy=True)
        query_embedding = self._embedding_model.encode([approximate_text.strip()], convert_to_numpy=True)[0]

        # Compute cosine similarities
        similarities = np.dot(block_embeddings, query_embedding) / (
            np.linalg.norm(block_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # Find best match
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        if best_score >= threshold:
            logger.debug(
                f"Best match: '{block_contents[best_idx][:50]}...' "
                f"(score: {best_score:.3f})"
            )
            return all_blocks[best_idx], best_score
        else:
            logger.debug(
                f"Best candidate score {best_score:.3f} below threshold {threshold}"
            )
            return None, 0.0

    def _build_full_context(
        self, block: LogseqBlock, root_blocks: List[LogseqBlock]
    ) -> str:
        """Build full-context text by walking up the AST tree.

        Finds all parent blocks and prepends their content to create
        full context. Includes [[Page Links]] from parents.

        Args:
            block: The target block
            root_blocks: Root-level blocks to search for parents

        Returns:
            Full context text with parent bullets prepended

        Example:
            Journal structure:
            - Working on [[Project X]]
              - Updated documentation
                - Added API examples

            For "Added API examples" block, returns:
            "[[Project X]]: Updated documentation - Added API examples"
        """
        # Find parent chain by walking up the tree
        parents = self._find_parent_chain(block, root_blocks)

        if not parents:
            # No parents, return full block content
            return block.get_full_content()

        # Build context from parents
        context_parts = []
        for parent in parents:
            # Extract [[Page Links]] and key phrases (use full content, normalized)
            content = parent.get_full_content(normalize_whitespace=True)

            # If parent has [[Page Link]], definitely include it
            if "[[" in content and "]]" in content:
                context_parts.append(content)
            # Otherwise, include if it's a meaningful heading/context
            elif len(content) > 0:
                context_parts.append(content)

        # Add the block's own full content (normalized)
        context_parts.append(block.get_full_content(normalize_whitespace=True))

        # Join with appropriate separator
        if len(context_parts) == 1:
            return context_parts[0]
        elif len(context_parts) == 2:
            # parent: child
            return f"{context_parts[0]}: {context_parts[1]}"
        else:
            # parent: intermediate - child
            return f"{context_parts[0]}: {' - '.join(context_parts[1:])}"

    def _build_hierarchical_context(
        self, block: LogseqBlock, root_blocks: List[LogseqBlock], indent_str: str = "  "
    ) -> str:
        """Build hierarchical Logseq markdown showing block with parent context.

        Creates a hierarchical representation showing parent bullets leading to
        the target block, preserving the tree structure for better LLM understanding.
        Uses LogseqOutline.render() to ensure consistent formatting.

        Args:
            block: The target block
            root_blocks: Root-level blocks to search for parents
            indent_str: Indentation string (e.g., "  " or "\t")

        Returns:
            Hierarchical markdown with parent bullets

        Example:
            Journal structure:
            - Working on [[Project X]]
              - Updated documentation
                - Added API examples

            For "Added API examples" block, returns:
            - Working on [[Project X]]
              - Updated documentation
                - Added API examples
        """
        from logsqueak.logseq.parser import LogseqOutline

        # Find parent chain by walking up the tree
        parents = self._find_parent_chain(block, root_blocks)

        if not parents:
            # No parents, create a simple outline with just the block
            outline = LogseqOutline(
                blocks=[LogseqBlock(content=block.content, indent_level=0, children=[])],
                source_text="",  # Not needed for rendering
                indent_str=indent_str,
                frontmatter=[]
            )
            return outline.render()

        # Build a hierarchical structure by nesting blocks
        # Start from the root parent and build down to the target block
        root_parent = LogseqBlock(
            content=parents[0].content.copy(),
            indent_level=0,
            children=[]
        )

        current = root_parent
        for parent in parents[1:]:
            child = LogseqBlock(
                content=parent.content.copy(),
                indent_level=current.indent_level + 1,
                children=[]
            )
            current.children.append(child)
            current = child

        # Add the target block as the final child
        target = LogseqBlock(
            content=block.content.copy(),
            indent_level=current.indent_level + 1,
            children=[]
        )
        current.children.append(target)

        # Create an outline and render it
        outline = LogseqOutline(
            blocks=[root_parent],
            source_text="",  # Not needed for rendering
            indent_str=indent_str,
            frontmatter=[]
        )

        return outline.render()

    def _find_parent_chain(
        self, target: LogseqBlock, blocks: List[LogseqBlock], parents: Optional[List[LogseqBlock]] = None
    ) -> List[LogseqBlock]:
        """Find the chain of parent blocks for a target block.

        Recursively walks the tree to find parents.

        Args:
            target: Block to find parents for
            blocks: Blocks to search
            parents: Accumulated parents (used in recursion)

        Returns:
            List of parent blocks from root to immediate parent
        """
        if parents is None:
            parents = []

        for block in blocks:
            # Check if target is a direct child
            if target in block.children:
                return parents + [block]

            # Recursively search children
            if block.children:
                result = self._find_parent_chain(target, block.children, parents + [block])
                if result:
                    return result

        return []


def _format_preview_with_frontmatter(outline) -> str:
    """Format page preview with frontmatter clearly marked.

    This wraps any frontmatter in XML tags to help the LLM distinguish
    between page-level properties (which provide context) and actual
    sections (which are valid placement targets).

    Args:
        outline: LogseqOutline to format

    Returns:
        Formatted preview with frontmatter in <frontmatter> tags
    """
    if not outline.frontmatter:
        return outline.render()

    # Separate frontmatter from blocks
    frontmatter_text = "\n".join(outline.frontmatter)

    # Render just the blocks (without frontmatter)
    lines = []
    for block in outline.blocks:
        lines.extend(_render_block(block, outline.indent_str))
    blocks_text = "\n".join(lines)

    # Combine with frontmatter wrapped in XML tags
    return f"<frontmatter>\n{frontmatter_text}\n</frontmatter>\n{blocks_text}"


def _render_block(block, indent_str: str) -> list[str]:
    """Helper to render a block and its children.

    Args:
        block: LogseqBlock to render
        indent_str: Indentation string

    Returns:
        List of rendered lines
    """
    lines = []
    indent = indent_str * block.indent_level

    # Render first line with bullet
    lines.append(f"{indent}- {block.content[0]}")

    # Render subsequent lines (continuation/properties) with "  " instead of "- "
    for line in block.content[1:]:
        lines.append(f"{indent}  {line}")

    for child in block.children:
        lines.extend(_render_block(child, indent_str))

    return lines
