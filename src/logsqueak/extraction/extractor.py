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

from logsqueak.llm.client import (
    ExtractionResult,
    LLMClient,
    PageCandidate,
    PageSelectionResult,
)
from logsqueak.llm.prompts import build_page_selection_messages
from logsqueak.llm.token_counter import TokenCounter
from logsqueak.logseq.parser import LogseqBlock
from logsqueak.models.journal import JournalEntry
from logsqueak.models.knowledge import ActionType, KnowledgeBlock, KnowledgePackage
from logsqueak.models.page import PageIndex, TargetPage
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
    """

    page_name: str
    action: ActionType
    target_id: Optional[str]
    new_content: str
    original_id: str  # For tracking back to journal block


class Extractor:
    """Orchestrates knowledge extraction from journal entries.

    Uses LLM to identify lasting knowledge (vs. activity logs) and
    extract knowledge blocks with confidence scores.
    """

    def __init__(self, llm_client: LLMClient, model: str):
        """Initialize the extractor.

        Args:
            llm_client: LLM client for knowledge extraction
            model: Model name for token counting
        """
        self.llm_client = llm_client
        self.token_counter = TokenCounter(model)

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
            exact_text = result.content  # LLM returns exact block text

            # Find the block in the journal AST
            block = self._find_block_by_content(journal.outline.blocks, exact_text)

            if block is None:
                logger.warning(f"Could not find block in AST for text: {exact_text[:50]}...")
                continue

            # Find parent chain for hybrid ID generation
            parents = self._find_parent_chain(block, journal.outline.blocks)

            # Generate original_id using hybrid ID system
            original_id = block.get_hybrid_id(parents=parents, indent_str=journal.outline.indent_str)

            # Build full_text with parent context
            full_text = self._build_full_context(block, journal.outline.blocks)

            packages.append(
                KnowledgePackage(
                    original_id=original_id,
                    exact_text=exact_text,
                    full_text=full_text,
                    confidence=result.confidence,
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
                    content=metadata.get("content", ""),
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
                    content=block.content,
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
                    knowledge_text=package.full_text,
                    candidate_chunks=[candidate_dict],
                )

                logger.debug(f"Decider action: {decision.action.value}, reasoning: {decision.reasoning[:100]}...")

                # Handle IGNORE actions - skip to next candidate
                if decision.action in (ActionType.IGNORE_ALREADY_PRESENT, ActionType.IGNORE_IRRELEVANT):
                    logger.debug(f"Ignoring candidate: {decision.action.value}")
                    continue

                # Phase 3.2: Call Reworder for UPDATE/APPEND actions
                logger.debug("Calling Reworder to generate clean content")
                rephrased = self.llm_client.rephrase_content(package.full_text)

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

    def select_target_page(
        self,
        knowledge_content: str,
        page_index: PageIndex,
        indent_str: str = "  ",
        token_budget: Optional[int] = None,
    ) -> PageSelectionResult:
        """Select target page and section for knowledge (Stage 2).

        This is the second stage of the two-stage extraction process.
        Uses RAG to find semantically similar pages, then LLM
        selects the best match with section and action.

        Uses token budget to determine how many candidates to include.
        More candidates = better LLM choice but more tokens.

        Args:
            knowledge_content: Extracted knowledge text
            page_index: PageIndex for RAG search
            indent_str: Indentation string from source (e.g., "  ", "\t")
            token_budget: Optional token budget for Stage 2 prompt (default: None = use top 5)

        Returns:
            Page selection result with target page, section, and action

        Raises:
            LLMError: If LLM request fails
        """
        # Determine how many candidates to retrieve based on token budget
        if token_budget is None:
            # No budget specified, use default of 5 candidates
            num_candidates = 5
            logger.info(
                f"No token budget specified, using default of {num_candidates} candidates"
            )
            similar_pages = page_index.find_similar(knowledge_content, top_k=num_candidates)
        else:
            # Use token budget to select candidates
            logger.info(f"Using token budget of {token_budget} tokens for candidate selection")
            similar_pages = self._select_candidates_within_budget(
                knowledge_content, page_index, token_budget, indent_str
            )

        # Convert to PageCandidate format for LLM
        candidates = []
        for page, similarity_score in similar_pages:
            # Get first 1000 characters of page content as preview
            # This matches the preview length used for RAG embeddings and gives
            # the LLM enough context to identify existing sections
            preview = _format_preview_with_frontmatter(page.outline)[:1000]
            candidates.append(
                PageCandidate(
                    page_name=page.name,
                    similarity_score=similarity_score,
                    preview=preview,
                )
            )

        # Call LLM to select best page from candidates
        selection = self.llm_client.select_target_page(
            knowledge_content=knowledge_content,
            candidates=candidates,
            indent_str=indent_str,
        )

        return selection

    def _select_candidates_within_budget(
        self,
        knowledge_content: str,
        page_index: PageIndex,
        token_budget: int,
        indent_str: str,
    ) -> list[tuple[TargetPage, float]]:
        """Select as many candidates as fit within token budget.

        Starts with top-ranked page and adds more until budget is exhausted.

        Args:
            knowledge_content: Knowledge block content
            page_index: PageIndex for RAG search
            token_budget: Maximum tokens for Stage 2 prompt
            indent_str: Indentation string for preview formatting

        Returns:
            List of (page, similarity_score) tuples that fit within budget
        """
        # Get all pages sorted by similarity (we'll take as many as fit)
        all_candidates = page_index.find_similar(knowledge_content, top_k=20)

        if not all_candidates:
            return []

        # Calculate base prompt tokens (system prompt + knowledge content + response overhead)
        base_tokens = self._calculate_base_prompt_tokens(knowledge_content, indent_str)

        # Reserve tokens for LLM response (JSON output ~200-300 tokens)
        response_overhead = 300
        available_tokens = token_budget - base_tokens - response_overhead

        if available_tokens <= 0:
            logger.warning(
                f"Token budget {token_budget} too small for base prompt ({base_tokens} tokens). "
                "Using single candidate."
            )
            return [all_candidates[0]]

        # Add candidates one by one until we run out of budget
        selected = []
        tokens_used = 0

        for page, similarity_score in all_candidates:
            # Calculate tokens for this candidate's preview
            preview = _format_preview_with_frontmatter(page.outline)[:1000]
            candidate_text = f"Page: {page.name}\nSimilarity: {similarity_score:.2f}\nPreview:\n{preview}\n\n"
            candidate_tokens = self.token_counter.count_tokens(candidate_text)

            if tokens_used + candidate_tokens <= available_tokens:
                selected.append((page, similarity_score))
                tokens_used += candidate_tokens
                logger.debug(
                    f"Added candidate '{page.name}' ({candidate_tokens} tokens, "
                    f"total: {tokens_used}/{available_tokens})"
                )
            else:
                logger.debug(
                    f"Skipping candidate '{page.name}' ({candidate_tokens} tokens) - "
                    f"would exceed budget ({tokens_used + candidate_tokens} > {available_tokens})"
                )
                break

        if not selected:
            # Edge case: even one candidate is too large, but we need at least one
            logger.warning(
                f"First candidate too large for budget, including it anyway "
                f"({all_candidates[0][0].name})"
            )
            selected = [all_candidates[0]]

        logger.info(
            f"Selected {len(selected)} candidates within {token_budget} token budget "
            f"(used {tokens_used + base_tokens + response_overhead} total)"
        )

        return selected

    def _calculate_base_prompt_tokens(self, knowledge_content: str, indent_str: str) -> int:
        """Calculate exact tokens for base Stage 2 prompt (without candidates).

        Uses the shared prompt builder to construct the actual prompts and counts
        their tokens. This ensures token counting matches what the LLM provider sends.

        Args:
            knowledge_content: Knowledge block content
            indent_str: Indentation string

        Returns:
            Exact token count for base prompt (without candidates)
        """
        # Build messages with empty candidates list to get base prompt tokens
        messages = build_page_selection_messages(
            knowledge_content=knowledge_content,
            candidates=[],  # Empty candidates for base calculation
            indent_str=indent_str,
        )

        return self.token_counter.count_messages_tokens(messages)

    def is_duplicate(self, knowledge_content: str, target_page: TargetPage) -> bool:
        """Check if knowledge already exists on target page (FR-017).

        Uses simple content matching to detect duplicates. This prevents
        re-adding the same knowledge if it was previously integrated.

        Args:
            knowledge_content: Extracted knowledge text
            target_page: Target page to check for duplicates

        Returns:
            True if duplicate found, False otherwise

        Examples:
            >>> page = TargetPage.load(graph_path, "Project X")
            >>> extractor.is_duplicate("Already on the page", page)
            True
            >>> extractor.is_duplicate("New unique knowledge", page)
            False
        """
        return target_page.has_duplicate(knowledge_content)

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
            # Strip whitespace for comparison
            if block.content.strip() == exact_text.strip():
                return block

            # Recursively search children
            if block.children:
                found = self._find_block_by_content(block.children, exact_text)
                if found:
                    return found

        return None

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
            # No parents, return exact text
            return block.content

        # Build context from parents
        context_parts = []
        for parent in parents:
            # Extract [[Page Links]] and key phrases
            content = parent.content.strip()

            # If parent has [[Page Link]], definitely include it
            if "[[" in content and "]]" in content:
                context_parts.append(content)
            # Otherwise, include if it's a meaningful heading/context
            elif len(content) > 0:
                context_parts.append(content)

        # Add the block's own content
        context_parts.append(block.content.strip())

        # Join with appropriate separator
        if len(context_parts) == 1:
            return context_parts[0]
        elif len(context_parts) == 2:
            # parent: child
            return f"{context_parts[0]}: {context_parts[1]}"
        else:
            # parent: intermediate - child
            return f"{context_parts[0]}: {' - '.join(context_parts[1:])}"

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
    lines.append(f"{indent}- {block.content}")

    if block.continuation_lines:
        lines.extend(block.continuation_lines)

    for child in block.children:
        lines.extend(_render_block(child, indent_str))

    return lines


def create_knowledge_block(
    extraction: ExtractionResult,
    source_date: date,
    target_page: str,
    target_section: List[str] | None,
    suggested_action: ActionType,
) -> KnowledgeBlock:
    """Create a KnowledgeBlock from extraction result and page selection.

    This is a helper function used after both Stage 1 (extraction) and
    Stage 2 (page selection) have completed.

    Args:
        extraction: Stage 1 extraction result
        source_date: Journal date
        target_page: Selected target page (from Stage 2)
        target_section: Selected section path (from Stage 2)
        suggested_action: How to integrate (from Stage 2)

    Returns:
        Complete KnowledgeBlock ready for integration
    """
    return KnowledgeBlock(
        content=extraction.content,
        source_date=source_date,
        confidence=extraction.confidence,
        target_page=target_page,
        target_section=target_section,
        suggested_action=suggested_action,
    )
