"""Knowledge extraction orchestrator.

This module implements the two-stage extraction process:
1. Stage 1: Extract knowledge blocks from journal (content + confidence)
2. Stage 2: Match knowledge to target pages using RAG candidates
"""

import logging
from datetime import date
from typing import List, Optional

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

logger = logging.getLogger(__name__)


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
