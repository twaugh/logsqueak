"""TUI-specific state models.

This module defines the data structures that manage state across all TUI screens
during the interactive knowledge extraction workflow.
"""

from dataclasses import dataclass, field
from typing import Literal

from logsqueak.models.config import Configuration
from logsqueak.llm.client import LLMClient
from logsqueak.models.journal import JournalEntry


@dataclass
class BlockState:
    """
    Classification state for a single journal block.

    Lifecycle:
    - Created on Phase1Screen mount (all blocks initialized as "pending")
    - Updated during LLM streaming (classification + confidence)
    - Updated by user overrides (classification changed, source="user")
    - Read in Phase 2/3 to filter knowledge blocks
    """

    block_id: str
    """
    Hybrid ID from journal block (id:: property or content hash).
    Must match LogseqBlock.id from parsed journal entry.
    """

    classification: Literal["pending", "knowledge", "activity"]
    """
    Current classification state:
    - "pending": Waiting for LLM analysis or user decision
    - "knowledge": Lasting insight worth integrating
    - "activity": Daily log entry (skip integration)
    """

    confidence: float | None
    """
    LLM confidence score (0.0-1.0) or None if user-marked.

    Guidelines:
    - 0.90-1.00: High confidence (normal display)
    - 0.75-0.89: Good confidence (normal display)
    - 0.60-0.74: Warning range (yellow color, suggest review)
    - 0.00-0.59: Low confidence (⚠ indicator, definitely review)
    - None: User-marked classification (no LLM confidence)
    """

    source: Literal["user", "llm"]
    """
    Origin of classification decision:
    - "user": User manually marked (cannot be overridden by LLM)
    - "llm": LLM classified (can be overridden by user)
    """

    llm_classification: Literal["knowledge", "activity"] | None = None
    """
    Original LLM classification (preserved when user overrides).

    - None: LLM hasn't classified this block yet
    - "knowledge"/"activity": LLM's original decision before user override

    Used for smart reset behavior (R key):
    - If llm_classification is not None: Reset to LLM's original decision
    - If llm_classification is None: Reset to "pending" (LLM hasn't decided yet)
    """

    llm_confidence: float | None = None
    """
    Original LLM confidence score (preserved when user overrides).

    - None: LLM hasn't classified this block yet, or had no confidence
    - 0.0-1.0: LLM's original confidence before user override

    Used to restore confidence when user resets to LLM decision.
    Always paired with llm_classification (both None or both set).
    """

    reason: str | None = None
    """
    Brief explanation (7-10 words) of why this block is knowledge.

    - None: Block is activity, or LLM hasn't classified yet
    - str: LLM's explanation for knowledge classification

    Example: "documents decision to use ChromaDB for vector storage"

    Used for UI display when block is highlighted (shown at bottom of screen).
    Only populated for knowledge blocks.
    """


@dataclass
class CandidatePage:
    """
    RAG search result for a single knowledge block.

    Lifecycle:
    - Created in Phase 2 (RAG search: semantic + hinted)
    - Updated by user in review mode (included toggled)
    - Read in Phase 3.1 to determine which pages to evaluate
    """

    page_name: str
    """
    Page name without extension.
    Example: "Software Architecture" (not "pages/Software Architecture.md")
    """

    similarity_score: float
    """
    Semantic match score from RAG vector search (0.0-1.0).

    Higher scores = more semantically similar to knowledge block content.
    Used for:
    - Sorting candidates (highest match first)
    - UI display (89% match)
    - User decision-making (exclude low-match pages)

    Note: Hinted pages (from [[Page Links]]) may have similarity_score=1.0
          to prioritize them over purely semantic matches.
    """

    included: bool = True
    """
    User selection state for Phase 3 evaluation.

    - True: Include in Phase 3 Decider/Reworder (default)
    - False: Exclude from Phase 3 (user deselected in review mode)

    Default: True (all candidates initially included)
    """

    blocks: list[dict] = field(default_factory=list)
    """
    Blocks within the page available for targeting.

    Each block dict contains:
    {
        "id": str,           # Hybrid ID (id:: property or content hash)
        "content": str,      # First line of block (for UI display)
        "depth": int,        # Nesting level (0=root, 1=child, etc.)
        "parent_id": str | None  # Parent block ID (None for root blocks)
    }

    Used for:
    - Phase 3 UI: "Pick location" dialog shows these blocks
    - Phase 3 Decider: LLM selects target_block_id from this list
    - Phase 4 Execution: find_block_by_id() looks up these IDs

    Empty list is valid (page has no blocks yet, only APPEND_ROOT allowed).
    """

    search_method: Literal["semantic", "hinted"] = "semantic"
    """
    How this page was discovered (informational only).

    - "semantic": From vector similarity search
    - "hinted": From [[Page Name]] link in journal entry

    Hinted pages are typically prioritized higher (similarity_score=1.0).
    """


@dataclass
class IntegrationDecision:
    """
    Decision about where and how to integrate a knowledge block.

    Lifecycle:
    - Created in Phase 3.1 (Decider) with action + confidence
    - Updated in Phase 3.2 (Reworder) with refined_text
    - Updated by user overrides (action/target changed, source="user")
    - Executed in Phase 4 (converted to WriteOperation)
    """

    knowledge_block_id: str
    """
    Hybrid ID of the source knowledge block from journal entry.
    Must reference a BlockState with classification="knowledge".
    """

    target_page: str
    """
    Destination page name (without extension).
    Example: "Software Architecture" (not "pages/Software Architecture.md")
    """

    action: Literal["skip", "add_section", "add_under", "replace"]
    """
    User-friendly action type (friendly terminology):
    - "skip": Don't integrate (duplicate, irrelevant, too vague)
    - "add_section": APPEND_ROOT - Add as top-level block
    - "add_under": APPEND_CHILD - Add as child of target_block_id
    - "replace": UPDATE - Replace content of target_block_id

    Technical mapping:
      skip → IGNORE_* (various ignore reasons)
      add_section → APPEND_ROOT
      add_under → APPEND_CHILD
      replace → UPDATE
    """

    target_block_id: str | None
    """
    Hybrid ID of target block for "add_under" or "replace" actions.

    - Required for: "add_under", "replace"
    - Must be None for: "skip", "add_section"
    - References a block within target_page (resolved via RAG chunk data)
    """

    target_block_title: str | None
    """
    Human-readable title/preview of target block for UI display.

    Example: "Bounded Context Pattern" (first line of target block)
    Used in UI: "Add under 'Bounded Context Pattern'" vs "Add under 'abc123...'"

    - Required for: "add_under", "replace"
    - Should be None for: "skip", "add_section"
    """

    confidence: float
    """
    LLM decision confidence (0.0-1.0).

    Even user-marked decisions store the original LLM confidence
    for reference. This allows users to see "I overrode an 85% confidence
    decision" in future UX improvements.
    """

    refined_text: str
    """
    Reworded content for evergreen knowledge base (Phase 3.2 output).

    - For action="skip": Empty string (no content to refine)
    - For other actions: LLM-reworded text, optionally user-edited

    Transformation:
      Journal: "Had great discussion with Sarah about microservices today"
      Refined: "Bounded contexts should drive microservice boundaries
                rather than technical layer separation."
    """

    source: Literal["user", "llm"]
    """
    Origin of decision:
    - "user": User modified action/target/text (locked against LLM override)
    - "llm": LLM-generated decision (can be overridden by user)
    """

    skip_reason: str | None = None
    """
    Optional explanation for skip actions (UI display only).

    Examples:
    - "Already covered in this page"
    - "Not relevant to page topic"
    - "Too vague to integrate"

    Not used for Phase 4 execution, purely informational.
    """


@dataclass
class ScreenState:
    """
    Shared state container for the entire extraction session.

    Lifecycle:
    - Created on ExtractionApp mount
    - Passed to each Screen via constructor
    - Mutated by screens as user progresses through phases
    - Consumed by Phase 4 for write execution
    """

    current_phase: int
    """
    Active phase number (1-4).

    Used for:
    - Navigation tracking (which screen to display)
    - Progress indicators (Phase 2/4)
    - Validation (can't skip to Phase 3 without completing Phase 1)
    """

    journal_entry: JournalEntry
    """
    Loaded journal data (parsed LogseqOutline).

    Contains:
    - date: Journal date (YYYY-MM-DD)
    - root_blocks: list[LogseqBlock] (hierarchical structure)

    Immutable after loading - used as source of truth for block IDs,
    content, and hierarchy throughout all phases.
    """

    block_states: dict[str, BlockState]
    """
    Map of block_id → BlockState for all blocks in journal entry.

    Populated in Phase 1 on mount (all initialized as "pending").
    Updated during:
    - Phase 1: LLM streaming + user overrides
    - Phase 2-4: Read-only (filter for classification="knowledge")

    Key invariant: len(block_states) == total_block_count(journal_entry)
    """

    candidates: dict[str, list[CandidatePage]]
    """
    Map of knowledge_block_id → list[CandidatePage] from RAG search.

    Populated in Phase 2 (RAG search: semantic + hinted).
    Updated during:
    - Phase 2: RAG retrieval + user review (toggling included)
    - Phase 3-4: Read-only (filter for included=True)

    Only contains entries for blocks with classification="knowledge".
    """

    decisions: dict[tuple[str, str], IntegrationDecision]
    """
    Map of (knowledge_block_id, target_page) → IntegrationDecision.

    Populated in Phase 3 (Decider + Reworder + user overrides).
    Updated during:
    - Phase 3.1: Decider creates initial decisions
    - Phase 3.2: Reworder adds refined_text
    - Phase 3: User overrides (action/target/text changes)
    - Phase 4: Read-only (convert to WriteOperations)

    Key structure allows O(1) lookup for any (block, page) pair.
    Multiple decisions can share same knowledge_block_id (multi-page integration).
    """

    config: Configuration
    """
    User configuration (from ~/.config/logsqueak/config.yaml).

    Used for:
    - LLM client initialization (endpoint, API key, models)
    - Graph path resolution
    - RAG parameters (top_k)

    Immutable after app initialization.
    """

    llm_client: LLMClient
    """
    Shared LLM client instance for all phases.

    Provides streaming methods:
    - stream_extract_ndjson() - Phase 1
    - stream_decisions_ndjson() - Phase 3.1 + 3.2

    Initialized once on app mount, reused across screens.
    """
