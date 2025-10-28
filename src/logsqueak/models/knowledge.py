"""Knowledge block model extracted from journal entries."""

from dataclasses import dataclass
from enum import Enum


@dataclass
class KnowledgePackage:
    """Knowledge extracted from journal with full context (M4.1+).

    This is the output of Phase 1 extraction in the new multi-stage pipeline.
    Contains the original block ID and full-context text (with parent context).

    Attributes:
        original_id: Hybrid ID of source journal block (id:: or content hash)
        exact_text: Exact block text as returned by LLM (no parent context)
        full_text: Full context text (exact_text + parent context from AST walk, flattened)
        hierarchical_text: Hierarchical Logseq markdown showing block + parents (for reworder)
        confidence: LLM confidence score (0.0-1.0) for classification
    """

    original_id: str
    exact_text: str
    full_text: str
    hierarchical_text: str
    confidence: float


class ActionType(Enum):
    """Type of action to take when integrating knowledge.

    Actions (M4+ multi-stage pipeline):
    - IGNORE_ALREADY_PRESENT: Knowledge already exists in target
    - IGNORE_IRRELEVANT: Knowledge not relevant to this candidate
    - UPDATE: Modify existing block content in place
    - APPEND_CHILD: Add as child to specific target block
    - APPEND_ROOT: Add at page root level
    """

    IGNORE_ALREADY_PRESENT = "ignore_already_present"
    IGNORE_IRRELEVANT = "ignore_irrelevant"
    UPDATE = "update"
    APPEND_CHILD = "append_child"
    APPEND_ROOT = "append_root"
