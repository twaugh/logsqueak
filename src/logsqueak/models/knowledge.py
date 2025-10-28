"""Knowledge block model extracted from journal entries."""

import hashlib
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Optional


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


@dataclass
class KnowledgeBlock:
    """A piece of information extracted from a journal with lasting value.

    DEPRECATED: This class is part of the old 2-stage pipeline.
    The new 5-phase pipeline uses KnowledgePackage instead.

    This represents knowledge that should be preserved (vs. temporary activity logs).

    Attributes:
        content: The knowledge text content
        source_date: Journal date this was extracted from
        confidence: LLM confidence score (0.0-1.0) for classification
        target_page: Page name where this should be integrated
        target_section: Hierarchical path to target location (e.g., ["Projects", "Timeline"])
        target_block_id: Hybrid ID for precise block targeting (either an explicit id:: or content hash)
        suggested_action: How to integrate
    """

    content: str
    source_date: date
    confidence: float
    target_page: str
    target_section: Optional[list[str]]
    target_block_id: Optional[str] = None
    suggested_action: ActionType = ActionType.APPEND_CHILD

    def content_hash(self) -> str:
        """Generate hash for duplicate detection (FR-017).

        Returns:
            First 8 characters of MD5 hash of content
        """
        return hashlib.md5(self.content.encode()).hexdigest()[:8]

    def provenance_link(self) -> str:
        """Generate Logseq page link to source journal (FR-003).

        Returns:
            Logseq page link format: [[YYYY-MM-DD]]
        """
        return f"[[{self.source_date.isoformat()}]]"

    def with_provenance(self) -> str:
        """Return content with provenance link appended.

        Returns:
            Content with provenance link: "content (see [[YYYY-MM-DD]] entry)"
        """
        return f"{self.content} (see {self.provenance_link()} entry)"

    def section_path(self) -> str:
        """Format section path for display.

        Returns:
            Human-readable section path (e.g., "Projects > Timeline")
            or "(page root)" if no section specified
        """
        if not self.target_section:
            return "(page root)"
        # Filter out None values in case LLM returns them
        valid_sections = [s for s in self.target_section if s is not None]
        if not valid_sections:
            return "(page root)"
        return " > ".join(valid_sections)
