"""Knowledge block model extracted from journal entries."""

import hashlib
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Optional


class ActionType(Enum):
    """Type of action to take when integrating knowledge."""

    ADD_CHILD = "add_child"  # Add as child bullet under existing section
    CREATE_SECTION = "create_section"  # Create new organizational bullet


@dataclass
class KnowledgeBlock:
    """A piece of information extracted from a journal with lasting value.

    This represents knowledge that should be preserved (vs. temporary activity logs).

    Attributes:
        content: The knowledge text content
        source_date: Journal date this was extracted from
        confidence: LLM confidence score (0.0-1.0) for classification
        target_page: Page name where this should be integrated
        target_section: Hierarchical path to target location (e.g., ["Projects", "Timeline"])
        suggested_action: How to integrate (ADD_CHILD or CREATE_SECTION)
    """

    content: str
    source_date: date
    confidence: float
    target_page: str
    target_section: Optional[list[str]]
    suggested_action: ActionType

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
