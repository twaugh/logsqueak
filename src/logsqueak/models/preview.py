"""Extraction preview model for showing proposed changes before applying."""

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Optional

from logsqueak.models.knowledge import KnowledgeBlock


class ActionStatus(Enum):
    """Status of a proposed action."""

    READY = "ready"  # Ready to integrate
    SKIPPED = "skipped"  # Skipped (duplicate or missing page)
    WARNING = "warning"  # Will proceed but user should review


@dataclass
class ProposedAction:
    """A single proposed integration action.

    Attributes:
        knowledge: Knowledge block to integrate
        status: Action status (READY/SKIPPED/WARNING)
        reason: Why skipped/warned (if applicable)
    """

    knowledge: KnowledgeBlock
    status: ActionStatus
    reason: Optional[str] = None

    def describe(self) -> str:
        """Generate human-readable description of action.

        Returns:
            Multi-line description for terminal display
        """
        kb = self.knowledge
        parts = [
            f'  "{kb.content[:60]}..."' if len(kb.content) > 60 else f'  "{kb.content}"',
            f"  → Target: {kb.target_page}",
            f"  → Section: {kb.section_path()}",
            f"  → Action: {kb.suggested_action.value}",
        ]

        if self.status == ActionStatus.SKIPPED:
            parts.append(f"  ⚠ SKIPPED: {self.reason}")
        elif self.status == ActionStatus.WARNING:
            parts.append(f"  ⚠ WARNING: {self.reason}")

        return "\n".join(parts)


@dataclass
class ExtractionPreview:
    """Summary of proposed changes shown to user before applying (FR-002).

    Attributes:
        journal_date: Source journal being processed
        knowledge_blocks: All extracted knowledge
        proposed_actions: What will happen to each block
        warnings: Issues found (missing pages, duplicates, etc.)
    """

    journal_date: date
    knowledge_blocks: list[KnowledgeBlock]
    proposed_actions: list[ProposedAction]
    warnings: list[str]

    def display(self) -> str:
        """Render preview for terminal display.

        Returns:
            Formatted preview text
        """
        lines = [
            f"Found {len(self.knowledge_blocks)} knowledge blocks in "
            f"journals/{self.journal_date.isoformat().replace('-', '_')}.md:",
            "",
        ]

        for i, action in enumerate(self.proposed_actions, 1):
            lines.append(f"{i}. {action.describe()}")
            lines.append("")

        if self.warnings:
            lines.append("Warnings:")
            for warning in self.warnings:
                lines.append(f"  ⚠ {warning}")
            lines.append("")

        # Summary
        ready_count = sum(1 for a in self.proposed_actions if a.status == ActionStatus.READY)
        skipped_count = sum(
            1 for a in self.proposed_actions if a.status == ActionStatus.SKIPPED
        )

        lines.append("Summary:")
        lines.append(f"  - Knowledge blocks found: {len(self.knowledge_blocks)}")
        lines.append(f"  - Will integrate: {ready_count}")
        lines.append(f"  - Skipped: {skipped_count}")
        lines.append("")

        return "\n".join(lines)
