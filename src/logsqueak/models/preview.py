"""Extraction preview model for showing proposed changes before applying."""

from dataclasses import dataclass
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Optional

from logsqueak.logseq.parser import LogseqOutline
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
        similarity_score: Semantic similarity score from RAG (0.0-1.0)
    """

    knowledge: KnowledgeBlock
    status: ActionStatus
    reason: Optional[str] = None
    similarity_score: Optional[float] = None

    def describe(self) -> str:
        """Generate human-readable description of action.

        Returns:
            Multi-line description for terminal display
        """
        kb = self.knowledge
        parts = [
            f'  "{kb.content[:60]}..."' if len(kb.content) > 60 else f'  "{kb.content}"',
        ]

        # Show target with similarity score if available
        if self.similarity_score is not None:
            parts.append(f"  → Target: {kb.target_page} (similarity: {self.similarity_score:.2f})")
        else:
            parts.append(f"  → Target: {kb.target_page}")

        parts.extend([
            f"  → Section: {kb.section_path()}",
            f"  → Action: {kb.suggested_action.value}",
        ])

        if self.status == ActionStatus.SKIPPED:
            parts.append(f"  ⚠ SKIPPED: {self.reason}")
        elif self.status == ActionStatus.WARNING:
            parts.append(f"  ⚠ WARNING: {self.reason}")

        return "\n".join(parts)

    def show_diff(self, target_page: "TargetPage") -> Optional[str]:
        """Generate diff showing what will be added to the page.

        Args:
            target_page: Target page to show diff for

        Returns:
            Diff string or None if page doesn't exist
        """
        from logsqueak.models.diff import generate_unified_diff
        from logsqueak.integration import writer
        from unittest.mock import Mock

        kb = self.knowledge

        # Create a copy of the page outline
        original_outline = target_page.outline

        # Create temporary page with copied outline
        temp_page = Mock()
        temp_page.outline = LogseqOutline(
            blocks=[b for b in original_outline.blocks],  # shallow copy of blocks
            source_text=original_outline.source_text,
            frontmatter=original_outline.frontmatter[:] if original_outline.frontmatter else [],
            indent_str=original_outline.indent_str,
        )

        # Add knowledge to temp page
        writer.add_knowledge_to_page(temp_page, kb)

        # Generate diff between original and modified
        original_text = original_outline.render()
        modified_text = temp_page.outline.render()

        return generate_unified_diff(
            original_text,
            modified_text,
            fromfile=f"pages/{kb.target_page}.md (current)",
            tofile=f"pages/{kb.target_page}.md (with new knowledge)",
        )


@dataclass
class ExtractionPreview:
    """Summary of proposed changes shown to user before applying (FR-002).

    Attributes:
        journal_date: Source journal being processed
        knowledge_blocks: All extracted knowledge
        proposed_actions: What will happen to each block
        warnings: Issues found (missing pages, duplicates, etc.)
        graph_path: Path to Logseq graph (optional, for diff generation)
    """

    journal_date: date
    knowledge_blocks: list[KnowledgeBlock]
    proposed_actions: list[ProposedAction]
    warnings: list[str]
    graph_path: Optional[Path] = None

    def display(self, show_diffs: bool = False) -> str:
        """Render preview for terminal display.

        Args:
            show_diffs: If True and graph_path is set, show diffs for each action

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

            # Show diff if requested and action is READY
            if show_diffs and self.graph_path and action.status == ActionStatus.READY:
                try:
                    from logsqueak.models.diff import generate_page_diff

                    diff = generate_page_diff(
                        self.graph_path,
                        action.knowledge.target_page,
                        action.knowledge,
                    )

                    if diff:
                        lines.append("")
                        lines.append("  Diff:")
                        # Indent each diff line - rstrip to remove any trailing newlines,
                        # since lines.append will add them back
                        for diff_line in diff.splitlines():
                            lines.append(f"  {diff_line.rstrip()}")
                    else:
                        lines.append("")
                        lines.append("  (No diff generated - page may not exist or no changes)")
                except Exception as e:
                    lines.append("")
                    lines.append(f"  Error generating diff: {e}")

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
