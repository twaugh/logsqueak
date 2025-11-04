"""Target page model for knowledge integration."""

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from logseq_outline import LogseqBlock, LogseqOutline

logger = logging.getLogger(__name__)


class ConventionType(Enum):
    """Page organizational convention."""

    PLAIN_BULLETS = "plain"  # Uses "- Section"
    HEADING_BULLETS = "heading"  # Uses "- ## Section"
    MIXED = "mixed"  # Mix of both styles


@dataclass
class TargetPage:
    """An existing Logseq page where knowledge will be integrated.

    Attributes:
        name: Page name (e.g., "Project X")
        file_path: Absolute path to page markdown file
        outline: Parsed outline structure
        organizational_convention: Detected page style
    """

    name: str
    file_path: Path
    outline: "LogseqOutline"
    organizational_convention: ConventionType

    @classmethod
    def load(cls, graph_path: Path, page_name: str) -> Optional["TargetPage"]:
        """Load target page from Logseq graph.

        Args:
            graph_path: Path to Logseq graph directory
            page_name: Name of the page to load

        Returns:
            TargetPage instance if found, None otherwise (FR-009)
        """
        from logseq_outline import LogseqOutline

        # Logseq stores pages as "Page Name.md"
        file_path = graph_path / "pages" / f"{page_name}.md"

        if not file_path.exists():
            logger.warning(f"Target page not found: {page_name}")
            return None

        try:
            outline = LogseqOutline.parse(file_path.read_text())
            convention = _detect_convention(outline)

            return cls(
                name=page_name,
                file_path=file_path,
                outline=outline,
                organizational_convention=convention,
            )
        except Exception as e:
            logger.error(f"Failed to load page {page_name}: {e}")
            return None

    def find_section(self, section_name: str) -> Optional["LogseqBlock"]:
        """Find section by heading text.

        Args:
            section_name: Name of section to find

        Returns:
            LogseqBlock if found, None otherwise
        """
        return self.outline.find_heading(section_name)

    def has_duplicate(self, content: str) -> bool:
        """Check if content already exists on page (FR-017).

        Simple content match - can be enhanced with semantic similarity later.

        Args:
            content: Content to check for duplicates

        Returns:
            True if duplicate found
        """
        return content in self.outline.render()


def _detect_convention(outline: "LogseqOutline") -> ConventionType:
    """Detect the organizational convention used in a page.

    Args:
        outline: Page outline structure

    Returns:
        Detected convention type
    """
    # Simple heuristic: check if any blocks contain markdown headings
    has_heading = False
    has_plain = False

    def check_blocks(blocks: list["LogseqBlock"]) -> None:
        nonlocal has_heading, has_plain
        for block in blocks:
            if "##" in block.content[0]:  # Check first line
                has_heading = True
            else:
                has_plain = True
            check_blocks(block.children)

    check_blocks(outline.blocks)

    if has_heading and has_plain:
        return ConventionType.MIXED
    elif has_heading:
        return ConventionType.HEADING_BULLETS
    else:
        return ConventionType.PLAIN_BULLETS
