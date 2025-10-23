"""Integration orchestrator for adding knowledge to pages.

This module coordinates the process of integrating extracted knowledge
into target pages, handling section matching, provenance links, and
safe file writing.
"""

from pathlib import Path
from typing import List, Optional

from logsqueak.integration import writer
from logsqueak.models.knowledge import ActionType, KnowledgeBlock
from logsqueak.models.page import PageIndex, TargetPage
from logsqueak.models.preview import ActionStatus, ProposedAction


class IntegrationResult:
    """Result of integration operation.

    Attributes:
        success: Whether integration succeeded
        actions_applied: Number of actions successfully applied
        actions_skipped: Number of actions skipped
        errors: List of error messages encountered
        modified_pages: List of page names that were modified
    """

    def __init__(
        self,
        success: bool,
        actions_applied: int = 0,
        actions_skipped: int = 0,
        errors: Optional[List[str]] = None,
        modified_pages: Optional[List[str]] = None,
    ):
        self.success = success
        self.actions_applied = actions_applied
        self.actions_skipped = actions_skipped
        self.errors = errors or []
        self.modified_pages = modified_pages or []


class Integrator:
    """Orchestrates integration of knowledge into pages.

    Coordinates:
    1. Loading target pages
    2. Finding target sections
    3. Adding knowledge with provenance
    4. Writing files safely
    5. Refreshing PageIndex
    """

    def __init__(self, graph_path: Path, page_index: Optional[PageIndex] = None):
        """Initialize the integrator.

        Args:
            graph_path: Path to Logseq graph root
            page_index: Optional PageIndex for refreshing after modifications
        """
        self.graph_path = graph_path
        self.page_index = page_index

    def integrate(
        self, proposed_actions: List[ProposedAction], dry_run: bool = True
    ) -> IntegrationResult:
        """Integrate knowledge blocks into target pages.

        Args:
            proposed_actions: List of actions to execute
            dry_run: If True, don't actually write files (default: True)

        Returns:
            IntegrationResult with statistics and errors
        """
        actions_applied = 0
        actions_skipped = 0
        errors = []
        modified_pages = []

        # Group actions by target page for efficient processing
        actions_by_page = self._group_by_page(proposed_actions)

        for page_name, actions in actions_by_page.items():
            try:
                # Load target page
                target_page = self._load_page(page_name)

                if target_page is None:
                    errors.append(f"Target page '{page_name}' not found")
                    actions_skipped += len(actions)
                    continue

                # Process each action for this page
                for action in actions:
                    if action.status != ActionStatus.READY:
                        actions_skipped += 1
                        continue

                    try:
                        # Add knowledge to page (will be implemented in writer module)
                        self._add_knowledge_to_page(target_page, action.knowledge)
                        actions_applied += 1

                    except Exception as e:
                        errors.append(
                            f"Failed to add knowledge to {page_name}: {str(e)}"
                        )
                        actions_skipped += 1

                # Write modified page to disk
                if actions_applied > 0 and not dry_run:
                    self._write_page(target_page)
                    modified_pages.append(page_name)

            except Exception as e:
                errors.append(f"Error processing page '{page_name}': {str(e)}")
                actions_skipped += len(actions)

        # Refresh PageIndex for modified pages
        if modified_pages and self.page_index and not dry_run:
            self._refresh_index(modified_pages)

        success = len(errors) == 0 and actions_applied > 0

        return IntegrationResult(
            success=success,
            actions_applied=actions_applied,
            actions_skipped=actions_skipped,
            errors=errors,
            modified_pages=modified_pages,
        )

    def _group_by_page(
        self, proposed_actions: List[ProposedAction]
    ) -> dict[str, List[ProposedAction]]:
        """Group actions by target page name.

        Args:
            proposed_actions: List of proposed actions

        Returns:
            Dictionary mapping page name to list of actions
        """
        grouped = {}

        for action in proposed_actions:
            page_name = action.knowledge.target_page

            if page_name not in grouped:
                grouped[page_name] = []

            grouped[page_name].append(action)

        return grouped

    def _load_page(self, page_name: str) -> Optional[TargetPage]:
        """Load a target page from the graph.

        Args:
            page_name: Name of the page to load

        Returns:
            TargetPage if found, None otherwise
        """
        # Use TargetPage.load() which handles path construction
        return TargetPage.load(self.graph_path, page_name)

    def _add_knowledge_to_page(
        self, target_page: TargetPage, knowledge: KnowledgeBlock
    ) -> None:
        """Add knowledge block to target page.

        Delegates to writer module which handles:
        - Finding target section (T036)
        - Adding provenance links (T037)
        - Adding child bullets (T038)
        - Fallback to page end (T039)

        Args:
            target_page: Page to add knowledge to
            knowledge: Knowledge block to add
        """
        writer.add_knowledge_to_page(target_page, knowledge)

    def _write_page(self, target_page: TargetPage) -> None:
        """Write modified page to disk safely.

        Args:
            target_page: Page to write

        Raises:
            IOError: If write fails
        """
        writer.write_page_safely(target_page)

    def _refresh_index(self, page_names: List[str]) -> None:
        """Refresh PageIndex embeddings for modified pages.

        Args:
            page_names: List of page names that were modified
        """
        if not self.page_index:
            return

        # TODO: Implement in T041
        # Will call page_index.refresh() for each modified page
        pass
