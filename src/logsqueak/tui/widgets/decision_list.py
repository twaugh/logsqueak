"""DecisionList widget for displaying integration decisions grouped by page.

This widget shows the output of Phase 3 (Decider + Reworder), organized by
destination page with expandable/collapsible sections for each page.
"""

from textual.app import ComposeResult
from textual.containers import Container, Vertical
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Collapsible, Label, Static

from logsqueak.tui.models import IntegrationDecision


class DecisionList(Widget):
    """
    Custom widget displaying integration decisions grouped by target page.

    This widget visualizes Phase 3 outputs:
    - Groups decisions by destination page
    - Shows action types (skip, add_section, add_under, replace)
    - Displays confidence scores with warnings for low confidence
    - Shows refined text (streaming or complete)
    - Collapses pages with all-skip decisions

    Visual Layout:
    ```
    ┌─ Software Architecture (3 blocks) ─────────────────┐
    │  ✓ Add as new section (confidence: 92%)            │
    │    "Bounded contexts should drive microservice..." │
    │  ⚠ Add under 'Design Patterns' (confidence: 58%)   │
    │    Refining... [streaming]                         │
    │  ✗ Skip - Already covered                          │
    └────────────────────────────────────────────────────┘
    ```

    State Management:
    - Reactive: decisions dict updates trigger re-render
    - Decisions grouped by target_page
    - Each page gets a Collapsible container
    - Pages with all skips are collapsed by default
    """

    decisions: reactive[dict[tuple[str, str], IntegrationDecision]] = reactive(dict)

    CSS = """
    DecisionList {
        height: 1fr;
        overflow-y: auto;
    }

    .page-section {
        margin-bottom: 1;
        border: solid $primary;
        padding: 1;
    }

    .page-header {
        background: $primary;
        color: $text;
        padding: 0 1;
        margin-bottom: 1;
    }

    .decision-item {
        margin-bottom: 1;
        padding-left: 2;
    }

    .action-skip {
        color: $text-muted;
    }

    .action-add {
        color: $success;
    }

    .action-replace {
        color: $warning;
    }

    .confidence-low {
        color: $warning;
    }

    .confidence-high {
        color: $success;
    }

    .refined-text {
        padding: 1;
        background: $panel;
        margin-top: 1;
    }

    .streaming-indicator {
        color: $accent;
    }
    """

    def __init__(self, decisions: dict[tuple[str, str], IntegrationDecision] = None):
        """
        Initialize DecisionList widget.

        Args:
            decisions: Map of (knowledge_block_id, target_page) -> IntegrationDecision
        """
        super().__init__()
        if decisions:
            self.decisions = decisions

    def compose(self) -> ComposeResult:
        """Compose widget layout with grouped decisions."""
        if not self.decisions:
            yield Label("No decisions yet. Waiting for Phase 3 streaming...")
            return

        # Group decisions by target page
        page_groups = self._group_by_page(self.decisions)

        for page_name, page_decisions in page_groups.items():
            # Count decision types
            skip_count = sum(1 for d in page_decisions if d.action == "skip")
            total_count = len(page_decisions)
            active_count = total_count - skip_count

            # Determine if page should be collapsed
            collapsed = (skip_count == total_count)

            # Build page section
            with Collapsible(
                title=self._format_page_header(page_name, active_count, total_count),
                collapsed=collapsed,
                classes="page-section",
            ):
                for decision in page_decisions:
                    yield from self._render_decision(decision)

    def _group_by_page(
        self, decisions: dict[tuple[str, str], IntegrationDecision]
    ) -> dict[str, list[IntegrationDecision]]:
        """
        Group decisions by target page.

        Args:
            decisions: Map of (knowledge_block_id, target_page) -> IntegrationDecision

        Returns:
            Map of page_name -> list of decisions for that page
        """
        page_groups = {}
        for decision in decisions.values():
            page_name = decision.target_page
            if page_name not in page_groups:
                page_groups[page_name] = []
            page_groups[page_name].append(decision)

        # Sort pages by name
        return dict(sorted(page_groups.items()))

    def _format_page_header(
        self, page_name: str, active_count: int, total_count: int
    ) -> str:
        """
        Format page section header with counts.

        Args:
            page_name: Name of the page
            active_count: Number of non-skip decisions
            total_count: Total number of decisions

        Returns:
            Formatted header string
        """
        if active_count == 0:
            return f"{page_name} (all skipped)"
        elif active_count == total_count:
            return f"{page_name} ({active_count} blocks)"
        else:
            return f"{page_name} ({active_count}/{total_count} active)"

    def _render_decision(self, decision: IntegrationDecision) -> ComposeResult:
        """
        Render a single integration decision.

        Args:
            decision: IntegrationDecision to display

        Yields:
            Textual widgets for the decision
        """
        # Determine action icon and label
        action_icon, action_label, action_class = self._format_action(decision)

        # Format confidence score
        confidence_str = self._format_confidence(decision.confidence)

        # Build action line
        action_line = f"{action_icon} {action_label} {confidence_str}"

        yield Static(action_line, classes=f"decision-item {action_class}")

        # Show refined text if available (skip actions don't have refined text)
        if decision.action != "skip":
            if decision.refined_text:
                # Check if this looks like streaming is in progress
                is_streaming = decision.refined_text.endswith("...")
                text_class = "streaming-indicator" if is_streaming else ""

                yield Static(
                    f'  "{decision.refined_text}"',
                    classes=f"refined-text {text_class}",
                )
            else:
                yield Static(
                    "  Refining... [streaming]",
                    classes="refined-text streaming-indicator",
                )
        elif decision.skip_reason:
            # Show skip reason
            yield Static(
                f"  ({decision.skip_reason})",
                classes="refined-text action-skip",
            )

    def _format_action(
        self, decision: IntegrationDecision
    ) -> tuple[str, str, str]:
        """
        Format action icon, label, and CSS class.

        Args:
            decision: IntegrationDecision to format

        Returns:
            Tuple of (icon, label, css_class)
        """
        if decision.action == "skip":
            return ("✗", "Skip", "action-skip")
        elif decision.action == "add_section":
            return ("✓", "Add as new section", "action-add")
        elif decision.action == "add_under":
            target_title = decision.target_block_title or "unknown block"
            return ("↳", f"Add under '{target_title}'", "action-add")
        elif decision.action == "replace":
            target_title = decision.target_block_title or "unknown block"
            return ("⟳", f"Replace '{target_title}'", "action-replace")
        else:
            return ("?", f"Unknown action: {decision.action}", "")

    def _format_confidence(self, confidence: float) -> str:
        """
        Format confidence score with visual indicators.

        Args:
            confidence: Confidence score (0.0-1.0)

        Returns:
            Formatted confidence string with class hints
        """
        percentage = int(confidence * 100)

        if confidence < 0.60:
            return f"[confidence-low]⚠ {percentage}%[/]"
        elif confidence < 0.75:
            return f"[confidence-low]{percentage}%[/]"
        else:
            return f"[confidence-high]{percentage}%[/]"

    async def watch_decisions(
        self, new_decisions: dict[tuple[str, str], IntegrationDecision]
    ) -> None:
        """
        React to decisions dict updates by re-rendering.

        This is called automatically by Textual when self.decisions changes.

        Args:
            new_decisions: Updated decisions dict
        """
        # Trigger full re-render by removing and re-adding children
        await self.remove_children()

        if not new_decisions:
            await self.mount(Label("No decisions yet. Waiting for Phase 3 streaming..."))
            return

        # Group decisions by target page
        page_groups = self._group_by_page(new_decisions)

        for page_name, page_decisions in page_groups.items():
            # Count decision types
            skip_count = sum(1 for d in page_decisions if d.action == "skip")
            total_count = len(page_decisions)
            active_count = total_count - skip_count

            # Determine if page should be collapsed
            collapsed = (skip_count == total_count)

            # Build page section
            collapsible = Collapsible(
                title=self._format_page_header(page_name, active_count, total_count),
                collapsed=collapsed,
                classes="page-section",
            )

            await self.mount(collapsible)

            # Add decision items to the collapsible
            for decision in page_decisions:
                widgets = list(self._render_decision(decision))
                await collapsible.mount(*widgets)
