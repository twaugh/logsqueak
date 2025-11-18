"""DecisionList widget for displaying multiple integration decisions per block.

This widget shows a list of integration decisions for the current knowledge block,
with visual indicators for status (pending, completed, failed).
"""

from pathlib import Path
from typing import List, Optional
from textual.widgets import Static
from textual.containers import VerticalScroll
from textual.reactive import reactive
from rich.text import Text
from rich.style import Style
from rich.color import Color
from rich.console import RenderableType
from logsqueak.models.integration_decision import IntegrationDecision


class DecisionItem(Static):
    """Individual decision item with tooltip support."""

    def __init__(
        self,
        decision: IntegrationDecision,
        is_current: bool,
        graph_path: Optional[Path] = None,
        *args,
        **kwargs
    ):
        """Initialize DecisionItem.

        Args:
            decision: Integration decision to display
            is_current: Whether this is the currently selected decision
            graph_path: Path to Logseq graph (for creating clickable links)
        """
        super().__init__("", *args, **kwargs)
        self.decision = decision
        self.is_current = is_current
        self.graph_path = graph_path

        # Set tooltip to show LLM's reasoning
        self.tooltip = decision.reasoning

    def render(self) -> RenderableType:
        """Render the decision item.

        Returns:
            Rich Text object with formatted decision
        """
        result = Text()

        # Status indicator
        if self.decision.action == "skip_exists":
            # Already exists - show as completed with different label
            indicator = "✓"
            style = "green dim"
        elif self.decision.write_status == "completed":
            indicator = "✓"
            style = "green"
        elif self.decision.write_status == "failed":
            indicator = "⚠"
            style = "red"
        else:  # pending
            indicator = "⊙"
            style = "yellow"

        # Highlight current selection
        if self.is_current:
            result.append("► ", style="bold blue")
        else:
            result.append("  ")

        # Status indicator
        result.append(f"{indicator} ", style=style)

        # Target page (with clickable link if graph_path is available)
        if self.graph_path:
            # Create logseq:// URL using shared utility
            from logsqueak.utils.logseq_urls import create_logseq_url
            logseq_url = create_logseq_url(self.decision.target_page, self.graph_path)

            # Use Rich Style with link parameter for proper clickable links
            # Explicit RGB blue color to override terminal's default link styling
            page_style = Style(
                bold=self.is_current,
                color=Color.from_rgb(100, 149, 237),  # Cornflower blue
                link=logseq_url
            )
            result.append(self.decision.target_page, style=page_style)
        else:
            result.append(f"{self.decision.target_page}", style="bold" if self.is_current else "")

        # Action type and confidence
        if self.decision.action == "skip_exists":
            result.append(" (Already Exists)", style="green dim")
        else:
            action_display = self.decision.action.replace("_", " ").title()
            result.append(f" ({action_display}, {self.decision.confidence:.0%})", style="dim")

        # Show target block title if applicable
        if self.decision.target_block_title:
            result.append(f" → {self.decision.target_block_title}", style="dim italic")

        return result


class DecisionList(VerticalScroll):
    """List of integration decisions with status indicators."""

    # Reactive attributes
    decisions = reactive([], always_update=True)
    current_index = reactive(0)

    def __init__(self, graph_path: Optional[Path] = None, *args, **kwargs):
        """Initialize DecisionList.

        Args:
            graph_path: Path to Logseq graph (for creating clickable links)
        """
        super().__init__(*args, id="decision-list", **kwargs)
        self.graph_path = graph_path

    def watch_decisions(self, new_decisions: List[IntegrationDecision]) -> None:
        """Update child widgets when decisions change."""
        self._rebuild_list()

    def watch_current_index(self, new_index: int) -> None:
        """Update child widgets when current index changes."""
        self._rebuild_list()

    def _rebuild_list(self) -> None:
        """Rebuild the list of decision items."""
        # Clear existing children
        self.remove_children()

        if not self.decisions:
            # Show empty state (no ID to avoid duplicates)
            empty_widget = Static("None available")
            empty_widget.styles.color = "gray"
            self.mount(empty_widget)
            return

        # Create DecisionItem widget for each decision
        for i, decision in enumerate(self.decisions):
            is_current = (i == self.current_index)
            item = DecisionItem(
                decision=decision,
                is_current=is_current,
                graph_path=self.graph_path
            )
            self.mount(item)

    def load_decisions(self, decisions: List[IntegrationDecision], current_index: int = 0) -> None:
        """Load decisions and set current selection.

        Args:
            decisions: List of integration decisions for current block
            current_index: Index of currently selected decision
        """
        self.decisions = decisions
        self.current_index = current_index

    def set_current_index(self, index: int) -> None:
        """Set the currently selected decision.

        Args:
            index: Index to select
        """
        if 0 <= index < len(self.decisions):
            self.current_index = index

    def get_current_decision(self) -> IntegrationDecision | None:
        """Get the currently selected decision.

        Returns:
            Currently selected decision or None
        """
        if 0 <= self.current_index < len(self.decisions):
            return self.decisions[self.current_index]
        return None

    def clear(self) -> None:
        """Clear decision list."""
        self.decisions = []
        self.current_index = 0
