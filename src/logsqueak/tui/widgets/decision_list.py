"""DecisionList widget for displaying multiple integration decisions per block.

This widget shows a list of integration decisions for the current knowledge block,
with visual indicators for status (pending, completed, failed).
"""

from textual.widgets import Static
from textual.reactive import reactive
from rich.text import Text
from rich.console import RenderableType
from typing import List
from logsqueak.models.integration_decision import IntegrationDecision


class DecisionList(Static):
    """List of integration decisions with status indicators."""

    # Reactive attributes
    decisions = reactive([], always_update=True)
    current_index = reactive(0)

    def __init__(self, *args, **kwargs):
        """Initialize DecisionList."""
        super().__init__("", *args, id="decision-list", **kwargs)

    def render(self) -> RenderableType:
        """Render the decision list with status indicators.

        Returns:
            Rich Text object with formatted decision list
        """
        if not self.decisions:
            return Text("None available", style="dim")

        result = Text()

        for i, decision in enumerate(self.decisions):
            # Status indicator
            if decision.write_status == "completed":
                indicator = "✓"
                style = "green"
            elif decision.write_status == "failed":
                indicator = "⚠"
                style = "red"
            else:  # pending
                indicator = "⊙"
                style = "yellow"

            # Highlight current selection
            if i == self.current_index:
                result.append("► ", style="bold blue")
            else:
                result.append("  ")

            # Status indicator and target page
            result.append(f"{indicator} ", style=style)
            result.append(f"{decision.target_page}", style="bold" if i == self.current_index else "")

            # Action type and confidence
            action_display = decision.action.replace("_", " ").title()
            result.append(f" ({action_display}, {decision.confidence:.0%})", style="dim")

            # Show target block title if applicable
            if decision.target_block_title:
                result.append(f" → {decision.target_block_title}", style="dim italic")

            # Add newline except for last item
            if i < len(self.decisions) - 1:
                result.append("\n")

        return result

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
