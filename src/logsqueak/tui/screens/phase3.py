"""Phase 3 Screen: Single-decision-per-screen review interface.

This screen shows one integration decision at a time with full context:
- Hierarchical journal context (where knowledge came from)
- Target page preview with visual diff
- Navigation between decisions
"""

import asyncio
import logging
from typing import Optional

from textual.app import ComposeResult
from textual.containers import Container, ScrollableContainer
from textual.screen import Screen
from textual.widgets import Footer, Header, Static

from logsqueak.tui.models import IntegrationDecision, ScreenState
from logsqueak.tui.utils import find_block_by_id, get_block_hierarchy

logger = logging.getLogger(__name__)


class Phase3Screen(Screen):
    """
    Phase 3: Integration Decisions Screen (Single-Decision View).

    Shows one decision at a time with:
    - Hierarchical journal context
    - Target page preview with visual diff
    - Navigation controls

    Keyboard Bindings:
    - n: Next decision
    - p: Previous decision
    - a: Accept decision
    - s: Skip decision
    - e: Edit refined text (future)
    - q: Quit application
    """

    BINDINGS = [
        ("n", "next_decision", "Next →"),
        ("p", "prev_decision", "← Previous"),
        ("a", "accept_decision", "Accept"),
        ("s", "skip_decision", "Skip"),
    ]

    CSS = """
    Phase3Screen {
        layout: vertical;
    }

    #header-container {
        dock: top;
        height: auto;
        background: $panel;
        padding: 1;
    }

    #content-container {
        height: 1fr;
        padding: 1;
    }

    .section-box {
        border: solid $primary;
        padding: 1;
        margin-bottom: 1;
    }

    .section-title {
        color: $accent;
        text-style: bold;
    }

    .knowledge-highlight {
        border-left: thick $success;
        padding-left: 1;
    }

    .new-content {
        border-left: thick $success;
        padding-left: 1;
        color: $success;
    }

    .replaced-content {
        text-style: strike;
    }

    .indent-1 { padding-left: 2; }
    .indent-2 { padding-left: 4; }
    .indent-3 { padding-left: 6; }
    .indent-4 { padding-left: 8; }
    .indent-5 { padding-left: 10; }
    """

    def __init__(self, state: ScreenState):
        """
        Initialize Phase3Screen.

        Args:
            state: Shared application state
        """
        super().__init__()
        self.state = state
        self.decision_task: Optional[asyncio.Task] = None
        self.rewording_task: Optional[asyncio.Task] = None
        self.decisions_complete = False
        self.rewording_complete = False
        self.current_decision_index = 0
        self.decision_list: list[tuple[tuple[str, str], IntegrationDecision]] = []

    def compose(self) -> ComposeResult:
        """Compose screen layout."""
        yield Header()
        with Container(id="header-container"):
            yield Static(
                "Phase 3: Streaming integration decisions...",
                id="status-label",
            )
        with ScrollableContainer(id="content-container"):
            yield Static("Waiting for decisions...", id="decision-content")
        yield Footer()

    async def on_mount(self) -> None:
        """
        Initialize Phase 3 and start decision + rewording streaming.
        """
        self.state.current_phase = 3

        # Start decision streaming (Phase 3.1)
        self.decision_task = asyncio.create_task(self._stream_decisions())

    async def _stream_decisions(self) -> None:
        """
        Stream Decider LLM decisions for all (knowledge block, candidate page) pairs.
        """
        try:
            # Get knowledge blocks
            knowledge_blocks = [
                (block_id, state)
                for block_id, state in self.state.block_states.items()
                if state.classification == "knowledge"
            ]

            if not knowledge_blocks:
                logger.warning("No knowledge blocks to process in Phase 3")
                self._update_status("No knowledge blocks to integrate.")
                self.decisions_complete = True
                await asyncio.sleep(1.5)
                return

            total_pairs = 0
            processed_pairs = 0

            # Count total (block, page) pairs
            for block_id, _ in knowledge_blocks:
                candidates = self.state.candidates.get(block_id, [])
                included_candidates = [c for c in candidates if c.included]
                total_pairs += len(included_candidates)

            if total_pairs == 0:
                logger.warning("No candidate pages to evaluate in Phase 3")
                self._update_status("No candidate pages found. Skipping to Phase 4.")
                self.decisions_complete = True
                await asyncio.sleep(1.5)
                return

            # Process each (knowledge block, candidate page) pair
            for block_id, block_state in knowledge_blocks:
                # Get block content from journal
                block = find_block_by_id(self.state.journal_entry.outline.blocks, block_id)
                if not block:
                    logger.warning(f"Block not found: {block_id}")
                    continue

                # Get full block text
                block_text = "\n".join(block.content)

                # Get hierarchical representation (block + parent context)
                hierarchical_text = self._get_hierarchical_text(block)

                # Build knowledge_block dict for LLM
                knowledge_block_dict = {
                    "block_id": block_id,
                    "content": block_text,
                    "hierarchical_text": hierarchical_text,
                }

                # Get candidate pages
                candidates = self.state.candidates.get(block_id, [])
                included_candidates = [c for c in candidates if c.included]

                # Build candidate_pages list for LLM
                candidate_pages_list = []
                for candidate in included_candidates:
                    chunks = [
                        {
                            "target_id": block_info["id"],
                            "content": block_info["content"],
                            "title": block_info["content"][:50] + "..." if len(block_info["content"]) > 50 else block_info["content"],
                        }
                        for block_info in candidate.blocks
                    ]

                    candidate_pages_list.append({
                        "page_name": candidate.page_name,
                        "similarity_score": candidate.similarity_score,
                        "chunks": chunks,
                    })

                # Stream decisions from LLM
                self._update_status(
                    f"Deciding actions for block {block_id[:8]}... ({processed_pairs + 1}/{total_pairs})"
                )

                async for decision_dict in self.state.llm_client.stream_decisions_ndjson(
                    knowledge_block=knowledge_block_dict,
                    candidate_pages=candidate_pages_list,
                ):
                    # Log the full raw decision response
                    logger.info(f"Decision response received: {decision_dict}")

                    # Parse decision and create IntegrationDecision
                    decision = self._parse_decision_dict(
                        decision_dict=decision_dict,
                        knowledge_block_id=block_id,
                    )

                    # Log the decision with details
                    logger.info(
                        f"Decision {processed_pairs + 1}/{total_pairs}:\n"
                        f"  Block ID: {block_id}\n"
                        f"  Target Page: {decision.target_page}\n"
                        f"  Action: {decision.action}\n"
                        f"  Confidence: {decision.confidence:.0%}\n"
                        f"  Target Block: {decision.target_block_title or 'N/A'}\n"
                        f"  Reasoning: {decision.skip_reason or 'N/A'}"
                    )

                    # Store in state.decisions
                    key = (block_id, decision.target_page)
                    self.state.decisions[key] = decision

                    # Update display
                    self._update_decision_display()

                    processed_pairs += 1

                # Allow UI to update
                await asyncio.sleep(0.05)

            # All decisions complete
            self.decisions_complete = True
            self._update_status(f"Decisions complete ({processed_pairs} evaluations). Starting rewording...")

            # Start rewording task
            self.rewording_task = asyncio.create_task(self._stream_rewording())

        except Exception as e:
            logger.error(f"Error during decision streaming: {e}", exc_info=True)
            self._update_status(f"Error: {e}. Press 'n' to continue anyway.")
            self.decisions_complete = True

    async def _stream_rewording(self) -> None:
        """
        Stream Reworder LLM refined text for all non-skip decisions.
        """
        try:
            # Filter for non-skip decisions
            accepted_decisions = [
                (key, decision)
                for key, decision in self.state.decisions.items()
                if decision.action != "skip"
            ]

            if not accepted_decisions:
                logger.info("No decisions to reword (all skipped)")
                self._update_status("All decisions skipped. Press 'n' to continue.")
                self.rewording_complete = True
                return

            # Build decisions list for rewording
            decisions_to_reword = []
            for (knowledge_block_id, target_page), decision in accepted_decisions:
                # Get block content from journal
                block = find_block_by_id(self.state.journal_entry.outline.blocks, knowledge_block_id)
                if not block:
                    continue

                block_text = "\n".join(block.content)
                hierarchical_text = self._get_hierarchical_text(block)

                decisions_to_reword.append({
                    "knowledge_block_id": knowledge_block_id,
                    "page": target_page,
                    "action": decision.action,
                    "full_text": block_text,
                    "hierarchical_text": hierarchical_text,
                })

            total = len(decisions_to_reword)
            processed = 0

            logger.info(f"Starting rewording for {total} decisions")

            # Stream rewording results from LLM
            async for reworded_dict in self.state.llm_client.stream_rewording_ndjson(
                decisions=decisions_to_reword
            ):
                # Log the full raw response
                logger.info(f"Rewording response received: {reworded_dict}")

                # Extract fields
                knowledge_block_id = reworded_dict["knowledge_block_id"]
                target_page = reworded_dict["page"]
                refined_text = reworded_dict["refined_text"]

                # Log the full rewording with details
                logger.info(
                    f"Rewording complete for decision {processed + 1}/{total}:\n"
                    f"  Block ID: {knowledge_block_id}\n"
                    f"  Target Page: {target_page}\n"
                    f"  Original: {decisions_to_reword[processed].get('full_text', '')[:100]}...\n"
                    f"  Refined: {refined_text}"
                )

                # Update decision in state
                key = (knowledge_block_id, target_page)
                if key in self.state.decisions:
                    original_action = self.state.decisions[key].action
                    self.state.decisions[key].refined_text = refined_text

                    logger.info(
                        f"Updated decision state for ({knowledge_block_id[:8]}..., {target_page}): "
                        f"action={original_action}, refined_text_length={len(refined_text)}"
                    )

                    # Update display
                    self._update_decision_display()

                    processed += 1
                    self._update_status(
                        f"Rewording complete for {processed}/{total} decisions"
                    )

                # Allow UI to update
                await asyncio.sleep(0.05)

            # All rewording complete
            self.rewording_complete = True
            self._update_status(
                f"Phase 3 complete! {processed} blocks ready. Press 'n' to continue."
            )

        except Exception as e:
            logger.error(f"Error during rewording streaming: {e}", exc_info=True)
            self._update_status(f"Error: {e}. Press 'n' to continue anyway.")
            self.rewording_complete = True

    def _parse_decision_dict(
        self, decision_dict: dict, knowledge_block_id: str
    ) -> IntegrationDecision:
        """
        Parse LLM decision dict into IntegrationDecision model.
        """
        # Map LLM action types to user-friendly action names
        action_map = {
            "skip": "skip",
            "add_section": "add_section",
            "add_under": "add_under",
            "replace": "replace",
        }

        action = action_map.get(decision_dict["action"], "skip")

        return IntegrationDecision(
            knowledge_block_id=knowledge_block_id,
            target_page=decision_dict["page"],
            action=action,
            target_block_id=decision_dict.get("target_id"),
            target_block_title=decision_dict.get("target_title"),
            confidence=decision_dict.get("confidence", 0.0),
            refined_text="",  # Populated by rewording task
            source="llm",
            skip_reason=decision_dict.get("reasoning") if action == "skip" else None,
        )

    def _get_hierarchical_text(self, block) -> str:
        """
        Get hierarchical markdown representation of block with parent context.
        """
        # For now, just return block content
        # TODO: Add parent context in future if needed
        return "\n".join(block.content)

    def _update_status(self, message: str) -> None:
        """
        Update status message display.
        """
        status_label = self.query_one("#status-label", Static)
        status_label.update(message)

    def _update_decision_display(self) -> None:
        """
        Update the decision display with current decision.
        """
        # Build list of decisions
        self.decision_list = list(self.state.decisions.items())

        # Ensure current index is valid
        if self.current_decision_index >= len(self.decision_list):
            self.current_decision_index = max(0, len(self.decision_list) - 1)

        # Render current decision
        if self.decision_list:
            content = self._render_current_decision()
        else:
            content = "No decisions yet. Waiting for streaming..."

        # Update content widget
        content_widget = self.query_one("#decision-content", Static)
        content_widget.update(content)

    def _render_current_decision(self) -> str:
        """
        Render the current decision as Rich markup.
        """
        if not self.decision_list:
            return "No decisions available."

        key, decision = self.decision_list[self.current_decision_index]
        knowledge_block_id, target_page = key

        # Build the display
        total = len(self.decision_list)
        current = self.current_decision_index + 1

        # Header
        lines = [
            f"[bold cyan]Decision {current} of {total}[/bold cyan]",
            f"Target: {target_page}",
            "",
        ]

        # Knowledge from journal (with hierarchy)
        lines.append("[bold]Knowledge from Journal:[/bold]")
        journal_context = self._render_journal_context(knowledge_block_id)
        lines.extend(journal_context)
        lines.append("")

        # Target page preview
        lines.append(f"[bold]Target Page: {target_page}[/bold]")
        lines.append(f"Action: {self._format_action_label(decision)} (Confidence: {decision.confidence:.0%})")
        lines.append("")

        if decision.action == "skip":
            lines.append(f"[dim]Reason: {decision.skip_reason or 'No reason given'}[/dim]")
        else:
            # Show page preview with highlighted new content
            page_preview = self._render_page_preview(decision)
            lines.extend(page_preview)

        return "\n".join(lines)

    def _render_journal_context(self, block_id: str) -> list[str]:
        """
        Render hierarchical journal context showing where knowledge came from.
        """
        # Get full hierarchy from root to this block
        hierarchy = get_block_hierarchy(self.state.journal_entry.outline.blocks, block_id)
        if not hierarchy:
            return ["[dim](Block not found)[/dim]"]

        lines = []
        # Add journal date
        lines.append(f"[dim]{self.state.journal_entry.date}[/dim]")

        # Render each level of hierarchy
        for depth, block in enumerate(hierarchy):
            indent = "  " * depth

            # Highlight the target block (last in hierarchy)
            if depth == len(hierarchy) - 1:
                # This is the knowledge block - show in bold with full content
                for i, line in enumerate(block.content):
                    if i == 0:
                        lines.append(f"{indent}• [bold]{line}[/bold]")
                    else:
                        lines.append(f"{indent}  [bold]{line}[/bold]")
            else:
                # Parent context - show dimmed, just first line
                content_preview = block.content[0] if block.content else "(empty)"
                # Truncate long content
                if len(content_preview) > 80:
                    content_preview = content_preview[:80] + "..."
                lines.append(f"{indent}• [dim]{content_preview}[/dim]")

        return lines

    def _render_page_preview(self, decision: IntegrationDecision) -> list[str]:
        """
        Render target page preview with visual diff highlighting.

        Shows the REFINED (reworded) content, not the original journal entry.
        """
        lines = []

        # Get candidate to show existing page blocks
        candidate = None
        for cand in self.state.candidates.get(decision.knowledge_block_id, []):
            if cand.page_name == decision.target_page:
                candidate = cand
                break

        if candidate and candidate.blocks:
            # Show existing blocks
            for block_info in candidate.blocks[:5]:
                content = block_info.get("content", "")
                if len(content) > 80:
                    content = content[:80] + "..."

                # Check if this block is being replaced
                if decision.action == "replace" and block_info.get("id") == decision.target_block_id:
                    # Show strikethrough for old content
                    lines.append(f"  • [strike]{content}[/strike]")
                    # Show new refined content below with green bar
                    if decision.refined_text:
                        # Split refined text into lines if it's multiline
                        for refined_line in decision.refined_text.split('\n'):
                            lines.append(f"  [green]┃[/green] {refined_line}")
                    else:
                        lines.append("  [dim]┃ (Refining...)[/dim]")
                else:
                    lines.append(f"  • {content}")

                # If adding under this block, show new REFINED content here
                if decision.action == "add_under" and block_info.get("id") == decision.target_block_id:
                    if decision.refined_text:
                        # Show refined text (reworded version), not original journal entry
                        for refined_line in decision.refined_text.split('\n'):
                            lines.append(f"    [green]┃[/green] {refined_line}")
                    else:
                        lines.append("    [dim]┃ (Refining...)[/dim]")

        # If add_section, show new REFINED content at root level
        if decision.action == "add_section":
            if decision.refined_text:
                # Show refined text, not original
                for refined_line in decision.refined_text.split('\n'):
                    lines.append(f"[green]┃[/green] {refined_line}")
            else:
                lines.append("[dim]┃ (Refining...)[/dim]")

        return lines

    def _format_action_label(self, decision: IntegrationDecision) -> str:
        """
        Format action label for display.
        """
        if decision.action == "skip":
            return "✗ Skip"
        elif decision.action == "add_section":
            return "✓ Add as new section"
        elif decision.action == "add_under":
            target = decision.target_block_title or "unknown block"
            return f"↳ Add under '{target}'"
        elif decision.action == "replace":
            target = decision.target_block_title or "unknown block"
            return f"⟳ Replace '{target}'"
        else:
            return f"? Unknown: {decision.action}"

    async def action_next_decision(self) -> None:
        """Navigate to next decision."""
        if self.current_decision_index < len(self.decision_list) - 1:
            self.current_decision_index += 1
            self._update_decision_display()
        else:
            # At the end, proceed to Phase 4
            await self.action_continue()

    async def action_prev_decision(self) -> None:
        """Navigate to previous decision."""
        if self.current_decision_index > 0:
            self.current_decision_index -= 1
            self._update_decision_display()

    async def action_accept_decision(self) -> None:
        """Accept current decision and move to next."""
        # Mark as accepted (already in state)
        await self.action_next_decision()

    async def action_skip_decision(self) -> None:
        """Skip current decision and move to next."""
        if self.decision_list:
            key, decision = self.decision_list[self.current_decision_index]
            decision.action = "skip"
            decision.skip_reason = "User skipped"
        await self.action_next_decision()

    async def action_continue(self) -> None:
        """
        Continue to Phase 4 (write operations).
        """
        # Check if we have any non-skip decisions
        non_skip_count = sum(
            1 for decision in self.state.decisions.values()
            if decision.action != "skip"
        )

        if non_skip_count == 0:
            logger.warning("No integrations to perform (all decisions skipped)")
            self._update_status(
                "No integrations to perform. Press 'n' again to exit."
            )
            return

        logger.info(f"Proceeding to Phase 4 with {non_skip_count} integrations")

        # TODO: Push Phase4Screen when implemented
        self._update_status(
            f"Phase 4 not yet implemented. Would write {non_skip_count} blocks. Press 'q' to quit."
        )

    async def on_unmount(self) -> None:
        """Cancel background streaming tasks on unmount."""
        if self.decision_task and not self.decision_task.done():
            self.decision_task.cancel()
            try:
                await self.decision_task
            except asyncio.CancelledError:
                pass

        if self.rewording_task and not self.rewording_task.done():
            self.rewording_task.cancel()
            try:
                await self.rewording_task
            except asyncio.CancelledError:
                pass
