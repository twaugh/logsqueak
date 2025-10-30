"""Phase 3 Screen: Integration decisions with streaming LLM feedback.

This screen orchestrates Phase 3.1 (Decider) and Phase 3.2 (Reworder) to show
users in real-time where knowledge will be integrated and how it will be reworded.
"""

import asyncio
import logging
from typing import Optional

from textual.app import ComposeResult
from textual.containers import Container, Vertical, ScrollableContainer
from textual.screen import Screen
from textual.widgets import Footer, Header, Label, Static

from logsqueak.tui.models import IntegrationDecision, ScreenState
from logsqueak.tui.utils import find_block_by_id
from logsqueak.tui.widgets.decision_list import DecisionList

logger = logging.getLogger(__name__)


class Phase3Screen(Screen):
    """
    Phase 3: Integration Decisions Screen.

    Streams LLM decisions (Decider + Reworder) showing where knowledge will be
    integrated and how it will be reworded. Organized by destination page.

    User Stories:
    - US2: See LLM integration decisions with streaming feedback (core feature)
    - US3: Override integration decisions and edit content (future: Phase 5)

    Workflow:
    1. On mount: Initialize decision streaming
    2. Phase 3.1 (Decider): For each (knowledge block, candidate page) pair:
       - Stream action decisions (skip, add_section, add_under, replace)
       - Show confidence scores with low-confidence warnings
    3. Phase 3.2 (Reworder): For non-skip decisions:
       - Stream refined text progressively
       - Show "Refining... [streaming]" while generating
    4. Group decisions by target page
    5. Collapse pages with all-skip decisions
    6. Press Enter to continue to Phase 4 when complete

    Keyboard Bindings:
    - n: Proceed when ready (write operations)
    - j/k, ↑/↓: Navigate (future: Phase 5 - US3)
    - Space: Cycle action (future: Phase 5 - US3)
    - E: Edit refined text (future: Phase 5 - US3)
    - q: Quit application
    """

    BINDINGS = [
        ("n", "continue", "Next →"),
    ]

    CSS = """
    Phase3Screen {
        layout: vertical;
    }

    #status-container {
        dock: top;
        height: auto;
        background: $panel;
        padding: 1;
    }

    #decisions-container {
        height: 1fr;
        overflow-y: scroll;
        padding: 1;
    }

    .status-message {
        color: $accent;
        margin-bottom: 1;
    }
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

    def compose(self) -> ComposeResult:
        """Compose screen layout."""
        yield Header()
        with Container(id="status-container"):
            yield Static(
                "Phase 3: Streaming integration decisions...",
                classes="status-message",
                id="status-label",
            )
        with ScrollableContainer(id="decisions-container"):
            yield DecisionList(decisions=self.state.decisions)
        yield Footer()

    async def on_mount(self) -> None:
        """
        Initialize Phase 3 and start decision + rewording streaming.

        Steps:
        1. Set current phase to 3
        2. Start Decider streaming task (Phase 3.1)
        3. Start Reworder streaming task (Phase 3.2)
        4. Tasks update state.decisions dict as results arrive
        5. DecisionList widget reactively re-renders on updates
        """
        self.state.current_phase = 3

        # Start decision streaming (Phase 3.1)
        self.decision_task = asyncio.create_task(self._stream_decisions())

        # Rewording task will start after decisions complete
        # (needs to know which decisions are non-skip)

    async def _stream_decisions(self) -> None:
        """
        Stream Decider LLM decisions for all (knowledge block, candidate page) pairs.

        This implements Phase 3.1 of the pipeline:
        - For each knowledge block (classification="knowledge"):
          - For each candidate page (included=True):
            - Call LLM to decide action (skip, add_section, add_under, replace)
            - Stream results as they arrive
            - Update state.decisions dict
            - Trigger UI refresh

        After all decisions complete, starts rewording task for non-skip decisions.
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
                    # Parse decision and create IntegrationDecision
                    decision = self._parse_decision_dict(
                        decision_dict=decision_dict,
                        knowledge_block_id=block_id,
                    )

                    # Log the decision
                    logger.info(
                        f"Decision for {decision.target_page}: {decision.action} "
                        f"(confidence: {decision.confidence:.0%})"
                    )

                    # Store in state.decisions
                    key = (block_id, decision.target_page)
                    self.state.decisions[key] = decision

                    # Trigger UI update
                    self._refresh_decision_list()

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
            self._update_status(f"Error: {e}. Press Enter to continue anyway.")
            self.decisions_complete = True

    async def _stream_rewording(self) -> None:
        """
        Stream Reworder LLM refined text for all non-skip decisions.

        This implements Phase 3.2 of the pipeline:
        - Filter decisions for action != "skip"
        - For each accepted decision:
          - Call LLM to rephrase knowledge into clean, evergreen content
          - Stream refined text as it arrives
          - Update decision.refined_text progressively
          - Trigger UI refresh

        After all rewording complete, enables Phase 4 continuation.
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
                self._update_status("All decisions skipped. Ready to continue.")
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

            # Stream rewording results from LLM
            async for reworded_dict in self.state.llm_client.stream_rewording_ndjson(
                decisions=decisions_to_reword
            ):
                # Extract fields
                knowledge_block_id = reworded_dict["knowledge_block_id"]
                target_page = reworded_dict["page"]
                refined_text = reworded_dict["refined_text"]

                # Log the rewording
                logger.info(
                    f"Reworded for {target_page}: {refined_text[:80]}..."
                )

                # Update decision in state
                key = (knowledge_block_id, target_page)
                if key in self.state.decisions:
                    self.state.decisions[key].refined_text = refined_text

                    # Trigger UI update
                    self._refresh_decision_list()

                    processed += 1
                    self._update_status(
                        f"Rewording complete for {processed}/{total} decisions"
                    )

                # Allow UI to update
                await asyncio.sleep(0.05)

            # All rewording complete
            self.rewording_complete = True
            self._update_status(
                f"Phase 3 complete! {processed} blocks ready to integrate. Press Enter to continue."
            )

        except Exception as e:
            logger.error(f"Error during rewording streaming: {e}", exc_info=True)
            self._update_status(f"Error: {e}. Press Enter to continue anyway.")
            self.rewording_complete = True

    def _parse_decision_dict(
        self, decision_dict: dict, knowledge_block_id: str
    ) -> IntegrationDecision:
        """
        Parse LLM decision dict into IntegrationDecision model.

        Args:
            decision_dict: Dict from LLM with page, action, confidence, etc.
            knowledge_block_id: Source knowledge block ID

        Returns:
            IntegrationDecision object
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

        Args:
            block: LogseqBlock from journal

        Returns:
            Hierarchical markdown text
        """
        # For now, just return block content
        # TODO: Add parent context in future if needed
        return "\n".join(block.content)

    def _update_status(self, message: str) -> None:
        """
        Update status message display.

        Args:
            message: Status message to show
        """
        status_label = self.query_one("#status-label", Static)
        status_label.update(message)

    def _refresh_decision_list(self) -> None:
        """
        Trigger DecisionList widget refresh.

        This forces the widget to re-render with updated decisions.
        """
        # Remove the old widget and mount a new one with updated data
        old_list = self.query_one(DecisionList)
        old_list.remove()

        new_list = DecisionList(decisions=self.state.decisions)
        container = self.query_one("#decisions-container")
        container.mount(new_list)

    async def action_continue(self) -> None:
        """
        Continue to Phase 4 (write operations).

        Validation:
        - At least one non-skip decision exists (optional - can proceed with zero)

        Navigation:
        - Push Phase4Screen onto screen stack
        """
        # Check if we have any non-skip decisions
        non_skip_count = sum(
            1 for decision in self.state.decisions.values()
            if decision.action != "skip"
        )

        if non_skip_count == 0:
            logger.warning("No integrations to perform (all decisions skipped)")
            self._update_status(
                "No integrations to perform. Press Enter again to exit."
            )
            # User can press Enter again to exit
            # For now, just log warning
            return

        logger.info(f"Proceeding to Phase 4 with {non_skip_count} integrations")

        # TODO: Push Phase4Screen when implemented (User Story 5)
        # For now, just show completion message
        self._update_status(
            f"Phase 4 not yet implemented. Would write {non_skip_count} blocks. Press 'q' to quit."
        )

        # from logsqueak.tui.screens.phase4 import Phase4Screen
        # await self.app.push_screen(Phase4Screen(self.state))

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
