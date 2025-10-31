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
from logsqueak.tui.markdown import render_markdown_to_markup

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
                "Deciding where to integrate your knowledge...",
                id="status-label",
            )
        with ScrollableContainer(id="content-container"):
            yield Static("Waiting for decisions...", id="decision-content", markup=True)
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
                    # Filter blocks to only matched ones (for semantic search)
                    # For hinted search (matched_block_ids=[]), use all blocks
                    if candidate.matched_block_ids:
                        # Semantic search: only send matched blocks
                        matched_set = set(candidate.matched_block_ids)
                        filtered_blocks = [
                            block_info for block_info in candidate.blocks
                            if block_info["id"] in matched_set
                        ]
                    else:
                        # Hinted search: send all blocks (whole page is relevant)
                        filtered_blocks = candidate.blocks

                    chunks = [
                        {
                            "target_id": block_info["id"],
                            "content": block_info["content"],
                            "title": block_info["content"][:50] + "..." if len(block_info["content"]) > 50 else block_info["content"],
                        }
                        for block_info in filtered_blocks
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

        # Clean target_id: LLM may return it wrapped in brackets like "[abc123]"
        target_id = decision_dict.get("target_id")
        if target_id and isinstance(target_id, str):
            target_id = target_id.strip("[]")

        return IntegrationDecision(
            knowledge_block_id=knowledge_block_id,
            target_page=decision_dict["page"],
            action=action,
            target_block_id=target_id,
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

    def _get_original_block_content(self, block_id: str) -> str:
        """
        Get the original journal content for a knowledge block.

        Args:
            block_id: Hybrid ID of the knowledge block

        Returns:
            Original block content as a string
        """
        block = find_block_by_id(self.state.journal_entry.outline.blocks, block_id)
        if block and block.content:
            return "\n".join(block.content)
        return "(empty block)"

    def _render_new_content(self, decision: IntegrationDecision, indent: str = "") -> list[str]:
        """
        Render new content with appropriate styling.

        Shows content with dim green bar and green text, regardless of whether
        it's refined text or original journal content.

        Args:
            decision: Integration decision containing refined_text and knowledge_block_id
            indent: Indentation string to prepend to each line

        Returns:
            List of formatted lines with green bars
        """
        lines = []

        # Get content to display (refined if available, otherwise original)
        if decision.refined_text:
            content_to_show = decision.refined_text
        else:
            content_to_show = self._get_original_block_content(decision.knowledge_block_id)

        # Render with consistent green styling and markdown formatting
        for content_line in content_to_show.split('\n'):
            rendered_line = render_markdown_to_markup(content_line, strip_id=True)
            lines.append(f"{indent}[dim green]┃[/dim green] [green]{rendered_line}[/green]")

        return lines

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
        if decision.target_block_id:
            lines.append(f"[dim]Target Block ID: {decision.target_block_id[:16]}...[/dim]")
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
                    rendered_line = render_markdown_to_markup(line, strip_id=True)
                    if i == 0:
                        lines.append(f"{indent}• [bold]{rendered_line}[/bold]")
                    else:
                        lines.append(f"{indent}  [bold]{rendered_line}[/bold]")
            else:
                # Parent context - show dimmed, just first line
                content_preview = block.content[0] if block.content else "(empty)"
                # Render markdown (Textual handles overflow with scrolling)
                rendered_preview = render_markdown_to_markup(content_preview, strip_id=True)
                lines.append(f"{indent}• [dim]{rendered_preview}[/dim]")

        return lines

    def _render_page_preview(self, decision: IntegrationDecision) -> list[str]:
        """
        Render target page preview with visual diff highlighting.

        Shows the REFINED (reworded) content when available, or the original
        journal content in dark green while rewording is in progress.
        Uses candidate blocks data for proper hybrid ID matching.
        """
        lines = []

        # Get candidate to access block IDs
        candidate = None
        for cand in self.state.candidates.get(decision.knowledge_block_id, []):
            if cand.page_name == decision.target_page:
                candidate = cand
                break

        if not candidate:
            # No candidate data - just show action description
            if decision.action == "add_section":
                lines.append("[dim]New page - adding first block:[/dim]")
                lines.extend(self._render_new_content(decision))
            return lines

        # Build ID map from candidate blocks
        id_to_block_info = {block_info["id"]: block_info for block_info in candidate.blocks}

        # Debug: Log candidate blocks and target
        logger.info(f"Rendering {decision.target_page}: {len(candidate.blocks)} blocks, target_id={decision.target_block_id}")
        for i, block_info in enumerate(candidate.blocks[:5]):
            logger.info(f"  Block {i}: id={block_info.get('id', 'N/A')[:16]}..., content={block_info.get('content', '')[:50]}")

        # Render blocks from candidate data with hierarchy
        lines.extend(self._render_candidate_blocks(
            candidate.blocks,
            decision,
            depth=0
        ))

        # If add_section, append new content at the end
        if decision.action == "add_section":
            lines.extend(self._render_new_content(decision))

        return lines

    def _render_candidate_blocks(
        self,
        blocks: list[dict],
        decision: IntegrationDecision,
        depth: int
    ) -> list[str]:
        """
        Render blocks from candidate data with proper hierarchy.

        Args:
            blocks: List of block info dicts from candidate
            decision: Current integration decision
            depth: Starting depth (usually 0)

        Returns:
            List of formatted lines
        """
        lines = []

        # Group blocks by parent to build hierarchy
        root_blocks = [b for b in blocks if b.get("parent_id") is None]
        children_map = {}
        for block in blocks:
            parent_id = block.get("parent_id")
            if parent_id:
                if parent_id not in children_map:
                    children_map[parent_id] = []
                children_map[parent_id].append(block)

        def render_block_and_children(block_info: dict, current_depth: int):
            block_lines = []
            block_id = block_info.get("id")
            content = block_info.get("content", "")

            # Apply markdown rendering (Textual handles overflow with scrolling)
            rendered_content = render_markdown_to_markup(content, strip_id=True)

            indent = "  " * current_depth

            # Check if this block is being replaced
            if decision.action == "replace" and block_id == decision.target_block_id:
                # Show strikethrough for old content
                block_lines.append(f"{indent}• [strike]{rendered_content}[/strike]")
                # Show new refined content below with green bar
                block_lines.extend(self._render_new_content(decision, indent))
            else:
                # Normal block display
                block_lines.append(f"{indent}• {rendered_content}")

            # Render children
            if block_id in children_map:
                for child in children_map[block_id]:
                    block_lines.extend(render_block_and_children(child, current_depth + 1))

            # If adding under this block, show new REFINED content as child AFTER children
            if decision.action == "add_under" and block_id == decision.target_block_id:
                child_indent = "  " * (current_depth + 1)
                logger.info(f"Found target block for add_under: block_id={block_id}, refined_text_length={len(decision.refined_text) if decision.refined_text else 0}")
                block_lines.extend(self._render_new_content(decision, child_indent))

            return block_lines

        # Render all root blocks
        for root_block in root_blocks:
            lines.extend(render_block_and_children(root_block, depth))

        return lines

    def _render_blocks_recursive(
        self,
        blocks: list,
        decision: IntegrationDecision,
        block_id_map: dict,
        depth: int,
        parents: list
    ) -> list[str]:
        """
        Recursively render blocks with proper indentation and modifications.

        Args:
            blocks: List of LogseqBlock objects
            decision: Current integration decision
            block_id_map: Map of hybrid_id -> block (for reverse lookup)
            depth: Current nesting depth
            parents: List of parent blocks (for hybrid ID generation)

        Returns:
            List of formatted lines
        """
        from logsqueak.logseq.context import generate_full_context, generate_content_hash

        lines = []
        indent = "  " * depth

        for block in blocks:
            # Generate hybrid ID for this block
            if block.block_id:
                hybrid_id = block.block_id
            else:
                full_context = generate_full_context(block, parents)
                hybrid_id = generate_content_hash(full_context, decision.target_page)

            # Debug: Log hybrid ID comparison
            if decision.action == "add_under":
                is_match = hybrid_id == decision.target_block_id
                logger.info(f"Block hybrid_id={hybrid_id[:16]}..., target={decision.target_block_id[:16] if decision.target_block_id else None}..., match={is_match}")

            # Get first line of content for display
            content = block.content[0] if block.content else "(empty)"

            # Apply markdown rendering (Textual handles overflow with scrolling)
            rendered_content = render_markdown_to_markup(content, strip_id=True)

            # Check if this block is being replaced
            if decision.action == "replace" and hybrid_id == decision.target_block_id:
                # Show strikethrough for old content
                lines.append(f"{indent}• [strike]{rendered_content}[/strike]")
                # Show new refined content below with green bar
                lines.extend(self._render_new_content(decision, indent))
            else:
                # Normal block display
                lines.append(f"{indent}• {rendered_content}")

            # Render children first
            if block.children:
                new_parents = parents + [block]
                lines.extend(self._render_blocks_recursive(
                    block.children,
                    decision,
                    block_id_map,
                    depth + 1,
                    new_parents
                ))

            # If adding under this block, show new REFINED content as child AFTER children
            if decision.action == "add_under" and hybrid_id == decision.target_block_id:
                child_indent = "  " * (depth + 1)
                logger.info(f"Found target block for add_under: {hybrid_id}, refined_text={decision.refined_text[:50] if decision.refined_text else None}")
                lines.extend(self._render_new_content(decision, child_indent))

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
            return "↳ Add new block"
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
