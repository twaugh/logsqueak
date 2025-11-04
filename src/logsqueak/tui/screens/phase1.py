"""Phase 1 Screen: Interactive knowledge classification.

This screen allows users to review and override LLM classification of journal blocks
as either "knowledge" (lasting information) or "activity" (temporary logs).
"""

import asyncio
import logging
from typing import Optional

from textual import events, on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.screen import Screen
from textual.widgets import Footer, Header, Label, Static, Tree
from textual.widgets.tree import TreeNode

from logseq_outline import LogseqBlock
from logsqueak.tui.models import BlockState, ScreenState
from logsqueak.tui.utils import find_block_by_id, generate_content_hash
from logsqueak.tui.widgets.block_tree import BlockTree

logger = logging.getLogger(__name__)


class Phase1Screen(Screen):
    """
    Phase 1: Knowledge Classification Screen.

    Displays journal blocks in hierarchical tree with real-time LLM classification
    streaming and user override capabilities.

    User Stories:
    - US1: Review and approve knowledge extraction with user override capability

    Workflow:
    1. On mount: Initialize all blocks as "pending"
    2. Start LLM streaming task (suggest classification for each block)
    3. LLM suggestions shown as "[LLM: ✓ 92%]" but blocks remain "pending"
    4. User must explicitly accept suggestions:
       - K: Accept as knowledge (or use LLM suggestion if available)
       - A: Accept as activity
       - Shift+A: Accept all LLM suggestions at once
    5. User can reset accepted blocks (R=reset to pending)
    6. Press Enter to continue when at least one knowledge block is accepted

    Keyboard Bindings:
    - j/↓: Navigate down
    - k/↑: Navigate up
    - Space: Toggle knowledge selection for current block
    - a: Accept all AI suggestions
    - r: Reset current block to AI suggestion
    - c: Clear all selections (mark all as activity)
    - Enter: Continue to Phase 2
    - q: Quit application
    - ?: Show help
    """

    BINDINGS = [
        ("j", "cursor_down", "Down"),
        ("k", "cursor_up", "Up"),
        ("down", "cursor_down", "Down"),
        ("up", "cursor_up", "Up"),
        Binding("space", "toggle_selection", "Toggle", priority=True),  # Override Tree's space binding
        ("a", "accept_all", "Accept AI"),
        ("r", "reset", "Reset"),
        ("c", "clear_all", "Clear All"),
        Binding("enter", "continue", "Continue →", priority=True),  # Override Tree's enter binding
    ]

    CSS = """
    Phase1Screen {
        layout: vertical;
    }

    #status-bar {
        dock: top;
        height: 3;
        background: $panel;
        padding: 1;
    }

    #tree-container {
        height: 1fr;
        overflow-y: auto;
    }

    BlockTree {
        height: 100%;
    }

    #reason-bar {
        dock: bottom;
        height: auto;
        min-height: 3;
        background: $boost;
        border-top: solid $primary;
        padding: 1;
    }

    #reason-text {
        color: $text;
    }
    """

    def __init__(self, state: ScreenState):
        """
        Initialize Phase1Screen.

        Args:
            state: Shared application state
        """
        super().__init__()
        self.state = state
        self.llm_task: Optional[asyncio.Task] = None
        self.block_tree: Optional[BlockTree] = None

    def compose(self) -> ComposeResult:
        """Compose screen layout."""
        yield Header()

        # Status bar showing progress
        with Vertical(id="status-bar"):
            yield Label(f"Pick blocks to integrate into pages - {self.state.journal_entry.date}")
            yield Static("Waiting for LLM...", id="llm-status")

        # Block tree container
        with Container(id="tree-container"):
            # Will be populated in on_mount with actual block data
            yield Label("Loading journal blocks...")

        # Reason bar (shows why highlighted block is knowledge)
        # Must come before Footer to be visible
        with Container(id="reason-bar"):
            yield Static("Select a knowledge block to see why it was classified as knowledge", id="reason-text")

        yield Footer()

    async def on_mount(self) -> None:
        """
        Initialize screen on mount.

        Steps:
        1. Initialize all block_states as "pending"
        2. Create and populate BlockTree widget
        3. Start LLM streaming task
        4. Update phase in shared state
        """
        logger.info(f"Phase1Screen mounted for journal {self.state.journal_entry.date}")

        # Initialize all blocks as pending
        self._initialize_block_states()

        # Create BlockTree widget
        self.block_tree = BlockTree(
            f"Journal {self.state.journal_entry.date}",
            self.state.block_states,
        )
        self.block_tree.root.expand()

        # Populate tree with journal blocks
        for root_block in self.state.journal_entry.outline.blocks:
            self.block_tree.add_block(root_block)

        # Replace loading message with tree
        container = self.query_one("#tree-container", Container)
        await container.query("Label").remove()
        await container.mount(self.block_tree)

        # Update phase
        self.state.current_phase = 1

        # Start LLM streaming
        self.llm_task = asyncio.create_task(self._stream_extraction())

    def _initialize_block_states(self) -> None:
        """
        Initialize block_states dict with all blocks from journal as "pending".

        Recursively walks LogseqBlock tree and creates BlockState for each block.
        """
        def walk_blocks(block: LogseqBlock) -> None:
            """Recursively walk blocks and initialize states."""
            block_id = block.block_id or generate_content_hash(block)

            self.state.block_states[block_id] = BlockState(
                block_id=block_id,
                classification="pending",
                confidence=None,
                source="llm",
                llm_classification=None,
                llm_confidence=None,
            )

            for child in block.children:
                walk_blocks(child)

        # Walk all root blocks
        for root_block in self.state.journal_entry.outline.blocks:
            walk_blocks(root_block)

        logger.info(f"Initialized {len(self.state.block_states)} block states")

    async def _stream_extraction(self) -> None:
        """
        Stream LLM classification results and update UI.

        Calls LLMClient.stream_extract_ndjson() with full journal content
        and updates block_states as results arrive.

        Handles:
        - Network errors (show error message)
        - Malformed responses (log and skip)
        - User overrides (skip updates for source="user")
        - UI refresh (post message to update tree)
        """
        classified_count = 0
        try:
            # Prepare journal content with IDs injected
            journal_content = self._prepare_journal_with_ids()

            logger.info(f"Starting LLM extraction stream for journal")
            logger.debug(f"Journal content length: {len(journal_content)} chars")

            # Update status
            status = self.query_one("#llm-status", Static)
            status.update("Classifying journal blocks...")

            # Stream classifications
            async for result in self.state.llm_client.stream_extract_ndjson(journal_content):
                # Log received JSON block
                logger.debug(f"Received NDJSON block: {result}")

                # Parse result - LLM only outputs knowledge blocks now
                block_id = result.get("block_id")
                confidence = result.get("confidence", 0.0)
                reason = result.get("reason", None)

                if block_id not in self.state.block_states:
                    logger.warning(f"Received classification for unknown block: {block_id}")
                    logger.warning(f"  Full JSON: {result}")
                    continue

                block_state = self.state.block_states[block_id]

                # Only update if not user-overridden
                if block_state.source != "user":
                    # LLM only outputs knowledge blocks (no is_knowledge field needed)
                    # Store LLM suggestion but keep block as "pending" until user accepts
                    block_state.llm_classification = "knowledge"
                    block_state.llm_confidence = confidence
                    block_state.reason = reason
                    # Keep classification as "pending" - user must explicitly accept
                    # block_state.classification remains "pending"
                    # block_state.confidence remains None
                    # block_state.source remains "llm"

                    classified_count += 1

                    # Get block content for logging
                    logseq_block = find_block_by_id(self.state.journal_entry.outline.blocks, block_id)
                    block_content = ""
                    if logseq_block:
                        first_line = logseq_block.content[0] if logseq_block.content else "(empty)"
                        block_content = first_line[:100]  # First 100 chars

                    # Log LLM suggestion with content and reason
                    reason_text = f" - {reason}" if reason else ""
                    logger.info(
                        f"LLM suggests KNOWLEDGE ({confidence:.0%}){reason_text}: {block_content}"
                    )
                    logger.debug(
                        f"  Block ID: {block_id[:8]}... | Full result: {result}"
                    )

                    # Update UI - find and refresh the tree node and ancestors
                    if self.block_tree:
                        node = self._find_tree_node(self.block_tree.root, block_id)
                        if node:
                            self._refresh_tree_node(node)
                            self._refresh_ancestors(node)
                            # Update reason bar if this is the currently highlighted block
                            if self.block_tree.cursor_node == node:
                                self._update_reason_bar()
                        else:
                            logger.warning(f"Could not find tree node for block {block_id[:8]}... (LLM suggestion: knowledge)")

            # Streaming complete - mark remaining pending blocks with activity suggestion
            unclassified_count = self._mark_unclassified_as_activity()

            logger.info(f"LLM suggestions complete: {classified_count} knowledge suggestions")
            if unclassified_count > 0:
                logger.info(f"Marked {unclassified_count} remaining blocks with activity suggestion")
            status.update(f"Suggestions complete: {classified_count} knowledge, {unclassified_count} activity (use Shift+A to accept all)")

        except asyncio.CancelledError:
            logger.info("LLM streaming task cancelled by user")
            raise

        except Exception as e:
            logger.error(f"Error during LLM streaming: {e}", exc_info=True)
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Classified {classified_count} blocks before error")
            status = self.query_one("#llm-status", Static)
            status.update(f"Error: {str(e)}")

    def _mark_unclassified_as_activity(self) -> int:
        """
        Mark all remaining pending blocks with LLM suggestion of activity.

        The LLM was instructed to output knowledge blocks only. Any blocks not
        explicitly identified are inferred to be activity logs (as suggestions).

        Returns:
            Number of blocks marked with activity suggestion
        """
        count = 0

        for block_id, block_state in self.state.block_states.items():
            # Only update blocks that are still pending and have no LLM classification yet
            if block_state.classification == "pending" and not block_state.llm_classification:
                # Set LLM suggestion to activity (but keep classification as pending)
                block_state.llm_classification = "activity"
                block_state.llm_confidence = 0.95  # High confidence - inferred from absence
                # classification remains "pending" - user must still accept
                count += 1

                # Update UI for this block
                if self.block_tree:
                    node = self._find_tree_node(self.block_tree.root, block_id)
                    if node:
                        self._refresh_tree_node(node)
                        self._refresh_ancestors(node)

        return count

    def _prepare_journal_with_ids(self) -> str:
        """
        Render journal entry as Logseq markdown with IDs injected.

        Ensures every block has an id:: property (using hash-based IDs where missing).

        Returns:
            Full journal content as Logseq markdown string
        """
        def inject_ids(block: LogseqBlock) -> None:
            """Recursively inject id:: properties where missing."""
            # If block doesn't have an id:: property, add hash-based ID
            if not block.block_id:
                hash_id = generate_content_hash(block)
                # Use set_property to add id:: property
                block.set_property("id", hash_id)

            # Recurse to children
            for child in block.children:
                inject_ids(child)

        # Clone the outline to avoid mutating the original
        import copy
        outline_copy = copy.deepcopy(self.state.journal_entry.outline)

        # Inject IDs in all blocks
        for root_block in outline_copy.blocks:
            inject_ids(root_block)

        # Render to markdown
        return outline_copy.render()

    def action_cursor_down(self) -> None:
        """Move cursor down (j or ↓)."""
        if self.block_tree:
            self.block_tree.action_cursor_down()
            self._update_reason_bar()

    def action_cursor_up(self) -> None:
        """Move cursor up (k or ↑)."""
        if self.block_tree:
            self.block_tree.action_cursor_up()
            self._update_reason_bar()

    def action_toggle_selection(self) -> None:
        """
        Toggle knowledge selection for current block (Space key).

        If block is currently "knowledge" → deselect (restore to LLM suggestion or activity)
        If block is currently "activity" or "pending" → mark as "knowledge"

        When deselecting:
        - If block has LLM suggestion, restore to pending state (shows blue background)
        - If no LLM suggestion, mark as activity

        Only affects the selected block (no cascading).
        """
        if not self.block_tree:
            return

        cursor_node = self.block_tree.cursor_node
        if cursor_node is None or cursor_node.data is None:
            return

        block_id = cursor_node.data
        if block_id not in self.state.block_states:
            return

        block_state = self.state.block_states[block_id]

        # Toggle logic
        if block_state.classification == "knowledge":
            # Deselecting knowledge block
            if block_state.llm_classification:
                # Has LLM suggestion - restore to pending (will show suggestion background)
                block_state.classification = "pending"
                block_state.source = "llm"
                block_state.confidence = None
                logger.info(f"User deselected block {block_id[:8]}..., restored to LLM suggestion (pending)")
            else:
                # No LLM suggestion - mark as activity
                block_state.classification = "activity"
                block_state.source = "user"
                block_state.confidence = None
                logger.info(f"User toggled block {block_id[:8]}... to activity")
        else:
            # Selecting as knowledge (from activity or pending)
            block_state.classification = "knowledge"
            block_state.source = "user"
            block_state.confidence = None
            logger.info(f"User toggled block {block_id[:8]}... to knowledge")

        # Refresh tree display (only this node and ancestors for knowledge count)
        self._refresh_tree_node(cursor_node)
        self._refresh_ancestors(cursor_node)
        self._update_reason_bar()

    def action_clear_all(self) -> None:
        """
        Clear all selections (c key).

        Resets all blocks back to pending state, preserving AI suggestions.
        - Affects both user-selected blocks (source="user") and accepted AI blocks
        - Resets classification to "pending" (AI suggestions will still show with blue background)
        - Preserves llm_classification and llm_confidence
        """
        cleared_count = 0

        for block_id, block_state in self.state.block_states.items():
            # Clear any block that's not already in pending state
            if block_state.classification != "pending":
                block_state.classification = "pending"
                block_state.source = "llm"
                block_state.confidence = None
                cleared_count += 1

        logger.info(f"Cleared {cleared_count} selections (AI suggestions preserved)")

        # Refresh entire tree
        if self.block_tree:
            self._refresh_entire_tree()
        self._update_reason_bar()


    def action_reset(self) -> None:
        """
        Reset current block to LLM classification (R key).

        If LLM has classified this block:
        - Restore llm_classification → classification
        - Restore llm_confidence → confidence
        - Set source = "llm"

        If LLM has not classified yet:
        - Reset to "pending"
        """
        if not self.block_tree:
            return

        cursor_node = self.block_tree.cursor_node
        if cursor_node is None or cursor_node.data is None:
            return

        block_id = cursor_node.data

        if block_id not in self.state.block_states:
            return

        block_state = self.state.block_states[block_id]

        # Reset to LLM classification if available
        if block_state.llm_classification is not None:
            block_state.classification = block_state.llm_classification
            block_state.confidence = block_state.llm_confidence
            block_state.source = "llm"
            logger.info(
                f"Reset block {block_id[:8]}... to LLM classification: "
                f"{block_state.llm_classification}"
            )
        else:
            # LLM hasn't classified yet, reset to pending
            block_state.classification = "pending"
            block_state.confidence = None
            block_state.source = "llm"
            logger.info(f"Reset block {block_id[:8]}... to pending (no LLM classification)")

        # Refresh tree node label and all ancestors (knowledge counts changed)
        self._refresh_tree_node(cursor_node)
        self._refresh_ancestors(cursor_node)
        self._update_reason_bar()

    def action_accept_all(self) -> None:
        """
        Accept all LLM suggestions (Shift+A).

        Converts all blocks with LLM suggestions to their suggested classification.
        - Blocks with llm_classification="knowledge" → classification="knowledge"
        - Blocks with llm_classification="activity" → classification="activity"
        - Preserves user-marked blocks (source="user")
        - Updates confidence and source appropriately
        """
        accepted_count = 0
        knowledge_count = 0

        for block_id, block_state in self.state.block_states.items():
            # Skip user-overridden blocks
            if block_state.source == "user":
                continue

            # Skip blocks without LLM suggestions
            if not block_state.llm_classification:
                continue

            # Accept the LLM suggestion
            block_state.classification = block_state.llm_classification
            block_state.confidence = block_state.llm_confidence
            block_state.source = "llm"
            accepted_count += 1

            # Count knowledge blocks specifically
            if block_state.llm_classification == "knowledge":
                knowledge_count += 1

        logger.info(f"Accepted {accepted_count} LLM suggestions ({knowledge_count} knowledge)")

        # Refresh entire tree to show accepted classifications
        if self.block_tree:
            self._refresh_entire_tree()

        # Update reason bar
        self._update_reason_bar()

        # Show notification - only show knowledge count to user
        self.app.notify(f"Accepted {knowledge_count} knowledge block{'s' if knowledge_count != 1 else ''}")

    def _refresh_entire_tree(self) -> None:
        """Refresh all nodes in the tree after bulk changes."""
        if not self.block_tree:
            return

        # Recursively refresh all nodes
        def refresh_node_and_children(node: TreeNode) -> None:
            self._refresh_tree_node(node)
            for child in node.children:
                refresh_node_and_children(child)

        refresh_node_and_children(self.block_tree.root)

    async def action_continue(self) -> None:
        """
        Continue to Phase 2 (Enter key).

        Validation:
        - At least one block must be marked as "knowledge"
        - Cancel LLM task if still running

        On success:
        - Push Phase2Screen
        """
        # Cancel LLM task if running
        if self.llm_task and not self.llm_task.done():
            logger.info("Cancelling LLM task before proceeding to Phase 2")
            self.llm_task.cancel()
            try:
                await self.llm_task
            except asyncio.CancelledError:
                pass

        # Validate at least one knowledge block
        knowledge_count = sum(
            1 for state in self.state.block_states.values()
            if state.classification == "knowledge"
        )

        if knowledge_count == 0:
            logger.warning("No knowledge blocks selected, cannot continue")
            # TODO: Show error modal
            status = self.query_one("#llm-status", Static)
            status.update("⚠ No knowledge blocks selected! Mark at least one block as knowledge (K)")
            return

        logger.info(f"Proceeding to Phase 2 with {knowledge_count} knowledge blocks")

        # Push Phase2Screen
        from logsqueak.tui.screens.phase2 import Phase2Screen
        await self.app.push_screen(Phase2Screen(self.state))


    def _refresh_tree_node(self, node: "TreeNode") -> None:
        """
        Refresh a tree node's label after state change.

        Re-generates the label based on current BlockState and updates the tree node.
        Also recalculates knowledge count for the subtree.

        Args:
            node: Tree node to refresh
        """
        if node.data is None:
            return

        block_id = node.data

        if block_id not in self.state.block_states:
            return

        block_state = self.state.block_states[block_id]

        # Find the corresponding LogseqBlock
        # We need to walk the journal entry to find the block with matching ID
        logseq_block = find_block_by_id(self.state.journal_entry.outline.blocks, block_id)

        if logseq_block is None:
            logger.warning(f"Could not find LogseqBlock for ID {block_id}")
            return

        # Format new label with updated knowledge count
        if self.block_tree:
            knowledge_count = self.block_tree._count_knowledge_in_subtree(logseq_block)
            new_label = self.block_tree.format_block_label(logseq_block, block_state, knowledge_count)
            old_label = str(node.label)
            node.set_label(new_label)
            logger.debug(f"Refreshed tree node {block_id[:8]}: '{old_label[:50]}' -> '{new_label[:50]}'")

    def _refresh_ancestors(self, node: "TreeNode") -> None:
        """
        Refresh all ancestor nodes' labels after a classification change.

        When a block's classification changes, all ancestor knowledge counts
        need to be updated.

        Args:
            node: Tree node whose ancestors should be refreshed
        """
        current = node.parent
        while current is not None and current != self.block_tree.root:
            self._refresh_tree_node(current)
            current = current.parent

    def _find_tree_node(self, node: TreeNode, block_id: str) -> Optional[TreeNode]:
        """
        Find tree node by block_id.

        Recursively searches tree for node with matching data (block_id).

        Args:
            node: Tree node to start searching from
            block_id: Block ID to find

        Returns:
            TreeNode if found, None otherwise
        """
        if node.data == block_id:
            return node

        for child in node.children:
            result = self._find_tree_node(child, block_id)
            if result is not None:
                return result

        return None

    def _update_reason_bar(self) -> None:
        """
        Update the reason bar to show why the current highlighted block is knowledge.

        Shows the LLM's brief explanation (7-10 words) for classifying the block as knowledge.
        Only displays for knowledge blocks; clears for activity/pending blocks.
        """
        if not self.block_tree:
            return

        # Get current cursor node
        cursor_node = self.block_tree.cursor_node
        if cursor_node is None or cursor_node.data is None:
            # Clear reason bar if no node selected
            reason_text = self.query_one("#reason-text", Static)
            reason_text.update("")
            return

        block_id = cursor_node.data
        block_state = self.state.block_states.get(block_id)

        if not block_state:
            return

        # Update reason bar based on classification and LLM suggestion
        reason_text = self.query_one("#reason-text", Static)

        # Show LLM suggestion if block is pending
        if block_state.classification == "pending" and block_state.llm_classification:
            llm_label = "knowledge" if block_state.llm_classification == "knowledge" else "activity"
            llm_conf = int(block_state.llm_confidence * 100) if block_state.llm_confidence else 0

            if block_state.reason:
                # Show LLM suggestion with reason
                reason_text.update(f"🤖 LLM suggests {llm_label} ({llm_conf}%): {block_state.reason}")
            else:
                # Show LLM suggestion without reason
                reason_text.update(f"🤖 LLM suggests {llm_label} ({llm_conf}%)")

        # Show accepted classification
        elif block_state.classification == "knowledge" and block_state.reason:
            # Show reason for accepted knowledge
            reason_text.update(f"💡 Why knowledge: {block_state.reason}")

        else:
            # Clear for activity or blocks without info
            reason_text.update("")

    @on(Tree.NodeHighlighted)
    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted) -> None:
        """Handle tree node highlight event - update reason bar."""
        self._update_reason_bar()

    async def on_unmount(self) -> None:
        """Cancel LLM task on screen unmount."""
        if self.llm_task and not self.llm_task.done():
            logger.info("Cancelling LLM task on unmount")
            self.llm_task.cancel()
            try:
                await self.llm_task
            except asyncio.CancelledError:
                pass
