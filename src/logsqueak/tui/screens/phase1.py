"""Phase 1 Screen: Interactive knowledge classification.

This screen allows users to review and override LLM classification of journal blocks
as either "knowledge" (lasting information) or "activity" (temporary logs).
"""

import asyncio
import logging
from typing import Optional

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Vertical
from textual.screen import Screen
from textual.widgets import Footer, Header, Label, Static
from textual.widgets.tree import TreeNode

from logsqueak.logseq.parser import LogseqBlock
from logsqueak.tui.models import BlockState, ScreenState
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
    2. Start LLM streaming task (classify each block asynchronously)
    3. Update UI in real-time as classifications arrive
    4. User can override any classification (K=knowledge, A=activity, R=reset)
    5. User-marked blocks are locked (LLM cannot override)
    6. Press Enter to continue to Phase 2 when satisfied

    Keyboard Bindings:
    - j/↓: Navigate down
    - k/↑: Navigate up
    - K: Mark current block as knowledge
    - A: Mark current block as activity
    - R: Reset to LLM classification (or pending if LLM hasn't classified yet)
    - Enter: Continue to Phase 2 (validate at least one knowledge block exists)
    - q: Quit application
    """

    BINDINGS = [
        ("j", "cursor_down", "Down"),
        ("k", "cursor_up", "Up"),
        ("down", "cursor_down", "Down"),
        ("up", "cursor_up", "Up"),
        ("K", "mark_knowledge", "Mark Knowledge"),
        ("A", "mark_activity", "Mark Activity"),
        ("R", "reset", "Reset"),
        ("enter", "continue", "Continue"),
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
            yield Label(f"Phase 1: Knowledge Classification - {self.state.journal_entry.date}")
            yield Static("Waiting for LLM...", id="llm-status")

        # Block tree container
        with Container(id="tree-container"):
            # Will be populated in on_mount with actual block data
            yield Label("Loading journal blocks...")

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
            block_id = block.block_id or self._generate_content_hash(block)

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

    def _generate_content_hash(self, block: LogseqBlock) -> str:
        """
        Generate content hash for block without id:: property.

        Uses MD5 hash of full block content (normalized).

        Args:
            block: LogseqBlock to hash

        Returns:
            MD5 hash string
        """
        import hashlib

        content = block.get_full_content(normalize_whitespace=True)
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    async def _stream_extraction(self) -> None:
        """
        Stream LLM classification results and update UI.

        Calls LLMClient.stream_extract_ndjson() with all journal blocks
        and updates block_states as results arrive.

        Handles:
        - Network errors (show error message)
        - Malformed responses (log and skip)
        - User overrides (skip updates for source="user")
        - UI refresh (post message to update tree)
        """
        try:
            # Prepare blocks for LLM
            blocks_payload = self._prepare_blocks_for_llm()

            logger.info(f"Starting LLM extraction stream for {len(blocks_payload)} blocks")
            logger.debug(f"Sending {len(blocks_payload)} blocks to LLM for classification")

            # Update status
            status = self.query_one("#llm-status", Static)
            status.update(f"Classifying {len(blocks_payload)} blocks...")

            # Stream classifications
            classified_count = 0
            async for result in self.state.llm_client.stream_extract_ndjson(blocks_payload):
                # Log received JSON block
                logger.debug(f"Received NDJSON block: {result}")

                # Parse result
                block_id = result.get("block_id")
                is_knowledge = result.get("is_knowledge", False)
                confidence = result.get("confidence", 0.0)

                if block_id not in self.state.block_states:
                    logger.warning(f"Received classification for unknown block: {block_id}")
                    logger.warning(f"  Full JSON: {result}")
                    continue

                block_state = self.state.block_states[block_id]

                # Only update if not user-overridden
                if block_state.source != "user":
                    classification = "knowledge" if is_knowledge else "activity"

                    # Store LLM classification
                    block_state.classification = classification
                    block_state.confidence = confidence
                    block_state.llm_classification = classification
                    block_state.llm_confidence = confidence

                    classified_count += 1

                    # Get block content for logging
                    logseq_block = self._find_logseq_block(block_id)
                    block_content = ""
                    if logseq_block:
                        first_line = logseq_block.content[0] if logseq_block.content else "(empty)"
                        block_content = first_line[:100]  # First 100 chars

                    # Log classification with content
                    logger.info(
                        f"Classified as {classification.upper()} ({confidence:.0%}): {block_content}"
                    )
                    logger.debug(
                        f"  Block ID: {block_id[:8]}... | Full classification: {result}"
                    )

                    # Update UI - find and refresh the tree node
                    if self.block_tree:
                        node = self._find_tree_node(self.block_tree.root, block_id)
                        if node:
                            self._refresh_tree_node(node)
                        else:
                            logger.warning(f"Could not find tree node for block {block_id[:8]}... (classification: {classification})")

            # Streaming complete
            logger.info(f"LLM classification complete: {classified_count} blocks classified")
            status.update(f"Classification complete ({classified_count} blocks)")

        except asyncio.CancelledError:
            logger.info("LLM streaming task cancelled by user")
            raise

        except Exception as e:
            logger.error(f"Error during LLM streaming: {e}", exc_info=True)
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Classified {classified_count} blocks before error")
            status = self.query_one("#llm-status", Static)
            status.update(f"Error: {str(e)}")

    def _prepare_blocks_for_llm(self) -> list[dict]:
        """
        Convert journal blocks to LLM API format.

        Returns:
            List of dicts with block_id, content, and hierarchy fields
        """
        blocks = []

        def walk_blocks(block: LogseqBlock, parents: list[str] = []) -> None:
            """Recursively walk blocks and collect data."""
            block_id = block.block_id or self._generate_content_hash(block)
            content = block.get_full_content()

            # Build hierarchical context (parent bullets)
            hierarchy = "\n".join(f"  - {p}" for p in parents)

            blocks.append({
                "block_id": block_id,
                "content": content,
                "hierarchy": hierarchy,
            })

            # Recurse with current content added to parents
            first_line = block.content[0] if block.content else "(empty)"
            for child in block.children:
                walk_blocks(child, parents + [first_line])

        # Walk all root blocks
        for root_block in self.state.journal_entry.outline.blocks:
            walk_blocks(root_block)

        return blocks

    def action_cursor_down(self) -> None:
        """Move cursor down (j or ↓)."""
        if self.block_tree:
            self.block_tree.action_cursor_down()

    def action_cursor_up(self) -> None:
        """Move cursor up (k or ↑)."""
        if self.block_tree:
            self.block_tree.action_cursor_up()

    def action_mark_knowledge(self) -> None:
        """
        Mark current block as knowledge (K key).

        Sets:
        - classification = "knowledge"
        - source = "user"
        - Preserves llm_classification and llm_confidence if present
        - Cascades to children unless they have user overrides
        """
        if not self.block_tree:
            return

        # Get current node
        cursor_node = self.block_tree.cursor_node
        if cursor_node is None or cursor_node.data is None:
            return

        block_id = cursor_node.data

        if block_id not in self.state.block_states:
            return

        block_state = self.state.block_states[block_id]

        # Mark as user-classified knowledge
        block_state.classification = "knowledge"
        block_state.source = "user"
        block_state.confidence = None  # User classifications don't have confidence

        logger.info(f"User marked block {block_id[:8]}... as knowledge")

        # Apply smart defaults: cascade to children
        self._cascade_classification(cursor_node, "knowledge")

        # Refresh tree node label
        self._refresh_tree_node(cursor_node)

    def action_mark_activity(self) -> None:
        """
        Mark current block as activity (A key).

        Sets:
        - classification = "activity"
        - source = "user"
        - Preserves llm_classification and llm_confidence if present
        - Cascades to children unless they have user overrides
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

        # Mark as user-classified activity
        block_state.classification = "activity"
        block_state.source = "user"
        block_state.confidence = None

        logger.info(f"User marked block {block_id[:8]}... as activity")

        # Apply smart defaults: cascade to children
        self._cascade_classification(cursor_node, "activity")

        # Refresh tree node label
        self._refresh_tree_node(cursor_node)

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

        # Refresh tree node label
        self._refresh_tree_node(cursor_node)

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

        # TODO: Push Phase2Screen when implemented
        # from logsqueak.tui.screens.phase2 import Phase2Screen
        # await self.app.push_screen(Phase2Screen(self.state))

        # For now, just show message
        status = self.query_one("#llm-status", Static)
        status.update(f"✓ Ready to proceed with {knowledge_count} knowledge blocks (Phase 2 not yet implemented)")

    def _cascade_classification(
        self,
        parent_node: "TreeNode",
        classification: str,
    ) -> None:
        """
        Cascade classification from parent to all children.

        Only updates children that don't have user overrides (source != "user").
        Recursively applies to all descendants.

        Args:
            parent_node: Tree node whose children should inherit classification
            classification: Classification to cascade ("knowledge" or "activity")
        """
        for child_node in parent_node.children:
            if child_node.data is None:
                continue

            child_block_id = child_node.data

            if child_block_id not in self.state.block_states:
                continue

            child_state = self.state.block_states[child_block_id]

            # Only cascade if child hasn't been user-overridden
            if child_state.source != "user":
                child_state.classification = classification
                child_state.confidence = None
                # Don't change source - keep as "llm" to allow LLM to override later
                # unless parent was explicitly user-marked

                logger.debug(
                    f"Cascaded {classification} to child block {child_block_id[:8]}..."
                )

                # Refresh child node label
                self._refresh_tree_node(child_node)

            # Recurse to grandchildren
            self._cascade_classification(child_node, classification)

    def _refresh_tree_node(self, node: "TreeNode") -> None:
        """
        Refresh a tree node's label after state change.

        Re-generates the label based on current BlockState and updates the tree node.

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
        logseq_block = self._find_logseq_block(block_id)

        if logseq_block is None:
            logger.warning(f"Could not find LogseqBlock for ID {block_id}")
            return

        # Format new label
        if self.block_tree:
            new_label = self.block_tree.format_block_label(logseq_block, block_state)
            old_label = str(node.label)
            node.set_label(new_label)
            logger.debug(f"Refreshed tree node {block_id[:8]}: '{old_label[:50]}' -> '{new_label[:50]}'")

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

    def _find_logseq_block(self, block_id: str) -> Optional[LogseqBlock]:
        """
        Find LogseqBlock by ID in journal entry.

        Args:
            block_id: Block ID to search for

        Returns:
            LogseqBlock if found, None otherwise
        """
        def walk_blocks(block: LogseqBlock) -> Optional[LogseqBlock]:
            """Recursively search for block."""
            block_block_id = block.block_id or self._generate_content_hash(block)

            if block_block_id == block_id:
                return block

            for child in block.children:
                result = walk_blocks(child)
                if result is not None:
                    return result

            return None

        # Search all root blocks
        for root_block in self.state.journal_entry.outline.blocks:
            result = walk_blocks(root_block)
            if result is not None:
                return result

        return None

    async def on_unmount(self) -> None:
        """Cancel LLM task on screen unmount."""
        if self.llm_task and not self.llm_task.done():
            logger.info("Cancelling LLM task on unmount")
            self.llm_task.cancel()
            try:
                await self.llm_task
            except asyncio.CancelledError:
                pass
