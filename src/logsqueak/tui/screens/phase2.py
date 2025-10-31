"""Phase 2 Screen: Candidate page retrieval via RAG search.

This screen orchestrates semantic + hinted search to find candidate pages
for each knowledge block. By default, it auto-proceeds to Phase 3 after
retrieval completes. Users can optionally press 'R' to enter review mode
to manually filter candidates.
"""

import asyncio
import logging
import re
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Container, Vertical
from textual.screen import Screen
from textual.widgets import Footer, Header, Label, Static

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

from logsqueak.logseq.parser import LogseqBlock, LogseqOutline
from logsqueak.rag.vector_store import VectorStore
from logsqueak.rag.vector_store import ChromaDBStore
from logsqueak.tui.models import CandidatePage, ScreenState
from logsqueak.tui.utils import find_block_by_id, generate_content_hash

logger = logging.getLogger(__name__)


class Phase2Screen(Screen):
    """
    Phase 2: Candidate Page Retrieval Screen.

    Performs RAG search (semantic + hinted) for each knowledge block to find
    candidate pages. Auto-proceeds to Phase 3 unless user enters review mode.

    User Stories:
    - US2: See LLM integration decisions with streaming feedback (auto-proceed portion)
    - US4: Review candidate pages before integration (optional review mode)

    Workflow:
    1. On mount: Start RAG search for each knowledge block
    2. Update progress indicator as candidates are retrieved
    3. If user presses 'R': Enter review mode (toggle candidates on/off)
    4. Otherwise: Auto-proceed to Phase 3 when retrieval completes

    Keyboard Bindings:
    - R: Enter review mode (optional - US4)
    - q: Quit application
    """

    BINDINGS = [
        ("R", "review_mode", "Review Candidates"),
    ]

    CSS = """
    Phase2Screen {
        layout: vertical;
    }

    #status-container {
        dock: top;
        height: auto;
        background: $panel;
        padding: 1;
    }

    #progress-container {
        height: 1fr;
        overflow-y: auto;
        padding: 1;
    }

    .progress-item {
        margin-bottom: 1;
    }
    """

    def __init__(self, state: ScreenState):
        """
        Initialize Phase2Screen.

        Args:
            state: Shared application state
        """
        super().__init__()
        self.state = state
        self.retrieval_task: Optional[asyncio.Task] = None
        self.review_mode_active = False
        self.embedding_model: Optional["SentenceTransformer"] = None
        self.progress_messages: list[str] = []

    def compose(self) -> ComposeResult:
        """Compose screen layout."""
        yield Header()
        with Container(id="status-container"):
            yield Label("Finding relevant pages for your knowledge...")
        with Container(id="progress-container"):
            yield Static(id="progress-display")
        yield Footer()

    async def on_mount(self) -> None:
        """
        Initialize Phase 2 and start RAG candidate retrieval.

        Steps:
        1. Filter block_states to get knowledge blocks only
        2. For each knowledge block:
           - Perform semantic search (vector store)
           - Perform hinted search ([[Page]] links)
           - Aggregate and deduplicate results
        3. Populate state.candidates dict
        4. Auto-proceed to Phase 3 (unless review mode requested)
        """
        self.state.current_phase = 2

        # Defer retrieval task until after screen is rendered
        # This ensures the screen appears immediately without blocking
        self.call_after_refresh(self._start_retrieval)

    def _start_retrieval(self) -> None:
        """Start the retrieval task after screen is visible."""
        self.retrieval_task = asyncio.create_task(self._retrieve_candidates())

    async def _retrieve_candidates(self) -> None:
        """
        Perform RAG search for all knowledge blocks.

        This task:
        1. Filters for knowledge blocks (classification="knowledge")
        2. For each knowledge block:
           a. Semantic search via vector store (top_k chunks)
           b. Hinted search via [[Page]] and ((block-ref)) patterns
           c. Aggregate, deduplicate, and group by page
        3. Populates state.candidates[knowledge_block_id] = [CandidatePage, ...]
        4. Auto-proceeds to Phase 3 when complete

        Error Handling:
        - Vector store errors: Log warning, skip semantic search for that block
        - Hinted search errors: Log warning, use semantic results only
        - If ALL searches fail: Show error, allow user to proceed anyway
        """
        try:
            # Show initial progress message immediately
            self._update_progress("Starting candidate retrieval...")
            await asyncio.sleep(0.1)  # Let UI render

            # Get knowledge blocks
            knowledge_blocks = [
                (block_id, state)
                for block_id, state in self.state.block_states.items()
                if state.classification == "knowledge"
            ]

            if not knowledge_blocks:
                logger.warning("No knowledge blocks to retrieve candidates for")
                self._update_progress("No knowledge blocks found. Skipping to Phase 3.")
                await asyncio.sleep(1.5)
                await self._proceed_to_phase3()
                return

            # Initialize embedding model (lazy-loaded, shared across all searches)
            self._update_progress("Loading embedding model (this may take a few seconds)...")
            await asyncio.sleep(0.1)  # Let UI update

            # Load model in executor to avoid blocking UI
            # Import here to avoid blocking the screen transition
            from sentence_transformers import SentenceTransformer

            loop = asyncio.get_event_loop()
            self.embedding_model = await loop.run_in_executor(
                None, lambda: SentenceTransformer("all-MiniLM-L6-v2")
            )

            # Initialize vector store
            self._update_progress("Initializing vector store...")
            await asyncio.sleep(0.1)  # Let UI update
            cache_dir = Path.home() / ".cache" / "logsqueak" / "chroma"
            vector_store = ChromaDBStore(persist_directory=cache_dir)

            total = len(knowledge_blocks)

            for idx, (block_id, block_state) in enumerate(knowledge_blocks, start=1):
                self._update_progress(
                    f"Retrieving candidates for block {idx}/{total} (ID: {block_id[:8]}...)"
                )

                # Get the actual block content from journal entry
                block = find_block_by_id(self.state.journal_entry.outline.blocks, block_id)
                if not block:
                    logger.warning(f"Block not found in journal: {block_id}")
                    continue

                # Perform semantic + hinted search
                # Use default top_k=20 for TUI (matches extractor.py default)
                candidates = await self._search_candidates(
                    block_id=block_id,
                    block=block,
                    vector_store=vector_store,
                    top_k=20,
                )

                # Store results
                self.state.candidates[block_id] = candidates

                # Allow UI to update
                await asyncio.sleep(0.05)

            # All retrieval complete
            self._update_progress(f"Retrieved candidates for {total} knowledge blocks.")
            await asyncio.sleep(1.0)

            # Auto-proceed unless review mode was requested
            if not self.review_mode_active:
                await self._proceed_to_phase3()

        except Exception as e:
            logger.error(f"Error during candidate retrieval: {e}", exc_info=True)
            self._update_progress(f"Error: {e}. Press Enter to continue anyway.")

    async def _search_candidates(
        self,
        block_id: str,
        block: LogseqBlock,
        vector_store: VectorStore,
        top_k: int,
    ) -> list[CandidatePage]:
        """
        Perform semantic + hinted search for a single knowledge block.

        Args:
            block_id: Hybrid ID of the knowledge block
            block: LogseqBlock from journal entry
            vector_store: Vector store for semantic search
            top_k: Number of semantic results to retrieve

        Returns:
            List of CandidatePage objects, sorted by similarity (highest first)
        """
        # Get full block text (concatenate all content lines)
        block_text = "\n".join(block.content)

        # 1. Semantic search
        semantic_candidates = await self._semantic_search(
            block_text=block_text,
            vector_store=vector_store,
            top_k=top_k,
        )

        # 2. Hinted search (parse [[Page Name]] links)
        hinted_candidates = await self._hinted_search(block_text=block_text)

        # 3. Aggregate and deduplicate (group by page)
        all_candidates = self._aggregate_candidates(
            semantic_candidates, hinted_candidates
        )

        return all_candidates

    async def _semantic_search(
        self,
        block_text: str,
        vector_store: VectorStore,
        top_k: int,
    ) -> list[CandidatePage]:
        """
        Perform semantic vector search for similar blocks.

        Args:
            block_text: Knowledge block content to search for
            vector_store: Vector store to query
            top_k: Number of results to return

        Returns:
            List of CandidatePage objects from semantic search
        """
        try:
            # Embed the query text
            query_embedding = self.embedding_model.encode(
                block_text, convert_to_numpy=True
            )

            # Query vector store
            ids, distances, metadatas = vector_store.query(
                query_embedding=query_embedding.tolist(), n_results=top_k
            )

            # Group chunks by page
            page_chunks = {}  # page_name -> [(similarity, block_id, metadata), ...]

            for i, chunk_id in enumerate(ids):
                metadata = metadatas[i]
                page_name = metadata["page_name"]
                similarity = 1.0 - distances[i]  # Convert distance to similarity

                if page_name not in page_chunks:
                    page_chunks[page_name] = []

                page_chunks[page_name].append((similarity, chunk_id, metadata))

            # Convert to CandidatePage objects
            candidates = []
            for page_name, chunks in page_chunks.items():
                # Use highest similarity score for the page
                max_similarity = max(sim for sim, _, _ in chunks)

                # Extract block info for targeting
                blocks = [
                    {
                        "id": chunk_id,
                        "content": metadata.get("block_content", ""),  # Don't truncate - let Phase 3 rendering handle it
                        "depth": metadata.get("depth", 0),
                        "parent_id": metadata.get("parent_id"),
                    }
                    for _, chunk_id, metadata in chunks
                ]

                candidates.append(
                    CandidatePage(
                        page_name=page_name,
                        similarity_score=max_similarity,
                        included=True,
                        blocks=blocks,
                        search_method="semantic",
                    )
                )

            return candidates

        except Exception as e:
            logger.warning(f"Semantic search failed: {e}")
            return []

    async def _hinted_search(self, block_text: str) -> list[CandidatePage]:
        """
        Perform hinted search by parsing [[Page Name]] links from block text.

        Args:
            block_text: Knowledge block content to extract page links from

        Returns:
            List of CandidatePage objects from hinted search (similarity_score=1.0)
        """
        try:
            # Extract [[Page Name]] patterns
            page_refs = re.findall(r'\[\[([^\]]+)\]\]', block_text)

            if not page_refs:
                return []

            candidates = []
            graph_path = Path(self.state.config.logseq.graph_path)

            for page_name in set(page_refs):  # Deduplicate
                # Check if page exists
                page_path = graph_path / "pages" / f"{page_name}.md"
                if not page_path.exists():
                    logger.debug(f"Hinted page not found: {page_name}")
                    continue

                # Load page to extract block info
                with open(page_path, "r", encoding="utf-8") as f:
                    page_content = f.read()

                outline = LogseqOutline.parse(page_content)

                # Extract all blocks for targeting
                blocks = self._extract_blocks_from_outline(outline)

                candidates.append(
                    CandidatePage(
                        page_name=page_name,
                        similarity_score=1.0,  # Explicit reference = highest priority
                        included=True,
                        blocks=blocks,
                        search_method="hinted",
                    )
                )

            return candidates

        except Exception as e:
            logger.warning(f"Hinted search failed: {e}")
            return []

    def _aggregate_candidates(
        self,
        semantic_candidates: list[CandidatePage],
        hinted_candidates: list[CandidatePage],
    ) -> list[CandidatePage]:
        """
        Merge and deduplicate semantic + hinted candidates.

        If the same page appears in both lists, prefer hinted (higher priority).

        Args:
            semantic_candidates: Results from semantic search
            hinted_candidates: Results from hinted search

        Returns:
            Deduplicated list sorted by similarity (highest first)
        """
        # Build map by page name
        page_map = {}

        # Add semantic candidates
        for candidate in semantic_candidates:
            page_map[candidate.page_name] = candidate

        # Override with hinted candidates (higher priority)
        for candidate in hinted_candidates:
            if candidate.page_name in page_map:
                # Merge blocks from both sources
                existing_blocks = page_map[candidate.page_name].blocks
                new_blocks = candidate.blocks
                merged_blocks = existing_blocks + [
                    b for b in new_blocks if b["id"] not in {eb["id"] for eb in existing_blocks}
                ]
                candidate.blocks = merged_blocks

            page_map[candidate.page_name] = candidate

        # Sort by similarity (highest first)
        candidates = sorted(
            page_map.values(),
            key=lambda c: c.similarity_score,
            reverse=True,
        )

        return candidates

    def _extract_blocks_from_outline(self, outline: LogseqOutline) -> list[dict]:
        """
        Extract block metadata from LogseqOutline for targeting.

        Args:
            outline: Parsed Logseq page outline

        Returns:
            List of block dicts with id, content, depth, parent_id
        """
        blocks = []

        def walk(block: LogseqBlock, depth: int = 0, parent_id: str | None = None):
            # Get hybrid ID (block_id property or content hash)
            block_hybrid_id = block.block_id or generate_content_hash(block)
            blocks.append({
                "id": block_hybrid_id,
                "content": "\n".join(block.content) if block.content else "",  # Include all lines
                "depth": depth,
                "parent_id": parent_id,
            })
            for child in block.children:
                walk(child, depth + 1, parent_id=block_hybrid_id)

        for root_block in outline.blocks:
            walk(root_block)

        return blocks

    def _update_progress(self, message: str) -> None:
        """
        Update progress display with current message.

        Args:
            message: Progress message to display
        """
        self.progress_messages.append(message)
        progress_display = self.query_one("#progress-display", Static)
        progress_display.update("\n".join(self.progress_messages))

    async def _proceed_to_phase3(self) -> None:
        """
        Navigate to Phase 3 (integration decisions).

        Pushes Phase3Screen onto the screen stack.
        """
        from logsqueak.tui.screens.phase3 import Phase3Screen
        await self.app.push_screen(Phase3Screen(self.state))

    def action_review_mode(self) -> None:
        """
        Enter review mode to manually filter candidates.

        This is part of User Story 4 (P3 - optional feature).
        For now, just set flag to prevent auto-proceed.
        Full implementation will be done in Phase 6 (US4 tasks).
        """
        self.review_mode_active = True
        logger.info("Review mode requested (full implementation pending US4)")

    async def on_unmount(self) -> None:
        """Cancel background retrieval task on unmount."""
        if self.retrieval_task and not self.retrieval_task.done():
            self.retrieval_task.cancel()
            try:
                await self.retrieval_task
            except asyncio.CancelledError:
                pass
