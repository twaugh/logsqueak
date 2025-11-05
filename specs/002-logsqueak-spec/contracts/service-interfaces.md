# Service Interfaces Contract

**Date**: 2025-11-05
**Feature**: 002-logsqueak-spec

## Overview

This document defines the Python service interfaces for the Logsqueak Interactive TUI application. All interfaces use async/await patterns with Pydantic models from `data-model.md` for type safety.

**Key Principles**:
- All LLM operations are async generators (streaming results)
- All file operations check modification times (concurrent modification detection)
- All services use Pydantic models for validation
- All errors are raised with descriptive messages

---

## Table of Contents

1. [LLMClient Interface](#llmclient-interface)
2. [PageIndexer Interface](#pageindexer-interface)
3. [RAGSearch Interface](#ragsearch-interface)
4. [FileMonitor Interface](#filemonitor-interface)
5. [Service Orchestration](#service-orchestration)

---

## LLMClient Interface

### Purpose
Unified client for streaming LLM API requests using httpx with NDJSON parsing.

### Class Definition

```python
from typing import AsyncIterator
from pydantic import BaseModel
import httpx
import structlog

logger = structlog.get_logger()


class LLMClient:
    """HTTP client for streaming LLM API requests."""

    def __init__(self, config: LLMConfig):
        """
        Initialize LLM client with configuration.

        Args:
            config: LLM configuration (endpoint, API key, model, etc.)
        """
        self.config = config
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=10.0,
                read=60.0,
                write=10.0,
                pool=10.0
            )
        )

    async def stream_ndjson(
        self,
        messages: list[dict],
        temperature: float = 0.5,
        request_id: str | None = None
    ) -> AsyncIterator[dict]:
        """
        Stream NDJSON chat completion from LLM API.

        Yields parsed JSON objects line-by-line from streaming response.
        Skips malformed JSON lines with warning log.

        Args:
            messages: Chat messages array (system + user prompts)
            temperature: Sampling temperature (0.0-1.0)
            request_id: Optional request ID for logging

        Yields:
            dict: Parsed JSON chunks from NDJSON stream

        Raises:
            httpx.HTTPStatusError: On 4xx/5xx HTTP errors
            httpx.ConnectError: On connection failures
            httpx.ReadTimeout: On read timeout during streaming
            httpx.ConnectTimeout: On connection timeout
        """
        import json
        import uuid

        if request_id is None:
            request_id = str(uuid.uuid4())

        # Build request body
        request_body = {
            "model": self.config.model,
            "messages": messages,
            "stream": True,
            "temperature": temperature
        }

        # Add Ollama-specific options if needed
        if self.config.num_ctx:
            request_body["options"] = {"num_ctx": self.config.num_ctx}

        headers = {"Authorization": f"Bearer {self.config.api_key}"}

        # Log request start
        logger.info(
            "llm_request_started",
            request_id=request_id,
            model=self.config.model,
            endpoint=str(self.config.endpoint),
            temperature=temperature,
            num_messages=len(messages)
        )

        chunk_count = 0
        try:
            async with self.client.stream(
                "POST",
                str(self.config.endpoint),
                json=request_body,
                headers=headers
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue  # Skip empty lines

                    try:
                        data = json.loads(line)
                        chunk_count += 1

                        logger.debug(
                            "llm_response_chunk",
                            request_id=request_id,
                            chunk_num=chunk_count,
                            data=data
                        )

                        yield data

                    except json.JSONDecodeError as e:
                        logger.warning(
                            "malformed_json_line",
                            request_id=request_id,
                            line=line[:100],  # Truncate for logging
                            error=str(e)
                        )
                        continue  # Skip bad line, process remaining stream

            logger.info(
                "llm_request_completed",
                request_id=request_id,
                total_chunks=chunk_count
            )

        except Exception as e:
            logger.error(
                "llm_request_failed",
                request_id=request_id,
                error_type=type(e).__name__,
                error_message=str(e),
                chunks_received=chunk_count
            )
            raise

    async def close(self) -> None:
        """Close HTTP client and release resources."""
        await self.client.aclose()


# Type-safe wrapper methods for each phase

async def classify_blocks(
    blocks: list[LogseqBlock],
    client: LLMClient,
    block_id_map: dict[str, str]  # LogseqBlock -> block_id mapping
) -> AsyncIterator[BlockState]:
    """
    Classify journal blocks as knowledge or activity.

    Streams BlockState updates as LLM identifies knowledge blocks.

    Args:
        blocks: List of LogseqBlock objects to classify
        client: LLM client for API requests
        block_id_map: Mapping of LogseqBlock to stable block IDs

    Yields:
        BlockState: Updated block states as classifications arrive

    Raises:
        httpx.HTTPError: On network/API errors
        pydantic.ValidationError: On malformed LLM response
    """
    from .models import KnowledgeClassificationChunk, BlockState

    # Format blocks for prompt
    formatted_blocks = "\n\n".join(
        f"Block ID: {block_id_map[block]}\n"
        f"Content: {block.get_full_content()}"
        for block in blocks
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI assistant that identifies lasting knowledge in journal entries. "
                "Classify each block as either 'knowledge' (lasting insight worth preserving) or "
                "'activity' (temporal event/task). For each block classified as knowledge, respond "
                "with a single-line JSON object containing: block_id (string), is_knowledge "
                "(boolean, always true for returned items), confidence (float 0-1), reason (string "
                "explaining why this is knowledge). Only return blocks that are knowledge - omit "
                "activity blocks from output. Output one JSON object per line (NDJSON format)."
            )
        },
        {
            "role": "user",
            "content": f"Analyze these journal blocks:\n\n{formatted_blocks}"
        }
    ]

    async for chunk_data in client.stream_ndjson(messages, temperature=0.3):
        # Validate and parse chunk
        chunk = KnowledgeClassificationChunk(**chunk_data)

        # Create BlockState
        yield BlockState(
            block_id=chunk.block_id,
            classification="knowledge",
            confidence=chunk.confidence,
            source="llm",
            llm_classification="knowledge",
            llm_confidence=chunk.confidence,
            reason=chunk.reason
        )


async def reword_content(
    edited_content: list[EditedContent],
    client: LLMClient
) -> AsyncIterator[tuple[str, str]]:
    """
    Generate reworded versions of knowledge blocks.

    Streams (block_id, reworded_content) tuples as LLM generates rewording.

    Args:
        edited_content: List of EditedContent to reword
        client: LLM client for API requests

    Yields:
        tuple[str, str]: (block_id, reworded_content)

    Raises:
        httpx.HTTPError: On network/API errors
        pydantic.ValidationError: On malformed LLM response
    """
    from .models import ContentRewordingChunk

    # Format blocks for prompt
    formatted_blocks = "\n\n".join(
        f"Block ID: {ec.block_id}\n"
        f"Original Content: {ec.original_content}"
        for ec in edited_content
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI assistant that transforms journal-style content into evergreen "
                "knowledge. Remove temporal context (dates, 'today', 'yesterday'), convert "
                "first-person to third-person or neutral, and make the content timeless. Preserve "
                "all technical details and insights. For each block, respond with a single-line "
                "JSON object containing: block_id (string), reworded_content (string). Output one "
                "JSON object per line (NDJSON format)."
            )
        },
        {
            "role": "user",
            "content": f"Reword these knowledge blocks:\n\n{formatted_blocks}"
        }
    ]

    async for chunk_data in client.stream_ndjson(messages, temperature=0.5):
        # Validate and parse chunk
        chunk = ContentRewordingChunk(**chunk_data)

        yield (chunk.block_id, chunk.reworded_content)


async def plan_integrations(
    edited_content: list[EditedContent],
    candidate_pages: dict[str, list[str]],  # block_id -> [page_name, ...]
    page_contents: dict[str, LogseqOutline],  # page_name -> LogseqOutline
    original_contexts: dict[str, str],  # block_id -> hierarchical context
    client: LLMClient
) -> AsyncIterator[IntegrationDecision]:
    """
    Generate integration decisions for knowledge blocks.

    Streams IntegrationDecision objects as LLM evaluates candidate pages.

    Args:
        edited_content: List of EditedContent to integrate
        candidate_pages: Mapping of block_id to candidate page names
        page_contents: Mapping of page_name to parsed LogseqOutline
        original_contexts: Mapping of block_id to original hierarchical context
        client: LLM client for API requests

    Yields:
        IntegrationDecision: Integration decisions as they arrive

    Raises:
        httpx.HTTPError: On network/API errors
        pydantic.ValidationError: On malformed LLM response
    """
    from .models import IntegrationDecisionChunk

    # Format knowledge blocks and candidate pages for prompt
    formatted_parts = []
    for ec in edited_content:
        block_candidates = candidate_pages.get(ec.block_id, [])
        if not block_candidates:
            continue

        part = f"Knowledge Block ID: {ec.block_id}\n"
        part += f"Content: {ec.current_content}\n"
        part += f"Original Context: {original_contexts.get(ec.block_id, '')}\n\n"
        part += "Candidate Pages:\n\n"

        for i, page_name in enumerate(block_candidates, 1):
            page_outline = page_contents.get(page_name)
            if not page_outline:
                continue

            part += f"{i}. Page: {page_name}\n"
            part += f"   Existing Structure:\n{page_outline.render()}\n\n"

        formatted_parts.append(part)

    formatted_content = "\n".join(formatted_parts)

    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI assistant that decides where to integrate knowledge into a "
                "knowledge base. For each (knowledge block, candidate page) pair, decide the best "
                "integration action. Actions: 'add_section' (create new top-level section), "
                "'add_under' (add as child under existing block), 'replace' (replace existing "
                "block content). Only return decisions for relevant integrations - omit irrelevant "
                "candidate pages. Return at most 2 decisions per (knowledge block, target page) "
                "pair. For each relevant integration, respond with a single-line JSON object "
                "containing: knowledge_block_id (string), target_page (string), action (string: "
                "add_section|add_under|replace), target_block_id (string or null), "
                "target_block_title (string or null), confidence (float 0-1), reasoning (string). "
                "Output one JSON object per line (NDJSON format)."
            )
        },
        {
            "role": "user",
            "content": f"Decide integration for these knowledge blocks and candidate pages:\n\n{formatted_content}"
        }
    ]

    async for chunk_data in client.stream_ndjson(messages, temperature=0.4):
        # Validate and parse chunk
        chunk = IntegrationDecisionChunk(**chunk_data)

        # Find corresponding EditedContent to get refined_text
        ec = next((e for e in edited_content if e.block_id == chunk.knowledge_block_id), None)
        if not ec:
            logger.warning(
                "integration_decision_orphaned",
                knowledge_block_id=chunk.knowledge_block_id,
                target_page=chunk.target_page
            )
            continue

        yield IntegrationDecision(
            knowledge_block_id=chunk.knowledge_block_id,
            target_page=chunk.target_page,
            action=chunk.action,
            target_block_id=chunk.target_block_id,
            target_block_title=chunk.target_block_title,
            confidence=chunk.confidence,
            refined_text=ec.current_content,  # From Phase 2 editing
            reasoning=chunk.reasoning,
            write_status="pending"
        )
```

---

## PageIndexer Interface

### Purpose
Build and maintain ChromaDB vector index for semantic search.

### Class Definition

```python
from pathlib import Path
from typing import Optional, Callable
import chromadb
from sentence_transformers import SentenceTransformer
from logseq_outline import LogseqOutline
from logseq_outline.context import generate_full_context
import structlog

logger = structlog.get_logger()


class PageIndexer:
    """Builds and maintains ChromaDB vector index for Logseq pages."""

    def __init__(
        self,
        graph_path: Path,
        db_path: Path,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize page indexer.

        Args:
            graph_path: Path to Logseq graph directory
            db_path: Path to ChromaDB persistent storage
            embedding_model: SentenceTransformer model name
        """
        self.graph_path = graph_path
        self.db_path = db_path
        self.encoder = SentenceTransformer(embedding_model)

        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=str(db_path))
        self.collection = self.chroma_client.get_or_create_collection(
            name="logsqueak_blocks",
            metadata={"hnsw:space": "cosine"}
        )

    async def build_index(
        self,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> None:
        """
        Build or update vector index for all pages in graph.

        Uses incremental indexing: only re-indexes modified pages.

        Args:
            progress_callback: Optional callback(current, total) for progress updates

        Raises:
            ValueError: If graph_path doesn't exist or contains no pages
            OSError: On file I/O errors
        """
        pages_dir = self.graph_path / "pages"
        if not pages_dir.exists():
            raise ValueError(f"Pages directory not found: {pages_dir}")

        # Get all .md files
        page_files = list(pages_dir.glob("*.md"))
        if not page_files:
            raise ValueError(f"No pages found in {pages_dir}")

        logger.info(
            "page_indexing_started",
            graph_path=str(self.graph_path),
            total_pages=len(page_files)
        )

        for idx, page_file in enumerate(page_files, 1):
            page_name = page_file.stem.replace("___", "/")

            # Check if page needs re-indexing
            if self._is_page_indexed(page_name, page_file):
                logger.debug("page_index_skip", page_name=page_name, reason="not_modified")
                if progress_callback:
                    progress_callback(idx, len(page_files))
                continue

            # Parse page
            outline = LogseqOutline.parse(page_file.read_text())

            # Index all blocks
            self._index_page_blocks(page_name, outline, page_file.stat().st_mtime)

            if progress_callback:
                progress_callback(idx, len(page_files))

            logger.debug("page_indexed", page_name=page_name, block_count=len(outline.blocks))

        logger.info("page_indexing_completed", total_pages=len(page_files))

    def _is_page_indexed(self, page_name: str, page_file: Path) -> bool:
        """Check if page is already indexed and unmodified."""
        # Query collection for page metadata
        results = self.collection.get(
            where={"page_name": page_name},
            limit=1
        )

        if not results["ids"]:
            return False

        # Check modification time
        stored_mtime = results["metadatas"][0].get("mtime")
        current_mtime = page_file.stat().st_mtime

        return stored_mtime is not None and current_mtime <= stored_mtime

    def _index_page_blocks(
        self,
        page_name: str,
        outline: LogseqOutline,
        mtime: float
    ) -> None:
        """Index all blocks from a page."""
        documents = []
        embeddings = []
        ids = []
        metadatas = []

        def process_block(block, parents):
            # Generate full hierarchical context
            context = generate_full_context(block, parents)

            # Generate embedding
            embedding = self.encoder.encode(context, convert_to_numpy=True)

            # Store
            block_id = block.block_id or self._generate_block_id(block, parents)
            documents.append(context)
            embeddings.append(embedding.tolist())
            ids.append(f"{page_name}::{block_id}")
            metadatas.append({
                "page_name": page_name,
                "block_id": block_id,
                "mtime": mtime
            })

            # Process children
            for child in block.children:
                process_block(child, parents + [block])

        # Process all root blocks
        for block in outline.blocks:
            process_block(block, [])

        # Add to collection (upsert)
        if documents:
            self.collection.upsert(
                documents=documents,
                embeddings=embeddings,
                ids=ids,
                metadatas=metadatas
            )

    def _generate_block_id(self, block, parents) -> str:
        """Generate content-based block ID."""
        from logseq_outline.context import generate_content_hash
        return generate_content_hash(block, parents)

    async def close(self) -> None:
        """Close ChromaDB client."""
        # ChromaDB client closes automatically
        pass
```

---

## RAGSearch Interface

### Purpose
Find candidate pages for knowledge blocks using semantic search.

### Class Definition

```python
from pathlib import Path
from typing import Optional
import chromadb
from sentence_transformers import SentenceTransformer
import structlog

logger = structlog.get_logger()


class RAGSearch:
    """Semantic search for candidate pages using ChromaDB."""

    def __init__(
        self,
        db_path: Path,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize RAG search.

        Args:
            db_path: Path to ChromaDB persistent storage
            embedding_model: SentenceTransformer model name (must match indexer)
        """
        self.db_path = db_path
        self.encoder = SentenceTransformer(embedding_model)

        # Load ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=str(db_path))
        self.collection = self.chroma_client.get_collection("logsqueak_blocks")

    async def find_candidates(
        self,
        edited_content: list[EditedContent],
        original_contexts: dict[str, str],  # block_id -> hierarchical context
        top_k: int = 10
    ) -> dict[str, list[str]]:
        """
        Find candidate pages for each knowledge block.

        Args:
            edited_content: List of EditedContent to find candidates for
            original_contexts: Mapping of block_id to original hierarchical context
            top_k: Number of candidate pages per block

        Returns:
            dict mapping block_id to list of candidate page names (ranked by relevance)

        Raises:
            ValueError: If ChromaDB collection doesn't exist
        """
        results = {}

        for ec in edited_content:
            context = original_contexts.get(ec.block_id, ec.original_content)

            # Generate embedding
            embedding = self.encoder.encode(context, convert_to_numpy=True)

            # Query ChromaDB
            query_results = self.collection.query(
                query_embeddings=[embedding.tolist()],
                n_results=top_k * 3  # Over-fetch to group by page
            )

            # Extract and rank pages
            candidate_pages = self._rank_pages(
                query_results,
                context,
                top_k
            )

            results[ec.block_id] = candidate_pages

            logger.debug(
                "rag_search_candidates",
                block_id=ec.block_id,
                num_candidates=len(candidate_pages)
            )

        return results

    def _rank_pages(
        self,
        query_results: dict,
        context: str,
        top_k: int
    ) -> list[str]:
        """
        Rank pages by total similarity score.

        Also boost pages mentioned in explicit links (e.g., [[Page Name]]).
        """
        import re
        from collections import defaultdict

        # Extract explicit page links from context
        explicit_links = set(re.findall(r'\[\[([^\]]+)\]\]', context))

        # Group results by page
        page_scores = defaultdict(float)
        for metadata, distance in zip(
            query_results["metadatas"][0],
            query_results["distances"][0]
        ):
            page_name = metadata["page_name"]
            similarity = 1.0 - distance  # Convert distance to similarity

            # Boost pages mentioned in explicit links
            if page_name in explicit_links or page_name.replace("/", "___") in explicit_links:
                similarity *= 1.5

            page_scores[page_name] += similarity

        # Sort by total score
        ranked_pages = sorted(
            page_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [page for page, score in ranked_pages[:top_k]]

    async def close(self) -> None:
        """Close ChromaDB client."""
        # ChromaDB client closes automatically
        pass
```

---

## FileMonitor Interface

### Purpose
Track file modification times and detect concurrent modifications.

### Class Definition

```python
from pathlib import Path
from typing import Dict
import structlog

logger = structlog.get_logger()


class FileMonitor:
    """Track file modification times for concurrent modification detection."""

    def __init__(self):
        """Initialize file monitor."""
        self._mtimes: Dict[Path, float] = {}

    def record(self, path: Path) -> None:
        """
        Record current modification time for file.

        Args:
            path: File path to track
        """
        self._mtimes[path] = path.stat().st_mtime

    def is_modified(self, path: Path) -> bool:
        """
        Check if file modified since last record.

        Args:
            path: File path to check

        Returns:
            bool: True if file modified or not tracked, False otherwise
        """
        if path not in self._mtimes:
            return True

        current_mtime = path.stat().st_mtime
        return current_mtime > self._mtimes[path]

    def refresh(self, path: Path) -> None:
        """
        Update recorded mtime after successful reload.

        Args:
            path: File path to refresh
        """
        self._mtimes[path] = path.stat().st_mtime

    def check_and_reload(
        self,
        path: Path,
        reload_func: callable
    ) -> bool:
        """
        Check if file modified, reload if necessary.

        Args:
            path: File path to check
            reload_func: Function to call if file modified (no args, no return)

        Returns:
            bool: True if file was reloaded, False otherwise
        """
        if self.is_modified(path):
            logger.info("file_modified_externally", path=str(path))
            reload_func()
            self.refresh(path)
            return True
        return False
```

---

## Service Orchestration

### Example: Phase 1 Background Tasks

```python
from textual.worker import Worker
from asyncio import Queue


class Phase1Screen(Screen):
    """Phase 1: Block Selection screen."""

    def on_mount(self) -> None:
        """Start background tasks on screen mount."""
        # Start LLM classification
        self.run_worker(
            self._classify_blocks_worker(),
            name="llm_classification"
        )

        # Start page indexing
        self.run_worker(
            self._build_index_worker(),
            name="page_indexing"
        )

    async def _classify_blocks_worker(self) -> None:
        """Worker: Stream LLM classifications."""
        try:
            async for block_state in classify_blocks(
                self.journal_blocks,
                self.llm_client,
                self.block_id_map
            ):
                # Update UI state
                self.update_block_state(block_state)

        except Exception as e:
            logger.error("classification_worker_failed", error=str(e))
            self.show_error(f"Classification failed: {e}")

    async def _build_index_worker(self) -> None:
        """Worker: Build page index with progress updates."""
        try:
            def progress_callback(current: int, total: int):
                self.update_progress("page_indexing", current, total)

            await self.page_indexer.build_index(progress_callback)
            self.mark_task_complete("page_indexing")

        except Exception as e:
            logger.error("indexing_worker_failed", error=str(e))
            self.show_error(f"Indexing failed: {e}")
```

---

## Summary

These service interfaces provide:

- **Type-safe async operations** using Pydantic models
- **Streaming-first design** for LLM interactions
- **Concurrent modification detection** for file operations
- **Incremental indexing** for efficient page index updates
- **Semantic search** with explicit link boosting
- **Structured logging** for debugging and audit

All interfaces align with constitution principles: simple (file-based), transparent (structured logging), non-destructive (modification tracking).
