# Internal Service Contracts

**Date**: 2025-11-04

**Feature**: 002-logsqueak-spec

## Overview

This document defines the interfaces for internal services used by the TUI application. All services are Python classes/functions, not REST APIs.

---

## Knowledge Classification Service

### Purpose
Classify journal blocks as knowledge vs activity using LLM.

### Interface

```python
from typing import AsyncIterator
from logseq_outline import LogseqBlock

async def classify_blocks(
    blocks: list[LogseqBlock],
    llm_client: LLMClient,
    config: LLMConfig
) -> AsyncIterator[BlockState]:
    """
    Classify journal blocks as knowledge or activity.

    Yields BlockState updates as LLM streaming response arrives.

    Args:
        blocks: List of blocks to classify
        llm_client: LLM client for API requests
        config: LLM configuration (endpoint, model, etc.)

    Yields:
        BlockState: Updated block states as classifications arrive

    Raises:
        httpx.HTTPError: On network/API errors
        ValueError: On malformed LLM response
    """
    ...

```

### Behavior

1. Format blocks into prompt (include block ID and content)
2. Send streaming request to LLM
3. Parse each chunk into `KnowledgeClassificationChunk`
4. Create/update `BlockState` and yield
5. Log all requests and chunks

---

## Content Rewriting Service

### Purpose
Generate reworded versions of knowledge blocks that remove temporal context.

### Interface

```python
from typing import AsyncIterator

async def rewrite_content(
    edited_content: list[EditedContent],
    llm_client: LLMClient,
    config: LLMConfig
) -> AsyncIterator[tuple[str, str]]:
    """
    Generate reworded versions of knowledge blocks.

    Yields (block_id, reworded_content) tuples as LLM streaming response arrives.

    Args:
        edited_content: List of EditedContent to rewrite
        llm_client: LLM client for API requests
        config: LLM configuration

    Yields:
        tuple[str, str]: (block_id, reworded_content)

    Raises:
        httpx.HTTPError: On network/API errors
        ValueError: On malformed LLM response
    """
    ...

```

### Behavior

1. Format blocks into prompt (include block ID and original content)
2. Send streaming request to LLM
3. Parse each chunk into `ContentRewordingChunk`
4. Yield (block_id, reworded_content) tuples
5. Log all requests and chunks

---

## Integration Planning Service

### Purpose
Generate integration decisions for where to place knowledge blocks in target pages.

### Interface

```python
from typing import AsyncIterator

async def plan_integrations(
    edited_content: list[EditedContent],
    candidate_pages: dict[str, list[str]],  # block_id -> [page_name, ...]
    page_contents: dict[str, LogseqOutline],  # page_name -> LogseqOutline
    llm_client: LLMClient,
    config: LLMConfig
) -> AsyncIterator[IntegrationDecision]:
    """
    Generate integration decisions for knowledge blocks.

    Yields IntegrationDecision objects as LLM streaming response arrives.

    Args:
        edited_content: List of EditedContent to integrate
        candidate_pages: Mapping of block_id to candidate page names
        page_contents: Mapping of page_name to parsed LogseqOutline
        llm_client: LLM client for API requests
        config: LLM configuration

    Yields:
        IntegrationDecision: Integration decisions as they arrive

    Raises:
        httpx.HTTPError: On network/API errors
        ValueError: On malformed LLM response
    """
    ...

```

### Behavior

1. For each (block, candidate_page) pair, format into prompt with:

   - Block ID and refined content
   - Original context (parent blocks, explicit page links)
   - Candidate page structure with block IDs

2. Send streaming request to LLM
3. Parse each chunk into `IntegrationDecisionChunk`
4. Create `IntegrationDecision` and yield
5. Log all requests and chunks

---

## Page Indexing Service

### Purpose
Build vector index of Logseq pages for semantic search.

### Interface

```python
from pathlib import Path
from typing import Optional

async def build_page_index(
    graph_path: Path,
    db_path: Path,
    progress_callback: Optional[callable] = None
) -> None:
    """
    Build or update ChromaDB vector index for Logseq pages.

    Args:
        graph_path: Path to Logseq graph directory
        db_path: Path to ChromaDB persistent storage
        progress_callback: Optional callback(current, total) for progress updates

    Raises:
        ValueError: If graph_path doesn't exist or contains no pages
        OSError: On file I/O errors
    """
    ...

```

### Behavior

1. Scan `{graph_path}/pages/` for `*.md` files
2. For each page:

   - Parse with LogseqOutline
   - For each block:

     - Generate full hierarchical context
     - Generate embedding using sentence-transformers
     - Store in ChromaDB with metadata: page_name, block_id, hierarchy, mtime

3. Call progress_callback after each page (if provided)
4. Persist ChromaDB collection to db_path

### Incremental Update

- Check file mtimes against stored metadata
- Re-index pages where mtime differs from stored metadata (accounts for git reverts/changes)
- Delete entries for removed pages

**Rationale**: Comparing mtimes with `!=` instead of `>` handles Logseq graphs stored in git, where file mtimes may decrease when reverting to earlier commits.

---

## RAG Search Service

### Purpose
Find candidate pages for knowledge blocks using semantic search.

### Interface

```python
from pathlib import Path

async def find_candidate_pages(
    edited_content: list[EditedContent],
    db_path: Path,
    top_k: int = 10
) -> dict[str, list[str]]:
    """
    Find candidate pages for each knowledge block using semantic search.

    Args:
        edited_content: List of EditedContent to find candidates for
        db_path: Path to ChromaDB persistent storage
        top_k: Number of candidate pages per block

    Returns:
        dict mapping block_id to list of candidate page names (ranked by relevance)

    Raises:
        ValueError: If ChromaDB index doesn't exist
    """
    ...

```

### Behavior

1. Load ChromaDB collection from db_path
2. For each EditedContent:

   - Extract original hierarchical context (includes explicit page links)
   - Generate embedding for context
   - Query ChromaDB for top_k similar blocks
   - Group results by page, rank by total similarity
   - Boost pages mentioned in explicit links (e.g., [[Page Name]])

3. Return dict of block_id -> [page_name, ...]

---

## File Operations Service

### Purpose
Read and write Logseq files with concurrent modification detection.

### Interface

```python
from pathlib import Path
from logseq_outline import LogseqOutline, LogseqBlock

class FileOperations:
    """Handles Logseq file I/O with modification tracking."""

    def __init__(self, graph_path: Path):
        self.graph_path = graph_path
        self._mtimes: dict[Path, float] = {}

    def read_journal(self, date: str) -> LogseqOutline:
        """
        Read journal entry for given date.

        Args:
            date: Date string in YYYY-MM-DD format

        Returns:
            LogseqOutline: Parsed journal entry

        Raises:
            FileNotFoundError: If journal doesn't exist
        """
        path = self.graph_path / "journals" / f"{date}.md"
        outline = LogseqOutline.parse(path.read_text())
        self._mtimes[path] = path.stat().st_mtime
        return outline

    def read_page(self, page_name: str) -> LogseqOutline:
        """
        Read page by name.

        Args:
            page_name: Page name (e.g., 'Python', 'Projects/Logsqueak')

        Returns:
            LogseqOutline: Parsed page

        Raises:
            FileNotFoundError: If page doesn't exist
        """
        # Convert hierarchical page names (Projects/Logsqueak -> Projects___Logsqueak.md)
        filename = page_name.replace("/", "___") + ".md"
        path = self.graph_path / "pages" / filename
        outline = LogseqOutline.parse(path.read_text())
        self._mtimes[path] = path.stat().st_mtime
        return outline

    def write_integration(
        self,
        decision: IntegrationDecision,
        journal_date: str
    ) -> None:
        """
        Write integration to target page and update journal provenance.

        This is an ATOMIC operation: both page write and journal update must succeed.

        Args:
            decision: Integration decision to execute
            journal_date: Journal date for provenance link

        Raises:
            FileModifiedError: If target file modified since last read
            ValueError: If target block doesn't exist (validation failed)
            OSError: On file I/O errors
        """
        # 1. Check page modification time
        page_path = self._get_page_path(decision.target_page)
        if self._is_modified(page_path):
            # Reload and re-validate
            page_outline = self.read_page(decision.target_page)
            self._validate_decision(decision, page_outline)

        # 2. Write to target page
        page_outline = self.read_page(decision.target_page)
        new_block_id = self._apply_integration(decision, page_outline)
        page_path.write_text(page_outline.render())
        self._mtimes[page_path] = page_path.stat().st_mtime

        # 3. Update journal provenance (ATOMIC with page write)
        journal_path = self.graph_path / "journals" / f"{journal_date}.md"
        if self._is_modified(journal_path):
            journal_outline = self.read_journal(journal_date)
        else:
            journal_outline = self.read_journal(journal_date)

        self._add_provenance(
            journal_outline,
            decision.knowledge_block_id,
            decision.target_page,
            new_block_id
        )
        journal_path.write_text(journal_outline.render())
        self._mtimes[journal_path] = journal_path.stat().st_mtime

    def _is_modified(self, path: Path) -> bool:
        """Check if file modified since last read."""
        if path not in self._mtimes:
            return True
        return path.stat().st_mtime > self._mtimes[path]

    def _validate_decision(
        self,
        decision: IntegrationDecision,
        page_outline: LogseqOutline
    ) -> None:
        """
        Validate that decision can still be applied to page.

        Raises:
            ValueError: If target block doesn't exist or structure changed
        """
        if decision.action == "add_under" or decision.action == "replace":
            block, _ = page_outline.find_block_by_id(decision.target_block_id)
            if block is None:
                raise ValueError(
                    f"Target block {decision.target_block_id} not found in {decision.target_page}"
                )

    def _apply_integration(
        self,
        decision: IntegrationDecision,
        page_outline: LogseqOutline
    ) -> str:
        """
        Apply integration decision to page outline.

        Returns:
            str: UUID of newly created block (for provenance)
        """
        import uuid
        new_block_id = str(uuid.uuid4())

        if decision.action == "add_section":
            # Add as new root-level block
            new_block = page_outline.blocks.append(
                LogseqBlock(
                    content=[decision.refined_text, f"id:: {new_block_id}"],
                    indent_level=0
                )
            )

        elif decision.action == "add_under":
            # Add as child of target block
            target_block, _ = page_outline.find_block_by_id(decision.target_block_id)
            target_block.add_child(decision.refined_text)
            # Add id:: property to the new child
            new_child = target_block.children[-1]
            new_child.set_property("id", new_block_id)

        elif decision.action == "replace":
            # Replace target block content (preserve children and ID)
            target_block, _ = page_outline.find_block_by_id(decision.target_block_id)
            target_block.content = [decision.refined_text]
            # Keep existing id:: if present, otherwise add new one
            if target_block.block_id is None:
                target_block.set_property("id", new_block_id)
            else:
                new_block_id = target_block.block_id

        return new_block_id

    def _add_provenance(
        self,
        journal_outline: LogseqOutline,
        knowledge_block_id: str,
        target_page: str,
        target_block_id: str
    ) -> None:
        """
        Add processed:: property to journal block.

        Format: processed:: [[Page Name]]((uuid)), [[Other Page]]((uuid2))
        """
        block, _ = journal_outline.find_block_by_id(knowledge_block_id)
        if block is None:
            raise ValueError(f"Knowledge block {knowledge_block_id} not found in journal")

        # Convert hierarchical page names (Projects___Logsqueak -> Projects/Logsqueak)
        display_name = target_page.replace("___", "/")
        provenance_link = f"[[{display_name}]](({target_block_id}))"

        # Get existing processed:: value or empty string
        existing = block.get_property("processed") or ""

        # Append new link (comma-separated)
        if existing:
            new_value = f"{existing}, {provenance_link}"
        else:
            new_value = provenance_link

        block.set_property("processed", new_value)

```

---

## LLM Client Service

### Purpose
Unified client for streaming LLM API requests.

### Interface

```python
from typing import AsyncIterator
import httpx

class LLMClient:
    """HTTP client for streaming LLM API requests."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=10.0,
                read=60.0,
                write=10.0,
                pool=10.0
            )
        )

    async def stream_chat_completion(
        self,
        messages: list[dict],
        temperature: float = 0.5
    ) -> AsyncIterator[dict]:
        """
        Stream chat completion from LLM API.

        Yields:
            dict: Parsed JSON chunks from streaming response

        Raises:
            httpx.HTTPError: On network/API errors
            ValueError: On malformed JSON in response
        """
        request_body = {
            "model": self.config.model,
            "messages": messages,
            "stream": True,
            "temperature": temperature
        }

        # Add Ollama-specific options if needed
        if "ollama" in self.config.endpoint.lower():
            request_body["options"] = {"num_ctx": self.config.num_ctx}

        headers = {"Authorization": f"Bearer {self.config.api_key}"}

        async with self.client.stream(
            "POST",
            self.config.endpoint,
            json=request_body,
            headers=headers
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if not line or line.startswith(":"):
                    continue

                # Handle OpenAI-compatible SSE format
                if line.startswith("data: "):
                    line = line[6:]  # Strip "data: " prefix

                if line == "[DONE]":
                    break

                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(
                        "malformed_json_chunk",
                        line=line,
                        error=str(e)
                    )
                    continue

    async def close(self) -> None:
        """Close HTTP client."""
        await self.client.aclose()

```

---

## Logging Service

### Purpose
Structured logging for debugging and audit.

### Interface

```python
import structlog

# Initialize logger
logger = structlog.get_logger()

# Usage examples:

# Log LLM request
logger.info(
    "llm_request_started",
    request_id="uuid",
    phase="classification",
    model="gpt-4-turbo-preview",
    num_blocks=15
)

# Log user action
logger.info(
    "user_action",
    screen="phase1_selection",
    action="toggle_selection",
    block_id="abc123"
)

# Log error
logger.error(
    "write_operation_failed",
    decision_id="uuid",
    target_page="Python/Concurrency",
    error="Target block not found"
)

```

### Configuration

```python
import structlog
from pathlib import Path

def configure_logging(log_dir: Path) -> None:
    """Configure structlog with JSON output to file."""
    log_file = log_dir / "logsqueak.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=open(log_file, "a")),
        cache_logger_on_first_use=True,
    )

```

---

## Summary

These service contracts provide:

- **Clear interfaces** for each major function
- **Async-first** design for responsive UI
- **Error handling** specifications
- **Streaming support** for LLM responses
- **File safety** via modification tracking
- **Atomic operations** for journal provenance

All services follow constitution principles: simple (file-based I/O, no ORM), transparent (structured logging), non-destructive (atomic journal updates with provenance).
