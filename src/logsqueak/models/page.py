"""Target page and page index models for knowledge integration."""

import logging
import pickle
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from logsqueak.logseq.parser import LogseqBlock, LogseqOutline
    import numpy as np
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class ConventionType(Enum):
    """Page organizational convention."""

    PLAIN_BULLETS = "plain"  # Uses "- Section"
    HEADING_BULLETS = "heading"  # Uses "- ## Section"
    MIXED = "mixed"  # Mix of both styles


@dataclass
class TargetPage:
    """An existing Logseq page where knowledge will be integrated.

    Attributes:
        name: Page name (e.g., "Project X")
        file_path: Absolute path to page markdown file
        outline: Parsed outline structure
        organizational_convention: Detected page style
    """

    name: str
    file_path: Path
    outline: "LogseqOutline"
    organizational_convention: ConventionType

    @classmethod
    def load(cls, graph_path: Path, page_name: str) -> Optional["TargetPage"]:
        """Load target page from Logseq graph.

        Args:
            graph_path: Path to Logseq graph directory
            page_name: Name of the page to load

        Returns:
            TargetPage instance if found, None otherwise (FR-009)
        """
        from logsqueak.logseq.parser import LogseqOutline

        # Logseq stores pages as "Page Name.md"
        file_path = graph_path / "pages" / f"{page_name}.md"

        if not file_path.exists():
            logger.warning(f"Target page not found: {page_name}")
            return None

        try:
            outline = LogseqOutline.parse(file_path.read_text())
            convention = _detect_convention(outline)

            return cls(
                name=page_name,
                file_path=file_path,
                outline=outline,
                organizational_convention=convention,
            )
        except Exception as e:
            logger.error(f"Failed to load page {page_name}: {e}")
            return None

    def find_section(self, section_name: str) -> Optional["LogseqBlock"]:
        """Find section by heading text.

        Args:
            section_name: Name of section to find

        Returns:
            LogseqBlock if found, None otherwise
        """
        return self.outline.find_heading(section_name)

    def has_duplicate(self, content: str) -> bool:
        """Check if content already exists on page (FR-017).

        Simple content match - can be enhanced with semantic similarity later.

        Args:
            content: Content to check for duplicates

        Returns:
            True if duplicate found
        """
        return content in self.outline.render()


@dataclass
class PageIndex:
    """Semantic index of all pages for RAG-based page matching.

    Uses per-page caching to avoid re-embedding unchanged pages.

    Attributes:
        pages: All pages in the graph
        embeddings: Embedding matrix (shape: [num_pages, embedding_dim])
        model: Embedding model instance
        page_texts: Text used for embeddings (name + preview)
        cache_dir: Directory for per-page embedding cache
    """

    pages: list["TargetPage"]
    embeddings: "np.ndarray"
    model: "SentenceTransformer"
    page_texts: list[str]
    cache_dir: Path

    @classmethod
    def build(cls, graph_path: Path, cache_dir: Optional[Path] = None) -> "PageIndex":
        """Build index from all pages in graph.

        Each page is represented as: "{page_name} {first_1000_chars}"

        Uses per-page caching:
        - Cache file: ~/.cache/logsqueak/embeddings/{page_name}.pkl
        - Contains: {'embedding': ndarray, 'mtime': float, 'text': str}
        - Invalidate: if page file mtime > cached mtime

        Args:
            graph_path: Path to Logseq graph
            cache_dir: Cache directory (default: ~/.cache/logsqueak/embeddings/)

        Returns:
            Built PageIndex
        """
        # Import heavy dependencies only when needed
        import numpy as np
        from sentence_transformers import SentenceTransformer

        # Load embedding model (80MB, cached by sentence-transformers)
        logger.info("Loading embedding model...")
        model = SentenceTransformer("all-MiniLM-L6-v2")

        # Setup cache directory
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "logsqueak" / "embeddings"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Scan all pages in graph
        pages_dir = graph_path / "pages"
        if not pages_dir.exists():
            logger.warning(f"Pages directory not found: {pages_dir}")
            return cls(
                pages=[],
                embeddings=np.array([]),
                model=model,
                page_texts=[],
                cache_dir=cache_dir,
            )

        pages = []
        page_texts = []
        embeddings_list = []
        cached_count = 0
        computed_count = 0

        page_files = list(pages_dir.glob("*.md"))
        logger.info(f"Building page index for {len(page_files)} pages...")

        for page_file in page_files:
            page = TargetPage.load(graph_path, page_file.stem)
            if not page:
                continue

            pages.append(page)

            # Combine page name + content preview for embedding
            preview = page.outline.render()[:1000]
            page_text = f"{page.name} {preview}"
            page_texts.append(page_text)

            # Try to load from cache
            cached_embedding = cls._load_cached_embedding(cache_dir, page_file, page_text)

            if cached_embedding is not None:
                embeddings_list.append(cached_embedding)
                cached_count += 1
            else:
                # Cache miss - compute and save
                embedding = model.encode(page_text, convert_to_numpy=True)
                cls._save_embedding(cache_dir, page_file, embedding, page_text)
                embeddings_list.append(embedding)
                computed_count += 1

        # Stack into matrix
        if embeddings_list:
            embeddings = np.vstack(embeddings_list)
        else:
            embeddings = np.array([])

        logger.info(
            f"✓ Indexed {len(pages)} pages ({cached_count} cached, {computed_count} computed)"
        )

        index = cls(
            pages=pages,
            embeddings=embeddings,
            model=model,
            page_texts=page_texts,
            cache_dir=cache_dir,
        )
        # Store stats as attributes for CLI reporting
        index.cached_count = cached_count  # type: ignore
        index.computed_count = computed_count  # type: ignore
        return index

    @staticmethod
    def _load_cached_embedding(
        cache_dir: Path, page_file: Path, page_text: str
    ) -> Optional["np.ndarray"]:
        """Load cached embedding if valid (mtime check).

        Args:
            cache_dir: Cache directory
            page_file: Page file path
            page_text: Current page text

        Returns:
            Cached embedding if valid, None otherwise
        """
        cache_file = cache_dir / f"{page_file.stem}.pkl"
        if not cache_file.exists():
            return None

        try:
            with cache_file.open("rb") as f:
                cached = pickle.load(f)

            # Validate: mtime must match and text must match
            if (
                cached["mtime"] >= page_file.stat().st_mtime
                and cached["text"] == page_text
            ):
                return cached["embedding"]
        except (pickle.PickleError, KeyError, OSError):
            pass

        return None

    @staticmethod
    def _save_embedding(
        cache_dir: Path, page_file: Path, embedding: "np.ndarray", page_text: str
    ) -> None:
        """Save embedding to per-page cache file.

        Args:
            cache_dir: Cache directory
            page_file: Page file path
            embedding: Embedding to cache
            page_text: Page text used for embedding
        """
        cache_file = cache_dir / f"{page_file.stem}.pkl"
        try:
            with cache_file.open("wb") as f:
                pickle.dump(
                    {
                        "embedding": embedding,
                        "mtime": page_file.stat().st_mtime,
                        "text": page_text,
                    },
                    f,
                )
        except (pickle.PickleError, OSError) as e:
            # Cache write failure is non-fatal (just slower next time)
            logger.warning(f"Failed to cache embedding for {page_file.stem}: {e}")

    def find_similar(
        self, text: str, top_k: int = 5
    ) -> list[tuple["TargetPage", float]]:
        """Find top-K most semantically similar pages.

        Args:
            text: Knowledge block content to match
            top_k: Number of candidates to return

        Returns:
            List of (page, similarity_score) tuples, sorted by score descending
        """
        import numpy as np

        if len(self.pages) == 0:
            return []

        # Embed the query text
        query_embedding = self.model.encode(text, convert_to_numpy=True)

        # Compute cosine similarity with all pages
        # Normalize embeddings for cosine similarity
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        page_norms = self.embeddings / np.linalg.norm(
            self.embeddings, axis=1, keepdims=True
        )

        similarities = np.dot(page_norms, query_norm)

        # Get top-K indices
        actual_k = min(top_k, len(self.pages))
        top_k_indices = np.argsort(similarities)[-actual_k:][::-1]

        # Return (page, score) tuples
        return [(self.pages[i], float(similarities[i])) for i in top_k_indices]

    def refresh(self, page_name: str) -> None:
        """Update embedding for a single page after modification.

        Re-embeds the page and updates cache. Called after we modify a page
        to keep the index current for subsequent extractions in same session.

        Args:
            page_name: Name of page to refresh
        """
        # Find page index
        for i, page in enumerate(self.pages):
            if page.name == page_name:
                # Re-embed this page
                preview = page.outline.render()[:1000]
                page_text = f"{page.name} {preview}"
                self.page_texts[i] = page_text

                # Compute new embedding
                embedding = self.model.encode(page_text, convert_to_numpy=True)
                self.embeddings[i] = embedding

                # Update cache
                page_file = page.file_path
                self._save_embedding(self.cache_dir, page_file, embedding, page_text)
                break


def _detect_convention(outline: "LogseqOutline") -> ConventionType:
    """Detect the organizational convention used in a page.

    Args:
        outline: Page outline structure

    Returns:
        Detected convention type
    """
    # Simple heuristic: check if any blocks contain markdown headings
    has_heading = False
    has_plain = False

    def check_blocks(blocks: list["LogseqBlock"]) -> None:
        nonlocal has_heading, has_plain
        for block in blocks:
            if "##" in block.content:
                has_heading = True
            else:
                has_plain = True
            check_blocks(block.children)

    check_blocks(outline.blocks)

    if has_heading and has_plain:
        return ConventionType.MIXED
    elif has_heading:
        return ConventionType.HEADING_BULLETS
    else:
        return ConventionType.PLAIN_BULLETS
