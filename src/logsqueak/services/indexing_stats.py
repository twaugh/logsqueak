"""Indexing statistics cache for time-based progress estimation.

This module manages historical timing data for page indexing operations,
enabling time-based progress feedback in the TUI and CLI.
"""

import json
from pathlib import Path
from typing import Optional
import structlog

logger = structlog.get_logger()


class IndexingStats:
    """Manages historical timing data for page indexing operations."""

    def __init__(self, graph_hash: str, cache_dir: Optional[Path] = None):
        """
        Initialize indexing stats manager.

        Args:
            graph_hash: Unique hash for the graph (from generate_graph_db_name)
            cache_dir: Optional cache directory. Defaults to ~/.cache/logsqueak
        """
        self.graph_hash = graph_hash

        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "logsqueak"

        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Stats file per graph: indexing_stats_{graph_hash}.json
        self.stats_file = self.cache_dir / f"indexing_stats_{graph_hash}.json"

        logger.debug(
            "indexing_stats_initialized",
            graph_hash=graph_hash,
            stats_file=str(self.stats_file)
        )

    def load(self) -> Optional[dict]:
        """
        Load historical timing statistics from cache.

        Returns:
            Dictionary with timing data, or None if no historical data exists

        Example:
            {
                "graph_hash": "abc123...",
                "last_indexing_duration_seconds": 45.2,
                "last_page_count": 150
            }
        """
        if not self.stats_file.exists():
            logger.debug("indexing_stats_not_found", stats_file=str(self.stats_file))
            return None

        try:
            with self.stats_file.open("r") as f:
                stats = json.load(f)

            # Validate graph_hash matches
            if stats.get("graph_hash") != self.graph_hash:
                logger.warning(
                    "indexing_stats_hash_mismatch",
                    expected=self.graph_hash,
                    found=stats.get("graph_hash")
                )
                return None

            logger.debug(
                "indexing_stats_loaded",
                duration=stats.get("last_indexing_duration_seconds"),
                page_count=stats.get("last_page_count")
            )

            return stats

        except (json.JSONDecodeError, OSError) as e:
            logger.warning("indexing_stats_load_failed", error=str(e))
            return None

    def save(self, duration_seconds: float, page_count: int) -> None:
        """
        Save timing statistics to cache.

        Args:
            duration_seconds: Total duration of indexing operation
            page_count: Number of pages indexed
        """
        stats = {
            "graph_hash": self.graph_hash,
            "last_indexing_duration_seconds": duration_seconds,
            "last_page_count": page_count
        }

        try:
            with self.stats_file.open("w") as f:
                json.dump(stats, f, indent=2)

            logger.debug(
                "indexing_stats_saved",
                duration=duration_seconds,
                page_count=page_count,
                stats_file=str(self.stats_file)
            )

        except OSError as e:
            logger.warning("indexing_stats_save_failed", error=str(e))

    def get_estimated_duration(self, current_page_count: Optional[int] = None) -> Optional[float]:
        """
        Get estimated duration for next indexing operation.

        If current_page_count is provided, scales the estimate proportionally
        based on the ratio of current to previous page counts.

        Args:
            current_page_count: Number of pages to index in current run (optional)

        Returns:
            Estimated duration in seconds, or None if no historical data

        Example:
            Previous: 150 pages took 45 seconds
            Current: 200 pages
            Estimate: (200/150) * 45 = 60 seconds
        """
        stats = self.load()
        if not stats:
            return None

        historical_duration = stats.get("last_indexing_duration_seconds")
        historical_page_count = stats.get("last_page_count")

        if historical_duration is None:
            return None

        # If current page count provided, scale proportionally
        if current_page_count is not None and historical_page_count:
            scaled_duration = (current_page_count / historical_page_count) * historical_duration
            return scaled_duration

        # Otherwise return raw historical duration
        return historical_duration
