"""Validation functions for wizard inputs and external systems."""

import asyncio
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import httpx

from logsqueak.wizard.providers import OllamaModel


@dataclass
class ValidationResult:
    """Result of a validation operation.

    Attributes:
        success: Whether validation passed
        error_message: Error details if failed (None if success)
        data: Additional data from validation (e.g., models list, disk space)
    """
    success: bool
    error_message: str | None = None
    data: Any | None = None


def validate_graph_path(path: str) -> ValidationResult:
    """Validate Logseq graph directory structure.

    Args:
        path: Path to validate

    Returns:
        ValidationResult with success=True if path exists and has journals/ and logseq/
    """
    expanded = Path(path).expanduser().resolve()

    if not expanded.exists():
        return ValidationResult(
            success=False,
            error_message=f"Path does not exist: {expanded}"
        )

    if not expanded.is_dir():
        return ValidationResult(
            success=False,
            error_message=f"Path is not a directory: {expanded}"
        )

    if not (expanded / "journals").exists():
        return ValidationResult(
            success=False,
            error_message=f"Missing journals/ directory in {expanded}"
        )

    if not (expanded / "logseq").exists():
        return ValidationResult(
            success=False,
            error_message=f"Missing logseq/ directory in {expanded}"
        )

    return ValidationResult(success=True, data={"path": str(expanded)})


def check_disk_space(required_mb: int = 1024) -> ValidationResult:
    """Check available disk space in cache directory.

    Args:
        required_mb: Minimum required space in megabytes

    Returns:
        ValidationResult with success=True if sufficient space,
        warning message if insufficient, data contains available_mb
    """
    # Check disk space in home directory (where cache will be)
    cache_dir = Path.home() / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    usage = shutil.disk_usage(cache_dir)
    available_mb = usage.free // (1024 * 1024)

    if available_mb < required_mb:
        return ValidationResult(
            success=False,
            error_message=f"Low disk space: {available_mb} MB available ({required_mb} MB recommended)",
            data={"available_mb": available_mb}
        )

    return ValidationResult(
        success=True,
        data={"available_mb": available_mb}
    )


async def test_ollama_connection(endpoint: str, timeout: int = 30) -> ValidationResult:
    """Test Ollama API connectivity and retrieve models.

    Args:
        endpoint: Ollama endpoint URL
        timeout: Request timeout in seconds

    Returns:
        ValidationResult with success=True and models data if connection succeeds

    Raises:
        asyncio.TimeoutError: If request exceeds timeout
    """
    # Normalize endpoint (remove /v1 suffix if present for /api/tags call)
    base_endpoint = endpoint.rstrip("/")
    if base_endpoint.endswith("/v1"):
        base_endpoint = base_endpoint[:-3]

    api_url = f"{base_endpoint}/api/tags"

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(api_url)
            response.raise_for_status()
            data = response.json()

            models = []
            for model_data in data.get("models", []):
                models.append(OllamaModel(
                    name=model_data["name"],
                    size=model_data["size"],
                    modified_at=model_data.get("modified_at", "")
                ))

            return ValidationResult(
                success=True,
                data={"models": models}
            )

    except httpx.ConnectError:
        return ValidationResult(
            success=False,
            error_message=f"Could not connect to Ollama at {endpoint}"
        )
    except httpx.HTTPStatusError as e:
        return ValidationResult(
            success=False,
            error_message=f"Ollama API error: {e.response.status_code}"
        )
    except Exception as e:
        return ValidationResult(
            success=False,
            error_message=f"Unexpected error: {str(e)}"
        )


def check_embedding_model_cached() -> bool:
    """Check if sentence-transformers embedding model is cached locally.

    Tries to load the model in offline mode - if it succeeds, the model is cached.

    Returns:
        True if model loads in offline mode, False otherwise
    """
    try:
        import os
        from sentence_transformers import SentenceTransformer

        # Set offline mode environment variable
        old_offline = os.environ.get("HF_HUB_OFFLINE")
        os.environ["HF_HUB_OFFLINE"] = "1"

        try:
            # Try to load model in offline mode
            SentenceTransformer("all-mpnet-base-v2")
            return True
        finally:
            # Restore original environment variable
            if old_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = old_offline
    except Exception:
        # If loading fails in offline mode, model is not cached
        return False


async def validate_embedding_model(
    progress_callback: Callable[[int, int], None] | None = None
) -> ValidationResult:
    """Validate embedding model loads successfully (downloads if needed).

    Args:
        progress_callback: Optional callback(current, total) for download progress

    Returns:
        ValidationResult with success=True if model loads successfully
    """
    try:
        # Import here to avoid loading at module level
        from sentence_transformers import SentenceTransformer

        # This will download if not cached
        # Run in executor to avoid blocking event loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            SentenceTransformer,
            "all-mpnet-base-v2"
        )

        return ValidationResult(success=True)

    except Exception as e:
        return ValidationResult(
            success=False,
            error_message=f"Failed to load embedding model: {str(e)}"
        )
