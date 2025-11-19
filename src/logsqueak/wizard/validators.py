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
            error_message=f"[PATH ERROR] Path does not exist\n\n"
                         f"The path {expanded} was not found on your system.\n\n"
                         f"→ Check the path is correct\n"
                         f"→ Or navigate to an existing Logseq graph directory"
        )

    if not expanded.is_dir():
        return ValidationResult(
            success=False,
            error_message=f"[PATH ERROR] Path is not a directory\n\n"
                         f"The path {expanded} exists but is a file, not a directory.\n\n"
                         f"→ Provide the path to your Logseq graph directory instead"
        )

    if not (expanded / "journals").exists():
        return ValidationResult(
            success=False,
            error_message=f"[INVALID GRAPH] Missing journals/ subdirectory\n\n"
                         f"The path {expanded} exists but does not contain a 'journals/' directory.\n"
                         f"A valid Logseq graph must contain this directory.\n\n"
                         f"→ Verify you've selected the correct Logseq graph directory\n"
                         f"→ Check that Logseq has been initialized in this location"
        )

    if not (expanded / "logseq").exists():
        return ValidationResult(
            success=False,
            error_message=f"[INVALID GRAPH] Missing logseq/ subdirectory\n\n"
                         f"The path {expanded} exists but does not contain a 'logseq/' directory.\n"
                         f"A valid Logseq graph must contain this directory.\n\n"
                         f"→ Verify you've selected the correct Logseq graph directory\n"
                         f"→ Check that Logseq has been initialized in this location"
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
            error_message=f"[DISK SPACE WARNING] Insufficient disk space\n\n"
                         f"Available: {available_mb} MB\n"
                         f"Recommended: {required_mb} MB\n\n"
                         f"→ Free up disk space before continuing\n"
                         f"→ Or proceed with caution (may fail during download)",
            data={"available_mb": available_mb}
        )

    return ValidationResult(
        success=True,
        data={"available_mb": available_mb}
    )


async def validate_ollama_connection(endpoint: str, timeout: int = 30) -> ValidationResult:
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
        # Disable SSL verification to support self-signed certificates
        # (common for local/development LLM servers)
        async with httpx.AsyncClient(timeout=timeout, verify=False) as client:
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
            error_message=f"[CONNECTION ERROR] Could not connect to Ollama\n\n"
                         f"Unable to reach Ollama at {endpoint}\n\n"
                         f"→ Start Ollama: ollama serve\n"
                         f"→ Or provide a different Ollama endpoint URL"
        )
    except httpx.HTTPStatusError as e:
        return ValidationResult(
            success=False,
            error_message=f"[API ERROR] Ollama API returned error {e.response.status_code}\n\n"
                         f"The Ollama server responded with an error.\n\n"
                         f"→ Check that Ollama is running correctly\n"
                         f"→ Try restarting Ollama: ollama serve"
        )
    except Exception as e:
        return ValidationResult(
            success=False,
            error_message=f"[UNEXPECTED ERROR] Failed to validate Ollama connection\n\n"
                         f"Error: {str(e)}\n\n"
                         f"→ Check network connectivity\n"
                         f"→ Verify the endpoint URL is correct"
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

    except OSError as e:
        # Disk space issues during download
        if "No space left on device" in str(e) or "Disk quota exceeded" in str(e):
            # Check current disk space
            disk_result = check_disk_space(0)  # Check any available space
            available_mb = disk_result.data.get("available_mb", 0) if disk_result.data else 0
            return ValidationResult(
                success=False,
                error_message=f"[DISK SPACE ERROR] Not enough space for download\n\n"
                             f"Disk space exhausted during model download.\n"
                             f"Available: {available_mb} MB\n\n"
                             f"→ Free up at least 500 MB of disk space\n"
                             f"→ Then retry the setup wizard"
            )
        else:
            return ValidationResult(
                success=False,
                error_message=f"[FILESYSTEM ERROR] File system error during download\n\n"
                             f"Error: {str(e)}\n\n"
                             f"→ Check filesystem permissions\n"
                             f"→ Ensure ~/.cache is writable"
            )

    except Exception as e:
        # Catch all other errors including corrupted downloads
        error_msg = str(e)
        if "pickle" in error_msg.lower() or "corrupt" in error_msg.lower() or "EOF" in error_msg:
            return ValidationResult(
                success=False,
                error_message=f"[CORRUPTED MODEL] Embedding model is corrupted\n\n"
                             f"The cached model appears to be damaged or incomplete.\n"
                             f"Error: {error_msg}\n\n"
                             f"→ Retry to re-download the model\n"
                             f"→ Or manually clear cache: rm -rf ~/.cache/torch/sentence_transformers"
            )
        return ValidationResult(
            success=False,
            error_message=f"[DOWNLOAD ERROR] Failed to load embedding model\n\n"
                         f"Error: {error_msg}\n\n"
                         f"→ Retry to re-download the model\n"
                         f"→ Check your internet connection\n"
                         f"→ Verify firewall allows downloads from huggingface.co"
        )


async def validate_openai_connection(
    endpoint: str, api_key: str, model: str, timeout: int = 30
) -> ValidationResult:
    """Test OpenAI API connection with minimal request.

    Args:
        endpoint: OpenAI API endpoint URL
        api_key: API key for authentication
        model: Model name to test
        timeout: Request timeout in seconds

    Returns:
        ValidationResult with success=True if connection succeeds

    Raises:
        asyncio.TimeoutError: If request exceeds timeout
    """
    # Normalize endpoint
    base_endpoint = endpoint.rstrip("/")

    # Use /v1/models endpoint for lightweight test
    api_url = f"{base_endpoint}/models"

    try:
        # Disable SSL verification to support self-signed certificates
        # (common for local/development LLM servers)
        async with httpx.AsyncClient(timeout=timeout, verify=False) as client:
            response = await client.get(
                api_url,
                headers={"Authorization": f"Bearer {api_key}"}
            )
            response.raise_for_status()

            return ValidationResult(success=True)

    except httpx.ConnectError as e:
        # More detailed connection error message
        return ValidationResult(
            success=False,
            error_message=(
                f"Could not connect to API at {endpoint}\n"
                f"  Error: {str(e)}\n"
                f"  Check that the endpoint URL is correct and the service is running"
            )
        )
    except httpx.TimeoutException:
        return ValidationResult(
            success=False,
            error_message=(
                f"Connection to {endpoint} timed out after {timeout} seconds\n"
                f"  The API may be slow to respond or unreachable"
            )
        )
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            return ValidationResult(
                success=False,
                error_message=(
                    "Invalid API key (401 Unauthorized)\n"
                    f"  The API at {endpoint} rejected your API key\n"
                    f"  Please check that your API key is correct"
                )
            )
        elif e.response.status_code == 403:
            return ValidationResult(
                success=False,
                error_message=(
                    "API key does not have permission (403 Forbidden)\n"
                    f"  Your API key is valid but lacks necessary permissions\n"
                    f"  Check your account settings at the provider"
                )
            )
        elif e.response.status_code == 404:
            return ValidationResult(
                success=False,
                error_message=(
                    f"API endpoint not found (404 Not Found)\n"
                    f"  The URL {api_url} does not exist\n"
                    f"  Check that your endpoint URL is correct\n"
                    f"  Expected format: https://api.example.com/v1"
                )
            )
        else:
            # Try to get response body for more context
            try:
                error_body = e.response.text
                return ValidationResult(
                    success=False,
                    error_message=(
                        f"API error {e.response.status_code}\n"
                        f"  Endpoint: {endpoint}\n"
                        f"  Response: {error_body[:200]}"
                    )
                )
            except Exception:
                return ValidationResult(
                    success=False,
                    error_message=f"API error: {e.response.status_code}"
                )
    except httpx.InvalidURL:
        return ValidationResult(
            success=False,
            error_message=(
                f"Invalid endpoint URL: {endpoint}\n"
                f"  Expected format: https://api.example.com/v1"
            )
        )
    except Exception as e:
        return ValidationResult(
            success=False,
            error_message=(
                f"Unexpected error while connecting to {endpoint}\n"
                f"  Error type: {type(e).__name__}\n"
                f"  Details: {str(e)}"
            )
        )
