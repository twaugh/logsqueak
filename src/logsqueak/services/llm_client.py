"""LLM client with NDJSON streaming support."""

import httpx
import json
from typing import AsyncIterator, Dict, Any, TypeVar, Type, Optional
from pydantic import BaseModel
import asyncio

from logsqueak.utils.logging import get_logger
from logsqueak.models.config import LLMConfig


logger = get_logger(__name__)

T = TypeVar('T', bound=BaseModel)


def _extract_content_from_openai_chunk(data: Dict[str, Any]) -> str | None:
    """
    Extract content from OpenAI-style streaming chunk.

    OpenAI/Ollama return chunks like:
    {
        "choices": [{
            "delta": {"content": "..."},
            "finish_reason": null
        }]
    }

    Args:
        data: Parsed JSON chunk from OpenAI API

    Returns:
        Content string if present, None otherwise
    """
    try:
        if "choices" in data and len(data["choices"]) > 0:
            choice = data["choices"][0]
            if "delta" in choice and "content" in choice["delta"]:
                return choice["delta"]["content"]
    except (KeyError, IndexError, TypeError):
        pass
    return None


def _extract_content_from_ollama_chunk(data: Dict[str, Any]) -> str | None:
    """
    Extract content from Ollama native streaming chunk.

    Ollama's /api/chat returns chunks like:
    {
        "model": "...",
        "message": {
            "role": "assistant",
            "content": "..."
        },
        "done": false
    }

    Args:
        data: Parsed JSON chunk from Ollama /api/chat

    Returns:
        Content string if present, None otherwise
    """
    try:
        if "message" in data and "content" in data["message"]:
            return data["message"]["content"]
    except (KeyError, TypeError):
        pass
    return None


class LLMClient:
    """
    HTTP client for LLM API with NDJSON streaming support.

    Supports OpenAI-compatible APIs (including Ollama) with automatic retry
    on transient errors.
    """

    def __init__(self, config: LLMConfig):
        """
        Initialize LLM client.

        Args:
            config: LLM configuration (endpoint, API key, model)
        """
        self.config = config
        self.timeout = httpx.Timeout(
            connect=10.0,
            read=60.0,  # Per-read timeout for streaming
            write=10.0,
            pool=10.0
        )
        self._is_ollama: bool | None = None  # Cached provider detection

    async def _detect_ollama(self) -> bool:
        """
        Detect if the LLM endpoint is Ollama by probing /api/version.

        This detection is cached after the first call.

        Returns:
            True if Ollama detected, False otherwise
        """
        if self._is_ollama is not None:
            return self._is_ollama

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0), verify=False) as client:
                # Probe Ollama-specific version endpoint
                base_url = str(self.config.endpoint).rstrip("/")
                # Remove /v1 suffix if present (OpenAI-compatible path)
                if base_url.endswith("/v1"):
                    base_url = base_url[:-3]

                version_url = f"{base_url}/api/version"

                logger.debug(
                    "llm_provider_detection",
                    version_url=version_url,
                )

                response = await client.get(version_url)

                # If we get a successful response, it's Ollama
                if response.status_code == 200:
                    logger.info(
                        "llm_provider_detected",
                        provider="ollama",
                        version_url=version_url,
                    )
                    self._is_ollama = True
                    return True

        except (httpx.HTTPError, Exception) as e:
            logger.debug(
                "llm_provider_detection_failed",
                error=str(e),
                assumed_provider="openai",
            )

        # Default to OpenAI-compatible if probe fails
        logger.info(
            "llm_provider_detected",
            provider="openai",
        )
        self._is_ollama = False
        return False

    async def stream_ndjson(
        self,
        prompt: str,
        system_prompt: str,
        chunk_model: Type[T],
        max_retries: int = 1,
        retry_delay: float = 2.0,
        temperature: float = 0.7,
        json_mode: bool = False,
        request_id: Optional[str] = None
    ) -> AsyncIterator[T]:
        """
        Stream responses from LLM API.

        Supports both NDJSON and SSE (Server-Sent Events) formats.
        Each line in the response is a complete JSON object parsed into chunk_model.

        SSE format (lines starting with "data: ") is automatically detected and handled.
        The special "data: [DONE]" message is recognized and skipped.

        Args:
            prompt: User prompt for the LLM
            system_prompt: System prompt for the LLM
            chunk_model: Pydantic model class to parse each JSON line
            max_retries: Number of automatic retries on transient errors (default: 1)
            retry_delay: Delay in seconds between retries (default: 2.0)
            temperature: Sampling temperature 0.0-1.0 (default: 0.7)
            json_mode: Enable JSON mode (forces JSON output, OpenAI only, default: False)
            request_id: Optional identifier for this request (for logging/tracing)

        Yields:
            Parsed chunk_model instances

        Raises:
            httpx.HTTPError: On network or HTTP errors after retries exhausted

        Example:
            >>> async for chunk in client.stream_ndjson(
            ...     prompt="Classify these blocks...",
            ...     system_prompt="You are a knowledge classifier",
            ...     chunk_model=KnowledgeClassificationChunk
            ... ):
            ...     print(f"Block {chunk.block_id}: confidence {chunk.confidence}")
        """
        # Use provided request_id, or fall back to asyncio task name
        if not request_id:
            current_task = asyncio.current_task()
            task_name = current_task.get_name() if current_task else None
            # Check for None, empty string, or the literal string "None"
            if task_name and task_name != "None":
                request_id = task_name
            else:
                request_id = "unknown"

        # Detect provider (cached after first call)
        is_ollama = await self._detect_ollama()

        # Prepare request payload
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "stream": True,
            "temperature": temperature,
        }

        # Add num_ctx for Ollama in options object
        if is_ollama and self.config.num_ctx:
            payload["options"] = {"num_ctx": self.config.num_ctx}

        # Add JSON mode for OpenAI (forces JSON output, reduces conversational responses)
        # Note: Ollama doesn't support response_format parameter
        if json_mode and not is_ollama:
            payload["response_format"] = {"type": "json_object"}

        # Log the complete request payload (raw input to LLM)
        logger.info(
            "llm_request_started",
            request_id=request_id,
            model=self.config.model,
            endpoint=str(self.config.endpoint),
            provider="ollama" if is_ollama else "openai",
            prompt_length=len(prompt),
            system_prompt_length=len(system_prompt),
            temperature=temperature,
        )

        logger.debug(
            "llm_request_payload",
            request_id=request_id,
            payload=payload,  # Full raw payload (includes messages with system/user prompts)
        )

        attempt = 0
        last_error = None

        while attempt <= max_retries:
            try:
                async with httpx.AsyncClient(timeout=self.timeout, verify=False) as client:
                    headers = {"Authorization": f"Bearer {self.config.api_key}"}

                    # Choose endpoint based on provider
                    if is_ollama:
                        # Use native Ollama endpoint (supports options)
                        base_url = str(self.config.endpoint).rstrip("/")
                        if base_url.endswith("/v1"):
                            base_url = base_url[:-3]
                        url = base_url + "/api/chat"
                    else:
                        # Use OpenAI-compatible endpoint
                        url = str(self.config.endpoint).rstrip("/") + "/chat/completions"

                    chunk_count = 0
                    accumulated_content = ""  # Accumulate content from OpenAI-style streaming

                    async with client.stream("POST", url, json=payload, headers=headers) as response:
                        response.raise_for_status()

                        # Log response headers (safely handle mock objects)
                        try:
                            response_headers = dict(response.headers) if hasattr(response.headers, '__iter__') else {}
                        except (TypeError, AttributeError):
                            response_headers = {}

                        logger.debug(
                            "llm_response_headers",
                            request_id=request_id,
                            status_code=response.status_code,
                            headers=response_headers,
                        )

                        # Check if response needs content extraction
                        content_type = response_headers.get("content-type", "")
                        # For Ollama native API, we always need to extract from message.content
                        # For OpenAI, we need to extract from choices[].delta.content if SSE
                        needs_content_extraction = is_ollama or "text/event-stream" in content_type

                        logger.debug(
                            "llm_response_format_detected",
                            request_id=request_id,
                            content_type=content_type,
                            needs_content_extraction=needs_content_extraction,
                            provider="ollama" if is_ollama else "openai",
                        )

                        async for line in response.aiter_lines():
                            if not line.strip():
                                continue  # Skip empty lines

                            # Handle SSE format (lines starting with "data: ")
                            # OpenAI-compatible APIs (including Ollama) may return SSE format
                            json_line = line
                            if line.startswith("data: "):
                                json_line = line[6:]  # Remove "data: " prefix

                                # Skip SSE control messages
                                if json_line == "[DONE]":
                                    logger.debug(
                                        "llm_response_sse_done",
                                        request_id=request_id,
                                    )
                                    continue

                            try:
                                # Parse JSON line
                                data = json.loads(json_line)

                                # Handle streaming formats that need content extraction
                                if needs_content_extraction:
                                    # Extract content using appropriate extractor
                                    if is_ollama:
                                        content_fragment = _extract_content_from_ollama_chunk(data)
                                    else:
                                        content_fragment = _extract_content_from_openai_chunk(data)

                                    if content_fragment:
                                        accumulated_content += content_fragment

                                        # Try to parse complete JSON objects from accumulated content
                                        # Split by newlines to find complete JSON lines
                                        lines = accumulated_content.split('\n')

                                        # Process all complete lines (all but the last, which may be incomplete)
                                        for complete_line in lines[:-1]:
                                            if complete_line.strip():
                                                try:
                                                    # Log the raw complete line before parsing
                                                    logger.debug(
                                                        "llm_response_complete_line",
                                                        request_id=request_id,
                                                        raw_line=complete_line,  # Raw NDJSON line
                                                    )

                                                    # Parse the complete JSON line as our custom format
                                                    custom_data = json.loads(complete_line)
                                                    chunk = chunk_model(**custom_data)
                                                    chunk_count += 1

                                                    # Log the successfully parsed chunk
                                                    logger.debug(
                                                        "llm_response_chunk",
                                                        request_id=request_id,
                                                        chunk_num=chunk_count,
                                                        chunk_type=chunk.type if hasattr(chunk, 'type') else None,
                                                        chunk_data=chunk.model_dump(),
                                                    )

                                                    yield chunk
                                                except (json.JSONDecodeError, Exception) as e:
                                                    logger.debug(
                                                        "llm_incomplete_json_line",
                                                        request_id=request_id,
                                                        line=complete_line,  # Full line (no truncation)
                                                        error=str(e),
                                                    )

                                        # Keep the last incomplete line for next iteration
                                        accumulated_content = lines[-1]

                                else:
                                    # Direct NDJSON format - parse immediately
                                    logger.debug(
                                        "llm_response_parsed_json",
                                        request_id=request_id,
                                        parsed_data=data,  # Full parsed JSON object
                                    )

                                    chunk = chunk_model(**data)
                                    chunk_count += 1

                                    logger.debug(
                                        "llm_response_chunk",
                                        request_id=request_id,
                                        chunk_num=chunk_count,
                                        chunk_type=chunk.type if hasattr(chunk, 'type') else None,
                                        chunk_data=chunk.model_dump(),  # Pydantic model as dict
                                    )

                                    yield chunk

                            except json.JSONDecodeError as e:
                                logger.error(
                                    "llm_malformed_json",
                                    request_id=request_id,
                                    line=line,  # Full line for debugging
                                    error=str(e)
                                )
                                # Skip bad line, continue processing stream
                                continue

                            except Exception as e:
                                logger.error(
                                    "llm_chunk_parse_error",
                                    request_id=request_id,
                                    line=line,  # Full line for debugging
                                    parsed_data=data if 'data' in locals() else None,
                                    error=str(e),
                                    error_type=type(e).__name__,
                                )
                                # Skip bad chunk, continue processing stream
                                continue

                        # After stream ends, try to parse any remaining accumulated content
                        if needs_content_extraction and accumulated_content.strip():
                            logger.debug(
                                "llm_processing_final_accumulated_content",
                                request_id=request_id,
                                content_length=len(accumulated_content),
                            )

                            # Split and process any remaining complete lines
                            lines = accumulated_content.split('\n')
                            for complete_line in lines:
                                if complete_line.strip():
                                    try:
                                        # Log the raw final line before parsing
                                        logger.debug(
                                            "llm_response_final_line",
                                            request_id=request_id,
                                            raw_line=complete_line,  # Raw NDJSON line
                                        )

                                        custom_data = json.loads(complete_line)
                                        chunk = chunk_model(**custom_data)
                                        chunk_count += 1

                                        logger.debug(
                                            "llm_response_chunk_final",
                                            request_id=request_id,
                                            chunk_num=chunk_count,
                                            chunk_data=chunk.model_dump(),
                                        )

                                        yield chunk
                                    except (json.JSONDecodeError, Exception) as e:
                                        logger.warning(
                                            "llm_final_content_parse_failed",
                                            request_id=request_id,
                                            line=complete_line,  # Full line (no truncation)
                                            error=str(e),
                                        )

                    logger.info(
                        "llm_request_completed",
                        request_id=request_id,
                        chunk_count=chunk_count
                    )

                    # Success - exit retry loop
                    return

            except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout) as e:
                last_error = e
                attempt += 1

                logger.warning(
                    "llm_request_retry",
                    request_id=request_id,
                    attempt=attempt,
                    max_retries=max_retries,
                    error=str(e),
                    retry_delay=retry_delay
                )

                if attempt <= max_retries:
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(
                        "llm_request_failed",
                        request_id=request_id,
                        attempts=attempt,
                        error=str(e)
                    )
                    raise

            except httpx.HTTPStatusError as e:
                # Don't retry on 4xx errors (bad request, auth, etc.)
                logger.error(
                    "llm_http_error",
                    request_id=request_id,
                    status_code=e.response.status_code,
                    error=str(e)
                )
                raise
