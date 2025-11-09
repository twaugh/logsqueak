"""LLM client with NDJSON streaming support."""

import httpx
import json
from typing import AsyncIterator, Dict, Any, TypeVar, Type
from pydantic import BaseModel
import asyncio

from logsqueak.utils.logging import get_logger
from logsqueak.models.config import LLMConfig


logger = get_logger(__name__)

T = TypeVar('T', bound=BaseModel)


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

    async def stream_ndjson(
        self,
        prompt: str,
        system_prompt: str,
        chunk_model: Type[T],
        max_retries: int = 1,
        retry_delay: float = 2.0
    ) -> AsyncIterator[T]:
        """
        Stream NDJSON responses from LLM API.

        Each line in the response is a complete JSON object parsed into chunk_model.

        Args:
            prompt: User prompt for the LLM
            system_prompt: System prompt for the LLM
            chunk_model: Pydantic model class to parse each NDJSON line
            max_retries: Number of automatic retries on transient errors (default: 1)
            retry_delay: Delay in seconds between retries (default: 2.0)

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
        request_id = str(asyncio.current_task().get_name() if asyncio.current_task() else "unknown")

        logger.info(
            "llm_request_started",
            request_id=request_id,
            model=self.config.model,
            endpoint=str(self.config.endpoint),
            prompt_length=len(prompt)
        )

        # Prepare request payload
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "stream": True,
            "temperature": 0.7,
        }

        # Add num_ctx for Ollama
        if self.config.num_ctx:
            payload["num_ctx"] = self.config.num_ctx

        attempt = 0
        last_error = None

        while attempt <= max_retries:
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    headers = {"Authorization": f"Bearer {self.config.api_key}"}

                    # Use chat completions endpoint
                    url = str(self.config.endpoint).rstrip("/") + "/chat/completions"

                    chunk_count = 0

                    async with client.stream("POST", url, json=payload, headers=headers) as response:
                        response.raise_for_status()

                        async for line in response.aiter_lines():
                            if not line.strip():
                                continue  # Skip empty lines

                            try:
                                # Parse JSON line
                                data = json.loads(line)

                                # Parse into Pydantic model
                                chunk = chunk_model(**data)
                                chunk_count += 1

                                logger.debug(
                                    "llm_response_chunk",
                                    request_id=request_id,
                                    chunk_num=chunk_count,
                                    chunk_type=chunk.type if hasattr(chunk, 'type') else None
                                )

                                yield chunk

                            except json.JSONDecodeError as e:
                                logger.error(
                                    "llm_malformed_json",
                                    request_id=request_id,
                                    line=line[:100],  # Truncate for logging
                                    error=str(e)
                                )
                                # Skip bad line, continue processing stream
                                continue

                            except Exception as e:
                                logger.error(
                                    "llm_chunk_parse_error",
                                    request_id=request_id,
                                    line=line[:100],
                                    error=str(e)
                                )
                                # Skip bad chunk, continue processing stream
                                continue

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
