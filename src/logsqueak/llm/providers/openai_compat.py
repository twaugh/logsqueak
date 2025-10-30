"""OpenAI-compatible LLM provider implementation.

This provider works with any OpenAI-compatible API (OpenAI, local models via llama.cpp, etc.)
using httpx for HTTP requests and JSON mode for structured outputs.
"""

import asyncio
import json
import logging
from datetime import date
from textwrap import dedent
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from logsqueak.llm.client import (
    DecisionResult,
    ExtractionResult,
    LLMAPIError,
    LLMClient,
    LLMError,
    LLMResponseError,
    LLMTimeoutError,
    PageCandidate,
    PageSelectionResult,
    RephrasedContent,
)
from logsqueak.llm.prompt_logger import PromptLogger
from logsqueak.llm.prompts import (
    build_decider_messages,
    build_page_selection_messages,
    build_reworder_messages,
)
from logsqueak.llm.streaming import parse_ndjson_stream
from logsqueak.models.knowledge import ActionType

logger = logging.getLogger(__name__)


class OpenAICompatibleProvider(LLMClient):
    """OpenAI-compatible API provider.

    Supports any API that implements the OpenAI chat completions format,
    including OpenAI, Azure OpenAI, and local models via llama.cpp or similar.

    Uses JSON mode (response_format: {type: "json_object"}) for structured outputs.
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        model: str,
        timeout: float = 60.0,
        prompt_logger: Optional[PromptLogger] = None,
        num_ctx: Optional[int] = None,
    ):
        """Initialize the provider.

        Args:
            endpoint: API endpoint URL
            api_key: API authentication key
            model: Model name to use
            timeout: Request timeout in seconds (default: 60s)
            prompt_logger: Optional logger for prompts and responses
            num_ctx: Context window size for Ollama (optional, controls VRAM usage)
        """
        self.endpoint = endpoint
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.prompt_logger = prompt_logger
        self.num_ctx = num_ctx
        self.client = httpx.Client(timeout=timeout)
        self._async_client: Optional[httpx.AsyncClient] = None

    def __del__(self):
        """Clean up HTTP client."""
        if hasattr(self, "client"):
            self.client.close()
        # Note: async client cleanup handled by context manager in async methods

    def extract_knowledge(
        self, journal_content: str, journal_date: date, indent_str: str = "  "
    ) -> List[ExtractionResult]:
        """Extract knowledge blocks from journal entry (Stage 1).

        Args:
            journal_content: Full text of journal entry
            journal_date: Date of the journal entry
            indent_str: Indentation string detected from source (e.g., "  ", "\t")

        Returns:
            List of extracted knowledge blocks with confidence scores

        Raises:
            LLMError: If API request fails or returns invalid response
        """
        # Describe the indentation style for the LLM
        indent_desc = repr(indent_str)  # This will show "\t" for tabs, "  " for 2 spaces, etc.

        system_prompt = dedent(f"""
            You are a knowledge extraction assistant for a personal knowledge management system.

            Your task is to identify pieces of information with lasting value from journal entries.

            FORMAT: The journal entries are in Logseq-flavored Markdown with these conventions:
            - Bullets start with "- " and can be nested with indentation
            - Page links use [[Page Name]] syntax
            - Properties use key:: value syntax
            - Indentation uses {indent_desc} per level to indicate hierarchy

            EXTRACT knowledge like:
            - Key decisions and rationale
            - Important insights or learnings
            - Project updates with meaningful context
            - Ideas worth preserving
            - Meeting outcomes and action items with context

            IGNORE activity logs like:
            - "Worked on X"
            - "Had meeting with Y"
            - "Completed task Z"
            - Routine todos without context
            - Temporary status updates

            CRITICAL: Return the EXACT text of each knowledge block as it appears in the journal!

            Do NOT:
            - Add parent context or reword the content
            - Remove words from the beginning (like "This means that")
            - Trim the end of sentences
            - Change ANY words at all

            You MUST copy the text character-for-character from the journal. Just identify which
            specific bullets contain lasting knowledge and return their EXACT, COMPLETE text.

            EXAMPLE JOURNAL:
            - Working on [[RHEL Documentation]]
              - Updated security guidelines
                - Added section on container scanning
            - This means that the service will be read-only

            GOOD EXTRACTION (exact text, character-for-character):
            "Added section on container scanning"
            "This means that the service will be read-only"

            BAD EXTRACTION (with added context):
            "[[RHEL Documentation]]: Added section on container scanning to security guidelines"

            BAD EXTRACTION (missing words from beginning):
            "The service will be read-only"  ← WRONG! Missing "This means that"

            RULES:
            1. Return the exact text of the knowledge bullet (no modifications)
            2. Do NOT include parent context or page links from parent bullets
            3. Parent context will be added automatically by the system
            4. Multiple knowledge blocks from the same parent are fine

            Return a JSON object with this structure:
            {{
              "knowledge_blocks": [
                {{
                  "exact_text": "The exact text of the bullet containing knowledge",
                  "confidence": 0.85
                }}
              ]
            }}

            Confidence score (0.0-1.0) indicates how certain you are this is lasting knowledge vs. activity log.
            Use confidence < 0.5 for borderline cases.
        """).strip()

        user_prompt = f"Journal entry from {journal_date.isoformat()}:\n\n{journal_content}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Log request if logger is enabled
        if self.prompt_logger:
            self.prompt_logger.log_request(
                stage="extraction",
                messages=messages,
                model=self.model,
                metadata={"journal_date": journal_date.isoformat()},
            )

        try:
            response = self._make_request(messages=messages)

            # Parse JSON response
            try:
                data = self._parse_json_response(response)
            except LLMResponseError as e:
                # Log response even when parsing fails (for debugging)
                if self.prompt_logger:
                    # Extract raw content for debugging
                    raw_content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
                    self.prompt_logger.log_response(
                        stage="extraction",
                        response=response,
                        error=e,
                        raw_content=raw_content,
                    )
                raise

            # Log successful response
            if self.prompt_logger:
                self.prompt_logger.log_response(
                    stage="extraction",
                    response=response,
                    parsed_content=data,
                )

            # Validate structure
            if "knowledge_blocks" not in data:
                raise LLMResponseError(
                    "Response missing 'knowledge_blocks' field"
                )

            results = []
            for block in data["knowledge_blocks"]:
                if "exact_text" not in block:
                    raise LLMResponseError(
                        "Knowledge block missing required field: exact_text"
                    )

                # Use default confidence of 0.5 if not provided (common with local models)
                confidence = float(block.get("confidence", 0.5))

                results.append(
                    ExtractionResult(
                        content=block["exact_text"],  # Map exact_text to content for now
                        confidence=confidence,
                    )
                )

            return results

        except httpx.TimeoutException as e:
            if self.prompt_logger:
                self.prompt_logger.log_response(stage="extraction", response={}, error=e)
            raise LLMTimeoutError(f"Request timeout: {e}") from e
        except httpx.HTTPStatusError as e:
            if self.prompt_logger:
                self.prompt_logger.log_response(stage="extraction", response={}, error=e)
            raise LLMAPIError(
                f"API error: {e.response.status_code} - {e.response.text}",
                status_code=e.response.status_code,
            ) from e
        except httpx.RequestError as e:
            if self.prompt_logger:
                self.prompt_logger.log_response(stage="extraction", response={}, error=e)
            raise LLMError(f"Network error: {e}") from e

    def select_target_page(
        self,
        knowledge_content: str,
        candidates: List[PageCandidate],
        indent_str: str = "  ",
    ) -> PageSelectionResult:
        """Select best target page and section from RAG candidates (Stage 2).

        Args:
            knowledge_content: The extracted knowledge text
            candidates: Top-5 candidate pages from RAG search
            indent_str: Indentation string detected from source (e.g., "  ", "\t")

        Returns:
            Selected target page, section path, and suggested action

        Raises:
            LLMError: If API request fails or returns invalid response
        """
        # Build messages using shared prompt builder
        messages = build_page_selection_messages(
            knowledge_content=knowledge_content,
            candidates=candidates,
            indent_str=indent_str,
        )

        # Log request if logger is enabled
        if self.prompt_logger:
            self.prompt_logger.log_request(
                stage="page_selection",
                messages=messages,
                model=self.model,
                metadata={
                    "knowledge_content": knowledge_content[:200] + "..." if len(knowledge_content) > 200 else knowledge_content,
                    "num_candidates": len(candidates),
                },
            )

        try:
            response = self._make_request(messages=messages)

            # Parse JSON response
            data = self._parse_json_response(response)

            # Log successful response
            if self.prompt_logger:
                self.prompt_logger.log_response(
                    stage="page_selection",
                    response=response,
                    parsed_content=data,
                )

            # Validate required structure
            required_fields = ["target_page", "suggested_action"]
            for field in required_fields:
                if field not in data:
                    raise LLMResponseError(f"Response missing '{field}' field")

            # Parse action type
            action_str = data["suggested_action"]
            try:
                suggested_action = ActionType(action_str)
            except ValueError:
                raise LLMResponseError(
                    f"Invalid action type: {action_str}. "
                    "Must be 'add_child' or 'create_section'"
                )

            return PageSelectionResult(
                target_page=data["target_page"],
                target_section=data.get("target_section"),
                suggested_action=suggested_action,
                reasoning=data.get("reasoning", ""),  # Optional for smaller models
            )

        except httpx.TimeoutException as e:
            if self.prompt_logger:
                self.prompt_logger.log_response(stage="page_selection", response={}, error=e)
            raise LLMTimeoutError(f"Request timeout: {e}") from e
        except httpx.HTTPStatusError as e:
            if self.prompt_logger:
                self.prompt_logger.log_response(stage="page_selection", response={}, error=e)
            raise LLMAPIError(
                f"API error: {e.response.status_code} - {e.response.text}",
                status_code=e.response.status_code,
            ) from e
        except httpx.RequestError as e:
            if self.prompt_logger:
                self.prompt_logger.log_response(stage="page_selection", response={}, error=e)
            raise LLMError(f"Network error: {e}") from e

    def decide_action(
        self,
        knowledge_text: str,
        candidate_chunks: List[dict],
    ) -> DecisionResult:
        """Decide what action to take with knowledge (Phase 3: Decider).

        Args:
            knowledge_text: The full-context knowledge text
            candidate_chunks: List of candidate chunks from Phase 2 RAG

        Returns:
            Decision with action type, target_id, and reasoning

        Raises:
            LLMError: If API request fails or returns invalid response
        """
        # Build messages using shared prompt builder
        messages = build_decider_messages(
            knowledge_text=knowledge_text,
            candidate_chunks=candidate_chunks,
        )

        # Log request if logger is enabled
        if self.prompt_logger:
            self.prompt_logger.log_request(
                stage="decider",
                messages=messages,
                model=self.model,
                metadata={
                    "knowledge_text": knowledge_text[:200] + "..." if len(knowledge_text) > 200 else knowledge_text,
                    "num_candidates": len(candidate_chunks),
                },
            )

        try:
            response = self._make_request(messages=messages)

            # Parse JSON response
            data = self._parse_json_response(response)

            # Log successful response
            if self.prompt_logger:
                self.prompt_logger.log_response(
                    stage="decider",
                    response=response,
                    parsed_content=data,
                )

            # Validate required structure
            required_fields = ["action", "reasoning"]
            for field in required_fields:
                if field not in data:
                    raise LLMResponseError(f"Response missing '{field}' field")

            # Parse action type
            action_str = data["action"]
            try:
                action = ActionType(action_str.lower())
            except ValueError:
                raise LLMResponseError(
                    f"Invalid action type: {action_str}. "
                    "Must be one of: IGNORE_ALREADY_PRESENT, IGNORE_IRRELEVANT, UPDATE, APPEND_CHILD, APPEND_ROOT"
                )

            # Get page_name (required only for APPEND_ROOT)
            page_name = data.get("page_name")

            # Get target_id (required for UPDATE, APPEND_CHILD, IGNORE_ALREADY_PRESENT)
            target_id = data.get("target_id")

            # Validate based on action type
            if action == ActionType.APPEND_ROOT and page_name is None:
                raise LLMResponseError("page_name is required for APPEND_ROOT action")

            if action in (ActionType.UPDATE, ActionType.APPEND_CHILD, ActionType.IGNORE_ALREADY_PRESENT):
                if target_id is None:
                    raise LLMResponseError(f"target_id is required for {action_str} action")

            return DecisionResult(
                action=action,
                page_name=page_name,
                target_id=target_id,
                reasoning=data.get("reasoning", ""),
            )

        except httpx.TimeoutException as e:
            if self.prompt_logger:
                self.prompt_logger.log_response(stage="decider", response={}, error=e)
            raise LLMTimeoutError(f"Request timeout: {e}") from e
        except httpx.HTTPStatusError as e:
            if self.prompt_logger:
                self.prompt_logger.log_response(stage="decider", response={}, error=e)
            raise LLMAPIError(
                f"API error: {e.response.status_code} - {e.response.text}",
                status_code=e.response.status_code,
            ) from e
        except httpx.RequestError as e:
            if self.prompt_logger:
                self.prompt_logger.log_response(stage="decider", response={}, error=e)
            raise LLMError(f"Network error: {e}") from e

    def rephrase_content(self, knowledge_full_text: str) -> RephrasedContent:
        """Rephrase knowledge into clean, evergreen content (Phase 3.2: Reworder).

        Args:
            knowledge_full_text: The full-context knowledge text from journal

        Returns:
            Rephrased content ready for integration

        Raises:
            LLMError: If API request fails or returns invalid response
        """
        # Build messages using shared prompt builder
        messages = build_reworder_messages(knowledge_full_text=knowledge_full_text)

        # Log request if logger is enabled
        if self.prompt_logger:
            self.prompt_logger.log_request(
                stage="reworder",
                messages=messages,
                model=self.model,
                metadata={
                    "knowledge_text": knowledge_full_text[:200] + "..." if len(knowledge_full_text) > 200 else knowledge_full_text,
                },
            )

        try:
            # Use plain text mode (not JSON) for reworder
            response = self._make_request(messages=messages, use_json_mode=False)

            # Extract plain text response
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

            if not content:
                raise LLMResponseError("Empty response from LLM")

            # Log successful response
            if self.prompt_logger:
                self.prompt_logger.log_response(
                    stage="reworder",
                    response=response,
                    parsed_content={"content": content},
                )

            return RephrasedContent(content=content)

        except httpx.TimeoutException as e:
            if self.prompt_logger:
                self.prompt_logger.log_response(stage="reworder", response={}, error=e)
            raise LLMTimeoutError(f"Request timeout: {e}") from e
        except httpx.HTTPStatusError as e:
            if self.prompt_logger:
                self.prompt_logger.log_response(stage="reworder", response={}, error=e)
            raise LLMAPIError(
                f"API error: {e.response.status_code} - {e.response.text}",
                status_code=e.response.status_code,
            ) from e
        except httpx.RequestError as e:
            if self.prompt_logger:
                self.prompt_logger.log_response(stage="reworder", response={}, error=e)
            raise LLMError(f"Network error: {e}") from e

    def _make_request(
        self, messages: List[Dict[str, str]], use_json_mode: bool = True
    ) -> Dict[str, Any]:
        """Make HTTP request to OpenAI-compatible or Ollama native API.

        Args:
            messages: List of chat messages
            use_json_mode: If True, request JSON formatted responses (default: True)

        Returns:
            Response JSON

        Raises:
            httpx exceptions on network/API errors
        """
        import logging
        logger = logging.getLogger(__name__)

        # Detect if this is an Ollama endpoint
        base_endpoint = str(self.endpoint).rstrip("/")
        is_ollama = base_endpoint.endswith("/v1") or "/v1/" in base_endpoint

        if is_ollama and self.num_ctx is not None:
            # Use Ollama's native /api/chat endpoint for num_ctx support
            logger.debug(f"Detected Ollama endpoint, using native API with num_ctx={self.num_ctx}")
            return self._make_ollama_request(messages, use_json_mode)
        else:
            # Use OpenAI-compatible endpoint
            return self._make_openai_request(messages, use_json_mode)

    def _make_openai_request(
        self, messages: List[Dict[str, str]], use_json_mode: bool = True
    ) -> Dict[str, Any]:
        """Make HTTP request to OpenAI-compatible API.

        Args:
            messages: List of chat messages
            use_json_mode: If True, request JSON formatted responses (default: True)

        Returns:
            Response JSON

        Raises:
            httpx exceptions on network/API errors
        """
        # Construct API endpoint URL
        endpoint = str(self.endpoint).rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            endpoint = f"{endpoint}/chat/completions"

        # Build request payload
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
        }

        # Add JSON mode if requested
        if use_json_mode:
            payload["response_format"] = {"type": "json_object"}

        # Make request
        response = self.client.post(
            endpoint,
            json=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

        # Raise for HTTP errors
        response.raise_for_status()

        return response.json()

    def _make_ollama_request(
        self, messages: List[Dict[str, str]], use_json_mode: bool = True
    ) -> Dict[str, Any]:
        """Make HTTP request to Ollama's native /api/chat endpoint.

        This endpoint supports the options field including num_ctx.

        Args:
            messages: List of chat messages
            use_json_mode: If True, request JSON formatted responses (default: True)

        Returns:
            Response JSON in OpenAI format (converted from Ollama format)

        Raises:
            httpx exceptions on network/API errors
        """
        import logging
        logger = logging.getLogger(__name__)

        # Construct Ollama native endpoint URL
        base_endpoint = str(self.endpoint).rstrip("/")
        # Remove /v1 suffix if present
        if base_endpoint.endswith("/v1"):
            base_endpoint = base_endpoint[:-3]
        endpoint = f"{base_endpoint}/api/chat"

        # Build Ollama native request payload
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,  # Get complete response
            "options": {
                "temperature": 0.7,
            }
        }

        # Add num_ctx to options
        if self.num_ctx is not None:
            payload["options"]["num_ctx"] = self.num_ctx
            logger.info(f"Using Ollama native API with num_ctx={self.num_ctx}")

        # Add format: json for JSON mode
        if use_json_mode:
            payload["format"] = "json"

        # Make request (Ollama doesn't require Authorization header)
        response = self.client.post(
            endpoint,
            json=payload,
            headers={
                "Content-Type": "application/json",
            },
        )

        # Raise for HTTP errors
        response.raise_for_status()

        ollama_response = response.json()

        # Convert Ollama response format to OpenAI format
        # Ollama format: {"message": {"role": "assistant", "content": "..."}}
        # OpenAI format: {"choices": [{"message": {"role": "assistant", "content": "..."}}]}
        return {
            "choices": [
                {
                    "message": ollama_response.get("message", {}),
                    "finish_reason": "stop",
                    "index": 0,
                }
            ]
        }

    def _parse_json_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse JSON content from OpenAI API response.

        Args:
            response: Full API response

        Returns:
            Parsed JSON content from message

        Raises:
            LLMResponseError: If response format is invalid
        """
        try:
            content = response["choices"][0]["message"]["content"]
            return json.loads(content)
        except (KeyError, IndexError) as e:
            raise LLMResponseError(
                f"Invalid response structure: {e}"
            ) from e
        except json.JSONDecodeError as e:
            # Include the malformed content in the error for debugging
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            preview = content[:500] + "..." if len(content) > 500 else content
            raise LLMResponseError(
                f"Failed to parse JSON content: {e}\n\nMalformed JSON preview:\n{preview}"
            ) from e

    async def _stream_http_ndjson(
        self, messages: List[Dict[str, str]], max_retries: int = 3
    ) -> AsyncIterator[str]:
        """Stream HTTP responses with retry logic and automatic protocol detection.

        Handles both OpenAI-compatible SSE format and Ollama's native streaming format.

        Args:
            messages: Chat messages to send
            max_retries: Maximum retry attempts for transient errors

        Yields:
            String chunks from the LLM response

        Raises:
            LLMError: On HTTP errors or after max retries
        """
        base_endpoint = str(self.endpoint).rstrip("/")
        is_ollama = base_endpoint.endswith("/v1") or "/v1/" in base_endpoint

        if is_ollama and self.num_ctx is not None:
            # Use Ollama native streaming
            async for chunk in self._stream_ollama_native(messages, max_retries):
                yield chunk
        else:
            # Use OpenAI-compatible streaming
            async for chunk in self._stream_openai_compatible(messages, max_retries):
                yield chunk

    async def _stream_openai_compatible(
        self, messages: List[Dict[str, str]], max_retries: int = 3
    ) -> AsyncIterator[str]:
        """Stream from OpenAI-compatible endpoint with SSE format.

        Args:
            messages: Chat messages to send
            max_retries: Maximum retry attempts

        Yields:
            String chunks from the LLM response
        """
        endpoint = str(self.endpoint).rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            endpoint = f"{endpoint}/chat/completions"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "stream": True,
        }

        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    async with client.stream(
                        "POST",
                        endpoint,
                        json=payload,
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                    ) as response:
                        response.raise_for_status()

                        async for line in response.aiter_lines():
                            # OpenAI SSE format: "data: {...}"
                            if line.startswith("data: "):
                                data_str = line[6:]  # Remove "data: " prefix
                                if data_str == "[DONE]":
                                    return

                                try:
                                    data = json.loads(data_str)
                                    delta = data.get("choices", [{}])[0].get("delta", {})
                                    content = delta.get("content", "")
                                    if content:
                                        yield content
                                except json.JSONDecodeError:
                                    logger.warning(f"Malformed SSE data: {data_str[:100]}")
                                    continue
                        return  # Success

            except (httpx.TimeoutException, httpx.RequestError) as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    backoff = 1.0 * (2 ** attempt)
                    logger.info(f"Retrying in {backoff:.1f}s...")
                    await asyncio.sleep(backoff)
                else:
                    raise LLMTimeoutError(f"Failed after {max_retries} attempts: {e}")

            except httpx.HTTPStatusError as e:
                raise LLMAPIError(
                    f"HTTP error {e.response.status_code}: {e.response.text}",
                    status_code=e.response.status_code,
                )

    async def _stream_ollama_native(
        self, messages: List[Dict[str, str]], max_retries: int = 3
    ) -> AsyncIterator[str]:
        """Stream from Ollama's native /api/chat endpoint.

        Args:
            messages: Chat messages to send
            max_retries: Maximum retry attempts

        Yields:
            String chunks from the LLM response
        """
        base_endpoint = str(self.endpoint).rstrip("/")
        if base_endpoint.endswith("/v1"):
            base_endpoint = base_endpoint[:-3]
        endpoint = f"{base_endpoint}/api/chat"

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": 0.7},
        }

        if self.num_ctx is not None:
            payload["options"]["num_ctx"] = self.num_ctx

        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    async with client.stream(
                        "POST",
                        endpoint,
                        json=payload,
                        headers={"Content-Type": "application/json"},
                    ) as response:
                        response.raise_for_status()

                        async for line in response.aiter_lines():
                            if not line.strip():
                                continue

                            try:
                                data = json.loads(line)
                                content = data.get("message", {}).get("content", "")
                                if content:
                                    yield content

                                # Check if done
                                if data.get("done", False):
                                    return

                            except json.JSONDecodeError:
                                logger.warning(f"Malformed Ollama response: {line[:100]}")
                                continue
                        return  # Success

            except (httpx.TimeoutException, httpx.RequestError) as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    backoff = 1.0 * (2 ** attempt)
                    logger.info(f"Retrying in {backoff:.1f}s...")
                    await asyncio.sleep(backoff)
                else:
                    raise LLMTimeoutError(f"Failed after {max_retries} attempts: {e}")

            except httpx.HTTPStatusError as e:
                raise LLMAPIError(
                    f"HTTP error {e.response.status_code}: {e.response.text}",
                    status_code=e.response.status_code,
                )

    async def stream_extract_ndjson(self, blocks: List[dict]) -> AsyncIterator[dict]:
        """Stream knowledge extraction results as NDJSON.

        Phase 1 of the extraction pipeline: Classify each journal block
        as either "knowledge" (lasting information) or "activity" (temporary log).

        Args:
            blocks: List of journal blocks to classify

        Yields:
            dict: Classification results as they arrive

        Raises:
            LLMError: Network error, API failure, or timeout
        """
        # Build prompt for NDJSON extraction
        system_prompt = dedent("""
            You are a knowledge extraction assistant for a personal knowledge management system.

            Your task is to classify journal blocks as either "knowledge" (lasting information)
            or "activity" (temporary logs).

            EXTRACT knowledge like:
            - Key decisions and rationale
            - Important insights or learnings
            - Project updates with meaningful context
            - Ideas worth preserving

            IGNORE activity logs like:
            - "Worked on X"
            - "Had meeting with Y"
            - Routine todos without context
            - Temporary status updates

            For each block, return ONE JSON object per line (NDJSON format):
            {"block_id": "...", "is_knowledge": true/false, "confidence": 0.0-1.0}

            IMPORTANT:
            - Output exactly ONE JSON object per line
            - Each line MUST end with a newline character
            - Do NOT wrap in a top-level array
            - Process blocks in any order (parallel processing is fine)
        """).strip()

        # Format blocks for the prompt
        blocks_text = "\n\n".join([
            f"Block ID: {b['block_id']}\nHierarchy: {b.get('hierarchy', '')}\nContent: {b['content']}"
            for b in blocks
        ])

        user_prompt = f"Classify these journal blocks:\n\n{blocks_text}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Stream HTTP response and parse NDJSON
        async for obj in parse_ndjson_stream(self._stream_http_ndjson(messages)):
            yield obj

    async def stream_decisions_ndjson(
        self, knowledge_block: dict, candidate_pages: List[dict]
    ) -> AsyncIterator[dict]:
        """Stream integration decisions for one knowledge block across candidate pages.

        Phase 3.1 (Decider): For each candidate page, decide what action to take
        (skip, add_section, add_under, replace).

        Args:
            knowledge_block: Knowledge block to integrate
            candidate_pages: Candidate pages from RAG search

        Yields:
            dict: Decision results as they arrive

        Raises:
            LLMError: Network error, API failure, or timeout
        """
        # Build prompt for decision making
        system_prompt = dedent("""
            You are an integration decision assistant for a knowledge management system.

            Your task is to decide where and how to integrate knowledge into existing pages.

            For each candidate page, choose ONE action:
            - "skip": Knowledge already covered or not relevant
            - "add_section": Add as new root-level section
            - "add_under": Add under a specific existing block
            - "replace": Replace an existing block's content

            For each candidate page, return ONE JSON object per line (NDJSON format):
            {"page": "...", "action": "...", "target_id": null/"...", "target_title": null/"...", "confidence": 0.0-1.0, "reasoning": "..."}

            Rules:
            - For "skip" and "add_section": target_id and target_title MUST be null
            - For "add_under" and "replace": target_id and target_title MUST be provided
            - confidence: 0.0-1.0 (how certain you are about this decision)

            IMPORTANT:
            - Output exactly ONE JSON object per line
            - Each line MUST end with a newline character
            - Do NOT wrap in a top-level array
        """).strip()

        # Format knowledge block and candidates
        kb_text = f"Knowledge: {knowledge_block['content']}\nHierarchy: {knowledge_block.get('hierarchical_text', '')}"

        candidates_text = "\n\n".join([
            f"Page: {p['page_name']} (similarity: {p['similarity_score']:.2f})\nBlocks:\n" +
            "\n".join([f"  - [{c['target_id']}] {c['title']}: {c['content'][:100]}" for c in p.get('chunks', [])])
            for p in candidate_pages
        ])

        user_prompt = f"{kb_text}\n\nCandidate Pages:\n{candidates_text}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Stream HTTP response and parse NDJSON
        async for obj in parse_ndjson_stream(self._stream_http_ndjson(messages)):
            yield obj

    async def stream_rewording_ndjson(self, decisions: List[dict]) -> AsyncIterator[dict]:
        """Stream reworded content for accepted integration decisions.

        Phase 3.2 (Reworder): Transform journal-specific knowledge into clean,
        evergreen content suitable for integration into permanent pages.

        Args:
            decisions: Accepted decisions from Phase 3.1

        Yields:
            dict: Reworded results as they arrive

        Raises:
            LLMError: Network error, API failure, or timeout
        """
        # Build prompt for rewording
        system_prompt = dedent("""
            You are a content rewording assistant for a knowledge management system.

            Your task is to transform journal-specific knowledge into clean, evergreen content.

            Remove:
            - Temporal references ("Today", "This week", dates)
            - Journal-specific context ("Had meeting with X")
            - First-person narrative ("I learned", "I discovered")

            Preserve:
            - [[Page Links]] and ((block refs))
            - Technical terminology and specifics
            - Core insights and knowledge

            For each decision, return ONE JSON object per line (NDJSON format):
            {"knowledge_block_id": "...", "page": "...", "refined_text": "..."}

            IMPORTANT:
            - refined_text should be PLAIN BLOCK CONTENT (no bullet markers, no indentation)
            - Output exactly ONE JSON object per line
            - Each line MUST end with a newline character
            - Do NOT wrap in a top-level array
        """).strip()

        # Format decisions for rewording
        decisions_text = "\n\n".join([
            f"Block ID: {d['knowledge_block_id']}\nPage: {d['page']}\nAction: {d['action']}\n"
            f"Original: {d['full_text']}\nHierarchy: {d.get('hierarchical_text', '')}"
            for d in decisions
        ])

        user_prompt = f"Rephrase these knowledge blocks:\n\n{decisions_text}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Stream HTTP response and parse NDJSON
        async for obj in parse_ndjson_stream(self._stream_http_ndjson(messages)):
            yield obj
