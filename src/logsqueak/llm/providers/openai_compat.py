"""OpenAI-compatible LLM provider implementation.

This provider works with any OpenAI-compatible API (OpenAI, local models via llama.cpp, etc.)
using httpx for HTTP requests and JSON mode for structured outputs.
"""

import json
from datetime import date
from textwrap import dedent
from typing import Any, Dict, List, Optional

import httpx

from logsqueak.llm.client import (
    ExtractionResult,
    LLMAPIError,
    LLMClient,
    LLMError,
    LLMResponseError,
    LLMTimeoutError,
    PageCandidate,
    PageSelectionResult,
)
from logsqueak.llm.prompt_logger import PromptLogger
from logsqueak.llm.prompts import build_page_selection_messages
from logsqueak.models.knowledge import ActionType


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
    ):
        """Initialize the provider.

        Args:
            endpoint: API endpoint URL
            api_key: API authentication key
            model: Model name to use
            timeout: Request timeout in seconds (default: 60s)
            prompt_logger: Optional logger for prompts and responses
        """
        self.endpoint = endpoint
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.prompt_logger = prompt_logger
        self.client = httpx.Client(timeout=timeout)

    def __del__(self):
        """Clean up HTTP client."""
        if hasattr(self, "client"):
            self.client.close()

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

            Do NOT add parent context or reword the content. Just identify which specific bullets
            contain lasting knowledge and return their exact text.

            EXAMPLE JOURNAL:
            - Working on [[RHEL Documentation]]
              - Updated security guidelines
                - Added section on container scanning

            GOOD EXTRACTION (exact text only):
            "Added section on container scanning"

            BAD EXTRACTION (with added context):
            "[[RHEL Documentation]]: Added section on container scanning to security guidelines"

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
            data = self._parse_json_response(response)

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
                if "exact_text" not in block or "confidence" not in block:
                    raise LLMResponseError(
                        "Knowledge block missing required fields (exact_text, confidence)"
                    )

                results.append(
                    ExtractionResult(
                        content=block["exact_text"],  # Map exact_text to content for now
                        confidence=float(block["confidence"]),
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

    def _make_request(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Make HTTP request to OpenAI-compatible API.

        Args:
            messages: List of chat messages

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
            "response_format": {"type": "json_object"},
            "temperature": 0.7,
        }

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
            raise LLMResponseError(
                f"Failed to parse JSON content: {e}"
            ) from e
