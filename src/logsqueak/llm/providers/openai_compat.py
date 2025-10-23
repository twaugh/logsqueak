"""OpenAI-compatible LLM provider implementation.

This provider works with any OpenAI-compatible API (OpenAI, local models via llama.cpp, etc.)
using httpx for HTTP requests and JSON mode for structured outputs.
"""

import json
from datetime import date
from textwrap import dedent
from typing import Any, Dict, List

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
from logsqueak.models.knowledge import ActionType


class OpenAICompatibleProvider(LLMClient):
    """OpenAI-compatible API provider.

    Supports any API that implements the OpenAI chat completions format,
    including OpenAI, Azure OpenAI, and local models via llama.cpp or similar.

    Uses JSON mode (response_format: {type: "json_object"}) for structured outputs.
    """

    def __init__(self, endpoint: str, api_key: str, model: str, timeout: float = 60.0):
        """Initialize the provider.

        Args:
            endpoint: API endpoint URL
            api_key: API authentication key
            model: Model name to use
            timeout: Request timeout in seconds (default: 60s)
        """
        self.endpoint = endpoint
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)

    def __del__(self):
        """Clean up HTTP client."""
        if hasattr(self, "client"):
            self.client.close()

    def extract_knowledge(
        self, journal_content: str, journal_date: date
    ) -> List[ExtractionResult]:
        """Extract knowledge blocks from journal entry (Stage 1).

        Args:
            journal_content: Full text of journal entry
            journal_date: Date of the journal entry

        Returns:
            List of extracted knowledge blocks with confidence scores

        Raises:
            LLMError: If API request fails or returns invalid response
        """
        system_prompt = dedent("""
            You are a knowledge extraction assistant for a personal knowledge management system.

            Your task is to identify pieces of information with lasting value from journal entries.

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

            CRITICAL: Preserve hierarchical context from nested bullets!

            When extracting from NESTED journal entries, INCLUDE parent context and [[Page Links]]:

            EXAMPLE JOURNAL:
            - Working on [[RHEL Documentation]]
              - Updated security guidelines
                - Added section on container scanning

            GOOD EXTRACTION:
            "[[RHEL Documentation]]: Added section on container scanning to security guidelines"

            BAD EXTRACTION:
            "Added section on container scanning"

            RULES:
            1. If a parent bullet contains [[Page Link]], include it in extracted knowledge
            2. Preserve the page reference even if the knowledge is nested deep
            3. Add brief parent context for clarity (but keep it concise)
            4. The extracted knowledge should be self-contained and clear

            Return a JSON object with this structure:
            {
              "knowledge_blocks": [
                {
                  "content": "The extracted knowledge text with parent context",
                  "confidence": 0.85
                }
              ]
            }

            Confidence score (0.0-1.0) indicates how certain you are this is lasting knowledge vs. activity log.
            Use confidence < 0.5 for borderline cases.
        """).strip()

        user_prompt = dedent(f"""
            Journal entry from {journal_date.isoformat()}:

            {journal_content}

            Extract knowledge blocks with lasting value.
        """).strip()

        try:
            response = self._make_request(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )

            # Parse JSON response
            data = self._parse_json_response(response)

            # Validate structure
            if "knowledge_blocks" not in data:
                raise LLMResponseError(
                    "Response missing 'knowledge_blocks' field"
                )

            results = []
            for block in data["knowledge_blocks"]:
                if "content" not in block or "confidence" not in block:
                    raise LLMResponseError(
                        "Knowledge block missing required fields"
                    )

                results.append(
                    ExtractionResult(
                        content=block["content"],
                        confidence=float(block["confidence"]),
                    )
                )

            return results

        except httpx.TimeoutException as e:
            raise LLMTimeoutError(f"Request timeout: {e}") from e
        except httpx.HTTPStatusError as e:
            raise LLMAPIError(
                f"API error: {e.response.status_code} - {e.response.text}",
                status_code=e.response.status_code,
            ) from e
        except httpx.RequestError as e:
            raise LLMError(f"Network error: {e}") from e

    def select_target_page(
        self,
        knowledge_content: str,
        candidates: List[PageCandidate],
    ) -> PageSelectionResult:
        """Select best target page and section from RAG candidates (Stage 2).

        Args:
            knowledge_content: The extracted knowledge text
            candidates: Top-5 candidate pages from RAG search

        Returns:
            Selected target page, section path, and suggested action

        Raises:
            LLMError: If API request fails or returns invalid response
        """
        system_prompt = dedent("""
            You are a knowledge organization assistant for a personal knowledge base.

            Your task is to determine the best location for a piece of knowledge within existing pages.

            You will receive:
            1. A knowledge block to organize
            2. Top 5 candidate pages (from semantic search)

            You must select:
            - Which page is most appropriate
            - Which section within that page (if applicable)
            - Whether to add as child bullet or create new section

            Return a JSON object with this structure:
            {
              "target_page": "Page Name",
              "target_section": ["Section", "Subsection"],
              "suggested_action": "add_child",
              "reasoning": "Brief explanation of your choice"
            }

            Notes:
            - target_section can be null if knowledge should go at page root
            - suggested_action must be either "add_child" or "create_section"
            - Use "add_child" when a suitable section exists
            - Use "create_section" when a new organizational section is needed
        """).strip()

        # Format candidates for prompt
        candidates_text = []
        for i, candidate in enumerate(candidates, 1):
            candidates_text.append(
                f"{i}. {candidate.page_name} (similarity: {candidate.similarity_score:.2f})\n"
                f"   Preview: {candidate.preview}"
            )

        user_prompt = dedent(f"""
            Knowledge to organize:
            {knowledge_content}

            Candidate pages:
            {chr(10).join(candidates_text)}

            Select the best page and section for this knowledge.
        """).strip()

        try:
            response = self._make_request(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )

            # Parse JSON response
            data = self._parse_json_response(response)

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
            raise LLMTimeoutError(f"Request timeout: {e}") from e
        except httpx.HTTPStatusError as e:
            raise LLMAPIError(
                f"API error: {e.response.status_code} - {e.response.text}",
                status_code=e.response.status_code,
            ) from e
        except httpx.RequestError as e:
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
