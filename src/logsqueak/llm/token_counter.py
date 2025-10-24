"""Token counting utilities for accurate LLM token budget management.

Uses tiktoken for exact token counts matching OpenAI's tokenization.
"""

import logging
from typing import Optional

import tiktoken

logger = logging.getLogger(__name__)


class TokenCounter:
    """Counts tokens using model-specific encodings.

    Uses tiktoken to provide exact token counts for OpenAI-compatible models.
    Falls back to reasonable estimates for unknown models.
    """

    def __init__(self, model: str):
        """Initialize token counter for a specific model.

        Args:
            model: Model name (e.g., "gpt-4", "gpt-3.5-turbo")
        """
        self.model = model
        self._encoding: Optional[tiktoken.Encoding] = None
        self._init_encoding()

    def _init_encoding(self) -> None:
        """Initialize tiktoken encoding for the model."""
        try:
            # Try to get encoding for the specific model
            self._encoding = tiktoken.encoding_for_model(self.model)
            logger.debug(f"Loaded tiktoken encoding for model: {self.model}")
        except KeyError:
            # Model not recognized, use cl100k_base (GPT-4/GPT-3.5 encoding)
            logger.warning(
                f"Model '{self.model}' not recognized by tiktoken, "
                "using cl100k_base encoding (GPT-4 tokenizer)"
            )
            self._encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Count exact number of tokens in text.

        Args:
            text: Text to tokenize

        Returns:
            Exact token count
        """
        if not self._encoding:
            # Fallback to character-based estimate (very rough: ~4 chars/token)
            return len(text) // 4

        return len(self._encoding.encode(text))

    def count_messages_tokens(self, messages: list[dict]) -> int:
        """Count tokens in a chat completion messages array.

        Includes overhead for message formatting (role, formatting tokens, etc.)
        Based on OpenAI's token counting for chat completions.

        Args:
            messages: List of message dicts with 'role' and 'content' keys

        Returns:
            Total token count including formatting overhead
        """
        if not self._encoding:
            # Fallback estimate
            total_text = " ".join(msg.get("content", "") for msg in messages)
            return len(total_text) // 4 + len(messages) * 4

        # Token counting based on OpenAI's documentation
        # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb

        tokens_per_message = 3  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = 1  # if there's a name, the role is omitted

        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                if isinstance(value, str):
                    num_tokens += len(self._encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name

        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>

        return num_tokens

    def estimate_response_tokens(self, max_output_length: int = 500) -> int:
        """Estimate tokens needed for response.

        Args:
            max_output_length: Expected maximum output length

        Returns:
            Estimated response token count
        """
        # Conservative estimate: assume max_output_length is in characters
        # Divide by ~3.5 chars/token (slightly conservative)
        return max_output_length // 3
