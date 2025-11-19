"""Provider-specific logic for LLM configuration."""

from dataclasses import dataclass


@dataclass
class OllamaModel:
    """Represents a model available in an Ollama instance.

    Attributes:
        name: Model identifier (e.g., "mistral:7b-instruct")
        size: Model size in bytes
        modified_at: Last modified timestamp (ISO format)
    """
    name: str
    size: int
    modified_at: str
