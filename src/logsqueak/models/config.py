"""Configuration models for Logsqueak."""

from pydantic import BaseModel, Field, HttpUrl, field_validator
from pathlib import Path
import yaml
import os
import stat


class LLMConfig(BaseModel):
    """Configuration for LLM API connection."""

    endpoint: HttpUrl = Field(
        ...,
        description="LLM API endpoint URL (OpenAI or Ollama compatible)"
    )

    api_key: str = Field(
        ...,
        description="API key for authentication"
    )

    model: str = Field(
        ...,
        description="Model identifier (e.g., 'gpt-4-turbo-preview', 'llama2')"
    )

    num_ctx: int = Field(
        default=32768,
        ge=1024,
        description="Context window size (Ollama-specific, controls VRAM usage)"
    )

    model_config = {"frozen": True}


class LogseqConfig(BaseModel):
    """Configuration for Logseq graph location."""

    graph_path: str = Field(
        ...,
        description="Path to Logseq graph directory"
    )

    @field_validator('graph_path')
    @classmethod
    def validate_graph_path(cls, v: str) -> str:
        """Validate graph path exists and is a directory."""
        path = Path(v).expanduser()
        if not path.exists():
            raise ValueError(
                f"Graph path does not exist: {path}\n"
                f"Please create the directory or update config.yaml"
            )
        if not path.is_dir():
            raise ValueError(
                f"Graph path is not a directory: {path}\n"
                f"Please provide a valid directory path"
            )
        return str(path)

    model_config = {"frozen": True}


class RAGConfig(BaseModel):
    """Configuration for RAG semantic search."""

    top_k: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Number of candidate chunks to retrieve from vector search"
    )

    model_config = {"frozen": True}


class Config(BaseModel):
    """Root configuration for Logsqueak application."""

    llm: LLMConfig = Field(..., description="LLM API settings")
    logseq: LogseqConfig = Field(..., description="Logseq graph settings")
    rag: RAGConfig = Field(default_factory=RAGConfig, description="RAG search settings")

    @classmethod
    def load(cls, path: Path) -> "Config":
        """
        Load configuration from YAML file.

        Validates file permissions before loading.
        Raises PermissionError if file is group/world readable.

        Args:
            path: Path to config.yaml file

        Returns:
            Validated Config instance

        Raises:
            PermissionError: If file permissions are too open
            FileNotFoundError: If config file doesn't exist
            ValueError: If YAML is invalid or validation fails
        """
        # Check file exists
        if not path.exists():
            raise FileNotFoundError(
                f"Configuration file not found at {path}\n\n"
                f"Please create the file with the following format:\n\n"
                f"llm:\n"
                f"  endpoint: https://api.openai.com/v1\n"
                f"  api_key: YOUR_API_KEY_HERE\n"
                f"  model: gpt-4-turbo-preview\n\n"
                f"logseq:\n"
                f"  graph_path: ~/Documents/logseq-graph\n\n"
                f"rag:\n"
                f"  top_k: 20\n"
            )

        # Check file permissions (must be 600)
        mode = os.stat(path).st_mode
        if mode & (stat.S_IRWXG | stat.S_IRWXO):
            raise PermissionError(
                f"Config file has overly permissive permissions: {oct(mode)}\n"
                f"Run: chmod 600 {path}"
            )

        # Load and parse YAML
        with open(path) as f:
            data = yaml.safe_load(f)

        return cls(**data)

    model_config = {"frozen": True}
