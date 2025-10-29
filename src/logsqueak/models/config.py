"""Configuration model for Logsqueak.

This module defines the configuration structure loaded from ~/.config/logsqueak/config.yaml
and validated using Pydantic.
"""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, HttpUrl, field_validator


class LLMConfig(BaseModel):
    """LLM API configuration."""

    endpoint: HttpUrl
    api_key: str = Field(..., min_length=1)
    model: str = Field(default="gpt-4-turbo-preview")
    decider_model: Optional[str] = Field(
        default=None,
        description="Model for Decider LLM (Phase 3.1). Defaults to 'model' if not specified."
    )
    reworder_model: Optional[str] = Field(
        default=None,
        description="Model for Reworder LLM (Phase 3.2). Defaults to 'model' if not specified."
    )
    num_ctx: Optional[int] = Field(
        default=None,
        ge=512,
        description="Context window size for Ollama models (num_ctx parameter). "
                    "Controls GPU VRAM usage. Typical values: 8192, 16384, 32768, 65536."
    )

    def get_decider_model(self) -> str:
        """Get the model to use for Decider (defaults to main model)."""
        return self.decider_model or self.model

    def get_reworder_model(self) -> str:
        """Get the model to use for Reworder (defaults to main model)."""
        return self.reworder_model or self.model


class LogseqConfig(BaseModel):
    """Logseq graph configuration."""

    graph_path: Path

    @field_validator("graph_path")
    @classmethod
    def path_must_exist(cls, v: Path) -> Path:
        """Validate that graph path exists and is a directory."""
        if not v.exists() or not v.is_dir():
            raise ValueError(f"Logseq graph path does not exist: {v}")
        return v


class RAGConfig(BaseModel):
    """RAG search configuration for tuning semantic search behavior.

    The token budget controls how many candidate pages are sent to the LLM
    for page selection (Stage 2). The system uses exact token counting (tiktoken)
    to fit as many candidates as possible within the budget.
    """

    token_budget: Optional[int] = Field(
        default=None,
        ge=500,
        description="Token budget for Stage 2 prompts (page selection). If None, uses top 5 candidates.",
    )

    @field_validator("token_budget")
    @classmethod
    def validate_token_budget(cls, v: Optional[int]) -> Optional[int]:
        """Validate that token budget is reasonable."""
        if v is not None and v < 500:
            raise ValueError(
                "token_budget must be at least 500 tokens (minimum for single candidate)"
            )
        return v


class Configuration(BaseModel):
    """Complete Logsqueak configuration."""

    llm: LLMConfig
    logseq: LogseqConfig
    rag: RAGConfig = Field(default_factory=RAGConfig)

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "Configuration":
        """Load configuration from YAML file.

        Args:
            config_path: Path to config file. If None, uses ~/.config/logsqueak/config.yaml

        Returns:
            Validated Configuration object

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file is invalid
        """
        import yaml

        if config_path is None:
            config_path = Path.home() / ".config" / "logsqueak" / "config.yaml"

        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found at {config_path}\n"
                "Create config file or set LOGSQUEAK_* environment variables."
            )

        with config_path.open() as f:
            data = yaml.safe_load(f)

        return cls(**data)
