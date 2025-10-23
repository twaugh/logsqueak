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


class Configuration(BaseModel):
    """Complete Logsqueak configuration."""

    llm: LLMConfig
    logseq: LogseqConfig

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
