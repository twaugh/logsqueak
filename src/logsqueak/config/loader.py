"""Configuration loader with YAML and environment variable support.

This module provides a configuration loader that reads from ~/.config/logsqueak/config.yaml
and allows environment variable overrides using LOGSQUEAK_* prefix.

Environment variables:
- LOGSQUEAK_LLM_ENDPOINT: Override LLM API endpoint
- LOGSQUEAK_LLM_API_KEY: Override LLM API key
- LOGSQUEAK_LLM_MODEL: Override LLM model name
- LOGSQUEAK_LOGSEQ_GRAPH_PATH: Override Logseq graph path
- LOGSQUEAK_RAG_TOKEN_BUDGET: Override RAG token budget for Stage 2 prompts
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from logsqueak.models.config import Configuration


def load_config(config_path: Optional[Path] = None) -> Configuration:
    """Load configuration from YAML file with environment variable overrides.

    Args:
        config_path: Path to config file. If None, uses ~/.config/logsqueak/config.yaml

    Returns:
        Validated Configuration object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file is invalid

    Environment Variables:
        LOGSQUEAK_LLM_ENDPOINT: Override llm.endpoint
        LOGSQUEAK_LLM_API_KEY: Override llm.api_key
        LOGSQUEAK_LLM_MODEL: Override llm.model
        LOGSQUEAK_LOGSEQ_GRAPH_PATH: Override logseq.graph_path
        LOGSQUEAK_RAG_TOKEN_BUDGET: Override rag.token_budget
    """
    if config_path is None:
        config_path = Path.home() / ".config" / "logsqueak" / "config.yaml"

    # Load base configuration from YAML if it exists
    if config_path.exists():
        with config_path.open() as f:
            data = yaml.safe_load(f) or {}
    else:
        # If no config file, start with empty dict (env vars will fill it in)
        data = {}

    # Apply environment variable overrides
    data = _apply_env_overrides(data)

    # Validate that required fields are present
    if not data:
        raise FileNotFoundError(
            f"Configuration file not found at {config_path} and no LOGSQUEAK_* environment variables set.\n"
            "Either create a config file or set environment variables."
        )

    # Pydantic will validate the structure
    return Configuration(**data)


def _apply_env_overrides(data: Dict[str, Any]) -> Dict[str, Any]:
    """Apply environment variable overrides to configuration data.

    Environment variables use the format: LOGSQUEAK_SECTION_KEY
    For example: LOGSQUEAK_LLM_ENDPOINT sets data['llm']['endpoint']

    Args:
        data: Base configuration dictionary from YAML

    Returns:
        Configuration dictionary with environment overrides applied
    """
    # Ensure nested structure exists
    if "llm" not in data:
        data["llm"] = {}
    if "logseq" not in data:
        data["logseq"] = {}
    if "rag" not in data:
        data["rag"] = {}

    # LLM configuration overrides
    if env_endpoint := os.getenv("LOGSQUEAK_LLM_ENDPOINT"):
        data["llm"]["endpoint"] = env_endpoint

    if env_api_key := os.getenv("LOGSQUEAK_LLM_API_KEY"):
        data["llm"]["api_key"] = env_api_key

    if env_model := os.getenv("LOGSQUEAK_LLM_MODEL"):
        data["llm"]["model"] = env_model

    # Logseq configuration overrides
    if env_graph_path := os.getenv("LOGSQUEAK_LOGSEQ_GRAPH_PATH"):
        data["logseq"]["graph_path"] = env_graph_path

    # RAG configuration overrides
    if env_token_budget := os.getenv("LOGSQUEAK_RAG_TOKEN_BUDGET"):
        try:
            data["rag"]["token_budget"] = int(env_token_budget)
        except ValueError:
            pass  # Invalid value, ignore

    return data
