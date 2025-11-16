"""Configuration management with lazy validation."""

from pathlib import Path
from functools import cached_property

from logsqueak.models.config import Config, LLMConfig, LogseqConfig, RAGConfig
from logsqueak.utils.logging import get_logger


logger = get_logger(__name__)


class ConfigManager:
    """
    Configuration manager with lazy validation.

    Loads config file and validates sections only when first accessed.
    This ensures fast startup and users only see errors for config they use.

    Example:
        >>> config_mgr = ConfigManager.load_default()
        >>> llm_config = config_mgr.llm  # Validates LLM config on first access
        >>> logseq_config = config_mgr.logseq  # Validates Logseq config on first access
    """

    def __init__(self, config: Config):
        """
        Initialize config manager with loaded config.

        Args:
            config: Loaded and validated Config instance
        """
        self._config = config

    @classmethod
    def load_default(cls) -> "ConfigManager":
        """
        Load configuration from default path (~/.config/logsqueak/config.yaml).

        Returns:
            ConfigManager instance with loaded config

        Raises:
            FileNotFoundError: If config file doesn't exist
            PermissionError: If config file has wrong permissions
            ValueError: If config is invalid
        """
        config_path = Path.home() / ".config" / "logsqueak" / "config.yaml"
        return cls.load_from_path(config_path)

    @classmethod
    def load_from_path(cls, path: Path) -> "ConfigManager":
        """
        Load configuration from specific path.

        Args:
            path: Path to config.yaml file

        Returns:
            ConfigManager instance with loaded config

        Raises:
            FileNotFoundError: If config file doesn't exist
            PermissionError: If config file has wrong permissions
            ValueError: If config is invalid
        """
        logger.info("config_loading", path=str(path))

        try:
            config = Config.load(path)
            logger.info("config_loaded", path=str(path))
            return cls(config)

        except FileNotFoundError as e:
            logger.error("config_not_found", path=str(path), error=str(e))
            raise

        except PermissionError as e:
            logger.error("config_permission_error", path=str(path), error=str(e))
            raise

        except Exception as e:
            logger.error("config_validation_error", path=str(path), error=str(e))
            raise ValueError(f"Configuration validation failed: {e}") from e

    @cached_property
    def llm(self) -> LLMConfig:
        """
        Get LLM configuration (lazy validation).

        Returns:
            Validated LLM configuration

        Raises:
            ValueError: If LLM config is invalid
        """
        try:
            return self._config.llm
        except Exception as e:
            logger.error("llm_config_invalid", error=str(e))
            raise ValueError(f"LLM configuration invalid: {e}") from e

    @cached_property
    def logseq(self) -> LogseqConfig:
        """
        Get Logseq configuration (lazy validation).

        Returns:
            Validated Logseq configuration

        Raises:
            ValueError: If Logseq config is invalid
        """
        try:
            return self._config.logseq
        except Exception as e:
            logger.error("logseq_config_invalid", error=str(e))
            raise ValueError(f"Logseq configuration invalid: {e}") from e

    @cached_property
    def rag(self) -> RAGConfig:
        """
        Get RAG configuration (lazy validation, uses defaults if not specified).

        Returns:
            Validated RAG configuration
        """
        return self._config.rag
