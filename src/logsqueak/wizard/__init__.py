"""Interactive setup wizard for Logsqueak configuration.

This package provides the `logsqueak init` command functionality,
guiding users through configuration setup with validation.
"""

from logsqueak.wizard.wizard import run_setup_wizard

__all__ = ["run_setup_wizard"]
