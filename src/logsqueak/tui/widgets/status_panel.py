"""StatusPanel widget for displaying background task progress.

This widget shows progress for multiple concurrent background tasks
such as LLM classification, page indexing, and RAG search.
"""

from typing import Dict
from textual.widgets import Static
from rich.text import Text
from logsqueak.models.background_task import BackgroundTask


class StatusPanel(Static):
    """Status widget for displaying background task progress."""

    def __init__(
        self,
        background_tasks: Dict[str, BackgroundTask],
        *args,
        **kwargs
    ):
        """Initialize StatusPanel.

        Args:
            background_tasks: Dictionary mapping task_type to BackgroundTask
        """
        super().__init__("", *args, id="status-panel", **kwargs)
        self.background_tasks = background_tasks

    def on_mount(self) -> None:
        """Set initial content when widget is mounted."""
        self.update_status()

    def update_status(self) -> None:
        """Update status display based on current background tasks."""
        if not self.background_tasks:
            self.update("Ready")
            return

        # Build status text from all active tasks
        status_parts = []

        for task_type, task in self.background_tasks.items():
            if task.status == "running":
                status_parts.append(self._format_task(task))
            elif task.status == "failed":
                status_parts.append(self._format_error(task))

        if status_parts:
            self.update(" | ".join(status_parts))
        else:
            # All tasks completed
            self.update("Ready")

    def _format_task(self, task: BackgroundTask) -> str:
        """Format running task for display.

        Args:
            task: BackgroundTask to format

        Returns:
            Formatted status string
        """
        # Map task types to human-readable labels
        labels = {
            "llm_classification": "Analyzing knowledge blocks",
            "model_preload": "Loading embedding model",
            "page_indexing": "Building page index",
            "rag_search": "Finding relevant pages",
            "llm_rewording": "Refining content",
            "llm_decisions": "Processing knowledge blocks"
        }

        label = labels.get(task.task_type, task.task_type)

        # Format progress based on available data
        if task.progress_percentage is not None:
            return f"{label} {task.progress_percentage:.0f}%"
        elif task.progress_current is not None and task.progress_total is not None:
            return f"{label} ({task.progress_current}/{task.progress_total})"
        else:
            return f"{label}..."

    def _format_error(self, task: BackgroundTask) -> str:
        """Format failed task for display.

        Args:
            task: BackgroundTask with failed status

        Returns:
            Formatted error string
        """
        labels = {
            "llm_classification": "Classification failed",
            "model_preload": "Model loading failed",
            "page_indexing": "Indexing failed",
            "rag_search": "Search failed",
            "llm_rewording": "Rewording failed",
            "llm_decisions": "Decision generation failed"
        }

        label = labels.get(task.task_type, "Task failed")

        if task.error_message:
            return f"⚠ {label}: {task.error_message}"
        else:
            return f"⚠ {label}"

    def add_task(self, task_type: str, task: BackgroundTask) -> None:
        """Add or update a background task.

        Args:
            task_type: Task type identifier
            task: BackgroundTask object
        """
        self.background_tasks[task_type] = task
        self.update_status()

    def remove_task(self, task_type: str) -> None:
        """Remove a background task from display.

        Args:
            task_type: Task type identifier to remove
        """
        if task_type in self.background_tasks:
            del self.background_tasks[task_type]
            self.update_status()

    def update_task_progress(
        self,
        task_type: str,
        **kwargs
    ) -> None:
        """Update progress for an existing task.

        Args:
            task_type: Task type identifier
            **kwargs: Fields to update (progress_current, progress_total, etc.)
        """
        if task_type in self.background_tasks:
            task = self.background_tasks[task_type]

            # Update fields
            if "progress_current" in kwargs:
                task.progress_current = kwargs["progress_current"]
            if "progress_total" in kwargs:
                task.progress_total = kwargs["progress_total"]
            if "progress_percentage" in kwargs:
                task.progress_percentage = kwargs["progress_percentage"]
            if "status" in kwargs:
                task.status = kwargs["status"]
            if "error_message" in kwargs:
                task.error_message = kwargs["error_message"]

            self.update_status()

    def get_task_status(self, task_type: str) -> str:
        """Get status of a specific task.

        Args:
            task_type: Task type identifier

        Returns:
            Task status ("running", "completed", "failed") or "not_found"
        """
        if task_type in self.background_tasks:
            return self.background_tasks[task_type].status
        return "not_found"

    def is_all_completed(self, task_types: list[str]) -> bool:
        """Check if all specified tasks are completed.

        Args:
            task_types: List of task type identifiers to check

        Returns:
            True if all tasks are completed, False otherwise
        """
        for task_type in task_types:
            if task_type in self.background_tasks:
                if self.background_tasks[task_type].status != "completed":
                    return False
            else:
                # Task not found means not started or removed
                return False

        return True
