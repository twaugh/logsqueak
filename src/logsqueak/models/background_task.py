"""BackgroundTask model for async operations."""

from enum import Enum
from pydantic import BaseModel, Field
from typing import Literal, Optional


class BackgroundTaskState(str, Enum):
    """Enum for background task states."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class BackgroundTask(BaseModel):
    """Background task state for async operations."""

    task_type: Literal[
        "llm_classification",
        "page_indexing",
        "rag_search",
        "llm_rewording",
        "llm_decisions"
    ] = Field(
        ...,
        description="Type of background task"
    )

    status: Literal["running", "completed", "failed"] = Field(
        default="running",
        description="Current task status"
    )

    progress_percentage: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Progress percentage if calculable (0.0-100.0)"
    )

    progress_current: Optional[int] = Field(
        default=None,
        description="Current item count (e.g., 3 blocks processed)"
    )

    progress_total: Optional[int] = Field(
        default=None,
        description="Total item count (e.g., 5 total blocks)"
    )

    error_message: Optional[str] = Field(
        default=None,
        description="Error details if status is 'failed'"
    )

    model_config = {"frozen": False}  # Allow mutation as task progresses
