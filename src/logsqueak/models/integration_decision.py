"""IntegrationDecision model for Phase 3 (Integration Review)."""

from pydantic import BaseModel, Field
from typing import Literal, Optional


class IntegrationDecision(BaseModel):
    """Integration decision for a knowledge block in Phase 3."""

    knowledge_block_id: str = Field(
        ...,
        description="Block ID of the knowledge being integrated"
    )

    target_page: str = Field(
        ...,
        description="Target page name (hierarchical pages use '/' separator)"
    )

    action: Literal["add_section", "add_under", "replace", "skip_exists"] = Field(
        ...,
        description="Type of integration action"
    )

    target_block_id: Optional[str] = Field(
        default=None,
        description="Target block ID for 'add_under', 'replace', or 'skip_exists' actions"
    )

    target_block_title: Optional[str] = Field(
        default=None,
        description="Human-readable title of target block (for display)"
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="LLM's confidence score for this integration (0.0-1.0)"
    )

    refined_text: str = Field(
        ...,
        description="The content to integrate (from Phase 2 EditedContent)"
    )

    reasoning: str = Field(
        ...,
        description="LLM's explanation for this integration decision"
    )

    write_status: Literal["pending", "completed", "failed"] = Field(
        default="pending",
        description="Status of write operation"
    )

    error_message: Optional[str] = Field(
        default=None,
        description="Error details if write_status is 'failed'"
    )

    model_config = {"frozen": False}  # Allow mutation when user accepts/writes
