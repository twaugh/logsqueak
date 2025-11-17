"""IntegrationDecision model for Phase 3 (Integration Review)."""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Literal, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from logsqueak.models.edited_content import EditedContent


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

    edited_content: "EditedContent" = Field(
        ...,
        description="Reference to EditedContent object (contains current_content)"
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

    @property
    def refined_text(self) -> str:
        """Get the current content from the referenced EditedContent.

        This property provides backward compatibility while ensuring we always
        have the latest edited content without manual synchronization.
        """
        return self.edited_content.current_content

    model_config = {"frozen": False, "arbitrary_types_allowed": True}  # Allow mutation and EditedContent reference
