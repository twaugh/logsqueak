"""EditedContent model for Phase 2 (Content Editing)."""

from pydantic import BaseModel, Field
from typing import Optional


class EditedContent(BaseModel):
    """Edited content state for a knowledge block in Phase 2."""

    block_id: str = Field(
        ...,
        description="Stable block identifier matching BlockState"
    )

    original_content: str = Field(
        ...,
        description="Original block content from journal (single block only, without parent context)"
    )

    hierarchical_context: str = Field(
        ...,
        description="Full hierarchical context including all parent blocks (shown in top read-only panel)"
    )

    reworded_content: Optional[str] = Field(
        default=None,
        description="LLM-generated reworded version of original_content (removes temporal context, single block only)"
    )

    current_content: str = Field(
        ...,
        description="Current editable content (starts as original_content, user can modify in bottom panel)"
    )

    rewording_complete: bool = Field(
        default=False,
        description="Whether LLM has finished generating reworded version"
    )

    model_config = {"frozen": False}  # Allow mutation during editing
