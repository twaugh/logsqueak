"""BlockState model for Phase 1 (Block Selection)."""

from pydantic import BaseModel, Field
from typing import Literal, Optional


class BlockState(BaseModel):
    """Selection state for a journal block in Phase 1."""

    block_id: str = Field(
        ...,
        description="Stable block identifier (explicit id:: property or content hash)"
    )

    classification: Literal["pending", "knowledge"] = Field(
        default="pending",
        description="Current classification status (user or LLM)"
    )

    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence score for current classification (0.0-1.0)"
    )

    source: Literal["user", "llm"] = Field(
        ...,
        description="Who set the current classification"
    )

    llm_classification: Optional[Literal["knowledge"]] = Field(
        default=None,
        description="LLM's classification if available (only 'knowledge' or None)"
    )

    llm_confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="LLM's confidence score if classification available"
    )

    reason: Optional[str] = Field(
        default=None,
        description="LLM's reasoning for why block is knowledge"
    )

    model_config = {"frozen": False}  # Allow mutation during user interaction
