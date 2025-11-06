"""Pydantic models for LLM NDJSON streaming chunks."""

from pydantic import BaseModel, Field
from typing import Literal, Optional


class KnowledgeClassificationChunk(BaseModel):
    """
    NDJSON chunk for knowledge classification streaming (Phase 1).

    Each line in the LLM response stream represents one classified block.
    """

    type: Literal["classification"] = Field(
        default="classification",
        description="Chunk type identifier"
    )

    block_id: str = Field(
        ...,
        description="Block identifier being classified"
    )

    is_knowledge: bool = Field(
        ...,
        description="Whether block contains lasting knowledge"
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score (0.0-1.0)"
    )

    reason: Optional[str] = Field(
        default=None,
        description="Reasoning for classification"
    )


class ContentRewordingChunk(BaseModel):
    """
    NDJSON chunk for content rewording streaming (Phase 2).

    Each line represents a reworded knowledge block.
    """

    type: Literal["rewording"] = Field(
        default="rewording",
        description="Chunk type identifier"
    )

    block_id: str = Field(
        ...,
        description="Block identifier being reworded"
    )

    reworded_content: str = Field(
        ...,
        description="Reworded content with temporal context removed"
    )


class IntegrationDecisionChunk(BaseModel):
    """
    NDJSON chunk for integration decision streaming (Phase 3).

    Each line represents one integration decision for a knowledge block.
    A single knowledge block may have multiple decisions (different target pages).
    """

    type: Literal["decision"] = Field(
        default="decision",
        description="Chunk type identifier"
    )

    knowledge_block_id: str = Field(
        ...,
        description="Block ID of the knowledge being integrated"
    )

    target_page: str = Field(
        ...,
        description="Target page name (hierarchical pages use '/' separator)"
    )

    action: Literal["add_section", "add_under", "replace"] = Field(
        ...,
        description="Type of integration action"
    )

    target_block_id: Optional[str] = Field(
        default=None,
        description="Target block ID for 'add_under' or 'replace' actions"
    )

    target_block_title: Optional[str] = Field(
        default=None,
        description="Human-readable title of target block (for display)"
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for this integration (0.0-1.0)"
    )

    reasoning: str = Field(
        ...,
        description="Explanation for this integration decision"
    )
