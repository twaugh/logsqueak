"""Pydantic models for LLM NDJSON streaming chunks."""

from pydantic import BaseModel, Field
from typing import Literal, Optional


class KnowledgeClassificationChunk(BaseModel):
    """
    NDJSON chunk for knowledge extraction streaming (Phase 1).

    Each line represents ONE INSIGHT extracted from ONE journal block.

    Insight-first approach:
    - LLM identifies valuable insights/knowledge in journal blocks
    - Each insight is already reworded (timeless, no temporal context)
    - One insight = One block (1:1 mapping)
    - One block can have at most one insight
    - The 'reasoning' field contains the reworded insight text

    This approach combines classification and rewording in a single pass,
    focusing on the insight first rather than mechanically classifying blocks.
    """

    type: Literal["classification"] = Field(
        default="classification",
        description="Chunk type identifier"
    )

    block_id: str = Field(
        ...,
        description="Single journal block ID containing this insight"
    )

    insight: str = Field(
        ...,
        description="The reworded insight suitable for knowledge base integration (timeless, no temporal context)"
    )

    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence score for this insight (0.0-1.0)"
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

    Note: knowledge_block_id is optional in LLM output (one block per call).
    It's set by the streaming handler using the block_id from the request context.
    """

    type: Literal["decision"] = Field(
        default="decision",
        description="Chunk type identifier"
    )

    knowledge_block_id: Optional[str] = Field(
        default=None,
        description="Block ID of the knowledge being integrated (set by handler, not LLM)"
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
        description="Confidence score for this integration (0.0-1.0)"
    )

    reasoning: str = Field(
        ...,
        description="Explanation for this integration decision"
    )
