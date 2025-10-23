"""Activity vs knowledge classifier.

This module implements confidence-based classification to distinguish
lasting knowledge from temporary activity logs (FR-001, FR-010).
"""

from logsqueak.llm.client import ExtractionResult


# Confidence threshold for knowledge classification
# Blocks with confidence >= KNOWLEDGE_THRESHOLD are considered lasting knowledge
# Blocks with confidence < KNOWLEDGE_THRESHOLD are activity logs (skipped)
KNOWLEDGE_THRESHOLD = 0.5


def is_knowledge(extraction: ExtractionResult, threshold: float = KNOWLEDGE_THRESHOLD) -> bool:
    """Determine if extraction is lasting knowledge vs. activity log.

    Uses confidence score from LLM to classify. The LLM is instructed to
    return confidence scores where:
    - High confidence (≥0.5): Lasting knowledge (decisions, insights, context)
    - Low confidence (<0.5): Activity log (meetings, tasks, status updates)

    Args:
        extraction: Extraction result from Stage 1
        threshold: Confidence threshold (default: 0.5)

    Returns:
        True if considered lasting knowledge, False if activity log

    Examples:
        >>> result = ExtractionResult("Decided to use PostgreSQL", 0.85)
        >>> is_knowledge(result)
        True

        >>> result = ExtractionResult("Attended meeting", 0.3)
        >>> is_knowledge(result)
        False
    """
    return extraction.confidence >= threshold


def classify_extractions(
    extractions: list[ExtractionResult], threshold: float = KNOWLEDGE_THRESHOLD
) -> tuple[list[ExtractionResult], list[ExtractionResult]]:
    """Classify extractions into knowledge and activity logs.

    Args:
        extractions: List of extraction results from Stage 1
        threshold: Confidence threshold (default: 0.5)

    Returns:
        Tuple of (knowledge_blocks, activity_logs)

    Examples:
        >>> extractions = [
        ...     ExtractionResult("Key decision made", 0.9),
        ...     ExtractionResult("Had meeting", 0.2),
        ...     ExtractionResult("Important insight", 0.85),
        ... ]
        >>> knowledge, activities = classify_extractions(extractions)
        >>> len(knowledge)
        2
        >>> len(activities)
        1
    """
    knowledge_blocks = []
    activity_logs = []

    for extraction in extractions:
        if is_knowledge(extraction, threshold):
            knowledge_blocks.append(extraction)
        else:
            activity_logs.append(extraction)

    return knowledge_blocks, activity_logs
