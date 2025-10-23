"""Unit tests for activity vs knowledge classifier.

Tests cover:
- Confidence threshold logic
- Classification of knowledge vs activity logs
- Edge cases (boundary values, empty lists)
- Custom threshold support
"""

import pytest

from logsqueak.extraction.classifier import (
    KNOWLEDGE_THRESHOLD,
    classify_extractions,
    is_knowledge,
)
from logsqueak.llm.client import ExtractionResult


# is_knowledge() Tests


def test_is_knowledge_high_confidence():
    """Test that high confidence extractions are classified as knowledge."""
    extraction = ExtractionResult(
        content="Decided to use PostgreSQL for persistence",
        confidence=0.9,
    )

    assert is_knowledge(extraction) is True


def test_is_knowledge_at_threshold():
    """Test that extractions at exactly threshold are classified as knowledge."""
    extraction = ExtractionResult(
        content="Borderline knowledge",
        confidence=KNOWLEDGE_THRESHOLD,  # Exactly 0.5
    )

    assert is_knowledge(extraction) is True


def test_is_knowledge_below_threshold():
    """Test that low confidence extractions are classified as activity logs."""
    extraction = ExtractionResult(
        content="Attended standup meeting",
        confidence=0.3,
    )

    assert is_knowledge(extraction) is False


def test_is_knowledge_just_below_threshold():
    """Test edge case just below threshold."""
    extraction = ExtractionResult(
        content="Possibly temporary info",
        confidence=0.49,
    )

    assert is_knowledge(extraction) is False


def test_is_knowledge_custom_threshold():
    """Test is_knowledge with custom threshold."""
    extraction = ExtractionResult(
        content="Medium confidence",
        confidence=0.6,
    )

    # With default threshold (0.5): True
    assert is_knowledge(extraction) is True

    # With higher threshold (0.7): False
    assert is_knowledge(extraction, threshold=0.7) is False


def test_is_knowledge_very_low_confidence():
    """Test very low confidence is classified as activity."""
    extraction = ExtractionResult(
        content="Checked email",
        confidence=0.1,
    )

    assert is_knowledge(extraction) is False


def test_is_knowledge_very_high_confidence():
    """Test very high confidence is classified as knowledge."""
    extraction = ExtractionResult(
        content="Critical architectural decision",
        confidence=0.95,
    )

    assert is_knowledge(extraction) is True


# classify_extractions() Tests


def test_classify_extractions_separates_correctly():
    """Test that classify_extractions separates knowledge from activity logs."""
    extractions = [
        ExtractionResult("Key decision made", 0.9),
        ExtractionResult("Had meeting", 0.2),
        ExtractionResult("Important insight", 0.85),
        ExtractionResult("Completed task", 0.3),
    ]

    knowledge, activities = classify_extractions(extractions)

    assert len(knowledge) == 2
    assert len(activities) == 2

    # Verify correct classification
    assert knowledge[0].content == "Key decision made"
    assert knowledge[1].content == "Important insight"
    assert activities[0].content == "Had meeting"
    assert activities[1].content == "Completed task"


def test_classify_extractions_all_knowledge():
    """Test classification when all extractions are knowledge."""
    extractions = [
        ExtractionResult("Decision 1", 0.9),
        ExtractionResult("Decision 2", 0.85),
        ExtractionResult("Decision 3", 0.75),
    ]

    knowledge, activities = classify_extractions(extractions)

    assert len(knowledge) == 3
    assert len(activities) == 0


def test_classify_extractions_all_activities():
    """Test classification when all extractions are activity logs."""
    extractions = [
        ExtractionResult("Meeting 1", 0.2),
        ExtractionResult("Task completed", 0.3),
        ExtractionResult("Email sent", 0.1),
    ]

    knowledge, activities = classify_extractions(extractions)

    assert len(knowledge) == 0
    assert len(activities) == 3


def test_classify_extractions_empty_list():
    """Test classification of empty extraction list."""
    knowledge, activities = classify_extractions([])

    assert knowledge == []
    assert activities == []


def test_classify_extractions_custom_threshold():
    """Test classify_extractions with custom threshold."""
    extractions = [
        ExtractionResult("Content 1", 0.6),
        ExtractionResult("Content 2", 0.75),
        ExtractionResult("Content 3", 0.4),
    ]

    # With default threshold (0.5)
    knowledge, activities = classify_extractions(extractions)
    assert len(knowledge) == 2
    assert len(activities) == 1

    # With higher threshold (0.7)
    knowledge, activities = classify_extractions(extractions, threshold=0.7)
    assert len(knowledge) == 1  # Only 0.75 passes
    assert len(activities) == 2


def test_classify_extractions_boundary_values():
    """Test classification with values at and around threshold."""
    extractions = [
        ExtractionResult("Just above", 0.51),
        ExtractionResult("At threshold", 0.5),
        ExtractionResult("Just below", 0.49),
    ]

    knowledge, activities = classify_extractions(extractions)

    assert len(knowledge) == 2  # 0.51 and 0.5
    assert len(activities) == 1  # 0.49

    assert knowledge[0].confidence == 0.51
    assert knowledge[1].confidence == 0.5
    assert activities[0].confidence == 0.49


def test_classify_extractions_preserves_order():
    """Test that classification preserves original order within each category."""
    extractions = [
        ExtractionResult("Knowledge 1", 0.9),
        ExtractionResult("Activity 1", 0.2),
        ExtractionResult("Knowledge 2", 0.8),
        ExtractionResult("Activity 2", 0.3),
        ExtractionResult("Knowledge 3", 0.7),
    ]

    knowledge, activities = classify_extractions(extractions)

    # Verify order is preserved
    assert knowledge[0].content == "Knowledge 1"
    assert knowledge[1].content == "Knowledge 2"
    assert knowledge[2].content == "Knowledge 3"

    assert activities[0].content == "Activity 1"
    assert activities[1].content == "Activity 2"
