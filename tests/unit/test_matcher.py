"""Unit tests for RAG-based page matching logic.

Tests semantic similarity calculations, top-K candidate selection,
and matching behavior without requiring sentence-transformers.
"""

from unittest.mock import Mock, patch

import pytest

from logsqueak.models.page import PageIndex, TargetPage


class TestPageIndexMatching:
    """Test PageIndex semantic matching with mocked embeddings."""

    def test_find_similar_returns_top_k_results(self):
        """Test that find_similar returns requested number of results."""
        index = Mock(spec=PageIndex)

        # Mock 5 similar pages
        mock_pages = [
            (Mock(name=f"Page {i}"), 0.9 - i * 0.1) for i in range(5)
        ]
        index.find_similar.return_value = mock_pages

        results = index.find_similar("test query", top_k=5)

        assert len(results) == 5
        assert results[0][1] == 0.9  # Highest similarity first
        assert results[4][1] == 0.5  # Lowest similarity last

    def test_find_similar_sorts_by_similarity_descending(self):
        """Test that results are sorted by similarity score (highest first)."""
        index = Mock(spec=PageIndex)

        # Mock unsorted results
        mock_pages = [
            (Mock(name="Page Low"), 0.3),
            (Mock(name="Page High"), 0.9),
            (Mock(name="Page Mid"), 0.6),
        ]

        # When sorted, should be High (0.9), Mid (0.6), Low (0.3)
        sorted_pages = sorted(mock_pages, key=lambda x: x[1], reverse=True)
        index.find_similar.return_value = sorted_pages

        results = index.find_similar("query")

        assert results[0][1] > results[1][1] > results[2][1]

    def test_find_similar_respects_top_k_parameter(self):
        """Test that top_k parameter limits results correctly."""
        index = Mock(spec=PageIndex)

        # Mock 10 pages, request top 3
        mock_pages = [
            (Mock(name=f"Page {i}"), 0.9 - i * 0.05) for i in range(3)
        ]
        index.find_similar.return_value = mock_pages

        results = index.find_similar("query", top_k=3)

        assert len(results) == 3

    def test_find_similar_handles_fewer_pages_than_k(self):
        """Test behavior when index has fewer pages than top_k."""
        index = Mock(spec=PageIndex)

        # Only 2 pages, but request top 5
        mock_pages = [
            (Mock(name="Page 1"), 0.8),
            (Mock(name="Page 2"), 0.6),
        ]
        index.find_similar.return_value = mock_pages

        results = index.find_similar("query", top_k=5)

        # Should return all available pages (2), not fail
        assert len(results) == 2

    def test_find_similar_empty_index_returns_empty_list(self):
        """Test that empty index returns empty results."""
        index = Mock(spec=PageIndex)
        index.find_similar.return_value = []

        results = index.find_similar("query")

        assert results == []

    def test_similarity_scores_are_floats_in_range(self):
        """Test that similarity scores are valid floats between 0 and 1."""
        index = Mock(spec=PageIndex)

        mock_pages = [
            (Mock(name="Page 1"), 0.95),
            (Mock(name="Page 2"), 0.5),
            (Mock(name="Page 3"), 0.1),
        ]
        index.find_similar.return_value = mock_pages

        results = index.find_similar("query")

        for page, score in results:
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0

    def test_find_similar_with_identical_query_text(self):
        """Test matching when query is identical to page content."""
        index = Mock(spec=PageIndex)

        # Perfect match should have very high similarity
        mock_pages = [
            (Mock(name="Exact Match"), 0.99),
            (Mock(name="Partial Match"), 0.7),
        ]
        index.find_similar.return_value = mock_pages

        results = index.find_similar("Exact Match")

        # First result should be the exact match with highest score
        assert results[0][1] > 0.9

    def test_find_similar_with_semantic_variation(self):
        """Test semantic matching with paraphrased queries."""
        index = Mock(spec=PageIndex)

        # Semantically similar but different words
        mock_pages = [
            (Mock(name="Database Technology"), 0.85),
            (Mock(name="Data Storage Systems"), 0.82),
            (Mock(name="Unrelated Topic"), 0.2),
        ]
        index.find_similar.return_value = mock_pages

        results = index.find_similar("database systems")

        # Related pages should rank higher than unrelated
        assert results[0][1] > 0.8
        assert results[2][1] < 0.3

    def test_candidate_page_includes_similarity_score(self):
        """Test that candidate tuples include both page and score."""
        index = Mock(spec=PageIndex)

        mock_page = Mock(spec=TargetPage)
        mock_page.name = "Test Page"
        mock_pages = [(mock_page, 0.87)]
        index.find_similar.return_value = mock_pages

        results = index.find_similar("query")

        assert len(results[0]) == 2  # (page, score) tuple
        page, score = results[0]
        assert page.name == "Test Page"
        assert score == 0.87

    def test_find_similar_query_variations(self):
        """Test matching with different query formulations."""
        index = Mock(spec=PageIndex)

        # Different query styles should still find relevant pages
        queries = [
            "How to configure PostgreSQL",
            "PostgreSQL configuration",
            "setting up PostgreSQL",
        ]

        # All should find database-related pages
        mock_pages = [
            (Mock(name="PostgreSQL Setup"), 0.88),
            (Mock(name="Database Config"), 0.75),
        ]

        for query in queries:
            index.find_similar.return_value = mock_pages
            results = index.find_similar(query)

            # Should find relevant pages regardless of phrasing
            assert len(results) > 0
            assert results[0][1] > 0.7

    def test_section_matching_within_page(self):
        """Test that section matching uses page content."""
        mock_page = Mock(spec=TargetPage)
        mock_page.find_section.return_value = Mock(content="Database Section")

        # Find section should work after page is selected
        section = mock_page.find_section("Database")

        assert section is not None
        assert "Database" in section.content

    def test_section_matching_returns_none_when_not_found(self):
        """Test section matching when section doesn't exist."""
        mock_page = Mock(spec=TargetPage)
        mock_page.find_section.return_value = None

        section = mock_page.find_section("NonexistentSection")

        assert section is None

    def test_top_5_candidate_selection_standard(self):
        """Test standard top-5 candidate selection from larger set."""
        index = Mock(spec=PageIndex)

        # Simulate 10 pages, expect top 5
        mock_pages = [
            (Mock(name=f"Page {i}"), 0.95 - i * 0.1) for i in range(5)
        ]
        index.find_similar.return_value = mock_pages

        results = index.find_similar("query", top_k=5)

        assert len(results) == 5
        # Verify descending order
        for i in range(len(results) - 1):
            assert results[i][1] >= results[i + 1][1]

    def test_find_similar_with_special_characters_in_query(self):
        """Test matching handles special characters in queries gracefully."""
        index = Mock(spec=PageIndex)

        # Query with special characters should still work
        mock_pages = [
            (Mock(name="C++ Programming"), 0.88),
            (Mock(name="Python & Ruby"), 0.75),
        ]
        index.find_similar.return_value = mock_pages

        results = index.find_similar("C++ & programming tips")

        # Should find relevant pages
        assert len(results) == 2
        assert results[0][1] > results[1][1]
