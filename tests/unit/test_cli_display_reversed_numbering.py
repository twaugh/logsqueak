"""Unit test for reversed numbering in search results display."""

from pathlib import Path
from unittest.mock import patch, MagicMock
from io import StringIO


def test_display_search_results_reversed_numbering():
    """Test that search results are numbered in reverse (most relevant = 1)."""
    from logsqueak.cli import _display_search_results

    # Create mock results (already in reversed order - least to most relevant)
    results = [
        {
            "page_name": "LeastRelevant",
            "block_id": "id1",
            "confidence": 50,
            "snippet": "- Some content"
        },
        {
            "page_name": "MiddleRelevant",
            "block_id": "id2",
            "confidence": 75,
            "snippet": "- Some other content"
        },
        {
            "page_name": "MostRelevant",
            "block_id": "id3",
            "confidence": 100,
            "snippet": "- The best content"
        }
    ]

    graph_path = Path("/fake/graph")

    # Capture output
    output_lines = []

    def mock_echo(msg=''):
        output_lines.append(msg)

    # Mock click.echo and the clickable link function
    with patch('logsqueak.cli.click.echo', side_effect=mock_echo):
        with patch('logsqueak.cli._create_clickable_link', side_effect=lambda name, path: name):
            _display_search_results(results, graph_path)

    # Join output for easier analysis
    output = '\n'.join(output_lines)

    # Parse the output to find numbered results
    result_numbers = []
    for i, line in enumerate(output_lines):
        if '. LeastRelevant' in line or '. MiddleRelevant' in line or '. MostRelevant' in line:
            # Extract the number before the dot
            number = int(line.split('.')[0])
            page_name = line.split('. ')[1]
            result_numbers.append((number, page_name))

    # Verify we found all 3 results
    assert len(result_numbers) == 3, f"Expected 3 results, got {len(result_numbers)}"

    # Verify numbering
    # First result shown (least relevant) should have highest number (3)
    assert result_numbers[0] == (3, "LeastRelevant"), \
        f"Expected (3, 'LeastRelevant'), got {result_numbers[0]}"

    # Middle result should have number 2
    assert result_numbers[1] == (2, "MiddleRelevant"), \
        f"Expected (2, 'MiddleRelevant'), got {result_numbers[1]}"

    # Last result shown (most relevant) should have number 1
    assert result_numbers[2] == (1, "MostRelevant"), \
        f"Expected (1, 'MostRelevant'), got {result_numbers[2]}"

    print("✓ Numbering is correctly reversed: most relevant result numbered as 1")


def test_display_search_results_reversed_numbering_with_two_results():
    """Test reversed numbering with only 2 results."""
    from logsqueak.cli import _display_search_results

    results = [
        {"page_name": "Less", "block_id": "id1", "confidence": 60, "snippet": "- Content"},
        {"page_name": "More", "block_id": "id2", "confidence": 90, "snippet": "- Better content"}
    ]

    graph_path = Path("/fake/graph")
    output_lines = []

    def mock_echo(msg=''):
        output_lines.append(msg)

    with patch('logsqueak.cli.click.echo', side_effect=mock_echo):
        with patch('logsqueak.cli._create_clickable_link', side_effect=lambda name, path: name):
            _display_search_results(results, graph_path)

    # Find numbered results
    result_numbers = []
    for line in output_lines:
        if '. Less' in line or '. More' in line:
            number = int(line.split('.')[0])
            page_name = line.split('. ')[1]
            result_numbers.append((number, page_name))

    # Verify
    assert result_numbers[0] == (2, "Less"), f"Expected (2, 'Less'), got {result_numbers[0]}"
    assert result_numbers[1] == (1, "More"), f"Expected (1, 'More'), got {result_numbers[1]}"

    print("✓ Two results correctly numbered: 2 (less relevant), 1 (more relevant)")
