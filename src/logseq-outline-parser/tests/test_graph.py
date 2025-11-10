"""Tests for Logseq graph path operations."""

import pytest
from pathlib import Path
from logseq_outline.graph import GraphPaths


@pytest.fixture
def temp_graph(tmp_path):
    """Create a temporary Logseq graph structure."""
    graph_path = tmp_path / "test-graph"
    graph_path.mkdir()

    # Create journals and pages directories
    (graph_path / "journals").mkdir()
    (graph_path / "pages").mkdir()

    return graph_path


class TestGraphPathsInitialization:
    """Tests for GraphPaths initialization."""

    def test_init_with_valid_path(self, temp_graph):
        """Test initialization with valid graph path."""
        graph = GraphPaths(temp_graph)
        assert graph.graph_path == temp_graph

    def test_init_with_nonexistent_path_raises_error(self, tmp_path):
        """Test initialization with nonexistent path raises ValueError."""
        nonexistent = tmp_path / "does-not-exist"
        with pytest.raises(ValueError, match="Graph path does not exist"):
            GraphPaths(nonexistent)

    def test_init_with_file_raises_error(self, tmp_path):
        """Test initialization with file path raises ValueError."""
        file_path = tmp_path / "file.txt"
        file_path.write_text("content")
        with pytest.raises(ValueError, match="Graph path is not a directory"):
            GraphPaths(file_path)


class TestGraphPathsProperties:
    """Tests for GraphPaths directory properties."""

    def test_journals_dir_property(self, temp_graph):
        """Test journals_dir property returns correct path."""
        graph = GraphPaths(temp_graph)
        assert graph.journals_dir == temp_graph / "journals"

    def test_pages_dir_property(self, temp_graph):
        """Test pages_dir property returns correct path."""
        graph = GraphPaths(temp_graph)
        assert graph.pages_dir == temp_graph / "pages"


class TestGetJournalPath:
    """Tests for get_journal_path method."""

    def test_get_journal_path_with_underscores(self, temp_graph):
        """Test get_journal_path with underscore format."""
        graph = GraphPaths(temp_graph)
        journal_path = graph.get_journal_path("2025_01_15")
        assert journal_path == temp_graph / "journals" / "2025_01_15.md"

    def test_get_journal_path_with_hyphens(self, temp_graph):
        """Test get_journal_path with hyphen format (auto-converts)."""
        graph = GraphPaths(temp_graph)
        journal_path = graph.get_journal_path("2025-01-15")
        # Should convert hyphens to underscores (Logseq convention)
        assert journal_path == temp_graph / "journals" / "2025_01_15.md"

    def test_get_journal_path_formats_are_equivalent(self, temp_graph):
        """Test both formats produce the same path."""
        graph = GraphPaths(temp_graph)
        path_with_hyphens = graph.get_journal_path("2025-11-06")
        path_with_underscores = graph.get_journal_path("2025_11_06")
        assert path_with_hyphens == path_with_underscores

    def test_get_journal_path_creates_correct_filename(self, temp_graph):
        """Test that journal path uses correct Logseq filename convention."""
        graph = GraphPaths(temp_graph)
        # Both inputs should produce underscore format
        assert graph.get_journal_path("2025-01-15").name == "2025_01_15.md"
        assert graph.get_journal_path("2025_01_15").name == "2025_01_15.md"


class TestJournalExists:
    """Tests for journal_exists method."""

    def test_journal_exists_returns_true_when_file_exists(self, temp_graph):
        """Test journal_exists returns True when journal file exists."""
        # Create a journal file
        journal_file = temp_graph / "journals" / "2025_01_15.md"
        journal_file.write_text("- Journal content")

        graph = GraphPaths(temp_graph)
        assert graph.journal_exists("2025_01_15") is True

    def test_journal_exists_returns_false_when_file_missing(self, temp_graph):
        """Test journal_exists returns False when journal file doesn't exist."""
        graph = GraphPaths(temp_graph)
        assert graph.journal_exists("2025_01_15") is False

    def test_journal_exists_accepts_both_formats(self, temp_graph):
        """Test journal_exists accepts both hyphen and underscore formats."""
        # Create a journal file
        journal_file = temp_graph / "journals" / "2025_01_15.md"
        journal_file.write_text("- Journal content")

        graph = GraphPaths(temp_graph)
        # Both formats should find the file
        assert graph.journal_exists("2025-01-15") is True
        assert graph.journal_exists("2025_01_15") is True


class TestGetPagePath:
    """Tests for get_page_path method."""

    def test_get_page_path_simple_name(self, temp_graph):
        """Test get_page_path with simple page name."""
        graph = GraphPaths(temp_graph)
        page_path = graph.get_page_path("Project X")
        assert page_path == temp_graph / "pages" / "Project X.md"

    def test_get_page_path_with_special_characters(self, temp_graph):
        """Test get_page_path with special characters in name."""
        graph = GraphPaths(temp_graph)
        page_path = graph.get_page_path("Project: Design & Architecture")
        assert page_path == temp_graph / "pages" / "Project: Design & Architecture.md"


class TestPageExists:
    """Tests for page_exists method."""

    def test_page_exists_returns_true_when_file_exists(self, temp_graph):
        """Test page_exists returns True when page file exists."""
        # Create a page file
        page_file = temp_graph / "pages" / "Project X.md"
        page_file.write_text("- Page content")

        graph = GraphPaths(temp_graph)
        assert graph.page_exists("Project X") is True

    def test_page_exists_returns_false_when_file_missing(self, temp_graph):
        """Test page_exists returns False when page file doesn't exist."""
        graph = GraphPaths(temp_graph)
        assert graph.page_exists("Nonexistent Page") is False


class TestListJournals:
    """Tests for list_journals method."""

    def test_list_journals_returns_sorted_paths(self, temp_graph):
        """Test list_journals returns journal files in sorted order."""
        # Create multiple journal files
        (temp_graph / "journals" / "2025_01_15.md").write_text("- Content")
        (temp_graph / "journals" / "2025_01_10.md").write_text("- Content")
        (temp_graph / "journals" / "2025_01_20.md").write_text("- Content")

        graph = GraphPaths(temp_graph)
        journals = graph.list_journals()

        assert len(journals) == 3
        # Check sorted order
        assert journals[0].name == "2025_01_10.md"
        assert journals[1].name == "2025_01_15.md"
        assert journals[2].name == "2025_01_20.md"

    def test_list_journals_empty_directory(self, temp_graph):
        """Test list_journals returns empty list when no journals exist."""
        graph = GraphPaths(temp_graph)
        journals = graph.list_journals()
        assert journals == []

    def test_list_journals_missing_directory(self, tmp_path):
        """Test list_journals returns empty list when journals dir doesn't exist."""
        graph_path = tmp_path / "graph"
        graph_path.mkdir()
        # Don't create journals directory

        graph = GraphPaths(graph_path)
        journals = graph.list_journals()
        assert journals == []


class TestListPages:
    """Tests for list_pages method."""

    def test_list_pages_returns_sorted_paths(self, temp_graph):
        """Test list_pages returns page files in sorted order."""
        # Create multiple page files
        (temp_graph / "pages" / "Project X.md").write_text("- Content")
        (temp_graph / "pages" / "Project A.md").write_text("- Content")
        (temp_graph / "pages" / "Project Z.md").write_text("- Content")

        graph = GraphPaths(temp_graph)
        pages = graph.list_pages()

        assert len(pages) == 3
        # Check sorted order
        assert pages[0].name == "Project A.md"
        assert pages[1].name == "Project X.md"
        assert pages[2].name == "Project Z.md"

    def test_list_pages_empty_directory(self, temp_graph):
        """Test list_pages returns empty list when no pages exist."""
        graph = GraphPaths(temp_graph)
        pages = graph.list_pages()
        assert pages == []

    def test_list_pages_missing_directory(self, tmp_path):
        """Test list_pages returns empty list when pages dir doesn't exist."""
        graph_path = tmp_path / "graph"
        graph_path.mkdir()
        # Don't create pages directory

        graph = GraphPaths(graph_path)
        pages = graph.list_pages()
        assert pages == []
