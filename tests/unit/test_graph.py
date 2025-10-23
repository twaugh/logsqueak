"""Unit tests for Logseq graph path operations."""

from pathlib import Path

import pytest

from logsqueak.logseq.graph import GraphPaths


class TestGraphPaths:
    """Tests for GraphPaths utility class."""

    def test_init_with_valid_path(self, tmp_path):
        """Test initializing with valid graph path."""
        graph = GraphPaths(tmp_path)
        assert graph.graph_path == tmp_path

    def test_init_with_nonexistent_path(self):
        """Test that nonexistent path raises ValueError."""
        with pytest.raises(ValueError, match="does not exist"):
            GraphPaths(Path("/nonexistent/path"))

    def test_init_with_file_path(self, tmp_path):
        """Test that file path (not directory) raises ValueError."""
        file_path = tmp_path / "file.txt"
        file_path.write_text("test")

        with pytest.raises(ValueError, match="not a directory"):
            GraphPaths(file_path)

    def test_journals_dir_property(self, tmp_path):
        """Test journals_dir property returns correct path."""
        graph = GraphPaths(tmp_path)
        expected = tmp_path / "journals"

        assert graph.journals_dir == expected

    def test_pages_dir_property(self, tmp_path):
        """Test pages_dir property returns correct path."""
        graph = GraphPaths(tmp_path)
        expected = tmp_path / "pages"

        assert graph.pages_dir == expected

    def test_get_journal_path(self, tmp_path):
        """Test getting journal file path."""
        graph = GraphPaths(tmp_path)
        path = graph.get_journal_path("2025_01_15")

        expected = tmp_path / "journals" / "2025_01_15.md"
        assert path == expected

    def test_get_page_path(self, tmp_path):
        """Test getting page file path."""
        graph = GraphPaths(tmp_path)
        path = graph.get_page_path("Project X")

        expected = tmp_path / "pages" / "Project X.md"
        assert path == expected

    def test_journal_exists_true(self, tmp_path):
        """Test journal_exists returns True when file exists."""
        journals_dir = tmp_path / "journals"
        journals_dir.mkdir()
        journal_file = journals_dir / "2025_01_15.md"
        journal_file.write_text("- test")

        graph = GraphPaths(tmp_path)
        assert graph.journal_exists("2025_01_15") is True

    def test_journal_exists_false(self, tmp_path):
        """Test journal_exists returns False when file doesn't exist."""
        journals_dir = tmp_path / "journals"
        journals_dir.mkdir()

        graph = GraphPaths(tmp_path)
        assert graph.journal_exists("2025_01_15") is False

    def test_page_exists_true(self, tmp_path):
        """Test page_exists returns True when file exists."""
        pages_dir = tmp_path / "pages"
        pages_dir.mkdir()
        page_file = pages_dir / "Project X.md"
        page_file.write_text("- test")

        graph = GraphPaths(tmp_path)
        assert graph.page_exists("Project X") is True

    def test_page_exists_false(self, tmp_path):
        """Test page_exists returns False when file doesn't exist."""
        pages_dir = tmp_path / "pages"
        pages_dir.mkdir()

        graph = GraphPaths(tmp_path)
        assert graph.page_exists("Project X") is False

    def test_list_journals_empty(self, tmp_path):
        """Test list_journals with no journals."""
        graph = GraphPaths(tmp_path)
        journals = graph.list_journals()

        assert journals == []

    def test_list_journals_with_files(self, tmp_path):
        """Test list_journals returns all journal files sorted."""
        journals_dir = tmp_path / "journals"
        journals_dir.mkdir()

        # Create journals out of order
        (journals_dir / "2025_01_15.md").write_text("- test")
        (journals_dir / "2025_01_10.md").write_text("- test")
        (journals_dir / "2025_01_20.md").write_text("- test")

        graph = GraphPaths(tmp_path)
        journals = graph.list_journals()

        # Should be sorted
        assert len(journals) == 3
        assert journals[0].name == "2025_01_10.md"
        assert journals[1].name == "2025_01_15.md"
        assert journals[2].name == "2025_01_20.md"

    def test_list_pages_empty(self, tmp_path):
        """Test list_pages with no pages."""
        graph = GraphPaths(tmp_path)
        pages = graph.list_pages()

        assert pages == []

    def test_list_pages_with_files(self, tmp_path):
        """Test list_pages returns all page files sorted."""
        pages_dir = tmp_path / "pages"
        pages_dir.mkdir()

        # Create pages
        (pages_dir / "Project X.md").write_text("- test")
        (pages_dir / "Project A.md").write_text("- test")
        (pages_dir / "Zulu Project.md").write_text("- test")

        graph = GraphPaths(tmp_path)
        pages = graph.list_pages()

        # Should be sorted alphabetically
        assert len(pages) == 3
        assert pages[0].name == "Project A.md"
        assert pages[1].name == "Project X.md"
        assert pages[2].name == "Zulu Project.md"

    def test_list_journals_ignores_non_md_files(self, tmp_path):
        """Test that list_journals only returns .md files."""
        journals_dir = tmp_path / "journals"
        journals_dir.mkdir()

        (journals_dir / "2025_01_15.md").write_text("- test")
        (journals_dir / "2025_01_16.txt").write_text("not markdown")
        (journals_dir / "readme.md").write_text("readme")

        graph = GraphPaths(tmp_path)
        journals = graph.list_journals()

        # Should only return .md files (all of them in journals/)
        md_files = [j.name for j in journals]
        assert "2025_01_15.md" in md_files
        assert "readme.md" in md_files
        assert "2025_01_16.txt" not in md_files
