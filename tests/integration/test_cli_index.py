"""Integration tests for CLI index commands (M2.7)."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from logsqueak.cli.main import cli


@pytest.fixture
def mock_embedding_model():
    """Create a mock embedding model for testing."""
    mock_model = Mock()

    def encode_mock(texts, convert_to_numpy=False):
        import numpy as np
        if isinstance(texts, str):
            return np.array([float(len(texts)) / 100.0] * 384)
        return np.array([[float(len(t)) / 100.0] * 384 for t in texts])

    mock_model.encode = encode_mock
    return mock_model


@pytest.fixture
def test_graph(tmp_path):
    """Create a test Logseq graph."""
    graph_path = tmp_path / "graph"
    pages_dir = graph_path / "pages"
    pages_dir.mkdir(parents=True)

    # Create test pages
    (pages_dir / "Page A.md").write_text("- Content A", encoding="utf-8")
    (pages_dir / "Page B.md").write_text("- Content B", encoding="utf-8")
    (pages_dir / "Page C.md").write_text("- Content C", encoding="utf-8")

    return graph_path


@pytest.fixture
def test_config(tmp_path, test_graph):
    """Create a test config file."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_file = config_dir / "config.yaml"

    config_content = f"""
llm:
  endpoint: http://localhost:11434/v1
  api_key: test-key
  model: test-model

logseq:
  graph_path: {test_graph}

rag:
  token_budget: null
"""
    config_file.write_text(config_content, encoding="utf-8")
    return config_file


class TestIndexRebuild:
    """Tests for 'logsqueak index rebuild' command."""

    def test_rebuild_basic(self, test_config, test_graph, mock_embedding_model):
        """Test basic index rebuild."""
        runner = CliRunner()

        # Patch IndexBuilder to use mock model
        from logsqueak.rag import indexer as indexer_module
        original_init = indexer_module.IndexBuilder.__init__

        def mock_init(self, vector_store, manifest, embedding_model=None):
            # Call original with mock embedding model
            original_init(self, vector_store, manifest, embedding_model=mock_embedding_model)

        with patch.object(indexer_module.IndexBuilder, '__init__', mock_init):
            result = runner.invoke(cli, ["--config", str(test_config), "index", "rebuild"])

        assert result.exit_code == 0, f"Output: {result.output}"
        assert "Building index for 3 pages" in result.output
        assert "✓ Index built" in result.output

    def test_rebuild_with_graph_override(self, test_config, tmp_path, mock_embedding_model):
        """Test rebuild with --graph override."""
        # Create alternate graph
        alt_graph = tmp_path / "alt_graph"
        pages_dir = alt_graph / "pages"
        pages_dir.mkdir(parents=True)
        (pages_dir / "Alt.md").write_text("- Alt content", encoding="utf-8")

        runner = CliRunner()

        from logsqueak.rag import indexer as indexer_module
        original_init = indexer_module.IndexBuilder.__init__

        def mock_init(self, vector_store, manifest, embedding_model=None):
            original_init(self, vector_store, manifest, embedding_model=mock_embedding_model)

        with patch.object(indexer_module.IndexBuilder, '__init__', mock_init):
            result = runner.invoke(
                cli, ["--config", str(test_config), "index", "rebuild", "--graph", str(alt_graph)]
            )

        assert result.exit_code == 0, f"Output: {result.output}"
        assert "Building index for 1 pages" in result.output

    def test_rebuild_verbose(self, test_config, test_graph, mock_embedding_model):
        """Test rebuild with --verbose flag."""
        runner = CliRunner()

        from logsqueak.rag import indexer as indexer_module
        original_init = indexer_module.IndexBuilder.__init__

        def mock_init(self, vector_store, manifest, embedding_model=None):
            original_init(self, vector_store, manifest, embedding_model=mock_embedding_model)

        with patch.object(indexer_module.IndexBuilder, '__init__', mock_init):
            result = runner.invoke(cli, ["--config", str(test_config), "--verbose", "index", "rebuild"])

        assert result.exit_code == 0, f"Output: {result.output}"
        assert "Loading configuration" in result.output


class TestIndexStatus:
    """Tests for 'logsqueak index status' command."""

    def test_status_no_index(self, test_config, test_graph, tmp_path, monkeypatch):
        """Test status when no index exists."""
        runner = CliRunner()

        # Redirect cache to tmp_path to avoid picking up real cache
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        result = runner.invoke(cli, ["--config", str(test_config), "index", "status"])

        assert result.exit_code == 0
        assert "Graph:" in result.output
        assert "Pages: 3" in result.output
        assert "No index found" in result.output

    def test_status_with_index(self, test_config, test_graph, mock_embedding_model, monkeypatch, tmp_path):
        """Test status with existing index."""
        runner = CliRunner()

        # Redirect cache to tmp_path for both build and status
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        # First, build an index
        from logsqueak.rag import indexer as indexer_module
        original_init = indexer_module.IndexBuilder.__init__

        def mock_init(self, vector_store, manifest, embedding_model=None):
            original_init(self, vector_store, manifest, embedding_model=mock_embedding_model)

        with patch.object(indexer_module.IndexBuilder, '__init__', mock_init):
            # Build the index first
            build_result = runner.invoke(cli, ["--config", str(test_config), "index", "rebuild"])
            assert build_result.exit_code == 0, f"Build output: {build_result.output}"

        # Check status (should now find the index)
        result = runner.invoke(cli, ["--config", str(test_config), "index", "status"])

        assert result.exit_code == 0
        assert "Graph:" in result.output
        assert "Pages: 3" in result.output
        assert "Indexed pages: 3" in result.output

    def test_status_with_graph_override(self, test_config, tmp_path):
        """Test status with --graph override."""
        # Create alternate graph
        alt_graph = tmp_path / "alt_graph"
        pages_dir = alt_graph / "pages"
        pages_dir.mkdir(parents=True)
        (pages_dir / "Alt1.md").write_text("- Alt content 1", encoding="utf-8")
        (pages_dir / "Alt2.md").write_text("- Alt content 2", encoding="utf-8")

        runner = CliRunner()

        result = runner.invoke(
            cli, ["--config", str(test_config), "index", "status", "--graph", str(alt_graph)]
        )

        assert result.exit_code == 0
        assert "Pages: 2" in result.output

    def test_status_verbose(self, test_config, test_graph):
        """Test status with --verbose flag."""
        runner = CliRunner()

        result = runner.invoke(cli, ["--config", str(test_config), "--verbose", "index", "status"])

        assert result.exit_code == 0
        assert "Graph:" in result.output


class TestIndexHelp:
    """Tests for index command help."""

    def test_index_help(self):
        """Test 'logsqueak index --help' shows available commands."""
        runner = CliRunner()

        result = runner.invoke(cli, ["index", "--help"])

        assert result.exit_code == 0
        assert "rebuild" in result.output
        assert "status" in result.output

    def test_rebuild_help(self):
        """Test 'logsqueak index rebuild --help'."""
        runner = CliRunner()

        result = runner.invoke(cli, ["index", "rebuild", "--help"])

        assert result.exit_code == 0
        assert "--force" in result.output
        assert "--graph" in result.output

    def test_status_help(self):
        """Test 'logsqueak index status --help'."""
        runner = CliRunner()

        result = runner.invoke(cli, ["index", "status", "--help"])

        assert result.exit_code == 0
        assert "--graph" in result.output
