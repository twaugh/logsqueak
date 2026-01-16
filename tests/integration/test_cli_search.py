"""Integration tests for CLI search command.

Performance optimization: Uses a shared SentenceTransformer encoder across all
tests to avoid the 56-second model loading delay on each test. The encoder is
loaded once at module scope and injected via monkeypatching.
"""

import pytest
from pathlib import Path
from click.testing import CliRunner
from logsqueak.cli import cli
from logsqueak.services.page_indexer import PageIndexer
from logsqueak.services.rag_search import RAGSearch
from logseq_outline.parser import LogseqOutline
from logseq_outline.graph import GraphPaths


@pytest.fixture(scope="module")
def shared_encoder():
    """Load SentenceTransformer model once for all tests in this module.

    Without this optimization, each test would load the model independently,
    taking ~56 seconds per test. With sharing, only the first test is slow.
    """
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-mpnet-base-v2")


@pytest.fixture
def use_shared_encoder(shared_encoder, monkeypatch):
    """Inject shared encoder into PageIndexer and RAGSearch.

    This avoids reloading the SentenceTransformer model on each test.
    Apply this fixture only to tests in this module by using it as a parameter.
    """
    # Patch PageIndexer.encoder property to return shared encoder
    monkeypatch.setattr(PageIndexer, "_encoder", shared_encoder, raising=False)
    # Patch RAGSearch.encoder property to return shared encoder
    monkeypatch.setattr(RAGSearch, "_encoder", shared_encoder, raising=False)


@pytest.fixture
def temp_graph_with_pages(tmp_path):
    """Create a temporary Logseq graph with test pages."""
    # Create graph structure
    graph_path = tmp_path / "test-graph"
    pages_dir = graph_path / "pages"
    pages_dir.mkdir(parents=True)

    # Create test pages with different content
    page1 = pages_dir / "Machine Learning.md"
    page1.write_text(
        "- Machine learning best practices\n"
        "  - Always validate your models\n"
        "  - Use cross-validation for hyperparameter tuning\n"
        "  - Document your experiments thoroughly\n"
    )

    page2 = pages_dir / "Python.md"
    page2.write_text(
        "- Python programming tips\n"
        "  - Use type hints for better code clarity\n"
        "  - Async/await for concurrent operations\n"
        "  - Virtual environments for project isolation\n"
    )

    page3 = pages_dir / "Debugging.md"
    page3.write_text(
        "- Debugging async code\n"
        "  - Use asyncio debug mode\n"
        "  - Add logging to trace execution flow\n"
        "  - Use breakpoint() with asyncio-aware debuggers\n"
    )

    return graph_path


@pytest.fixture
def temp_config(tmp_path, temp_graph_with_pages):
    """Create a temporary config file."""
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "config.yaml"

    config_content = f"""
llm:
  endpoint: http://localhost:11434/v1
  api_key: dummy-key
  model: test-model

logseq:
  graph_path: {temp_graph_with_pages}

rag:
  top_k: 5
"""
    config_path.write_text(config_content)
    # Set correct permissions
    config_path.chmod(0o600)

    return config_path


def test_search_builds_index_on_first_run(temp_config, temp_graph_with_pages, use_shared_encoder, monkeypatch):
    """Test that search builds index on first run."""
    # Monkeypatch config path
    monkeypatch.setenv("HOME", str(temp_config.parent.parent))
    config_home = temp_config.parent.parent / ".config" / "logsqueak"
    config_home.mkdir(parents=True, exist_ok=True)
    (config_home / "config.yaml").write_text(temp_config.read_text())
    (config_home / "config.yaml").chmod(0o600)

    runner = CliRunner()
    result = runner.invoke(cli, ["search", "machine learning"])

    # Should build index on first run
    assert result.exit_code == 0
    assert "Building search index" in result.output or "Rebuilding search index" in result.output
    assert "Searching for: machine learning" in result.output


def test_search_with_reindex_flag(temp_config, temp_graph_with_pages, use_shared_encoder, monkeypatch):
    """Test search with --reindex flag forces rebuild."""
    # Monkeypatch config path
    monkeypatch.setenv("HOME", str(temp_config.parent.parent))
    config_home = temp_config.parent.parent / ".config" / "logsqueak"
    config_home.mkdir(parents=True, exist_ok=True)
    (config_home / "config.yaml").write_text(temp_config.read_text())
    (config_home / "config.yaml").chmod(0o600)

    runner = CliRunner()

    # First run to build index
    result1 = runner.invoke(cli, ["search", "test query"])
    assert result1.exit_code == 0

    # Second run with --reindex should rebuild
    result2 = runner.invoke(cli, ["search", "test query", "--reindex"])
    assert result2.exit_code == 0
    assert "Rebuilding search index" in result2.output


def test_search_returns_relevant_results(temp_config, temp_graph_with_pages, use_shared_encoder, monkeypatch):
    """Test that search returns relevant results."""
    # Monkeypatch config path
    monkeypatch.setenv("HOME", str(temp_config.parent.parent))
    config_home = temp_config.parent.parent / ".config" / "logsqueak"
    config_home.mkdir(parents=True, exist_ok=True)
    (config_home / "config.yaml").write_text(temp_config.read_text())
    (config_home / "config.yaml").chmod(0o600)

    runner = CliRunner()
    result = runner.invoke(cli, ["search", "machine learning best practices"])

    assert result.exit_code == 0
    # Should find the Machine Learning page
    assert "Machine Learning" in result.output or "Machine%20Learning" in result.output
    # Should show relevance percentage
    assert "Relevance:" in result.output
    assert "%" in result.output


def test_search_no_results(temp_config, temp_graph_with_pages, use_shared_encoder, monkeypatch):
    """Test search with query that has no matches."""
    # Monkeypatch config path
    monkeypatch.setenv("HOME", str(temp_config.parent.parent))
    config_home = temp_config.parent.parent / ".config" / "logsqueak"
    config_home.mkdir(parents=True, exist_ok=True)
    (config_home / "config.yaml").write_text(temp_config.read_text())
    (config_home / "config.yaml").chmod(0o600)

    runner = CliRunner()
    # Search for something completely unrelated
    result = runner.invoke(cli, ["search", "quantum physics advanced topology"])

    assert result.exit_code == 0
    # Either no results or very low confidence results
    # (ChromaDB always returns something, but confidence should be low)
    assert "Searching for:" in result.output


def test_search_missing_config(monkeypatch, tmp_path):
    """Test search fails gracefully with missing config."""
    # Point to non-existent config
    monkeypatch.setenv("HOME", str(tmp_path))

    runner = CliRunner()
    result = runner.invoke(cli, ["search", "test query"])

    assert result.exit_code != 0
    assert "Error:" in result.output


def test_search_displays_snippets(temp_config, temp_graph_with_pages, use_shared_encoder, monkeypatch):
    """Test that search displays content snippets."""
    # Monkeypatch config path
    monkeypatch.setenv("HOME", str(temp_config.parent.parent))
    config_home = temp_config.parent.parent / ".config" / "logsqueak"
    config_home.mkdir(parents=True, exist_ok=True)
    (config_home / "config.yaml").write_text(temp_config.read_text())
    (config_home / "config.yaml").chmod(0o600)

    runner = CliRunner()
    result = runner.invoke(cli, ["search", "async debugging"])

    assert result.exit_code == 0
    # Should show snippets from the Debugging page
    # The snippet should contain hierarchical context
    assert "Searching for:" in result.output


def test_search_respects_top_k_config(temp_config, temp_graph_with_pages, use_shared_encoder, monkeypatch):
    """Test that search respects top_k from config."""
    # Monkeypatch config path
    monkeypatch.setenv("HOME", str(temp_config.parent.parent))
    config_home = temp_config.parent.parent / ".config" / "logsqueak"
    config_home.mkdir(parents=True, exist_ok=True)

    # Modify config to set top_k to 2
    modified_config = temp_config.read_text().replace("top_k: 5", "top_k: 2")
    (config_home / "config.yaml").write_text(modified_config)
    (config_home / "config.yaml").chmod(0o600)

    runner = CliRunner()
    result = runner.invoke(cli, ["search", "programming"])

    assert result.exit_code == 0
    # Count numbered results (1., 2., 3., etc.)
    # Should have at most 2 results
    result_count = result.output.count("\n1.") + result.output.count("\n2.") + result.output.count("\n3.")
    assert result_count <= 2


def test_search_displays_reversed_numbering(temp_config, temp_graph_with_pages, use_shared_encoder, monkeypatch):
    """Test that search displays results in reverse order with reversed numbering.

    Most relevant result should appear last (closest to command prompt) and be numbered "1".
    Less relevant results should appear earlier with higher numbers.
    """
    # Monkeypatch config path
    monkeypatch.setenv("HOME", str(temp_config.parent.parent))
    config_home = temp_config.parent.parent / ".config" / "logsqueak"
    config_home.mkdir(parents=True, exist_ok=True)
    (config_home / "config.yaml").write_text(temp_config.read_text())
    (config_home / "config.yaml").chmod(0o600)

    runner = CliRunner()
    result = runner.invoke(cli, ["search", "programming"])

    # Print output for debugging
    print(f"\n=== CLI OUTPUT ===")
    print(result.output)
    print(f"=== EXIT CODE: {result.exit_code} ===\n")

    if result.exit_code != 0:
        print(f"=== EXCEPTION ===")
        if result.exception:
            import traceback
            traceback.print_exception(type(result.exception), result.exception, result.exception.__traceback__)
        print("=================\n")

    assert result.exit_code == 0

    # Parse the output to find result numbers and their relevance scores
    lines = result.output.split('\n')
    results = []

    for i, line in enumerate(lines):
        # Look for numbered results (e.g., "1. PageName" or "2. PageName")
        if line and line[0].isdigit() and '. ' in line:
            number = int(line.split('.')[0])
            # Next line should have "Relevance: XX%"
            if i + 1 < len(lines) and "Relevance:" in lines[i + 1]:
                relevance_str = lines[i + 1].split("Relevance:")[1].strip().rstrip('%')
                relevance = int(relevance_str)
                results.append((number, relevance))

    # Should have at least 2 results for meaningful test
    assert len(results) >= 2, f"Expected at least 2 results, got {len(results)}"

    # Most relevant result should be last in the list
    last_result = results[-1]
    first_result = results[0]

    # The last result (most relevant) should have number "1"
    assert last_result[0] == 1, f"Expected last result to be numbered 1, got {last_result[0]}"

    # The first result (least relevant) should have the highest number
    assert first_result[0] == len(results), f"Expected first result to be numbered {len(results)}, got {first_result[0]}"

    # Relevance should decrease as we go from last to first (reversed display)
    assert last_result[1] >= first_result[1], f"Expected last result ({last_result[1]}%) to be more relevant than first ({first_result[1]}%)"
