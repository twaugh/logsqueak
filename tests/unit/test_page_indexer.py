"""Unit tests for PageIndexer service."""

import pytest
from pathlib import Path
import tempfile
import shutil
from logsqueak.services.page_indexer import PageIndexer, generate_graph_db_name
from logseq_outline.graph import GraphPaths
from logseq_outline.parser import LogseqOutline


@pytest.fixture
def temp_graph(tmp_path):
    """Create a temporary Logseq graph for testing."""
    graph_path = tmp_path / "test-graph"
    graph_path.mkdir()

    pages_dir = graph_path / "pages"
    pages_dir.mkdir()

    # Create some test pages
    page1 = pages_dir / "Test Page 1.md"
    page1.write_text("""- First block in page 1
  - Child block
- Second block in page 1
""")

    page2 = pages_dir / "Test Page 2.md"
    page2.write_text("""- Knowledge about Python
  - Python is a programming language
- Another block
""")

    page3 = pages_dir / "Projects___Acme.md"  # Hierarchical page
    page3.write_text("""- Project overview
- Timeline
  - Q1 2025
  - Q2 2025
""")

    return graph_path


@pytest.fixture
def temp_db(tmp_path):
    """Create temporary ChromaDB directory."""
    return tmp_path / "chromadb"


def test_generate_graph_db_name_basic():
    """Test basic database directory name generation."""
    graph_path = Path("/home/user/Documents/my-graph")
    db_name = generate_graph_db_name(graph_path)

    # Should contain basename
    assert "my-graph" in db_name or "my_graph" in db_name

    # Should end with hyphen + 16-character hex hash
    assert "-" in db_name
    parts = db_name.rsplit("-", 1)
    assert len(parts) == 2
    assert len(parts[1]) == 16
    assert all(c in "0123456789abcdef" for c in parts[1])


def test_generate_graph_db_name_preserves_basename():
    """Test that basename is preserved as-is (since Path already validates it)."""
    graph_path = Path("/home/user/Documents/my_graph")
    db_name = generate_graph_db_name(graph_path)

    # Should contain the exact basename
    assert "my_graph" in db_name

    # Should have the expected format
    assert db_name.startswith("my_graph-")
    hash_part = db_name.split("-", 1)[1]
    assert len(hash_part) == 16
    assert all(c in "0123456789abcdef" for c in hash_part)


def test_generate_graph_db_name_different_paths_different_hashes():
    """Test that different graph paths produce different hashes."""
    graph_path1 = Path("/home/user/Documents/graph1")
    graph_path2 = Path("/home/user/Documents/graph2")

    name1 = generate_graph_db_name(graph_path1)
    name2 = generate_graph_db_name(graph_path2)

    # Different paths should produce different directory names
    assert name1 != name2

    # Hashes (last part) should be different
    hash1 = name1.rsplit("-", 1)[-1]
    hash2 = name2.rsplit("-", 1)[-1]
    assert hash1 != hash2


def test_generate_graph_db_name_same_path_same_hash():
    """Test that same path produces same hash consistently."""
    graph_path = Path("/home/user/Documents/my-graph")

    name1 = generate_graph_db_name(graph_path)
    name2 = generate_graph_db_name(graph_path)

    # Same path should produce identical directory names
    assert name1 == name2


def test_generate_graph_db_name_same_basename_different_paths():
    """Test that graphs with same basename but different paths get different directories."""
    graph_path1 = Path("/home/user/Documents/my-graph")
    graph_path2 = Path("/home/other/Projects/my-graph")

    name1 = generate_graph_db_name(graph_path1)
    name2 = generate_graph_db_name(graph_path2)

    # Should have different directory names due to different full paths
    assert name1 != name2

    # But both should contain the basename
    assert "my-graph" in name1 or "my_graph" in name1
    assert "my-graph" in name2 or "my_graph" in name2


@pytest.mark.asyncio
async def test_page_indexer_initialization(temp_graph, temp_db, shared_sentence_transformer):
    """Test PageIndexer initialization."""
    graph_paths = GraphPaths(temp_graph)
    indexer = PageIndexer(graph_paths, db_path=temp_db, encoder=shared_sentence_transformer)

    assert indexer.graph_paths.graph_path == temp_graph
    # db_path should be a subdirectory under temp_db
    assert indexer.db_path.parent == temp_db
    assert indexer.collection is not None

    await indexer.close()


@pytest.mark.asyncio
async def test_page_indexer_uses_per_graph_directory(temp_graph, temp_db, shared_sentence_transformer):
    """Test that PageIndexer creates per-graph ChromaDB directories."""
    graph_paths = GraphPaths(temp_graph)
    indexer = PageIndexer(graph_paths, db_path=temp_db, encoder=shared_sentence_transformer)

    # Database path should be under the base path with graph-specific subdirectory
    assert indexer.db_path.parent == temp_db

    # Directory name should be based on graph path
    expected_db_name = generate_graph_db_name(temp_graph)
    assert indexer.db_path.name == expected_db_name

    # Directory should contain the graph basename
    assert "test-graph" in indexer.db_path.name or "test_graph" in indexer.db_path.name

    # Directory should exist
    assert indexer.db_path.exists()
    assert indexer.db_path.is_dir()

    await indexer.close()


@pytest.mark.asyncio
async def test_page_indexer_different_graphs_different_directories(tmp_path, shared_sentence_transformer):
    """Test that different graphs get different ChromaDB directories."""
    # Create two different graphs
    graph1 = tmp_path / "graph1"
    graph1.mkdir()
    (graph1 / "pages").mkdir()
    (graph1 / "pages" / "test.md").write_text("- Test content")

    graph2 = tmp_path / "graph2"
    graph2.mkdir()
    (graph2 / "pages").mkdir()
    (graph2 / "pages" / "test.md").write_text("- Test content")

    # Create shared db base path
    db_base = tmp_path / "chromadb"

    # Initialize indexers for both graphs
    graph_paths1 = GraphPaths(graph1)
    indexer1 = PageIndexer(graph_paths1, db_path=db_base, encoder=shared_sentence_transformer)

    graph_paths2 = GraphPaths(graph2)
    indexer2 = PageIndexer(graph_paths2, db_path=db_base, encoder=shared_sentence_transformer)

    # Should have different directories
    assert indexer1.db_path != indexer2.db_path

    # Both should be under the base path
    assert indexer1.db_path.parent == db_base
    assert indexer2.db_path.parent == db_base

    # Both directories should exist
    assert indexer1.db_path.exists()
    assert indexer2.db_path.exists()

    await indexer1.close()
    await indexer2.close()


@pytest.mark.asyncio
async def test_build_index_creates_embeddings(temp_graph, temp_db, shared_sentence_transformer):
    """Test that build_index creates embeddings for all pages."""
    graph_paths = GraphPaths(temp_graph)
    indexer = PageIndexer(graph_paths, temp_db, encoder=shared_sentence_transformer)

    # Build index
    await indexer.build_index()

    # Verify collection has entries
    count = indexer.collection.count()
    assert count > 0  # Should have indexed some blocks

    # Check that we can query the collection
    results = indexer.collection.get()
    assert len(results["ids"]) > 0

    await indexer.close()


@pytest.mark.asyncio
async def test_build_index_with_progress_callback(temp_graph, temp_db, shared_sentence_transformer):
    """Test that progress callback is called during indexing."""
    graph_paths = GraphPaths(temp_graph)
    indexer = PageIndexer(graph_paths, temp_db, encoder=shared_sentence_transformer)

    progress_updates = []

    def progress_callback(current: int, total: int):
        progress_updates.append((current, total))

    await indexer.build_index(progress_callback)

    # Verify progress callback was called
    assert len(progress_updates) > 0

    # Verify final progress shows completion
    final_current, final_total = progress_updates[-1]
    assert final_current == final_total

    await indexer.close()


@pytest.mark.asyncio
async def test_incremental_indexing_skips_unmodified_pages(temp_graph, temp_db, shared_sentence_transformer):
    """Test that incremental indexing skips unmodified pages."""
    graph_paths = GraphPaths(temp_graph)
    indexer = PageIndexer(graph_paths, temp_db, encoder=shared_sentence_transformer)

    # First build
    await indexer.build_index()
    first_count = indexer.collection.count()

    # Second build (nothing changed)
    await indexer.build_index()
    second_count = indexer.collection.count()

    # Count should be the same (no duplicates)
    assert first_count == second_count

    await indexer.close()


@pytest.mark.asyncio
async def test_incremental_indexing_updates_modified_pages(temp_graph, temp_db, shared_sentence_transformer):
    """Test that incremental indexing updates modified pages."""
    graph_paths = GraphPaths(temp_graph)
    indexer = PageIndexer(graph_paths, temp_db, encoder=shared_sentence_transformer)

    # First build
    await indexer.build_index()

    # Modify a page
    page1 = temp_graph / "pages" / "Test Page 1.md"
    page1.write_text("""- Modified first block
- New second block
- Another new block
""")

    # Second build
    await indexer.build_index()

    # Query for the page
    results = indexer.collection.get(
        where={"page_name": "Test Page 1"}
    )

    # Should have results for the modified page
    assert len(results["ids"]) > 0

    await indexer.close()


@pytest.mark.asyncio
async def test_index_hierarchical_pages(temp_graph, temp_db, shared_sentence_transformer):
    """Test that hierarchical pages (with ___) are indexed correctly."""
    graph_paths = GraphPaths(temp_graph)
    indexer = PageIndexer(graph_paths, temp_db, encoder=shared_sentence_transformer)

    await indexer.build_index()

    # Query for hierarchical page
    results = indexer.collection.get(
        where={"page_name": "Projects/Acme"}
    )

    # Should have indexed blocks from Projects/Acme page
    assert len(results["ids"]) > 0

    await indexer.close()


@pytest.mark.asyncio
async def test_build_index_raises_on_missing_pages_dir(temp_graph, temp_db, shared_sentence_transformer):
    """Test that build_index raises error if pages directory doesn't exist."""
    # Remove pages directory
    pages_dir = temp_graph / "pages"
    shutil.rmtree(pages_dir)

    graph_paths = GraphPaths(temp_graph)
    indexer = PageIndexer(graph_paths, temp_db, encoder=shared_sentence_transformer)

    with pytest.raises(ValueError, match="Pages directory not found"):
        await indexer.build_index()

    await indexer.close()


@pytest.mark.asyncio
async def test_build_index_raises_on_empty_pages_dir(temp_graph, temp_db, shared_sentence_transformer):
    """Test that build_index raises error if pages directory is empty."""
    # Remove all pages
    pages_dir = temp_graph / "pages"
    for page_file in pages_dir.glob("*.md"):
        page_file.unlink()

    graph_paths = GraphPaths(temp_graph)
    indexer = PageIndexer(graph_paths, temp_db, encoder=shared_sentence_transformer)

    with pytest.raises(ValueError, match="No pages found"):
        await indexer.build_index()

    await indexer.close()


@pytest.mark.asyncio
async def test_index_blocks_with_children(temp_graph, temp_db, shared_sentence_transformer):
    """Test that nested blocks are indexed correctly."""
    graph_paths = GraphPaths(temp_graph)
    indexer = PageIndexer(graph_paths, temp_db, encoder=shared_sentence_transformer)

    await indexer.build_index()

    # Check that child blocks were indexed
    results = indexer.collection.get()

    # Should have multiple blocks (including children)
    assert len(results["ids"]) >= 6  # At least the blocks we created

    await indexer.close()


@pytest.mark.asyncio
async def test_embeddings_are_stored(temp_graph, temp_db, shared_sentence_transformer):
    """Test that embeddings are actually stored in the collection."""
    graph_paths = GraphPaths(temp_graph)
    indexer = PageIndexer(graph_paths, temp_db, encoder=shared_sentence_transformer)

    await indexer.build_index()

    # Query and check embeddings
    results = indexer.collection.get(include=["embeddings"])

    # Should have embeddings
    assert results["embeddings"] is not None
    assert len(results["embeddings"]) > 0

    # Each embedding should be a list or array of floats
    first_embedding = results["embeddings"][0]
    assert isinstance(first_embedding, (list, type(results["embeddings"][0])))
    assert len(first_embedding) > 0  # Non-empty vector

    await indexer.close()
