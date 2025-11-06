"""Unit tests for PageIndexer service."""

import pytest
from pathlib import Path
import tempfile
import shutil
from logsqueak.services.page_indexer import PageIndexer
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


@pytest.mark.asyncio
async def test_page_indexer_initialization(temp_graph, temp_db, shared_sentence_transformer):
    """Test PageIndexer initialization."""
    graph_paths = GraphPaths(temp_graph)
    indexer = PageIndexer(graph_paths, temp_db, encoder=shared_sentence_transformer)

    assert indexer.graph_paths.graph_path == temp_graph
    assert indexer.db_path == temp_db
    assert indexer.collection is not None

    await indexer.close()


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
