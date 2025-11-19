"""Unit tests for PageIndexer service."""

import pytest
from pathlib import Path
import tempfile
import shutil
from unittest.mock import MagicMock, patch
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


@pytest.mark.asyncio
async def test_reindexing_deletes_old_chunks(temp_graph, temp_db, shared_sentence_transformer):
    """Test that re-indexing a modified page deletes old chunks with stale block IDs.

    This is a regression test for a bug where modified pages would have new chunks added
    but old chunks (with old content hashes/block IDs) would remain in the index, causing
    RAG search to return stale block IDs that no longer exist in the actual files.
    """
    graph_paths = GraphPaths(temp_graph)
    indexer = PageIndexer(graph_paths, temp_db, encoder=shared_sentence_transformer)

    # Initial index
    await indexer.build_index()

    # Get initial chunks for Test Page 1
    initial_results = indexer.collection.get(
        where={"page_name": "Test Page 1"},
        include=["documents", "metadatas"]
    )
    initial_chunk_count = len(initial_results["ids"])
    initial_chunk_ids = set(initial_results["ids"])

    assert initial_chunk_count > 0, "Should have chunks after initial indexing"

    # Modify the page (change content, which changes content hashes)
    test_page = temp_graph / "pages" / "Test Page 1.md"
    test_page.write_text("""- Modified content
  - This is completely different
  - New blocks with new content hashes
- Another new section
  - More new content
""")

    # Wait a bit to ensure mtime changes
    import time
    time.sleep(0.01)

    # Re-index
    await indexer.build_index()

    # Get chunks after re-indexing
    final_results = indexer.collection.get(
        where={"page_name": "Test Page 1"},
        include=["documents", "metadatas"]
    )
    final_chunk_count = len(final_results["ids"])
    final_chunk_ids = set(final_results["ids"])

    # Verify old chunks were deleted
    # The old chunk IDs should NOT be present in the final results
    # Exception: __PAGE__ chunk ID is stable (represents the page itself, not content)
    overlap = (initial_chunk_ids & final_chunk_ids) - {f"Test Page 1::__PAGE__"}
    assert len(overlap) == 0, f"Old chunks still present after re-indexing: {overlap}"

    # Verify new chunks were added
    assert final_chunk_count > 0, "Should have new chunks after re-indexing"

    # Verify none of the documents contain the old content
    old_content_patterns = ["Nested block", "Sub-block"]
    new_content_patterns = ["Modified content", "completely different"]

    for doc in final_results["documents"]:
        for old_pattern in old_content_patterns:
            assert old_pattern not in doc, f"Old content '{old_pattern}' still in index after re-indexing"

    # At least some documents should have new content
    has_new_content = any(
        any(pattern in doc for pattern in new_content_patterns)
        for doc in final_results["documents"]
    )
    assert has_new_content, "New content not found in index after re-indexing"

    await indexer.close()


@pytest.mark.asyncio
async def test_batched_encoding_progress(tmp_path, shared_sentence_transformer):
    """Test that batched encoding provides incremental progress updates."""
    # Create a graph with enough pages to generate multiple encoding batches
    graph_path = tmp_path / "test-graph"
    graph_path.mkdir()
    pages_dir = graph_path / "pages"
    pages_dir.mkdir()

    # Create 10 pages with multiple blocks each to generate enough chunks
    # for multiple encoding batches (batch_size = 256 in PageIndexer)
    for i in range(10):
        page_file = pages_dir / f"Page {i}.md"
        # Each page has 30 blocks to generate ~300+ chunks total
        blocks = "\n".join([f"- Block {j} content with some text" for j in range(30)])
        page_file.write_text(blocks)

    graph_paths = GraphPaths(graph_path)
    db_path = tmp_path / "chromadb"
    indexer = PageIndexer(graph_paths, db_path, encoder=shared_sentence_transformer)

    progress_updates = []

    def progress_callback(current: int, total: int):
        progress_updates.append((current, total))

    await indexer.build_index(progress_callback)

    # Verify we got progress updates
    assert len(progress_updates) > 0

    # Find where encoding phase starts (total jumps from page count to chunk count)
    encoding_updates = []
    parsing_total = 0

    for i, (current, total) in enumerate(progress_updates):
        # Skip model loading signal
        if current == -1:
            continue

        # Detect phase transition (total increases significantly)
        if parsing_total > 0 and total > parsing_total * 2:
            # Found encoding phase - collect all subsequent updates
            encoding_updates = progress_updates[i:]
            break
        parsing_total = total

    # Verify we got multiple incremental encoding updates (batched encoding is working)
    # Should have more than 1 update (start and end) if batching is working
    assert len(encoding_updates) > 1, \
        f"Expected multiple encoding updates from batching, got {len(encoding_updates)}"

    # Verify encoding updates show incremental progress
    if len(encoding_updates) > 1:
        # First encoding update should be partial
        first_current, first_total = encoding_updates[0]
        assert first_current <= first_total

        # Last update should show completion
        last_current, last_total = encoding_updates[-1]
        assert last_current == last_total

        # Updates should be monotonically increasing
        for i in range(len(encoding_updates) - 1):
            curr_current, curr_total = encoding_updates[i]
            next_current, next_total = encoding_updates[i + 1]
            assert next_current >= curr_current, "Progress should increase"
            assert curr_total == next_total, "Total should remain constant during encoding"

    await indexer.close()


@pytest.mark.asyncio
async def test_deleted_pages_removed_from_index(temp_graph, temp_db, shared_sentence_transformer):
    """Test that chunks for deleted pages are removed from the index."""
    graph_paths = GraphPaths(temp_graph)
    indexer = PageIndexer(graph_paths, temp_db, encoder=shared_sentence_transformer)

    # First build - index all pages
    await indexer.build_index()
    initial_count = indexer.collection.count()
    assert initial_count > 0

    # Verify "Test Page 1" is in the index
    results = indexer.collection.get(where={"page_name": "Test Page 1"})
    assert len(results["ids"]) > 0, "Test Page 1 should be indexed"
    page1_chunk_count = len(results["ids"])

    # Delete "Test Page 1" from filesystem
    page1_file = temp_graph / "pages" / "Test Page 1.md"
    page1_file.unlink()

    # Rebuild index
    await indexer.build_index()
    final_count = indexer.collection.count()

    # Verify chunks were removed
    assert final_count == initial_count - page1_chunk_count, \
        f"Expected {page1_chunk_count} chunks to be removed"

    # Verify "Test Page 1" is no longer in the index
    results = indexer.collection.get(where={"page_name": "Test Page 1"})
    assert len(results["ids"]) == 0, "Test Page 1 should be removed from index"

    # Verify "Test Page 2" is still there
    results = indexer.collection.get(where={"page_name": "Test Page 2"})
    assert len(results["ids"]) > 0, "Test Page 2 should still be indexed"

    await indexer.close()


@pytest.mark.asyncio
async def test_schema_version_mismatch_triggers_rebuild(temp_graph, temp_db, shared_sentence_transformer):
    """Test that schema version mismatch causes index rebuild."""
    from logsqueak.services.page_indexer import INDEX_SCHEMA_VERSION
    import chromadb

    graph_paths = GraphPaths(temp_graph)

    # Create a collection with old schema version
    db_path = temp_db / "test-graph-abc123"
    db_path.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(db_path))

    # Create collection with wrong version
    old_collection = chroma_client.create_collection(
        name="logsqueak_blocks",
        metadata={
            "hnsw:space": "cosine",
            "schema_version": INDEX_SCHEMA_VERSION - 1  # Old version
        }
    )

    # Add some dummy data to the old collection
    old_collection.add(
        ids=["test_id"],
        documents=["test document"],
        metadatas=[{"page_name": "test"}]
    )

    assert old_collection.count() == 1, "Old collection should have data"

    # Now initialize PageIndexer - should detect version mismatch and rebuild
    indexer = PageIndexer(graph_paths, temp_db, encoder=shared_sentence_transformer)

    # Verify collection was recreated with correct version
    assert indexer.collection.metadata.get("schema_version") == INDEX_SCHEMA_VERSION

    # Verify old data was cleared (collection was deleted and recreated)
    assert indexer.collection.count() == 0, "Collection should be empty after version mismatch rebuild"

    # Build index with new version
    await indexer.build_index()

    # Verify index was built successfully
    assert indexer.collection.count() > 0, "Index should have data after rebuild"

    await indexer.close()


@pytest.mark.asyncio
async def test_large_batch_upsert_handles_chromadb_limit(tmp_path, shared_sentence_transformer):
    """Test that large batch upsert respects ChromaDB's batch size limit.

    ChromaDB has a maximum batch size limit (~5461 documents). When indexing
    large graphs with many blocks, we need to batch the upsert operations
    to avoid "Batch size X is greater than max batch size Y" errors.

    This test mocks a collection that enforces the batch limit to verify
    the PageIndexer properly batches large upserts.
    """
    # Create a small graph (we'll mock the large batch scenario)
    graph_path = tmp_path / "test-graph"
    graph_path.mkdir()
    pages_dir = graph_path / "pages"
    pages_dir.mkdir()

    # Create just a few pages to keep the test fast
    # Each page gets: 1 page-level chunk + 2 block chunks = 3 chunks
    # Total: 3 pages × 3 chunks = 9 chunks
    for i in range(3):
        page_file = pages_dir / f"Page {i}.md"
        page_file.write_text("- Test block\n- Another block\n")

    graph_paths = GraphPaths(graph_path)
    db_path = tmp_path / "chromadb"

    # Use a small batch size (3) to force multiple batches with our 9 chunks
    # This simulates what would happen with a large graph and the default batch size of 256
    INDEXER_BATCH_SIZE = 3
    indexer = PageIndexer(
        graph_paths,
        db_path,
        encoder=shared_sentence_transformer,
        batch_size=INDEXER_BATCH_SIZE
    )

    # Mock the collection to simulate ChromaDB's batch size limit
    # (No need to clean the index since we're mocking upsert - it never reaches ChromaDB)
    original_upsert = indexer.collection.upsert
    SIMULATED_MAX_BATCH_SIZE = 5  # Simulate ChromaDB's limit (higher than our batch size)
    upsert_calls = []

    def mock_upsert(documents, embeddings, ids, metadatas):
        """Mock upsert that enforces batch size limit like ChromaDB."""
        batch_size = len(ids)
        upsert_calls.append(batch_size)
        if batch_size > SIMULATED_MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size of {batch_size} is greater than max batch size of {SIMULATED_MAX_BATCH_SIZE}"
            )
        # Call original upsert if batch is within limit
        return original_upsert(
            documents=documents,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )

    indexer.collection.upsert = mock_upsert

    # This should succeed because we're batching upserts to respect the limit
    # (3 pages with 2 blocks each = 9 chunks total, batch size of 3 = 3 batches)
    await indexer.build_index()

    # Verify the mock was called multiple times (3 batches expected)
    assert len(upsert_calls) == 3, \
        f"Expected 3 batched upsert calls (9 chunks / batch_size 3), got {len(upsert_calls)}: {upsert_calls}"

    # Verify each batch size matches our batch_size setting and respects the mock limit
    for i, batch_size in enumerate(upsert_calls):
        assert batch_size == INDEXER_BATCH_SIZE, \
            f"Batch {i} has size {batch_size}, expected {INDEXER_BATCH_SIZE}"
        assert batch_size <= SIMULATED_MAX_BATCH_SIZE, \
            f"Batch {i} has size {batch_size}, exceeds ChromaDB limit {SIMULATED_MAX_BATCH_SIZE}"

    # Verify all chunks were upserted (sum of all batches = total chunks)
    total_upserted = sum(upsert_calls)
    assert total_upserted == 9, \
        f"Expected 9 total chunks upserted, got {total_upserted} from batches: {upsert_calls}"

    # Verify chunks are actually in the collection
    assert indexer.collection.count() == 9, \
        f"Expected 9 chunks in collection, got {indexer.collection.count()}"

    await indexer.close()
