# Implementation Plan: Current State → FUTURE-STATE

## Executive Summary

This plan bridges the gap between our **current implementation** (basic 2-stage pipeline with RAG) and the **FUTURE-STATE** vision (comprehensive 5-phase pipeline with vector store, hybrid IDs, and advanced semantic merge).

**Current State**: Working proof-of-concept (69% complete)
- 2-stage LLM pipeline (extract → RAG match)
- Basic integration (ADD_CHILD only)
- No persistent vector store (embeddings cached per session)
- Simple content hashing for duplicates

**Future State**: Production-ready semantic merge system
- 5-phase pipeline with persistent vector store
- Hybrid-ID system (persistent IDs + content hashes)
- Full CRUD operations (UPDATE, APPEND, CREATE_SECTION)
- Round-trip-safe AST with block targeting

---

## Key Architectural Changes Needed

### 1. **Persistent Vector Store** (Phase 0)
**Current**: Per-page embedding cache (pkl files), no block-level indexing
**Future**: ChromaDB with hybrid-ID system for every block

**Gap**: Need to implement full graph indexing with incremental updates

### 2. **Hybrid-ID System**
**Current**: No persistent block IDs, content-based matching
**Future**: Dual ID system (`id::` properties OR content hashes)

**Gap**: Parser needs to extract/preserve `id::`, writer needs to generate UUIDs

### 3. **AST-Based Targeting** (Phase 4)
**Current**: Simple section finding by heading text
**Future**: `find_target_node_by_id()` for precise block targeting

**Gap**: Need block-level ID generation and lookup by hybrid ID

### 4. **Round-Trip Safety**
**Current**: Parser → Renderer preserves property order ✅
**Future**: Must also preserve formatting, maintain node identity

**Gap**: Enhance parser to track source positions, test round-trip fidelity

### 5. **LLM Pipeline Stages**
**Current**: 2 stages (Stage 1: extract, Stage 2: select page)
**Future**: 5 phases (Phase 0-4 as specified)

**Gap**: Add Phase 3 (Decider + Reworder), Phase 4 (execution with cleanup)

---

## Incremental Implementation Plan

### **Milestone 1: Hybrid-ID Foundation** ✅ (5 tasks - COMPLETE)

Build the infrastructure for persistent block IDs and full-context chunk generation. This lays the groundwork for both hybrid IDs and the chunking system needed in M2.

**Status**: Complete (commits: 41c14c9, d1162d8, 332c4e5, 8b72c5c, 03d21c1)

#### M1.1: Enhance Parser to Extract `id::` Properties ✅
- **File**: `src/logsqueak/logseq/parser.py`
- **Task**: ✅ Modified `LogseqBlock` to store `block_id: Optional[str]`
- **Task**: ✅ Parse `id::` property and populate `block_id` field
- **Test**: ✅ 7 unit tests for parsing blocks with/without `id::`
- **Commit**: 41c14c9

#### M1.2: Implement Full-Context Generation and Hashing ✅
- **File**: New `src/logsqueak/logseq/context.py`
- **Task**: ✅ Created `generate_full_context(block, parents) -> str` helper (includes bullet markers)
- **Task**: ✅ Recursive traversal to build full context (prepend parent context)
- **Task**: ✅ Created `generate_content_hash(full_context: str) -> str` using MD5
- **Task**: ✅ Created `generate_chunks(outline) -> List[Tuple[block, full_context, hybrid_id]]`
  - This will be reused by M2.3 for chunking
- **Test**: ✅ 18 unit tests for full-context generation with nested blocks
- **Test**: ✅ Unit test for hash stability and collision resistance
- **Commit**: d1162d8 (amended to include bullet markers)

#### M1.3: Add Hybrid ID to LogseqBlock ✅
- **File**: `src/logsqueak/logseq/parser.py`
- **Task**: ✅ Added method `get_hybrid_id() -> str` to `LogseqBlock`
- **Task**: ✅ Returns `block_id` if present, else content hash
- **Task**: ✅ Added `find_block_by_id()` to LogseqOutline (optimized with generate_chunks)
- **Test**: ✅ 8 unit tests for hybrid ID resolution
- **Commit**: 332c4e5

#### M1.4: Implement UUID Generation in Writer ✅
- **File**: `src/logsqueak/integration/writer.py`
- **Task**: ✅ Generate new UUID for each integrated block
- **Task**: ✅ Append `id:: <uuid>` to new blocks (in both properties dict and continuation_lines)
- **Test**: ✅ 8 unit tests for ID generation and formatting
- **Commit**: 8b72c5c

#### M1.5: Round-Trip Safety Tests ✅
- **File**: `tests/integration/test_parsing_roundtrip.py`
- **Task**: ✅ 8 tests for parse → modify → render → parse preserves IDs
- **Task**: ✅ Test that existing `id::` properties are never modified
- **Test**: ✅ Property order preservation (already exists ✅)
- **Commit**: 03d21c1

---

### **Milestone 2: Persistent Vector Store** ✅ (6 tasks - COMPLETE)

Replace session-based embedding cache with ChromaDB for block-level indexing. Reuses full-context generation from M1.

**Status**: Complete (commits: f63c5e1, 3b8ce3f, bbb065c, e3589d1, de0a269, 2169316, 62892f5)

#### M2.1: Add ChromaDB Dependency ✅
- **File**: `pyproject.toml`
- **Task**: ✅ Added `chromadb>=0.4.0` to dependencies
- **Commit**: f63c5e1

#### M2.2: Create VectorStore Abstraction ✅
- **File**: `src/logsqueak/rag/vector_store.py`
- **Task**: ✅ Defined `VectorStore` interface (add, delete, query, close)
- **Task**: ✅ Implemented `ChromaDBStore` with collection management
- **Test**: ✅ 10 unit tests for store initialization and CRUD operations
- **Commit**: 3b8ce3f

#### M2.3: Create Chunk Dataclass and Page Chunker ✅
- **File**: `src/logsqueak/rag/chunker.py`
- **Task**: ✅ Defined `Chunk` dataclass with fields: `full_context_text`, `hybrid_id`, `page_name`, `metadata`
- **Task**: ✅ Implemented `chunk_page(outline, page_name) -> List[Chunk]`
  - Reuses `generate_chunks()` from M1.2 (`logseq/context.py`)
  - Wraps results in `Chunk` dataclass with page metadata
- **Test**: ✅ 16 unit tests for Chunk serialization and page chunking
- **Commit**: bbb065c

#### M2.4: Create Cache Manifest System ✅
- **File**: `src/logsqueak/rag/manifest.py`
- **Task**: ✅ Implemented `CacheManifest` class (load/save JSON)
- **Task**: ✅ Tracks `{page_name: mtime}` mappings
- **Test**: ✅ 20 unit tests for manifest persistence
- **Commit**: e3589d1

#### M2.5: Implement Incremental Index Builder ✅
- **File**: `src/logsqueak/rag/indexer.py`
- **Task**: ✅ Implemented `IndexBuilder.build_incremental(graph_path, vector_store)`
- **Task**: ✅ Detects deletions (in manifest but not on disk)
- **Task**: ✅ Detects updates (mtime changed) and additions (not in manifest)
- **Test**: ✅ 8 integration tests for incremental updates
- **Commit**: de0a269

#### M2.6: Migrate PageIndex to Use VectorStore ✅
- **File**: `src/logsqueak/models/page.py`
- **Task**: ✅ Created `PageIndex.build_with_vector_store()` transitional API
- **Task**: ✅ Maintained backward compatibility with existing API
- **Test**: ✅ 6 integration tests for RAG search with ChromaDB
- **Commit**: 2169316

#### M2.7: CLI Command for Index Management ✅
- **File**: `src/logsqueak/cli/main.py`
- **Task**: ✅ Added `logsqueak index rebuild` command
- **Task**: ✅ Added `logsqueak index status` command (show cache stats)
- **Test**: ✅ 10 CLI integration tests
- **Commit**: 62892f5

---

### **Milestone 3: Block-Level Targeting** ✅ (4 tasks - COMPLETE)

Implement precise block targeting using hybrid IDs for UPDATE and APPEND operations. Simplifies current section-path approach.

**Status**: Complete (commit: ba475e2)

#### M3.2: Implement `find_target_node_by_id()` ✅
- **File**: `src/logsqueak/logseq/parser.py`
- **Task**: Add `LogseqOutline.find_block_by_id(target_id) -> Optional[LogseqBlock]`
- **Task**: Traverse AST, comparing hybrid IDs
- **Test**: ✅ Unit test for finding blocks by `id::` and content hash
- **Note**: Implementation already existed from M1.3

#### M3.3: Replace Section Paths with Block ID Targeting ✅
- **File**: `src/logsqueak/models/knowledge.py`
- **Task**: Replace `target_section: List[str]` with `target_block_id: Optional[str]`
- **Task**: Update all references to use direct block ID instead of section paths
- **Task**: Simplify `_find_target_section()` to use `find_block_by_id()`
- **Rationale**: FUTURE-STATE uses single `target_id` (simpler, more precise than section paths)
- **Test**: ✅ Updated tests to use block IDs instead of section paths
- **Note**: Added `target_block_id` field; section paths still supported for legacy compatibility

#### M3.4: Implement Block Modification (UPDATE) ✅
- **File**: `src/logsqueak/integration/writer.py`
- **Task**: Implement `update_block(target_block, new_content, preserve_id=True)`
- **Task**: Replace block content while preserving existing `id::`
- **Test**: ✅ Unit test for UPDATE operation

#### M3.5: Implement APPEND Operations ✅
- **File**: `src/logsqueak/integration/writer.py`
- **Task**: Implement `append_to_block(target_block, new_content, new_id)`
  - Add child to specific target block
- **Task**: Implement `append_to_root(page_outline, new_content, new_id)`
  - Add block to page root (when target_id == "root")
- **Task**: Update `add_knowledge_to_page()` to handle both APPEND modes
- **Rationale**: FUTURE-STATE only has UPDATE, APPEND (as child), APPEND (to root) - no CREATE_SECTION
- **Test**: ✅ Unit test for both APPEND operations

---

### **Milestone 4: Multi-Stage LLM Pipeline** (12 tasks)

Implement Phase 1 (Knowledge Extraction changes), Phase 2 (Enhanced RAG), Phase 3 (Decider + Reworder) and Phase 4 (Execution + Cleanup) from FUTURE-STATE.

#### M4.0: Remove Backward-Compatible PageIndex API (moved from M3.1)
- **File**: `src/logsqueak/models/page.py`, `src/logsqueak/cli/main.py`
- **Task**: Remove `PageIndex.build_with_vector_store()` transitional API (added in M2.6)
- **Task**: Remove `_find_similar_with_vector_store()` method from PageIndex
- **Task**: Update CLI to query VectorStore directly instead of via PageIndex wrapper
- **Task**: Update extraction pipeline to use block-level search directly
- **Rationale**: M2.6 created backward-compatible API for transition; cleanup before M4 pipeline implementation needs direct VectorStore access
- **Test**: Ensure all existing integration tests pass with direct VectorStore usage
- **Note**: Moved from M3 to M4.0 as prerequisite for M4.2 (Enhanced RAG needs direct block-level access)

#### M4.1: Update Phase 1 Extraction to Return Exact Block Text
- **File**: `src/logsqueak/extraction/extractor.py`, `src/logsqueak/llm/prompts.py`
- **Task**: Modify extraction prompt to return exact block text (not pre-contextualized)
- **Task**: Add post-extraction step to walk AST and add parent context to each block
- **Task**: Generate `original_id` for each extracted block (using id:: or content hash)
- **Task**: Return "Knowledge Packages" with `{original_id, full_text}` instead of just content
- **Rationale**: Separates LLM responsibility (identify blocks) from code responsibility (add context)
- **Test**: Verify extracted blocks get proper context from parent bullets

#### M4.2: Implement Phase 2 Candidate Retrieval (Enhanced)
- **File**: `src/logsqueak/extraction/extractor.py`
- **Task**: Implement semantic search - query vector store for Top-K similar chunks
- **Task**: Implement hinted search - parse [[Page Links]] from knowledge text using regex
- **Task**: Aggregate and deduplicate results from both semantic + hinted searches
- **Task**: For each candidate page, gather all relevant chunks with {page_name, target_id, content}
- **Task**: Return augmented Knowledge Package with candidates list
- **Rationale**: Combines RAG similarity with explicit page references for better targeting
- **Test**: Unit test for semantic search, hinted search, and deduplication

#### M4.3: Create Decider Prompt (Phase 3.1)
- **File**: `src/logsqueak/llm/prompts.py`
- **Task**: Add `build_decider_prompt(knowledge, targetable_chunks)`
- **Task**: LLM chooses: IGNORE_*, UPDATE, APPEND (with `target_id`)
- **Test**: Prompt inspection test

#### M4.4: Implement Decider LLM Call
- **File**: `src/logsqueak/llm/client.py`
- **Task**: Add `decide_action(knowledge, targetable_chunks) -> DecisionResult`
- **Task**: Parse JSON response with action and `target_id`
- **Test**: Mock LLM test for decision parsing

#### M4.5: Create Reworder Prompt (Phase 3.2)
- **File**: `src/logsqueak/llm/prompts.py`
- **Task**: Add `build_reworder_prompt(knowledge_full_text)`
- **Task**: Instructions: remove journal context, preserve links, create evergreen block
- **Test**: Prompt inspection test

#### M4.6: Implement Reworder LLM Call
- **File**: `src/logsqueak/llm/client.py`
- **Task**: Add `reword_knowledge(full_text) -> str`
- **Task**: Use high-quality model (configurable)
- **Test**: Mock LLM test for reword output

#### M4.7: Wire Up Phase 3 in Pipeline
- **File**: `src/logsqueak/cli/main.py`
- **Task**: Initialize empty "Write List" and "Processed Journal Blocks" map
- **Task**: Implement nested loop: for each Knowledge Package, loop through each of its candidates
- **Task**: For each (knowledge, candidate) pair, call Decider LLM
- **Task**: If action is IGNORE_*, skip to next candidate
- **Task**: If action is UPDATE/APPEND, call Reworder LLM with knowledge full_text
- **Task**: Add to Write List: {page_name, action, target_id, new_content}
- **Task**: Track in Processed Journal Blocks map: original_id -> [(page_name, None)] (new_id added in Phase 4)
- **Rationale**: Nested loop structure evaluates each knowledge block against ALL candidate pages
- **Test**: Integration test for Phase 3 pipeline with multiple candidates

#### M4.8: Implement Phase 4 Execution Logic
- **File**: `src/logsqueak/integration/executor.py` (new file)
- **Task**: Group all operations in Write List by page_name (minimize file I/O)
- **Task**: For each page: parse to AST, apply all operations, serialize back
- **Task**: For each operation: generate NEW UUID during write (not before)
- **Task**: Use `find_block_by_id(target_id)` to locate target node in AST
- **Task**: Handle UPDATE: set target_node content to `{new_content}\n  id:: {new_block_id}`
  - NOTE: handle indentation correctly! The id:: block property should be on a continuation line.
- **Task**: Handle APPEND (as child): add child to target_node with new UUID
- **Task**: Handle APPEND (to root): add root-level block when target_id == "root"
- **Task**: Update Processed Journal Blocks map with new_block_id for each write
- **Rationale**: Batching by page reduces I/O, UUID generation during write ensures uniqueness
- **Test**: Unit tests for UPDATE/APPEND operations, grouping logic

#### M4.9: Implement Journal Cleanup (Phase 4.5)
- **File**: `src/logsqueak/integration/journal_cleanup.py`
- **Task**: Read and parse journal file into AST
- **Task**: For each original_id in Processed Journal Blocks map, find source block using `find_block_by_id()`
- **Task**: Gather all new links for that block: [(page_name, new_id), ...]
- **Task**: Format each link: remove `.md`, replace `___` with `/`, format as `[page](((uuid)))`
  - NOTE: This is markdown link with block ref target, NOT Logseq `[[page]]` syntax
- **Task**: Create processed marker: `processed:: [page1](((uuid1))), [page2](((uuid2)))`
- **Task**: Add processed marker as child block to original journal block
- **Task**: Serialize modified journal AST and write back to file
- **Rationale**: Links user back to where knowledge was integrated
- **Test**: Unit test for link formatting, AST modification, round-trip safety

#### M4.10: Wire Up Phase 4 in Pipeline
- **File**: `src/logsqueak/cli/main.py`
- **Task**: After Phase 3, call Phase 4 execution with Write List
- **Task**: Collect updated Processed Journal Blocks map with new_block_ids
- **Task**: Call journal cleanup with updated map
- **Test**: Integration test for end-to-end Phase 4 execution + cleanup

#### M4.11: Add Configuration for Model Selection
- **File**: `src/logsqueak/models/config.py`
- **Task**: Add optional `llm.decider_model` and `llm.reworder_model` config fields
- **Task**: Both default to `llm.model` if not specified (allows using different models for speed vs quality)
- **Test**: Config validation test

---

### **Milestone 5: Testing & Refinement** (5-7 tasks)

Comprehensive testing and polish for the new pipeline.

#### M5.1: Integration Test: Full Pipeline (Phase 0-4)
- **File**: `tests/integration/test_full_pipeline.py`
- **Task**: End-to-end test from journal → index → extract → integrate → cleanup
- **Task**: Verify hybrid IDs, `processed::` markers, round-trip safety
- **Test**: Use fixture journal and pages

#### M5.2: Performance Testing
- **File**: `tests/performance/test_indexing.py`
- **Task**: Benchmark incremental indexing (add 5 pages to 500-page graph)
- **Task**: Compare ChromaDB vs old pkl cache performance
- **Test**: Document performance characteristics

#### M5.3: Error Recovery Tests
- **File**: `tests/integration/test_error_recovery.py`
- **Task**: Test corrupted vector store recovery
- **Task**: Test LLM failures (retries, fallback strategies)
- **Task**: Test write failures (atomic operations)

#### M5.4: Update Documentation
- **File**: `README.md`, `FUTURE-STATE.md`
- **Task**: Document new configuration options
- **Task**: Update architecture diagrams
- **Task**: Add troubleshooting section for vector store

#### M5.5: CLI Enhancements
- **File**: `src/logsqueak/cli/main.py`
- **Task**: Add `--rebuild-index` flag to force full rebuild
- **Task**: Add progress bars for indexing
- **Task**: Improve error messages for common issues

#### M5.6: Migration Guide
- **File**: `MIGRATION.md`
- **Task**: Document migration from old pkl cache to ChromaDB
- **Task**: Provide script to migrate existing embeddings
- **Task**: Breaking changes and compatibility notes

---

## Timeline Estimates

| Milestone | Tasks | Estimated Days | Status | Actual Time |
|-----------|-------|----------------|--------|-------------|
| M1: Hybrid-ID Foundation | 5 | 3-5 days | ✅ Complete | ~1 day |
| M2: Persistent Vector Store | 6 | 4-6 days | ✅ Complete | ~1 day |
| M3: Block-Level Targeting | 4 | 2-3 days | ✅ Complete | <1 day |
| M4: Multi-Stage Pipeline | 12 | 10-14 days | ⏳ Next | - |
| M5: Testing & Refinement | 6 | 4-6 days | Pending | - |
| **Total** | **32 tasks** | **23-35 days** | 44% | ~3/23-35 |

---

## Risk Mitigation

### Risk 1: ChromaDB Integration Complexity
**Mitigation**: Start with simple in-memory ChromaDB, then add persistence
**Fallback**: Keep pkl cache as backup option (config flag)

### Risk 2: Round-Trip Safety Edge Cases
**Mitigation**: Extensive property testing with real Logseq files
**Fallback**: Warn users about non-standard markdown syntax

### Risk 3: LLM Cost for Reworder
**Mitigation**: Make reworder model configurable (use cheaper local model)
**Fallback**: Add `--skip-reword` flag to use raw journal content

### Risk 4: Performance Degradation
**Mitigation**: Benchmark after each milestone
**Fallback**: Add caching layers, lazy loading, async processing

---

## Success Criteria

### Milestone 1 Complete When:
- [x] Parser extracts `id::` properties ✅
- [x] Full-context chunk generation works (shared code for M1 and M2) ✅
- [x] Content hashing generates stable hybrid IDs ✅
- [x] Writer adds UUID to new blocks ✅
- [x] Round-trip tests pass for IDs ✅

### Milestone 2 Complete When:
- [x] ChromaDB stores block-level embeddings ✅
- [x] Incremental indexing detects changes ✅
- [x] RAG search works with new vector store ✅
- [x] Index rebuild command available ✅

### Milestone 3 Complete When:
- [x] `find_target_node_by_id()` works for hybrid IDs ✅
- [x] UPDATE operation modifies blocks precisely ✅
- [x] APPEND operations work (to block and to root) ✅
- [x] Block ID targeting infrastructure in place ✅

### Milestone 4 Complete When:
- [ ] Decider LLM selects action + target ✅
- [ ] Reworder generates clean content ✅
- [ ] Journal cleanup adds `processed::` markers ✅
- [ ] Full Phase 0-4 pipeline runs end-to-end ✅

### Milestone 5 Complete When:
- [ ] All integration tests pass ✅
- [ ] Performance benchmarks documented ✅
- [ ] Migration guide written ✅
- [ ] README updated with new features ✅

---

## Next Steps

1. **Review this plan** - Validate assumptions, adjust estimates
2. **Create M1 branch** - `git checkout -b milestone-1-hybrid-ids`
3. **Start with M1.1** - Enhance parser for `id::` extraction
4. **Test incrementally** - Don't merge without tests ✅
5. **Document as you go** - Update PLAN.md with learnings

**Ready to begin Milestone 1?** 🚀
