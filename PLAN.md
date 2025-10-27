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

### **Milestone 1: Hybrid-ID Foundation** (5 tasks)

Build the infrastructure for persistent block IDs and full-context chunk generation. This lays the groundwork for both hybrid IDs and the chunking system needed in M2.

#### M1.1: Enhance Parser to Extract `id::` Properties
- **File**: `src/logsqueak/logseq/parser.py`
- **Task**: Modify `LogseqBlock` to store `block_id: Optional[str]`
- **Task**: Parse `id::` property and populate `block_id` field
- **Test**: Unit test for parsing blocks with/without `id::`

#### M1.2: Implement Full-Context Generation and Hashing
- **File**: New `src/logsqueak/logseq/context.py`
- **Task**: Create `generate_full_context(block, parents) -> str` helper
- **Task**: Recursive traversal to build full context (prepend parent context)
- **Task**: Create `generate_content_hash(full_context: str) -> str` using MD5
- **Task**: Create `generate_chunks(outline) -> List[Tuple[block, full_context, hybrid_id]]`
  - This will be reused by M2.3 for chunking
- **Test**: Unit test for full-context generation with nested blocks
- **Test**: Unit test for hash stability and collision resistance

#### M1.3: Add Hybrid ID to LogseqBlock
- **File**: `src/logsqueak/logseq/parser.py`
- **Task**: Add method `get_hybrid_id() -> str` to `LogseqBlock`
- **Task**: Returns `block_id` if present, else content hash
- **Test**: Unit test for hybrid ID resolution

#### M1.4: Implement UUID Generation in Writer
- **File**: `src/logsqueak/integration/writer.py`
- **Task**: Generate new UUID for each integrated block
- **Task**: Append `id:: <uuid>` to new blocks
- **Test**: Integration test for ID generation and formatting

#### M1.5: Round-Trip Safety Tests
- **File**: `tests/integration/test_roundtrip.py`
- **Task**: Test parse → modify → render → parse preserves IDs
- **Task**: Test that existing `id::` properties are never modified
- **Test**: Property order preservation (already exists ✅)

---

### **Milestone 2: Persistent Vector Store** (6 tasks)

Replace session-based embedding cache with ChromaDB for block-level indexing. Reuses full-context generation from M1.

#### M2.1: Add ChromaDB Dependency
- **File**: `requirements.txt`, `pyproject.toml`
- **Task**: Add `chromadb>=0.4.0` to dependencies
- **Task**: Update setup documentation

#### M2.2: Create VectorStore Abstraction
- **File**: `src/logsqueak/rag/vector_store.py`
- **Task**: Define `VectorStore` interface (add, delete, query, close)
- **Task**: Implement `ChromaDBStore` with collection management
- **Test**: Unit test for store initialization and CRUD operations

#### M2.3: Create Chunk Dataclass and Page Chunker
- **File**: `src/logsqueak/rag/chunker.py`
- **Task**: Define `Chunk` dataclass with fields: `full_context_text`, `hybrid_id`, `page_name`, `metadata`
- **Task**: Implement `chunk_page(outline, page_name) -> List[Chunk]`
  - Reuses `generate_chunks()` from M1.2 (`logseq/context.py`)
  - Wraps results in `Chunk` dataclass with page metadata
- **Test**: Unit test for Chunk serialization and page chunking

#### M2.4: Create Cache Manifest System
- **File**: `src/logsqueak/rag/manifest.py`
- **Task**: Implement `CacheManifest` class (load/save JSON)
- **Task**: Track `{page_name: mtime}` mappings
- **Test**: Unit test for manifest persistence

#### M2.5: Implement Incremental Index Builder
- **File**: `src/logsqueak/rag/indexer.py`
- **Task**: Implement `IndexBuilder.build_incremental(graph_path, vector_store)`
- **Task**: Detect deletions (in manifest but not on disk)
- **Task**: Detect updates (mtime changed) and additions (not in manifest)
- **Test**: Integration test for incremental updates

#### M2.6: Migrate PageIndex to Use VectorStore
- **File**: `src/logsqueak/models/page.py`
- **Task**: Refactor `PageIndex.build()` to use `VectorStore`
- **Task**: Maintain backward compatibility with existing API
- **Test**: Integration test for RAG search with ChromaDB

#### M2.7: CLI Command for Index Management
- **File**: `src/logsqueak/cli/main.py`
- **Task**: Add `logsqueak index rebuild` command
- **Task**: Add `logsqueak index status` command (show cache stats)
- **Test**: CLI test for index commands

---

### **Milestone 3: Block-Level Targeting** (4 tasks)

Implement precise block targeting using hybrid IDs for UPDATE and APPEND operations. Simplifies current section-path approach.

#### M3.1: Implement `find_target_node_by_id()`
- **File**: `src/logsqueak/logseq/parser.py`
- **Task**: Add `LogseqOutline.find_block_by_id(target_id) -> Optional[LogseqBlock]`
- **Task**: Traverse AST, comparing hybrid IDs
- **Test**: Unit test for finding blocks by `id::` and content hash

#### M3.2: Replace Section Paths with Block ID Targeting
- **File**: `src/logsqueak/models/knowledge.py`
- **Task**: Replace `target_section: List[str]` with `target_block_id: Optional[str]`
- **Task**: Update all references to use direct block ID instead of section paths
- **Task**: Simplify `_find_target_section()` to use `find_block_by_id()`
- **Rationale**: FUTURE-STATE uses single `target_id` (simpler, more precise than section paths)
- **Test**: Update tests to use block IDs instead of section paths

#### M3.3: Implement Block Modification (UPDATE)
- **File**: `src/logsqueak/integration/writer.py`
- **Task**: Implement `update_block(target_block, new_content, preserve_id=True)`
- **Task**: Replace block content while preserving existing `id::`
- **Test**: Unit test for UPDATE operation

#### M3.4: Implement APPEND Operations
- **File**: `src/logsqueak/integration/writer.py`
- **Task**: Implement `append_to_block(target_block, new_content, new_id)`
  - Add child to specific target block
- **Task**: Implement `append_to_root(page_outline, new_content, new_id)`
  - Add block to page root (when target_id == "root")
- **Task**: Update `add_knowledge_to_page()` to handle both APPEND modes
- **Rationale**: FUTURE-STATE only has UPDATE, APPEND (as child), APPEND (to root) - no CREATE_SECTION
- **Test**: Unit test for both APPEND operations

---

### **Milestone 4: Multi-Stage LLM Pipeline** (7-9 tasks)

Implement Phase 3 (Decider + Reworder) and Phase 4 (Execution + Cleanup) from FUTURE-STATE.

#### M4.1: Implement Phase 2 Candidate Retrieval (Enhanced)
- **File**: `src/logsqueak/extraction/extractor.py`
- **Task**: Update `select_target_page()` to return block-level candidates
- **Task**: Include `target_id` for each relevant chunk
- **Test**: Unit test for block-level candidate retrieval

#### M4.2: Create Decider Prompt (Phase 3.1)
- **File**: `src/logsqueak/llm/prompts.py`
- **Task**: Add `build_decider_prompt(knowledge, targetable_chunks)`
- **Task**: LLM chooses: IGNORE_*, UPDATE, APPEND (with `target_id`)
- **Test**: Prompt inspection test

#### M4.3: Implement Decider LLM Call
- **File**: `src/logsqueak/llm/client.py`
- **Task**: Add `decide_action(knowledge, targetable_chunks) -> DecisionResult`
- **Task**: Parse JSON response with action and `target_id`
- **Test**: Mock LLM test for decision parsing

#### M4.4: Create Reworder Prompt (Phase 3.2)
- **File**: `src/logsqueak/llm/prompts.py`
- **Task**: Add `build_reworder_prompt(knowledge_full_text)`
- **Task**: Instructions: remove journal context, preserve links, create evergreen block
- **Test**: Prompt inspection test

#### M4.5: Implement Reworder LLM Call
- **File**: `src/logsqueak/llm/client.py`
- **Task**: Add `reword_knowledge(full_text) -> str`
- **Task**: Use high-quality model (configurable)
- **Test**: Mock LLM test for reword output

#### M4.6: Wire Up Phase 3 in Pipeline
- **File**: `src/logsqueak/cli/main.py`
- **Task**: Call Decider for each candidate
- **Task**: Call Reworder only if action is UPDATE/APPEND
- **Task**: Build "Write List" with `{page, decision, new_content}`
- **Test**: Integration test for Phase 3 pipeline

#### M4.7: Implement Journal Cleanup (Phase 4.5)
- **File**: `src/logsqueak/integration/journal_cleanup.py`
- **Task**: Add `processed::` markers to journal blocks
- **Task**: Format: `processed:: [page1](((uuid1))), [page2](((uuid2)))`
  - NOTE: Regular markdown links with block reference targets, NOT Logseq `[[page]]` syntax
- **Task**: Handle page name formatting (remove `.md`, replace `___` with `/`)
- **Test**: Unit test for cleanup formatting and link syntax

#### M4.8: Wire Up Phase 4 Cleanup
- **File**: `src/logsqueak/cli/main.py`
- **Task**: After file writes, update journal with `processed::` markers
- **Task**: Track `{original_id: [(page, new_id), ...]}`
- **Test**: Integration test for end-to-end cleanup

#### M4.9: Add Configuration for Model Selection
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

| Milestone | Tasks | Estimated Days | Dependencies |
|-----------|-------|----------------|--------------|
| M1: Hybrid-ID Foundation | 5 | 3-5 days | None (builds on current parser) |
| M2: Persistent Vector Store | 6 | 4-6 days | M1 (needs chunk generation from M1.2) |
| M3: Block-Level Targeting | 4 | 2-4 days | M1 (needs hybrid IDs) |
| M4: Multi-Stage Pipeline | 9 | 7-10 days | M2, M3 (needs targeting + index) |
| M5: Testing & Refinement | 6 | 4-6 days | M4 (needs full pipeline) |
| **Total** | **30 tasks** | **20-31 days** | |

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
- [ ] Parser extracts `id::` properties ✅
- [ ] Full-context chunk generation works (shared code for M1 and M2) ✅
- [ ] Content hashing generates stable hybrid IDs ✅
- [ ] Writer adds UUID to new blocks ✅
- [ ] Round-trip tests pass for IDs ✅

### Milestone 2 Complete When:
- [ ] ChromaDB stores block-level embeddings ✅
- [ ] Incremental indexing detects changes ✅
- [ ] RAG search works with new vector store ✅
- [ ] Index rebuild command available ✅

### Milestone 3 Complete When:
- [ ] `find_target_node_by_id()` works for hybrid IDs ✅
- [ ] UPDATE operation modifies blocks precisely ✅
- [ ] CREATE_SECTION adds nested headings ✅
- [ ] Block-level APPEND works ✅

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
