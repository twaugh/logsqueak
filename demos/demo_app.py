"""Demo for Full Logsqueak App Integration.

This demonstrates the complete workflow:
- Phase 1: Block Selection
- Phase 2: Content Editing
- Phase 3: Integration Review

Run: python demos/demo_app.py
"""

from pathlib import Path
from logsqueak.tui.app import LogsqueakApp
from logsqueak.models.config import Config, LLMConfig, LogseqConfig
from logsqueak.services.llm_client import LLMClient
from logsqueak.services.page_indexer import PageIndexer
from logsqueak.services.rag_search import RAGSearch
from logsqueak.services.file_monitor import FileMonitor
from logseq_outline.parser import LogseqOutline, LogseqBlock
from logseq_outline.graph import GraphPaths


def create_sample_journal() -> LogseqOutline:
    """Create a sample journal outline for demonstration."""
    blocks = [
        LogseqBlock(
            content=["2025-01-15 - Tuesday"],
            indent_level=0,
            block_id="root",
            children=[
                # Morning routine (activity log)
                LogseqBlock(
                    content=["Morning routine"],
                    indent_level=1,
                    block_id="block-1",
                    children=[
                        LogseqBlock(
                            content=["Reviewed emails and Slack messages"],
                            indent_level=2,
                            block_id="block-1-1",
                            children=[]
                        ),
                    ]
                ),
                # Knowledge: Python async patterns
                LogseqBlock(
                    content=["Learned about Python asyncio patterns #async"],
                    indent_level=1,
                    block_id="block-2",
                    children=[
                        LogseqBlock(
                            content=["asyncio.create_task() enables concurrent operations"],
                            indent_level=2,
                            block_id="block-2-1",
                            children=[]
                        ),
                    ]
                ),
                # Knowledge: Docker networking
                LogseqBlock(
                    content=["Docker containers can communicate by service name #docker"],
                    indent_level=1,
                    block_id="block-4",
                    children=[
                        LogseqBlock(
                            content=["Docker DNS handles name resolution automatically"],
                            indent_level=2,
                            block_id="block-4-1",
                            children=[]
                        ),
                    ]
                ),
            ]
        ),
    ]

    return LogseqOutline(
        blocks=blocks,
        source_text="# Demo journal content",
        indent_str="  ",
        frontmatter=[]
    )


def main():
    """Run the demo app."""
    import time

    # Create a temporary directory for demo
    import tempfile
    temp_dir = Path(tempfile.mkdtemp(prefix="logsqueak_demo_"))

    print(f"[{time.time():.3f}] Creating config...")
    # Create mock config
    config = Config(
        llm=LLMConfig(
            endpoint="http://localhost:11434/v1",
            api_key="demo-key",
            model="llama2",
        ),
        logseq=LogseqConfig(graph_path=str(temp_dir)),
    )

    print(f"[{time.time():.3f}] Creating services...")
    # Create services (all will use temp directory)
    llm_client = LLMClient(config=config.llm)
    graph_paths = GraphPaths(temp_dir)
    page_indexer = PageIndexer(
        graph_paths=graph_paths,
        db_path=temp_dir / "chroma",
    )
    rag_search = RAGSearch(
        db_path=temp_dir / "chroma",
    )
    file_monitor = FileMonitor()

    print(f"[{time.time():.3f}] Creating journal...")
    # Create sample journal
    journal_outline = create_sample_journal()

    print(f"[{time.time():.3f}] Creating app...")
    # Create and run app
    app = LogsqueakApp(
        journal_outline=journal_outline,
        journal_date="2025-01-15",
        config=config,
        llm_client=llm_client,
        page_indexer=page_indexer,
        rag_search=rag_search,
        file_monitor=file_monitor,
    )

    print("=" * 60)
    print("Logsqueak Demo - Full App Integration")
    print("=" * 60)
    print()
    print("Instructions:")
    print("  Phase 1 (Block Selection):")
    print("    - Navigate: j/k (or arrow keys)")
    print("    - Select block: Space")
    print("    - Next phase: n (when blocks selected)")
    print()
    print("  Phase 2 (Content Editing):")
    print("    - Navigate blocks: j/k")
    print("    - Focus editor: Tab")
    print("    - Accept LLM version: a")
    print("    - Revert to original: r")
    print("    - Next phase: n (after RAG completes)")
    print()
    print("  Phase 3 (Integration Review):")
    print("    - Navigate decisions: j/k")
    print("    - Accept decision: y")
    print("    - Next block: n")
    print("    - Back: q")
    print()
    print("  Global:")
    print("    - Quit: Ctrl+C")
    print()
    print("Note: LLM workers are enabled but will fail (no real LLM server)")
    print("      You can still manually select and navigate blocks.")
    print("=" * 60)
    print(f"[{time.time():.3f}] Starting app.run()...")

    app.run()

    print(f"[{time.time():.3f}] App finished.")


if __name__ == "__main__":
    main()
