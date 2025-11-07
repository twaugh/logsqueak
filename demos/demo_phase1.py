"""Demo for Phase 1: Block Selection Screen.

This demonstrates the interactive block selection workflow where users:
1. View journal blocks in a tree structure
2. See LLM classification suggestions (knowledge vs. activity logs)
3. Manually select/deselect blocks for extraction

Run: python demos/demo_phase1.py
"""

from textual.app import App, ComposeResult
from logsqueak.tui.screens.block_selection import Phase1Screen
from logsqueak.models.block_state import BlockState
from logseq_outline.parser import LogseqBlock


def create_sample_blocks() -> list[LogseqBlock]:
    """Create realistic sample journal blocks for demonstration."""
    return [
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
                        LogseqBlock(
                            content=["Updated sprint board with completed tasks"],
                            indent_level=2,
                            block_id="block-1-2",
                            children=[]
                        ),
                    ]
                ),
                # Knowledge: Python async patterns
                LogseqBlock(
                    content=["Learned about Python asyncio patterns while debugging slow API calls"],
                    indent_level=1,
                    block_id="block-2",
                    children=[
                        LogseqBlock(
                            content=["asyncio.create_task() enables concurrent operations without blocking"],
                            indent_level=2,
                            block_id="block-2-1",
                            children=[]
                        ),
                        LogseqBlock(
                            content=["This is different from await which blocks until the coroutine completes"],
                            indent_level=2,
                            block_id="block-2-2",
                            children=[]
                        ),
                        LogseqBlock(
                            content=["Use asyncio.gather() to wait for multiple tasks concurrently"],
                            indent_level=2,
                            block_id="block-2-3",
                            children=[]
                        ),
                    ]
                ),
                # Meeting notes (activity log)
                LogseqBlock(
                    content=["Team standup at 10am"],
                    indent_level=1,
                    block_id="block-3",
                    children=[
                        LogseqBlock(
                            content=["Alice is blocked on code review"],
                            indent_level=2,
                            block_id="block-3-1",
                            children=[]
                        ),
                        LogseqBlock(
                            content=["Bob will help with database migration"],
                            indent_level=2,
                            block_id="block-3-2",
                            children=[]
                        ),
                    ]
                ),
                # Knowledge: Docker networking
                LogseqBlock(
                    content=["Docker containers on same bridge network can communicate by service name"],
                    indent_level=1,
                    block_id="block-4",
                    children=[
                        LogseqBlock(
                            content=["No need to use IP addresses - Docker DNS handles name resolution"],
                            indent_level=2,
                            block_id="block-4-1",
                            children=[]
                        ),
                    ]
                ),
                # Lunch (activity log)
                LogseqBlock(
                    content=["Had lunch with the team at the new Thai place"],
                    indent_level=1,
                    block_id="block-5",
                    children=[]
                ),
                # Knowledge: Git workflow tip
                LogseqBlock(
                    content=["TIL: git rebase -i HEAD~N lets you interactively squash commits"],
                    indent_level=1,
                    block_id="block-6",
                    children=[
                        LogseqBlock(
                            content=["Use 'fixup' instead of 'squash' to discard commit messages"],
                            indent_level=2,
                            block_id="block-6-1",
                            children=[]
                        ),
                    ]
                ),
                # TODO (activity log)
                LogseqBlock(
                    content=["TODO Review PR #456 before EOD"],
                    indent_level=1,
                    block_id="block-7",
                    children=[]
                ),
            ]
        ),
    ]


def create_sample_block_states() -> dict[str, BlockState]:
    """Create sample block states with LLM classifications.

    This simulates what the LLM would suggest after analyzing the blocks:
    - Knowledge blocks: Python asyncio, Docker networking, Git workflow
    - Activity logs: Morning routine, meetings, lunch, TODOs (no llm_classification)
    """
    return {
        # Morning routine - activity log (LLM classified as NOT knowledge)
        "block-1": BlockState(
            block_id="block-1",
            classification="pending",
            source="llm",
            llm_classification=None,  # LLM did not classify as knowledge
        ),
        "block-1-1": BlockState(
            block_id="block-1-1",
            classification="pending",
            source="llm",
            llm_classification=None,
        ),
        "block-1-2": BlockState(
            block_id="block-1-2",
            classification="pending",
            source="llm",
            llm_classification=None,
        ),
        # Python asyncio - KNOWLEDGE
        "block-2": BlockState(
            block_id="block-2",
            classification="knowledge",
            confidence=0.95,
            source="llm",
            llm_classification="knowledge",
            llm_confidence=0.95,
            reason="Technical learning about asyncio patterns and debugging"
        ),
        "block-2-1": BlockState(
            block_id="block-2-1",
            classification="knowledge",
            confidence=0.93,
            source="llm",
            llm_classification="knowledge",
            llm_confidence=0.93,
            reason="Explains asyncio.create_task() concurrency pattern"
        ),
        "block-2-2": BlockState(
            block_id="block-2-2",
            classification="knowledge",
            confidence=0.91,
            source="llm",
            llm_classification="knowledge",
            llm_confidence=0.91,
            reason="Clarifies difference between async patterns"
        ),
        "block-2-3": BlockState(
            block_id="block-2-3",
            classification="knowledge",
            confidence=0.89,
            source="llm",
            llm_classification="knowledge",
            llm_confidence=0.89,
            reason="Describes asyncio.gather() usage pattern"
        ),
        # Team standup - activity log (LLM classified as NOT knowledge)
        "block-3": BlockState(
            block_id="block-3",
            classification="pending",
            source="llm",
            llm_classification=None,
        ),
        "block-3-1": BlockState(
            block_id="block-3-1",
            classification="pending",
            source="llm",
            llm_classification=None,
        ),
        "block-3-2": BlockState(
            block_id="block-3-2",
            classification="pending",
            source="llm",
            llm_classification=None,
        ),
        # Docker networking - KNOWLEDGE
        "block-4": BlockState(
            block_id="block-4",
            classification="knowledge",
            confidence=0.94,
            source="llm",
            llm_classification="knowledge",
            llm_confidence=0.94,
            reason="Technical insight about Docker networking"
        ),
        "block-4-1": BlockState(
            block_id="block-4-1",
            classification="knowledge",
            confidence=0.92,
            source="llm",
            llm_classification="knowledge",
            llm_confidence=0.92,
            reason="Explains Docker DNS resolution mechanism"
        ),
        # Lunch - activity log (LLM classified as NOT knowledge)
        "block-5": BlockState(
            block_id="block-5",
            classification="pending",
            source="llm",
            llm_classification=None,
        ),
        # Git workflow - KNOWLEDGE
        "block-6": BlockState(
            block_id="block-6",
            classification="knowledge",
            confidence=0.96,
            source="llm",
            llm_classification="knowledge",
            llm_confidence=0.96,
            reason="Git workflow tip with practical application"
        ),
        "block-6-1": BlockState(
            block_id="block-6-1",
            classification="knowledge",
            confidence=0.93,
            source="llm",
            llm_classification="knowledge",
            llm_confidence=0.93,
            reason="Specific git rebase technique"
        ),
        # TODO - activity log (LLM classified as NOT knowledge)
        "block-7": BlockState(
            block_id="block-7",
            classification="pending",
            source="llm",
            llm_classification=None,
        ),
    }


class Phase1DemoApp(App):
    """Demo application for Phase 1 Block Selection screen."""

    TITLE = "Logsqueak - Phase 1: Block Selection Demo"
    SUB_TITLE = "Interactive Knowledge Block Selection"

    def __init__(self):
        super().__init__()
        self.blocks = create_sample_blocks()
        self.block_states = create_sample_block_states()

    def on_mount(self) -> None:
        """Push Phase1Screen on mount."""
        screen = Phase1Screen(
            blocks=self.blocks,
            journal_date="2025-01-15",
            initial_block_states=self.block_states,
            auto_start_workers=False  # Don't start LLM/indexing workers in demo
        )
        self.push_screen(screen)


if __name__ == "__main__":
    app = Phase1DemoApp()
    app.run()
