#!/usr/bin/env python3
"""Demo for Phase 3: Integration Review Screen.

Demonstrates the integration review workflow using the actual Phase3Screen class.
"""

from textual.app import App, ComposeResult
from textual.widgets import Header
from logsqueak.tui.screens.integration_review import Phase3Screen
from logsqueak.models.integration_decision import IntegrationDecision
from logsqueak.models.edited_content import EditedContent
from logseq_outline.parser import LogseqOutline, LogseqBlock
import logging

# Set up file logging
logging.basicConfig(
    filename='/tmp/demo_phase3.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True  # Force reconfiguration
)
logger = logging.getLogger(__name__)


# Sample data - knowledge blocks with their decisions
KNOWLEDGE_BLOCKS = [
    {
        "id": "kb1",
        "number": 1,
        "total": 3,
        "context": "Journal: 2025-01-15\n  - Morning session\n    - Python learning",
        "content": "**List comprehensions** are a concise way to create lists in Python. They combine map() and filter() operations into a single expression: `[x*2 for x in range(10) if x % 2 == 0]`",
        "decisions": [
            IntegrationDecision(
                knowledge_block_id="kb1",
                target_page="Python",
                action="add_under",
                target_block_id="target-1",
                target_block_title="List Operations",
                refined_text="**List comprehensions** are a concise way to create lists in Python. They combine map() and filter() operations into a single expression: `[x*2 for x in range(10) if x % 2 == 0]`",
                confidence=0.92,
                reasoning="This is a fundamental Python feature that belongs under list operations",
                write_status="pending"
            ),
            IntegrationDecision(
                knowledge_block_id="kb1",
                target_page="Functional Programming",
                action="add_section",
                target_block_id=None,
                target_block_title=None,
                refined_text="**List comprehensions** are a concise way to create lists in Python. They combine map() and filter() operations into a single expression: `[x*2 for x in range(10) if x % 2 == 0]`",
                confidence=0.78,
                reasoning="List comprehensions are a functional programming concept",
                write_status="pending"
            ),
        ],
    },
    {
        "id": "kb2",
        "number": 2,
        "total": 3,
        "context": "Journal: 2025-01-15\n  - Afternoon\n    - Docker setup",
        "content": "Docker **multi-stage builds** reduce image size by using multiple FROM statements. Build artifacts from earlier stages can be copied to later stages with `COPY --from=builder`, eliminating build dependencies from the final image.",
        "decisions": [
            IntegrationDecision(
                knowledge_block_id="kb2",
                target_page="Docker",
                action="add_under",
                target_block_id="target-2",
                target_block_title="Best Practices",
                refined_text="Docker **multi-stage builds** reduce image size by using multiple FROM statements. Build artifacts from earlier stages can be copied to later stages with `COPY --from=builder`, eliminating build dependencies from the final image.",
                confidence=0.95,
                reasoning="Multi-stage builds are a Docker best practice for production images",
                write_status="pending"
            ),
        ],
    },
    {
        "id": "kb3",
        "number": 3,
        "total": 3,
        "context": "Journal: 2025-01-16\n  - Reading notes\n    - CI/CD article",
        "content": "**Continuous deployment** differs from continuous delivery in that code changes automatically deploy to production after passing tests, without human approval. This requires high confidence in automated testing and monitoring.",
        "decisions": [
            IntegrationDecision(
                knowledge_block_id="kb3",
                target_page="CI/CD",
                action="add_section",
                target_block_id=None,
                target_block_title=None,
                refined_text="**Continuous deployment** differs from continuous delivery in that code changes automatically deploy to production after passing tests, without human approval. This requires high confidence in automated testing and monitoring.",
                confidence=0.88,
                reasoning="This clarifies an important distinction in CI/CD terminology",
                write_status="pending"
            ),
            IntegrationDecision(
                knowledge_block_id="kb3",
                target_page="DevOps",
                action="add_under",
                target_block_id="target-3",
                target_block_title="Deployment Strategies",
                refined_text="**Continuous deployment** differs from continuous delivery in that code changes automatically deploy to production after passing tests, without human approval. This requires high confidence in automated testing and monitoring.",
                confidence=0.82,
                reasoning="Continuous deployment is a key DevOps practice",
                write_status="pending"
            ),
        ],
    },
]

# Sample target page content (what would be shown in preview)
TARGET_PAGES = {
    "Python": """- Python Programming
  - Core Concepts
    - Variables and Types
    - Control Flow
  - List Operations
    - Creating Lists
    - Indexing and Slicing
    - List Methods
  - Functions
    - Defining Functions
    - Lambda Functions""",
    "Functional Programming": """- Functional Programming Paradigm
  - Core Principles
    - Immutability
    - Pure Functions
    - First-Class Functions
  - Techniques
    - Map and Filter
    - Reduce Operations""",
    "Docker": """- Docker Containerization
  - Images
    - Building Images
    - Image Layers
  - Best Practices
    - Security
    - Performance Optimization
  - Networking
    - Bridge Networks
    - Host Networking""",
    "CI/CD": """- Continuous Integration and Deployment
  - CI Fundamentals
    - Automated Testing
    - Build Automation
  - CD Fundamentals
    - Deployment Pipelines
    - Environment Management""",
    "DevOps": """- DevOps Practices
  - Deployment Strategies
    - Blue-Green Deployment
    - Canary Releases
  - Monitoring
    - Metrics Collection
    - Alerting""",
}


def create_demo_data():
    """Create demo data for Phase3Screen.

    Returns:
        tuple: (journal_blocks, edited_content, decisions)
    """
    # Create journal blocks (mock LogseqBlock objects)
    journal_blocks = []
    edited_content_list = []
    all_decisions = []

    for kb in KNOWLEDGE_BLOCKS:
        # Create a mock LogseqBlock
        block = LogseqBlock(
            content=[kb["content"]],
            indent_level=0,
            block_id=kb["id"]
        )
        journal_blocks.append(block)

        # Create EditedContent for this block
        ec = EditedContent(
            block_id=kb["id"],
            original_content=kb["content"],
            hierarchical_context=kb["context"],
            reworded_content=kb["content"],  # Same as original for demo
            current_content=kb["content"],
            rewording_complete=True
        )
        edited_content_list.append(ec)

        # Add the decisions for this block
        all_decisions.extend(kb["decisions"])

    return journal_blocks, edited_content_list, all_decisions


class Phase3Demo(App):
    """Demo app that uses the actual Phase3Screen class."""

    def compose(self) -> ComposeResult:
        """Compose the demo app with just a header."""
        yield Header()

    async def on_mount(self) -> None:
        """Push the Phase3Screen when app mounts."""
        # Create demo data
        journal_blocks, edited_content, decisions = create_demo_data()

        # Create and push Phase3Screen
        screen = Phase3Screen(
            journal_blocks=journal_blocks,
            edited_content=edited_content,
            decisions=decisions,
            journal_date="2025-01-15",
            graph_paths=None,
            file_monitor=None,
            auto_start_workers=False  # Don't start background workers in demo
        )
        await self.push_screen(screen)


if __name__ == "__main__":
    app = Phase3Demo()
    app.run()
