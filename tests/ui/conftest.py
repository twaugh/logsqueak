"""Shared fixtures for UI tests."""

import pytest
from logseq_outline.parser import LogseqBlock, LogseqOutline


@pytest.fixture
def sample_journals():
    """Create sample journals dict for testing.

    This fixture provides a default journals dictionary with a single
    journal entry for 2025-01-15. Individual test files can override
    this fixture with custom journal data if needed.
    """
    blocks = [
        LogseqBlock(
            content=["Knowledge block about Python async"],
            indent_level=0,
            block_id="journal-block-1",
            children=[]
        ),
    ]
    outline = LogseqOutline(blocks=blocks, source_text="", frontmatter=[])
    return {"2025-01-15": outline}
