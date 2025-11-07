"""SVG snapshot tests for TargetPagePreview widget.

Tests visual rendering of:
- Logseq markdown with various syntax (links, tags, bold, italic)
- Word wrapping with hanging indent
- Multi-line blocks with properties
- Green bar highlighting for specific blocks
"""

import pytest
from textual.app import App, ComposeResult
from logsqueak.tui.widgets.target_page_preview import TargetPagePreview


class TargetPagePreviewTestApp(App):
    """Test app wrapper for TargetPagePreview widget."""

    CSS = """
    TargetPagePreview {
        width: 100%;
        height: 100%;
    }
    """

    def __init__(self, content: str = "", highlight_block_id: str | None = None):
        super().__init__()
        self.test_content = content
        self.test_highlight_id = highlight_block_id
        self.content_loaded = False

    def compose(self) -> ComposeResult:
        """Compose the test app with TargetPagePreview."""
        yield TargetPagePreview()

    async def on_mount(self) -> None:
        """Load preview content on mount."""
        # Wait for compose to complete
        await self.workers.wait_for_complete()

        # Load the content
        preview = self.query_one(TargetPagePreview)
        await preview.load_preview(self.test_content, self.test_highlight_id)
        self.content_loaded = True


def test_target_preview_simple_content(snap_compare):
    """Snapshot test for simple content without highlighting."""
    content = """- Python Programming
  - Core Concepts
    - Variables and Types
    - Control Flow
  - Functions
    - Defining Functions
    - Lambda Functions"""

    app = TargetPagePreviewTestApp(content=content)
    assert snap_compare(app, terminal_size=(80, 24))


def test_target_preview_with_formatting(snap_compare):
    """Snapshot test for content with inline formatting (bold, italic, links)."""
    content = """- [[Python]] Best Practices
  Following established conventions helps maintain code quality.
  - Use **type hints** for clarity
    Introduced in Python 3.5, type hints help catch bugs early.
  - Follow #PEP8 style guide
  - See [official docs](https://docs.python.org)
  - Related: [[Code Quality]]"""

    app = TargetPagePreviewTestApp(content=content)
    assert snap_compare(app, terminal_size=(80, 24))


def test_target_preview_with_properties(snap_compare):
    """Snapshot test for content with Logseq properties."""
    content = """- Project: Logsqueak
  tags:: #project #logseq #tui
  status:: in-progress
  - **Phase 1**: Block Selection ✅
  - **Phase 2**: Content Editing ✅
  - **Phase 3**: Integration Review 🚧
    - Implement [[TargetPagePreview]] widget
    - Add green bar insertion indicator"""

    app = TargetPagePreviewTestApp(content=content)
    assert snap_compare(app, terminal_size=(80, 24))


def test_target_preview_with_mixed_tags(snap_compare):
    """Snapshot test for content with various tag types (#tag, #[[Link Tag]])."""
    content = """- Learning Resources
  category:: #learning
  topics:: #[[Machine Learning]], #python, #[[Data Science]]
  - **Deep Learning** fundamentals
  - #tutorial on neural networks
  - See [[Python]] and [[TensorFlow]]"""

    app = TargetPagePreviewTestApp(content=content)
    assert snap_compare(app, terminal_size=(80, 24))


def test_target_preview_with_wrapping(snap_compare):
    """Snapshot test for word wrapping with hanging indent."""
    content = """- This is a very long block that should definitely wrap when rendered in a narrow terminal window to demonstrate proper hanging indent behavior
  property:: This is also a very long property value that will wrap and should maintain proper indentation throughout
  - Child block with even more extremely verbose content that goes on and on to ensure we test multi-line wrapping properly"""

    app = TargetPagePreviewTestApp(content=content)
    assert snap_compare(app, terminal_size=(60, 24))


def test_target_preview_with_highlight_explicit_id(snap_compare):
    """Snapshot test for green bar highlighting with explicit block ID.

    Note: id:: property is not rendered, but block is still highlighted.
    """
    content = """- Parent block
  - First child
  - Second child to highlight
    id:: highlight-me
  - Third child"""

    app = TargetPagePreviewTestApp(content=content, highlight_block_id="highlight-me")
    assert snap_compare(app, terminal_size=(80, 24))


def test_target_preview_with_highlight_and_properties(snap_compare):
    """Snapshot test for highlighting block with properties (all lines highlighted).

    Note: id:: property is not rendered, but tags and status are shown.
    """
    content = """- Parent block
  - Child one
  - Child two with properties
    id:: block-with-props
    tags:: #important #demo
    status:: complete
  - Child three"""

    app = TargetPagePreviewTestApp(content=content, highlight_block_id="block-with-props")
    assert snap_compare(app, terminal_size=(80, 24))


def test_target_preview_nested_with_highlight(snap_compare):
    """Snapshot test for deeply nested structure with highlighted block.

    Note: id:: property is not rendered, but description property is shown.
    """
    content = """- Web Development
  - Frontend
    - React
      - Hooks
        id:: react-hooks
        description:: Functional component state management
      - Context API
  - Backend
    - REST APIs"""

    app = TargetPagePreviewTestApp(content=content, highlight_block_id="react-hooks")
    assert snap_compare(app, terminal_size=(80, 24))


def test_target_preview_empty_content(snap_compare):
    """Snapshot test for empty content placeholder."""
    app = TargetPagePreviewTestApp(content="")
    assert snap_compare(app, terminal_size=(80, 24))


def test_target_preview_narrow_terminal(snap_compare):
    """Snapshot test for narrow terminal with aggressive wrapping."""
    content = """- [[Python]] Programming
  - **List comprehensions** are concise
  - Following #PEP8 guidelines
  - See [[Documentation]]"""

    app = TargetPagePreviewTestApp(content=content)
    assert snap_compare(app, terminal_size=(40, 24))
