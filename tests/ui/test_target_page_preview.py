"""Snapshot tests for TargetPagePreview widget.

Tests for visual rendering of the target page preview with green bar
indicators at insertion points.
"""

import pytest
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Header, Footer
from logsqueak.tui.widgets.target_page_preview import TargetPagePreview


class PreviewTestApp(App):
    """Test app wrapper for TargetPagePreview widget."""

    CSS = """
    TargetPagePreview {
        height: 100%;
        border: solid white;
    }
    """

    def __init__(self, preview: TargetPagePreview):
        super().__init__()
        self.preview = preview

    def compose(self) -> ComposeResult:
        """Compose test app with preview widget."""
        yield Header()
        yield Container(self.preview)
        yield Footer()


@pytest.fixture
def simple_page_content():
    """Simple page content for testing."""
    return """- Python Concepts
  - Functions
  - Decorators
  - Generators"""


@pytest.fixture
def page_with_properties():
    """Page content with properties."""
    return """- Configuration Management
  tags:: programming, devops
  created:: 2025-01-15
  - Use environment variables for secrets
  - Keep config separate from code"""


@pytest.fixture
def nested_page_content():
    """Deeply nested page structure."""
    return """- Web Development
  - Frontend
    - React
      - Hooks
      - Context API
    - CSS
      - Flexbox
      - Grid
  - Backend
    - REST APIs
    - GraphQL"""


@pytest.fixture
def page_with_integrated_block():
    """Page content with a newly integrated block (marked at insertion)."""
    return """- Python Concurrency
  - Threading
  - Multiprocessing
  - Async/await pattern for concurrent file I/O operations
  - Event loops"""


def test_preview_initial_empty_state(snap_compare):
    """Test preview widget starts empty with helpful message."""
    preview = TargetPagePreview()
    app = PreviewTestApp(preview)

    assert snap_compare(app, terminal_size=(80, 24))


def test_preview_simple_content_no_insertion(snap_compare, simple_page_content):
    """Test preview displays simple content without insertion marker."""
    preview = TargetPagePreview()
    preview.load_preview(simple_page_content)
    app = PreviewTestApp(preview)

    assert snap_compare(app, terminal_size=(80, 24))


def test_preview_with_insertion_marker(snap_compare, page_with_integrated_block):
    """Test preview displays green bar at insertion line."""
    preview = TargetPagePreview()
    # Line 2 (0-indexed) is where "Async/await pattern..." was inserted
    preview.load_preview(page_with_integrated_block, insertion_line=2)
    app = PreviewTestApp(preview)

    assert snap_compare(app, terminal_size=(80, 24))


def test_preview_with_properties(snap_compare, page_with_properties):
    """Test preview displays content with properties correctly."""
    preview = TargetPagePreview()
    preview.load_preview(page_with_properties)
    app = PreviewTestApp(preview)

    assert snap_compare(app, terminal_size=(80, 24))


def test_preview_nested_structure(snap_compare, nested_page_content):
    """Test preview displays deeply nested content."""
    preview = TargetPagePreview()
    preview.load_preview(nested_page_content)
    app = PreviewTestApp(preview)

    assert snap_compare(app, terminal_size=(80, 30))


def test_preview_insertion_at_top(snap_compare, simple_page_content):
    """Test preview with insertion marker at first line."""
    preview = TargetPagePreview()
    preview.load_preview(simple_page_content, insertion_line=0)
    app = PreviewTestApp(preview)

    assert snap_compare(app, terminal_size=(80, 24))


def test_preview_insertion_at_nested_level(snap_compare, nested_page_content):
    """Test preview with insertion marker in nested structure."""
    preview = TargetPagePreview()
    # Insert at line 4 (under "React")
    preview.load_preview(nested_page_content, insertion_line=4)
    app = PreviewTestApp(preview)

    assert snap_compare(app, terminal_size=(80, 30))


def test_preview_clear_content(snap_compare, simple_page_content):
    """Test preview clear() method resets to empty state."""
    preview = TargetPagePreview()
    preview.load_preview(simple_page_content, insertion_line=2)
    preview.clear()
    app = PreviewTestApp(preview)

    assert snap_compare(app, terminal_size=(80, 24))


def test_preview_focused_border(snap_compare, simple_page_content):
    """Test preview changes border when focused."""
    preview = TargetPagePreview()
    preview.load_preview(simple_page_content, insertion_line=1)
    app = PreviewTestApp(preview)

    # Manually trigger focus state
    preview.on_focus()

    assert snap_compare(app, terminal_size=(80, 24))


def test_preview_long_content_scrollable(snap_compare):
    """Test preview handles long content that requires scrolling."""
    long_content = "\n".join([
        "- Very Long Document",
        *[f"  - Section {i}\n    - Detail {i}" for i in range(1, 21)]
    ])

    preview = TargetPagePreview()
    preview.load_preview(long_content, insertion_line=10)
    app = PreviewTestApp(preview)

    assert snap_compare(app, terminal_size=(80, 30))


def test_preview_multiple_insertions_same_line(snap_compare):
    """Test preview with insertion marker on line with existing content."""
    content = """- Existing Section
  - Item A
  - NEW ITEM INSERTED HERE
  - Item B"""

    preview = TargetPagePreview()
    preview.load_preview(content, insertion_line=2)
    app = PreviewTestApp(preview)

    assert snap_compare(app, terminal_size=(80, 24))


def test_preview_edge_case_empty_lines(snap_compare):
    """Test preview handles content with empty lines."""
    content = """- Section 1

  - Subsection with gap

- Section 2"""

    preview = TargetPagePreview()
    preview.load_preview(content, insertion_line=2)
    app = PreviewTestApp(preview)

    assert snap_compare(app, terminal_size=(80, 24))
