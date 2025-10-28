"""Diff generation for preview of proposed changes."""

import difflib
from pathlib import Path
from typing import Optional

from logsqueak.logseq.parser import LogseqBlock, LogseqOutline
from logsqueak.models.knowledge import KnowledgeBlock
from logsqueak.models.page import TargetPage


def simulate_integration(
    page: TargetPage, knowledge: KnowledgeBlock
) -> str:
    """Simulate what page would look like after integrating knowledge.

    Creates a copy of the page's outline, adds the knowledge block,
    and renders the result.

    Args:
        page: Target page to modify
        knowledge: Knowledge block to add

    Returns:
        Rendered markdown of modified page
    """
    # Create a deep copy of the outline by parsing rendered content
    # This ensures we don't modify the original
    original_rendered = page.outline.render()
    modified_outline = LogseqOutline.parse(original_rendered)

    # Add knowledge with provenance
    content_with_provenance = knowledge.with_provenance()

    # Find target location and add content
    if knowledge.target_section:
        # Navigate to the target section
        current_block = None
        for section_name in knowledge.target_section:
            # Skip None values (LLM may return them)
            if section_name is None:
                continue

            if current_block is None:
                # Search at root level
                current_block = modified_outline.find_heading(section_name)
            else:
                # Search in children
                for child in current_block.children:
                    if section_name.lower() in child.content[0].lower():
                        current_block = child
                        break

        if current_block:
            # Add as child under the section
            # Don't add "- " prefix - render() will add it
            current_block.add_child(content_with_provenance)
        else:
            # Section not found - add at end of page
            # Don't add "- " prefix - render() will add it
            modified_outline.blocks.append(
                LogseqBlock(
                    content=[content_with_provenance],
                    indent_level=0,
                )
            )
    else:
        # No section specified - add at page root
        # Don't add "- " prefix - render() will add it
        modified_outline.blocks.append(
            LogseqBlock(
                content=[content_with_provenance],
                indent_level=0,
            )
        )

    return modified_outline.render()


def generate_unified_diff(
    original: str,
    modified: str,
    fromfile: str = "original",
    tofile: str = "modified",
    context_lines: int = 3,
) -> str:
    """Generate unified diff between original and modified content.

    Lines that differ only by the presence/absence of a trailing newline
    are treated as identical to avoid showing spurious differences.

    Args:
        original: Original content
        modified: Modified content
        fromfile: Label for original file
        tofile: Label for modified file
        context_lines: Number of context lines to show

    Returns:
        Unified diff as string
    """
    # Normalize: ensure all lines end with newline for consistent comparison
    # This prevents differences that are only about trailing newlines
    original_lines = original.splitlines(keepends=True)
    modified_lines = modified.splitlines(keepends=True)

    # Ensure every line ends with newline (normalize trailing newlines)
    original_lines = [line if line.endswith('\n') else line + '\n' for line in original_lines]
    modified_lines = [line if line.endswith('\n') else line + '\n' for line in modified_lines]

    diff_lines = list(difflib.unified_diff(
        original_lines,
        modified_lines,
        fromfile=fromfile,
        tofile=tofile,
        n=context_lines,
    ))

    # Ensure each line ends with newline for consistent splitting later
    # Some lines from unified_diff may not have trailing newlines
    result_lines = []
    for line in diff_lines:
        if line.endswith('\n'):
            result_lines.append(line)
        else:
            result_lines.append(line + '\n')

    return "".join(result_lines)


def generate_page_diff(
    graph_path: Path,
    page_name: str,
    knowledge: KnowledgeBlock,
) -> Optional[str]:
    """Generate diff showing what would change on a page.

    Args:
        graph_path: Path to Logseq graph
        page_name: Name of target page
        knowledge: Knowledge block to integrate

    Returns:
        Unified diff string, or None if page doesn't exist
    """
    # Load the target page
    page = TargetPage.load(graph_path, page_name)
    if not page:
        return None

    # Get original and modified content
    original = page.outline.render()
    modified = simulate_integration(page, knowledge)

    # Generate diff
    return generate_unified_diff(
        original,
        modified,
        fromfile=f"a/{page_name}.md",
        tofile=f"b/{page_name}.md",
        context_lines=3,
    )
