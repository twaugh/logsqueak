#!/usr/bin/env python3
"""Extract anonymized test cases from real Logseq graph for round-trip testing.

This script scans your Logseq graph, tests parser round-trip correctness on
each file, and reports failures with anonymized content for debugging.

It will:
1. Parse each markdown file with LogseqOutline.parse()
2. Render it back with outline.render()
3. Compare with original
4. Report failures with anonymized content

Usage:
    python scripts/extract_roundtrip_test_cases.py /path/to/logseq/graph
    python scripts/extract_roundtrip_test_cases.py /path/to/logseq/graph --output failures.txt
"""

import argparse
import difflib
import hashlib
import re
import sys
from pathlib import Path
from typing import List, Tuple, Optional

# Add src to path so we can import logsqueak modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from logsqueak.logseq.parser import LogseqOutline


def anonymize_text(text: str, preserve_structure: bool = True) -> str:
    """Anonymize text while preserving structure for testing.

    Preserves:
    - Punctuation
    - Content inside {braces}
    - Markdown structure (headings, code blocks, links, properties)
    - Leading/trailing whitespace

    Args:
        text: Original text
        preserve_structure: If True, preserve markdown structure markers

    Returns:
        Anonymized version
    """
    if not text.strip():
        return text

    # Preserve special markers
    if preserve_structure:
        # Preserve property syntax (key:: value) - anonymize both sides
        if '::' in text and not text.strip().startswith('```'):
            parts = text.split('::', 1)
            key_anon = anonymize_words(parts[0])
            value_anon = anonymize_words(parts[1]) if len(parts) > 1 else ""
            return f"{key_anon}::{value_anon}"

        # Preserve code block markers exactly
        if text.strip().startswith('```'):
            return text  # Keep code fence exactly as-is

        # Preserve headings
        if text.strip().startswith('#'):
            match = re.match(r'^(#+)\s*(.*)', text.strip())
            if match:
                level, heading_text = match.groups()
                return level + ' ' + anonymize_words(heading_text)

        # Preserve page links [[...]]
        text = re.sub(
            r'\[\[([^\]]+)\]\]',
            lambda m: f"[[{anonymize_words(m.group(1))}]]",
            text
        )

        # Preserve renderer syntax and content in braces
        # Keep everything inside {{ }} exactly as-is
        if '{{' in text:
            return text

    # Anonymize words while preserving punctuation and braces
    return anonymize_words(text)


def anonymize_words(text: str) -> str:
    """Anonymize words while preserving punctuation and content in braces.

    Args:
        text: Text to anonymize

    Returns:
        Anonymized text with punctuation and {braces} preserved
    """
    if not text.strip():
        return text

    result = []
    i = 0
    while i < len(text):
        # Preserve content inside curly braces
        if text[i] == '{':
            # Find matching closing brace
            depth = 1
            j = i + 1
            while j < len(text) and depth > 0:
                if text[j] == '{':
                    depth += 1
                elif text[j] == '}':
                    depth -= 1
                j += 1
            result.append(text[i:j])
            i = j
            continue

        # Check if this is start of a word (alphanumeric or underscore)
        if text[i].isalnum() or text[i] == '_':
            # Extract the whole word
            j = i
            while j < len(text) and (text[j].isalnum() or text[j] == '_'):
                j += 1
            word = text[i:j]

            # Hash the word
            word_hash = hashlib.md5(word.encode()).hexdigest()[:6]
            result.append(f"w{word_hash}")
            i = j
        else:
            # Preserve punctuation, whitespace, and special chars
            result.append(text[i])
            i += 1

    return ''.join(result)


def has_interesting_structure(content: str) -> Tuple[bool, List[str]]:
    """Check if content has interesting structural patterns worth testing.

    Returns:
        (is_interesting, list_of_reasons)
    """
    reasons = []

    # Check for code blocks
    if '```' in content:
        reasons.append("code_block")

    # Check for multi-line content (non-bullet continuation lines)
    lines = content.split('\n')
    has_multiline = False
    for i, line in enumerate(lines):
        if i > 0 and not line.lstrip().startswith('-') and line.strip():
            has_multiline = True
            break
    if has_multiline:
        reasons.append("multiline_content")

    # Check for properties on continuation lines
    if '::' in content:
        for line in lines:
            if '::' in line and not line.lstrip().startswith('-'):
                reasons.append("continuation_properties")
                break

    # Check for tab indentation
    if '\t' in content:
        reasons.append("tab_indentation")

    # Check for deeply nested structure (3+ levels)
    max_indent = 0
    for line in lines:
        if line.lstrip().startswith('-'):
            leading = line[:len(line) - len(line.lstrip())]
            indent = leading.count('  ') + leading.count('\t')
            max_indent = max(max_indent, indent)
    if max_indent >= 3:
        reasons.append("deep_nesting")

    # Check for renderers
    if '{{renderer' in content:
        reasons.append("renderer")

    return len(reasons) > 0, reasons


def test_roundtrip(content: str) -> Tuple[bool, Optional[str]]:
    """Test if content survives parse -> render round-trip.

    Args:
        content: Original markdown content

    Returns:
        (success, error_message) - error message if failed or exception
    """
    try:
        outline = LogseqOutline.parse(content)
        rendered = outline.render()

        if content == rendered:
            return True, None

        # Report data loss statistics
        orig_lines = len(content.splitlines())
        rend_lines = len(rendered.splitlines())
        lost_lines = orig_lines - rend_lines
        lost_bytes = len(content) - len(rendered)

        return False, f"Lost {lost_lines} lines ({lost_bytes} bytes)"

    except Exception as e:
        return False, f"Exception during parsing/rendering: {e}"


def extract_test_cases(graph_path: Path, max_cases: int = 20, include_passing: bool = False) -> List[Tuple[str, List[str], str, bool, Optional[str]]]:
    """Extract test cases from Logseq graph and test round-trip.

    Args:
        graph_path: Path to Logseq graph
        max_cases: Maximum number of failing test cases to extract
        include_passing: If True, include passing test cases in output

    Returns:
        List of (anonymized_content, reasons, original_file, passed, diff_or_error) tuples
    """
    pages_dir = graph_path / "pages"
    journals_dir = graph_path / "journals"

    if not pages_dir.exists() and not journals_dir.exists():
        print(f"Error: No pages or journals directory found in {graph_path}", file=sys.stderr)
        return []

    test_cases = []
    seen_patterns = set()  # Avoid duplicate patterns

    total_files = 0
    total_passed = 0
    total_failed = 0

    # Scan all markdown files
    all_files = []
    if pages_dir.exists():
        all_files.extend(pages_dir.glob("*.md"))
    if journals_dir.exists():
        all_files.extend(journals_dir.glob("*.md"))

    for md_file in all_files:
        # Stop collecting after max_cases failures (but keep counting stats)
        if len([tc for tc in test_cases if not tc[3]]) >= max_cases and not include_passing:
            # Keep processing to get accurate stats
            pass

        total_files += 1

        try:
            content = md_file.read_text(encoding='utf-8')
        except Exception as e:
            print(f"Warning: Could not read {md_file}: {e}", file=sys.stderr)
            continue

        # Test round-trip
        passed, diff_or_error = test_roundtrip(content)

        if passed:
            total_passed += 1
            if not include_passing:
                continue
        else:
            total_failed += 1

        # Check if interesting (for categorization)
        is_interesting, reasons = has_interesting_structure(content)
        if not reasons:
            reasons = ["simple_structure"]

        # Create pattern signature to avoid duplicates
        pattern_sig = ','.join(sorted(reasons))
        if pattern_sig in seen_patterns and not include_passing:
            continue

        seen_patterns.add(pattern_sig)

        # Anonymize content
        anonymized = anonymize_content(content)

        # Only add to results if we haven't hit max_cases for failures
        if len([tc for tc in test_cases if not tc[3]]) < max_cases or include_passing:
            test_cases.append((anonymized, reasons, str(md_file), passed, diff_or_error))

    print(f"\nRound-trip test statistics:", file=sys.stderr)
    print(f"  Total files: {total_files}", file=sys.stderr)
    print(f"  Passed: {total_passed} ({total_passed*100//total_files if total_files else 0}%)", file=sys.stderr)
    print(f"  Failed: {total_failed} ({total_failed*100//total_files if total_files else 0}%)", file=sys.stderr)

    return test_cases


def anonymize_content(content: str) -> str:
    """Anonymize markdown content while preserving structure.

    Args:
        content: Original markdown content

    Returns:
        Anonymized content with same structure
    """
    lines = content.split('\n')
    result_lines = []

    for line in lines:
        if not line.strip():
            result_lines.append(line)
            continue

        # Preserve leading whitespace
        leading = line[:len(line) - len(line.lstrip())]
        stripped = line.lstrip()

        # Process bullet lines
        if stripped.startswith('- '):
            bullet_content = stripped[2:]
            anonymized = anonymize_text(bullet_content, preserve_structure=True)
            result_lines.append(f"{leading}- {anonymized}")
        else:
            # Continuation line
            anonymized = anonymize_text(stripped, preserve_structure=True)
            result_lines.append(f"{leading}{anonymized}")

    return '\n'.join(result_lines)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test parser round-trip on Logseq graph and extract failing cases"
    )
    parser.add_argument(
        "graph_path",
        type=Path,
        help="Path to Logseq graph directory"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output file (default: stdout)"
    )
    parser.add_argument(
        "--max-cases", "-n",
        type=int,
        default=20,
        help="Maximum number of failing test cases to extract (default: 20)"
    )
    parser.add_argument(
        "--include-passing",
        action="store_true",
        help="Include passing test cases in output (default: only failures)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    if not args.graph_path.exists():
        print(f"Error: Graph path does not exist: {args.graph_path}", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print(f"Testing round-trip on {args.graph_path}...", file=sys.stderr)

    test_cases = extract_test_cases(
        args.graph_path,
        args.max_cases,
        args.include_passing
    )

    if args.verbose:
        failures = [tc for tc in test_cases if not tc[3]]
        print(f"\nReporting {len(failures)} failing test cases", file=sys.stderr)

    # Format output
    output_lines = []
    output_lines.append("# Round-Trip Test Results")
    output_lines.append("# Generated from real Logseq graph")
    output_lines.append("# Content has been anonymized")
    output_lines.append("")

    failures = [tc for tc in test_cases if not tc[3]]
    passes = [tc for tc in test_cases if tc[3]]

    if failures:
        output_lines.append(f"## FAILURES ({len(failures)})")
        output_lines.append("")

        for i, (content, reasons, source_file, passed, error_msg) in enumerate(failures, 1):
            output_lines.append(f"### Failure {i}")
            output_lines.append(f"**Source:** `{Path(source_file).name}`")
            output_lines.append(f"**Patterns:** {', '.join(reasons)}")
            if error_msg:
                output_lines.append(f"**Error:** {error_msg}")
            output_lines.append("")
            output_lines.append("**Original (anonymized):**")
            output_lines.append("```markdown")
            output_lines.append(content)
            output_lines.append("```")
            output_lines.append("")
            output_lines.append("---")
            output_lines.append("")

    if passes and args.include_passing:
        output_lines.append(f"## PASSES ({len(passes)})")
        output_lines.append("")

        for i, (content, reasons, source_file, passed, _) in enumerate(passes, 1):
            output_lines.append(f"### Pass {i}")
            output_lines.append(f"**Source:** `{Path(source_file).name}`")
            output_lines.append(f"**Patterns:** {', '.join(reasons)}")
            output_lines.append("")
            output_lines.append("```markdown")
            output_lines.append(content)
            output_lines.append("```")
            output_lines.append("")
            output_lines.append("---")
            output_lines.append("")

    # Write output
    output_text = '\n'.join(output_lines)

    if args.output:
        args.output.write_text(output_text, encoding='utf-8')
        print(f"\nWrote results to {args.output}", file=sys.stderr)
    else:
        print(output_text)


if __name__ == "__main__":
    main()
