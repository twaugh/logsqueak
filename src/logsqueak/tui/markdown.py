"""Shared markdown rendering utilities for TUI widgets.

This module provides functions to render Logseq markdown with rich text formatting.
Used across multiple screens and widgets for consistent formatting.
"""

from rich.text import Text


def render_markdown_to_markup(content: str, strip_id: bool = True) -> str:
    """
    Render Logseq markdown to Textual markup string.

    This version returns a markup string (e.g., "[bold]text[/bold]") suitable
    for use with Static widgets that accept markup strings.

    Args:
        content: Raw markdown content
        strip_id: If True, hide 'id::' properties (default: True)

    Returns:
        Markup string with Textual markup tags
    """
    if strip_id and content.lstrip().startswith("id::"):
        return ""  # Don't show id properties

    # Handle TODO/DONE/CANCELLED markers with checkbox indicators
    if content.startswith("TODO "):
        return f"[bold yellow]☐ TODO[/bold yellow] {_markdown_to_markup(content[5:])}"
    elif content.startswith("DONE "):
        return f"[bold green]☑ DONE[/bold green] {_markdown_to_markup(content[5:])}"
    elif content.startswith("CANCELLED ") or content.startswith("CANCELED "):
        prefix_len = 10 if content.startswith("CANCELLED ") else 9
        prefix = content[:prefix_len]
        rest = content[prefix_len:]
        return f"[bold dim]☒[/bold dim] [dim strike]{prefix}[/dim strike]{_markdown_to_markup(rest)}"

    return _markdown_to_markup(content)


def _markdown_to_markup(content: str) -> str:
    """Convert markdown syntax to Textual markup."""
    result = []
    i = 0
    while i < len(content):
        # Check for [[page links]]
        if content[i:i+2] == "[[":
            end = content.find("]]", i+2)
            if end != -1:
                link_text = content[i+2:end]
                result.append(f"[bold cyan underline]{link_text}[/bold cyan underline]")
                i = end + 2
                continue

        # Check for [markdown](url) style links
        if content[i] == "[":
            # Find matching closing bracket, accounting for nested brackets
            bracket_count = 1
            bracket_end = i + 1
            while bracket_end < len(content) and bracket_count > 0:
                if content[bracket_end] == "[":
                    bracket_count += 1
                elif content[bracket_end] == "]":
                    bracket_count -= 1
                bracket_end += 1
            bracket_end -= 1  # Point to the closing bracket, not past it

            if bracket_count == 0 and bracket_end + 1 < len(content) and content[bracket_end + 1] == "(":
                paren_end = content.find(")", bracket_end + 2)
                if paren_end != -1:
                    link_text = content[i+1:bracket_end]
                    url = content[bracket_end+2:paren_end]
                    result.append(f"[bold cyan underline]{link_text}[/bold cyan underline]")
                    i = paren_end + 1
                    continue

            # Not a markdown link - escape the bracket for Rich markup
            result.append("\\[")
            i += 1
            continue

        # Check for **bold**
        if content[i:i+2] == "**":
            end = content.find("**", i+2)
            if end != -1:
                bold_text = content[i+2:end]
                result.append(f"[bold]{bold_text}[/bold]")
                i = end + 2
                continue

        # Check for *italic*
        if content[i] == "*" and (i == 0 or content[i-1] != "*"):
            end = content.find("*", i+1)
            if end != -1 and (end+1 >= len(content) or content[end+1] != "*"):
                italic_text = content[i+1:end]
                result.append(f"[italic]{italic_text}[/italic]")
                i = end + 1
                continue

        # Check for `code`
        if content[i] == "`":
            end = content.find("`", i+1)
            if end != -1:
                code_text = content[i+1:end]
                result.append(f"[bold cyan on #222222]{code_text}[/bold cyan on #222222]")
                i = end + 1
                continue

        # Check for ~~strikethrough~~
        if content[i:i+2] == "~~":
            end = content.find("~~", i+2)
            if end != -1:
                strike_text = content[i+2:end]
                result.append(f"[strike dim]{strike_text}[/strike dim]")
                i = end + 2
                continue

        # Check for ~strikethrough~
        if content[i] == "~" and (i == 0 or content[i-1] != "~"):
            end = content.find("~", i+1)
            if end != -1 and (end+1 >= len(content) or content[end+1] != "~"):
                strike_text = content[i+1:end]
                result.append(f"[strike dim]{strike_text}[/strike dim]")
                i = end + 1
                continue

        # Check for unmatched closing bracket ] - escape it for Rich markup
        if content[i] == "]":
            result.append("\\]")
            i += 1
            continue

        # Regular character
        result.append(content[i])
        i += 1

    return "".join(result)


def render_markdown(content: str, strip_id: bool = True) -> Text:
    """
    Render Logseq markdown content to Rich Text with formatting.

    Supports:
    - **bold**
    - *italic*
    - `code`
    - [[page links]]
    - TODO/DONE/CANCELLED markers (with checkboxes)
    - ~~strikethrough~~

    Args:
        content: Raw markdown content
        strip_id: If True, hide 'id::' properties (default: True)

    Returns:
        Rich Text object with styled content
    """
    if strip_id and content.lstrip().startswith("id::"):
        return Text("")  # Don't show id properties

    text = Text()
    _render_markdown_to_text(content, text)
    return text


def _render_markdown_to_text(content: str, text: Text) -> None:
    """
    Render Logseq markdown to Rich Text with styles.

    Args:
        content: Raw markdown content
        text: Rich Text object to append styled text to
    """
    if not content or content == "":
        return

    # Handle TODO/DONE/CANCELLED markers with checkbox indicators
    if content.startswith("TODO "):
        text.append("☐ ", style="bold yellow")
        text.append("TODO ", style="yellow")
        content = content[5:]
    elif content.startswith("DONE "):
        text.append("☑ ", style="bold green")
        text.append("DONE ", style="green")
        content = content[5:]
    elif content.startswith("CANCELLED ") or content.startswith("CANCELED "):
        prefix_len = 10 if content.startswith("CANCELLED ") else 9
        text.append("☒ ", style="bold dim")
        text.append(content[:prefix_len], style="dim strike")
        content = content[prefix_len:]

    # Process the rest with simple parser
    i = 0
    while i < len(content):
        # Check for [[page links]]
        if content[i:i+2] == "[[":
            end = content.find("]]", i+2)
            if end != -1:
                link_text = content[i+2:end]
                text.append(link_text, style="bold cyan underline")
                i = end + 2
                continue

        # Check for [markdown](url) style links
        if content[i] == "[":
            # Find matching closing bracket, accounting for nested brackets
            bracket_count = 1
            bracket_end = i + 1
            while bracket_end < len(content) and bracket_count > 0:
                if content[bracket_end] == "[":
                    bracket_count += 1
                elif content[bracket_end] == "]":
                    bracket_count -= 1
                bracket_end += 1
            bracket_end -= 1  # Point to the closing bracket, not past it

            if bracket_count == 0 and bracket_end + 1 < len(content) and content[bracket_end + 1] == "(":
                paren_end = content.find(")", bracket_end + 2)
                if paren_end != -1:
                    link_text = content[i+1:bracket_end]
                    url = content[bracket_end+2:paren_end]
                    text.append(link_text, style="bold cyan underline")
                    i = paren_end + 1
                    continue

        # Check for **bold**
        if content[i:i+2] == "**":
            end = content.find("**", i+2)
            if end != -1:
                bold_text = content[i+2:end]
                text.append(bold_text, style="bold")
                i = end + 2
                continue

        # Check for *italic*
        if content[i] == "*" and (i == 0 or content[i-1] != "*"):
            end = content.find("*", i+1)
            if end != -1 and (end+1 >= len(content) or content[end+1] != "*"):
                italic_text = content[i+1:end]
                text.append(italic_text, style="italic")
                i = end + 1
                continue

        # Check for `code`
        if content[i] == "`":
            end = content.find("`", i+1)
            if end != -1:
                code_text = content[i+1:end]
                text.append(code_text, style="bold cyan on #222222")
                i = end + 1
                continue

        # Check for ~~strikethrough~~
        if content[i:i+2] == "~~":
            end = content.find("~~", i+2)
            if end != -1:
                strike_text = content[i+2:end]
                text.append(strike_text, style="strike dim")
                i = end + 2
                continue

        # Check for ~strikethrough~
        if content[i] == "~" and (i == 0 or content[i-1] != "~"):
            end = content.find("~", i+1)
            if end != -1 and (end+1 >= len(content) or content[end+1] != "~"):
                strike_text = content[i+1:end]
                text.append(strike_text, style="strike dim")
                i = end + 1
                continue

        # Regular character
        text.append(content[i])
        i += 1
