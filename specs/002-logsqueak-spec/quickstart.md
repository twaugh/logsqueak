# Quickstart: Logsqueak Interactive TUI

**Feature**: 002-logsqueak-spec

**Status**: Specification

**Date**: 2025-11-05

This guide walks you through installing, configuring, and using the Logsqueak Interactive TUI for extracting lasting knowledge from your Logseq journal entries.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [First Run](#first-run)
5. [Walkthrough](#walkthrough)
   - [Phase 1: Block Selection](#phase-1-block-selection)
   - [Phase 2: Content Editing](#phase-2-content-editing)
   - [Phase 3: Integration Decisions](#phase-3-integration-decisions)

6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before installing Logsqueak, ensure you have:

### 1. Python 3.11 or Higher

Check your Python version:

```bash
python3 --version
# Should show: Python 3.11.x or higher

```

If you need to install Python 3.11+, visit [python.org](https://www.python.org/downloads/).

### 2. Logseq Graph with Journal Entries

- You should have an existing Logseq graph with journal entries
- Journal entries should contain a mix of tasks, activities, and knowledge blocks
- Knowledge blocks are insights, learnings, or notes you want to preserve beyond the daily journal

### 3. LLM API Access

Choose one of the following:

**Option A: OpenAI API** (Recommended for best quality)

- Sign up at [platform.openai.com](https://platform.openai.com/)
- Generate an API key
- Recommended model: `gpt-4-turbo-preview` or `gpt-4o`

**Option B: Ollama** (Local, free, privacy-focused)

- Install Ollama from [ollama.ai](https://ollama.ai/)
- Pull a model: `ollama pull llama2` or `ollama pull mistral`
- Ollama runs on `http://localhost:11434/v1` by default

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/logsqueak.git
cd logsqueak

```

### 2. Set Up Virtual Environment

Create and activate a Python virtual environment:

```bash
# Create virtual environment
python3 -m venv venv

# Activate on Linux/macOS
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate

```

Your terminal prompt should now show `(venv)` prefix.

### 3. Install Dependencies

Install Logsqueak and all dependencies from `pyproject.toml`:

```bash
pip install -e .

```

This will install:

- `textual` - TUI framework
- `httpx` - HTTP client for LLM streaming
- `pydantic` - Data validation
- `click` - CLI framework
- `chromadb` - Vector database for RAG search
- `sentence-transformers` - Text embeddings (large download ~400MB)
- `markdown-it-py` - Markdown rendering
- `pyyaml` - YAML configuration parsing

**Note**: First-time installation downloads the sentence-transformers model (`all-MiniLM-L6-v2`, ~90MB), which may take a few minutes.

### 4. Verify Installation

Check that the `logsqueak` command is available:

```bash
logsqueak --help

```

You should see usage information.

---

## Configuration

### 1. Create Configuration Directory

```bash
mkdir -p ~/.config/logsqueak

```

### 2. Create Configuration File

Create `~/.config/logsqueak/config.yaml` with your settings:

#### Option A: OpenAI Configuration

```yaml
llm:
  endpoint: https://api.openai.com/v1
  api_key: sk-proj-YOUR_OPENAI_API_KEY_HERE
  model: gpt-4-turbo-preview

logseq:
  graph_path: ~/Documents/logseq-graph

rag:
  top_k: 10  # Optional: Number of candidate pages for RAG search

```

#### Option B: Ollama Configuration

```yaml
llm:
  endpoint: http://localhost:11434/v1
  api_key: ollama  # Ollama doesn't use API keys, but field is required
  model: llama2    # Or mistral, llama3, etc.
  num_ctx: 32768   # Optional: Context window size (controls VRAM usage)

logseq:
  graph_path: ~/Documents/logseq-graph

rag:
  top_k: 10  # Optional

```

**Configuration Fields**:

| Section | Field | Required | Description | Default |
|---------|-------|----------|-------------|---------|
| `llm` | `endpoint` | Yes | LLM API endpoint URL | - |
| `llm` | `api_key` | Yes | API key for authentication | - |
| `llm` | `model` | Yes | Model identifier | - |
| `llm` | `num_ctx` | No | Context window size (Ollama only) | 32768 |
| `logseq` | `graph_path` | Yes | Path to your Logseq graph directory | - |
| `rag` | `top_k` | No | Number of candidate pages to retrieve | 10 |

### 3. Set File Permissions

**IMPORTANT**: The config file MUST have restrictive permissions (mode 600) to protect your API key:

```bash
chmod 600 ~/.config/logsqueak/config.yaml

```

Verify permissions:

```bash
ls -l ~/.config/logsqueak/config.yaml
# Should show: -rw------- (600)

```

If permissions are too open (e.g., 644), Logsqueak will refuse to start with a permission error.

### 4. Verify Configuration

Test your configuration:

```bash
logsqueak --version

```

If there are any configuration errors, Logsqueak will display a clear error message indicating which setting is invalid.

---

## First Run

### CLI Command Syntax

```bash
# Extract from today's journal
logsqueak extract

# Extract from a specific date
logsqueak extract 2025-01-15

# Extract from a date range
logsqueak extract 2025-01-10..2025-01-15

```

### Example: Extract from Today's Journal

```bash
logsqueak extract

```

**What happens on first run**:

1. **Configuration Loading**: Logsqueak reads `~/.config/logsqueak/config.yaml` and checks file permissions
2. **Graph Validation**: Verifies that the Logseq graph path exists and is accessible
3. **Journal Loading**: Parses today's journal entry (e.g., `journals/2025-11-05.md`)
4. **Phase 1 Start**: Displays the interactive TUI with your journal blocks in a tree view
5. **Background Tasks Begin**: LLM classification and page indexing start automatically

**Expected Output**:

```
Loading configuration from ~/.config/logsqueak/config.yaml
Graph path: /home/user/Documents/logseq-graph
Loading journal: 2025-11-05

[Interactive TUI launches - see Phase 1 below]

```

### Common First-Run Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `Config file not found` | Missing config file | Create `~/.config/logsqueak/config.yaml` |
| `Overly permissive permissions` | File mode is not 600 | Run `chmod 600 ~/.config/logsqueak/config.yaml` |
| `Graph path does not exist` | Invalid Logseq graph path | Update `logseq.graph_path` in config |
| `Connection refused` | Ollama not running | Start Ollama: `ollama serve` |
| `Invalid API key` | Wrong OpenAI API key | Update `llm.api_key` in config |

---

## Walkthrough

This section walks through a complete knowledge extraction session, showing what to expect at each phase and which keyboard shortcuts to use.

### Phase 1: Block Selection

**Purpose**: Review your journal blocks and select which ones contain lasting knowledge worth extracting.

#### What You'll See

```
┌──────────────────────────────────────────────────────────────────────┐
│ Phase 1: Block Selection                                             │
├──────────────────────────────────────────────────────────────────────┤
│ Journal Blocks (2025-11-05)                                          │
│ ┌────────────────────────────────────────────────────────────────┐   │
│ │ ▼ 2025-11-05                                                   │   │
│ │   ▶ TODO Review PR #123                                        │   │
│ │ 🤖 ▶ Learned that asyncio.create_task() enables concurrent o...│   │
│ │   🤖 └─ This is different from await which blocks and makes... │   │
│ │   ▶ Had lunch with the team                                    │   │
│ │ 🤖 ▶ Python type hints improve code clarity and help catch b...│   │
│ │   ▶ Reviewed architecture docs                                 │   │
│ └────────────────────────────────────────────────────────────────┘   │
│                                                                      │
│ Selected Block Details                                               │
│ ┌────────────────────────────────────────────────────────────────┐   │
│ │ This is different from await which blocks and makes the code   │   │
│ │ execute sequentially rather than concurrently.                 │   │
│ │                                                                │   │
│ │ LLM Reasoning: Explains a key distinction between async        │   │
│ │ patterns in Python that helps understand concurrency concepts. │   │
│ │ Confidence: 88%                                                │   │
│ └────────────────────────────────────────────────────────────────┘   │
│                                                                      │
│ Status: Analyzing knowledge blocks (3/15) | Building page index 45%  │
├──────────────────────────────────────────────────────────────────────┤
│ j/k: Navigate | Space: Select | a: Accept all | n: Next | q: Quit    │
└──────────────────────────────────────────────────────────────────────┘

```

**Note on display**:

- 🤖 Robot emoji appears in left margin (before the `▶` tree icon)
- Block content alignment is consistent regardless of emoji presence
- Long lines are truncated with `...` at screen edge in tree view
- Selected Block Details panel shows full wrapped content with all lines
- Multi-line blocks wrap naturally in the details panel

#### Visual Indicators

- **🤖 Robot emoji**: Block suggested by LLM as knowledge
- **Green highlight**: Block selected by you for extraction
- **Yellow highlight**: Block suggested by LLM (not yet selected by you)
- **Bottom panel**: Shows full content and LLM reasoning for selected block

#### Keyboard Controls

| Key | Action |
|-----|--------|
| `j` or `↓` | Navigate to next block |
| `k` or `↑` | Navigate to previous block |
| `Shift+j` or `Shift+↓` | Jump to next LLM-suggested knowledge block |
| `Shift+k` or `Shift+↑` | Jump to previous LLM-suggested knowledge block |
| `Space` | Toggle selection on current block |
| `a` | Accept all LLM suggestions |
| `r` | Reset current block to LLM suggestion |
| `c` | Clear all selections |
| `n` | Continue to Phase 2 (only available if at least 1 block selected) |
| `q` | Quit application |

#### Step-by-Step Example

1. **Start**: TUI opens showing all blocks in tree view
2. **Wait for LLM**: Watch as blocks are highlighted (yellow) as LLM identifies knowledge
   - Status widget shows: "Analyzing knowledge blocks (3/15)"
3. **Navigate**: Press `j` to move down through blocks
4. **Review**: Bottom panel shows full content and LLM reasoning
5. **Quick navigation**: Press `Shift+j` to jump directly to next LLM-suggested block
6. **Select**: Press `Space` to select current block (highlight turns green)
7. **Accept all**: Press `a` to quickly select all LLM suggestions
8. **Proceed**: Press `n` to move to Phase 2

#### Tips

- **Let the LLM finish**: While you can start selecting immediately, waiting a few seconds lets the LLM identify knowledge blocks automatically
- **Review suggestions**: Check the LLM reasoning (bottom panel) to understand why a block was suggested
- **Manual selection**: You can select any block, even if LLM didn't suggest it
- **Use Shift+navigation**: Quickly review only LLM-suggested blocks with `Shift+j`/`Shift+k`

---

### Phase 2: Content Editing

**Purpose**: Refine the selected knowledge blocks by removing temporal context (e.g., "today", "yesterday") and making content evergreen.

#### What You'll See

```
┌─────────────────────────────────────────────────────────────────────┐
│ Phase 2: Content Editing                                            │
├─────────────────────────────────────────────────────────────────────┤
│ Knowledge Block 1 of 3                                              │
│                                                                     │
│ Original Context (from journal):                                    │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │ - Daily Standup                                                 │ │
│ │   - Learned that asyncio.create_task() enables concurrency      │ │
│ │     - This is different from await which blocks                 │ │
│ └─────────────────────────────────────────────────────────────────┘ │
│                                                                     │
│ LLM-Reworded Version:                                               │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │ Using asyncio.create_task() enables concurrent operations in    │ │
│ │ Python, unlike await which executes sequentially.               │ │
│ └─────────────────────────────────────────────────────────────────┘ │
│                                                                     │
│ Current Content (editable):                                         │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │ Using asyncio.create_task() enables concurrent operations in    │ │
│ │ Python, unlike await which executes sequentially.               │ │
│ │ [Cursor here - press Tab to edit]                               │ │
│ └─────────────────────────────────────────────────────────────────┘ │
│                                                                     │
│ Status: Refining content (2/3) | Building page index: 78%           │
│         Finding relevant pages (1/3)                                │
├─────────────────────────────────────────────────────────────────────┤
│ j/k: Navigate | Tab: Edit | a: Accept LLM | r: Revert | n: Next     │
└─────────────────────────────────────────────────────────────────────┘

```

#### Sections Explained

1. **Original Context**: Shows the journal hierarchy (read-only, dimmed) so you remember where this knowledge came from
2. **LLM-Reworded Version**: Shows the LLM's suggested rewrite that removes temporal context
3. **Current Content**: Editable text field - this is what will be integrated into your knowledge base

#### Keyboard Controls

| Key | Action |
|-----|--------|
| `j` or `↓` | Navigate to next knowledge block (saves current edits) |
| `k` or `↑` | Navigate to previous knowledge block (saves current edits) |
| `Tab` | Focus text editor for manual editing / Unfocus to navigate |
| `a` | Accept LLM-reworded version (replaces editor content) |
| `r` | Revert to original journal content |
| `n` | Continue to Phase 3 (only available after page indexing + RAG search complete) |
| `q` | Back to Phase 1 |

**While editing** (text editor focused):

- Arrow keys, Home, End: Navigate within text
- Standard text editing (insert, delete, copy/paste)
- Enter: New line
- Tab: Unfocus editor to enable navigation

#### Step-by-Step Example

1. **Review original**: See the journal hierarchy context (top panel)
2. **Review LLM version**: Read the reworded version (middle panel)
3. **Accept or edit**:
   - Press `a` to accept LLM version (updates current content)
   - OR press `Tab` to manually edit the text
4. **Manual editing** (if Tab pressed):
   - Type to modify content
   - Press `Tab` again to unfocus and return to navigation
5. **Next block**: Press `j` to move to next knowledge block (auto-saves edits)
6. **Wait for RAG**: Status shows "Finding relevant pages (2/3)"
7. **Proceed**: Press `n` when RAG search completes

#### Tips

- **LLM suggestions are good**: The LLM is quite good at removing "today", "learned", "realized" temporal phrases
- **Review for accuracy**: Always check that the reworded version preserves the actual meaning
- **Manual tweaks**: Sometimes you'll want to add context the LLM removed, or fix awkward phrasing
- **Original is preserved**: The journal entry is never modified - you can always revert to original
- **Background tasks**: Page indexing and RAG search run in background while you edit

---

### Phase 3: Integration Decisions

**Purpose**: Review where each knowledge block will be integrated, see previews in context, and approve integrations.

#### What You'll See

```
┌─────────────────────────────────────────────────────────────────────┐
│ Phase 3: Integration Decisions                                      │
├─────────────────────────────────────────────────────────────────────┤
│ Knowledge Block 1 of 3 (Decision 1 of 2: ⊙ pending)                 │
│                                                                     │
│ Journal Context (source):                                           │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │ - Daily Standup                                                 │ │
│ │   - Learned that asyncio.create_task() enables concurrency      │ │
│ └─────────────────────────────────────────────────────────────────┘ │
│                                                                     │
│ Content to Integrate:                                               │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │ Using asyncio.create_task() enables concurrent operations in    │ │
│ │ Python, unlike await which executes sequentially.               │ │
│ └─────────────────────────────────────────────────────────────────┘ │
│                                                                     │
│ Target Page: Programming Notes/Python (Confidence: 87%)             │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │ - Async Patterns                                                │ │
│ │   - Event loops handle concurrent tasks                         │ │
│ │   - Use async/await for I/O-bound operations                    │ │
│ │  ┃- Using asyncio.create_task() enables concurrent operations   │ │
│ │  ┃  in Python, unlike await which executes sequentially.        │ │
│ │   - Common pitfalls with asyncio                                │ │
│ │ - Type Hints                                                     │ │
│ │   ...                                                            │ │
│ └─────────────────────────────────────────────────────────────────┘ │
│                                                                     │
│ Reasoning: This insight fits well under the 'Async Patterns'        │
│ section as it directly explains task concurrency.                   │
├─────────────────────────────────────────────────────────────────────┤
│ j/k: Navigate decisions | y: Accept | n: Next block | a: Accept all │
└─────────────────────────────────────────────────────────────────────┘

```

**Note on Target Page Preview:**
- Preview automatically scrolls to show the insertion point (green bar `┃`)
- For long pages, press `Tab` to focus the preview widget
- When focused, use `↑`/`↓` arrows to scroll and see more context
- Press `Tab` again to return focus to decision navigation
- `j`/`k` navigate between decisions (different target pages/insertion points)

#### Visual Indicators

- **Green bar (`┃`)**: Shows where new content will be inserted in target page
- **⊙ Pending**: Decision not yet accepted
- **✓ Completed**: Decision accepted and written successfully
- **⚠ Failed**: Write operation failed (with error details)
- **▶ Selected**: Currently selected decision in list

#### Keyboard Controls

| Key | Action |
|-----|--------|
| `j` | Navigate to next decision for current knowledge block |
| `k` | Navigate to previous decision for current knowledge block |
| `Tab` | Toggle focus on target page preview (enables scrolling with arrows) |
| `↑`/`↓` | Scroll preview when focused (otherwise same as `k`/`j`) |
| `y` | Accept current decision and write immediately |
| `n` | Skip to next knowledge block (leaves remaining decisions unwritten) |
| `a` | Accept and write ALL pending decisions for current block, then advance |
| `Enter` | Advance to next knowledge block (when all decisions processed) |
| `q` | Back to Phase 2 |

#### Step-by-Step Example

1. **Wait for processing**: Status shows "Processing knowledge blocks..." while LLM evaluates candidate pages
2. **Review knowledge block**: See journal context (where it came from) and refined content (what will be written)
3. **View first decision**: Display shows "Decision 1 of 2" with target page preview
4. **Review preview**: Target page automatically scrolls to show insertion point (green bar `┃`)
   - If you need more context, press `Tab` to focus the preview
   - Use `↑`/`↓` arrows to scroll and see surrounding content
   - Press `Tab` again to return to decision navigation
5. **Accept or navigate**:
   - Press `y` to accept this decision and write immediately
   - Press `j` to see the next decision (same block, different page or location)
6. **Multiple decisions for same block**:
   - Each `j`/`k` press cycles through decisions for current knowledge block
   - Preview automatically updates and scrolls to show each decision's insertion point
   - You can accept multiple decisions to integrate same knowledge to different pages
7. **Track progress**: Decision status shows ⊙ pending → ✓ completed (or ⚠ failed)
8. **Next block**: Press `n` to advance to next knowledge block

#### Integration Actions

| Action | Description | Example |
|--------|-------------|---------|
| **Add Section** | Create new top-level section in page | `- New Section\n  - [knowledge block]` |
| **Add Under** | Add as child under specific block | Insert under "Async Patterns" heading |
| **Replace** | Replace existing block content | Update outdated information |

#### Understanding Confidence Scores

- **85-100%**: High confidence - LLM is very certain this is a good fit
- **70-84%**: Medium confidence - Probably a good fit, but review carefully
- **50-69%**: Low confidence - Less certain, may be tangentially related

You can accept any decision regardless of confidence - the score is just guidance.

#### Completion Summary

When all knowledge blocks are processed, you'll see:

```
┌─────────────────────────────────────────────────────────────────────┐
│ Extraction Complete                                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ Summary:                                                            │
│ - 3 knowledge blocks processed                                      │
│ - 5 integrations written successfully                               │
│ - 0 failed integrations                                             │
│                                                                     │
│ Journal entry updated with provenance markers:                      │
│ journals/2025-11-05.md                                              │
│                                                                     │
│ Press Enter to exit                                                 │
└─────────────────────────────────────────────────────────────────────┘

```

#### Tips

- **Review previews carefully**: The green bar (`┃`) shows exactly where content will appear
- **Multiple pages**: Same knowledge can be relevant to multiple pages - accept all that make sense
- **Skip irrelevant**: Press `n` to skip if LLM suggestions don't fit
- **No relevant pages**: If LLM finds no good match, it shows "No relevant pages found" - press `n` to skip
- **Atomic writes**: Each `y` press writes immediately and marks journal - if you quit mid-session, completed integrations remain
- **Provenance**: Check your journal entry after completion - `processed::` property links to all integrated blocks

---

## Troubleshooting

### Configuration Errors

#### Error: Config file not found

```
Error: Configuration file not found at ~/.config/logsqueak/config.yaml

Please create the file with the following format:

llm:
  endpoint: https://api.openai.com/v1
  api_key: YOUR_API_KEY_HERE
  model: gpt-4-turbo-preview

logseq:
  graph_path: ~/Documents/logseq-graph

rag:
  top_k: 10
```

**Solution**: Create the config file with your settings (see [Configuration](#configuration) section above).

---

#### Error: Overly permissive file permissions

```
Error: Config file has overly permissive permissions: 0o644
Run: chmod 600 ~/.config/logsqueak/config.yaml
```

**Solution**:

```bash
chmod 600 ~/.config/logsqueak/config.yaml
```

---

#### Error: Graph path does not exist

```
Error: Graph path does not exist: /home/user/Documents/logseq-graph
Please create the directory or update config.yaml
```

**Solution**: Update `logseq.graph_path` in config.yaml with the correct path to your Logseq graph.

---

### Network Errors

#### Error: Connection refused (Ollama)

```
Error: Failed to connect to LLM API at http://localhost:11434/v1
Connection refused

Auto-retrying in 2 seconds...
```

**Solution**: Start Ollama:

```bash
ollama serve
```

If already running, check the endpoint in config matches Ollama's actual endpoint.

---

#### Error: Invalid API key (OpenAI)

```
Error: LLM API request failed with status 401 Unauthorized
Invalid API key provided

Please check your API key in ~/.config/logsqueak/config.yaml
```

**Solution**:

1. Verify your API key at [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Update `llm.api_key` in config.yaml
3. Restart Logsqueak

---

#### Error: Request timeout

```
Error: LLM request timed out after 60 seconds

Auto-retrying in 2 seconds...
```

**Solution**:

- **Ollama**: Your model may be too slow - try a smaller model or increase VRAM allocation
- **OpenAI**: Network issues - check your internet connection
- **Both**: If retry also fails, you'll be prompted to retry manually or skip

---

### Validation Errors

#### Error: Invalid YAML syntax

```
Error: Failed to parse configuration file
YAML syntax error at line 5: mapping values are not allowed here
```

**Solution**: Check your YAML syntax - common issues:

- Missing spaces after colons (should be `key: value`, not `key:value`)
- Incorrect indentation (use 2 spaces, not tabs)
- Missing quotes around special characters

---

#### Error: Target block not found (Phase 3)

```
Error writing integration decision:
Target block 'section-async-patterns' not found in page 'Programming Notes/Python'

File may have been modified externally.
```

**Solution**:

- The target page was edited in Logseq while TUI was running
- Logsqueak automatically reloaded the file but the target block no longer exists
- Decision marked as failed (⚠) - you can continue with other decisions
- Consider manually adding the knowledge block to the page in Logseq

---

### Checking Logs

If you encounter unexpected behavior, check the logs for detailed diagnostic information:

```bash
# View recent logs
tail -f ~/.cache/logsqueak/logs/logsqueak.log

# Search for errors
grep ERROR ~/.cache/logsqueak/logs/logsqueak.log

# View logs in JSON pretty-print
cat ~/.cache/logsqueak/logs/logsqueak.log | jq .
```

Log levels:

- **DEBUG**: LLM response chunks, state changes
- **INFO**: User actions, phase transitions, LLM requests
- **WARNING**: Partial failures, retry attempts
- **ERROR**: Operation failures, validation errors

---

### Common Questions

**Q: Can I process multiple journal entries at once?**

A: Yes! Use date range syntax:

```bash
logsqueak extract 2025-01-10..2025-01-15
```

The tree view will show entries grouped by date.

**Q: What if I quit mid-session?**

A: Integrations already written (✓ Completed) remain in your pages. The journal's `processed::` property tracks which blocks were integrated. Blocks you hadn't reached or skipped are not modified.

**Q: Can I edit the LLM's suggestions?**

A: Yes! In Phase 2, press `Tab` to manually edit any knowledge block content. In Phase 3, you can skip decisions you disagree with.

**Q: How do I undo an integration?**

A: Currently no built-in undo. You'll need to manually remove the integrated block from the target page in Logseq. The journal's `processed::` property shows which blocks were integrated and where.

**Q: Why isn't the 'n' key working in Phase 2?**

A: The 'n' key (continue to Phase 3) is disabled until background tasks complete. Wait for "Building page index" and "Finding relevant pages" to finish (status widget shows progress).

**Q: Can I cancel during a write operation?**

A: Pressing Ctrl+C during Phase 3 will show a warning about partial journal state (some blocks already marked as processed). You'll be asked to confirm cancellation.

---

## Next Steps

After completing your first extraction session:

1. **Review integrated blocks**: Open the target pages in Logseq and verify integrations look correct
2. **Check journal provenance**: Open the journal entry and see the `processed::` property with links to integrated blocks
3. **Process more journals**: Run `logsqueak extract` on different dates or date ranges
4. **Refine workflow**: Experiment with accepting all LLM suggestions (`a` key) vs manually reviewing each block

---

## Getting Help

- **Documentation**: See `specs/002-logsqueak-spec/spec.md` for complete feature specification
- **Issues**: Report bugs or request features on GitHub
- **Logs**: Always check `~/.cache/logsqueak/logs/logsqueak.log` for diagnostic information

---

**Happy knowledge extraction!**
