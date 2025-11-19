# Logsqueak

[![Tests](https://github.com/twaugh/logsqueak/actions/workflows/test.yml/badge.svg)](https://github.com/twaugh/logsqueak/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/twaugh/logsqueak/branch/main/graph/badge.svg)](https://codecov.io/gh/twaugh/logsqueak)

**Turn your Logseq journal chaos into organized knowledge.**

Logsqueak helps you extract lasting insights from daily journal entries using AI. Review what the AI finds, refine the content, and integrate it into your knowledge base—all through an interactive keyboard-driven interface.

![Logsqueak Demo](demo/demo.gif)

---

## Quick Start

**Get running in 5 minutes:**

```bash
# 1. Clone and install
git clone https://github.com/twaugh/logsqueak.git
cd logsqueak
./setup-dev.sh

# 2. Try it with the included test graph
source venv/bin/activate
logsqueak init  # Follow the interactive setup wizard

# 3. Extract knowledge from a sample journal
logsqueak extract 2025-01-15
```

**What happens next:**
- **Phase 1**: AI identifies knowledge blocks (you can select/deselect)
- **Phase 2**: AI suggests better wording (you can edit or accept)
- **Phase 3**: AI suggests where to save it (you approve each one)

That's it! Your knowledge is now organized in your Logseq graph.

---

## What is Logsqueak?

If you use Logseq journals to capture ideas during your day, you've probably noticed:
- Great insights get buried in daily logs
- Finding that one useful tip from last month is hard
- Your knowledge base stays empty while journals pile up

**Logsqueak solves this** by:

1. **Finding knowledge** - AI reads your journals and identifies valuable content (technical tips, lessons learned, insights)
2. **Cleaning it up** - AI removes temporal context ("today I learned...") and improves clarity
3. **Organizing it** - AI suggests where to save it in your knowledge base (you review and approve)

All through a keyboard-driven terminal interface—no mouse needed.

---

## Before You Start

**You'll need:**

- ✓ **Python 3.11 or later**
- ✓ **A Logseq graph** with journal entries
  - *Don't have one?* Use the included `test-graph/` directory to try it out
- ✓ **Access to an AI assistant** (choose one):
  - **Free**: [Ollama](https://ollama.com/) running locally (recommended for beginners)
  - **Paid**: OpenAI API key
- ✓ **~500MB disk space** for dependencies

**New to Ollama?** It's free software that runs AI models on your computer. [Install guide →](https://ollama.com/download)

---

## Installation

### Step 1: Install Logsqueak

**Recommended: Automated setup**

```bash
# Clone the repository
git clone https://github.com/twaugh/logsqueak.git
cd logsqueak

# Run setup script (creates virtual environment and installs everything)
./setup-dev.sh
```

**Manual setup** (if you prefer):

<details>
<summary>Click to expand manual installation steps</summary>

```bash
# Clone the repository
git clone https://github.com/twaugh/logsqueak.git
cd logsqueak

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -e .
pip install -e src/logseq-outline-parser/

# Verify installation
pytest -v
```

</details>

### Step 2: Set Up Your AI Assistant

**Option A: Ollama (Free, runs locally)**

```bash
# 1. Install Ollama from https://ollama.com/download

# 2. Pull the recommended model:
ollama pull mistral:7b-instruct

# 3. Make sure Ollama is running
ollama serve
```

**Option B: OpenAI (Paid, cloud-based)**

Requires an API key from [platform.openai.com/api-keys](https://platform.openai.com/api-keys)

Or: use any service provide an OpenAI-compatible API.

### Step 3: Configure Logsqueak

**Interactive setup wizard** (recommended):

```bash
source venv/bin/activate
logsqueak init
```

The wizard will guide you through:
1. Selecting your Logseq graph location
2. Configuring your AI assistant (Ollama or OpenAI-compatible)
3. Setting up semantic search

**Manual configuration** (advanced):

<details>
<summary>Click to expand manual config instructions</summary>

Create `~/.config/logsqueak/config.yaml`:

```bash
mkdir -p ~/.config/logsqueak
nano ~/.config/logsqueak/config.yaml
```

**For Ollama (local AI):**
```yaml
llm:
  endpoint: http://localhost:11434/v1
  api_key: ollama  # Any string works for local Ollama
  model: mistral:7b-instruct
  num_ctx: 32768  # Optional: controls VRAM usage

logseq:
  graph_path: ~/Documents/logseq-graph  # Path to your graph
```

**For OpenAI:**
```yaml
llm:
  endpoint: https://api.openai.com/v1
  api_key: sk-proj-xxxxxxxxxxxxx  # Your API key
  model: your-chosen-model

logseq:
  graph_path: ~/Documents/logseq-graph
```

Set correct permissions:
```bash
chmod 600 ~/.config/logsqueak/config.yaml
```

</details>

---

## Usage

### Try It with the Test Graph

The repository includes a sample Logseq graph with realistic journal entries:

```bash
source venv/bin/activate

# Configure to use test-graph (if not already done)
logsqueak init  # Point to /path/to/logsqueak/test-graph

# Extract knowledge from a sample journal entry
logsqueak extract 2025-01-15
```

**What you'll see:**

**Phase 1 - Block Selection**
```
The AI reads the journal and highlights blocks like:
✓ "Python 3.12 type hints improvements..." (knowledge)
✗ "Morning standup at 9am" (activity log)

Navigate with j/k, press Space to select/deselect, then 'n' to continue.
```

**Phase 2 - Content Editing**
```
Original: "Learned about TDD best practices..."
AI suggests: "Test-Driven Development (TDD) best practices include..."

Press 'a' to accept AI version, 'r' to revert, or Tab to edit manually.
```

**Phase 3 - Integration Review**
```
AI suggests: Add to "TDD" page under "Best Practices" section

You'll see a preview with the insertion point marked in green.
Press 'y' to accept, 's' to skip.
```

### Use with Your Own Graph

```bash
# Extract from today's journal
logsqueak extract

# Extract from specific date
logsqueak extract 2025-01-20

# Extract from date range
logsqueak extract 2025-01-15..2025-01-20
```

### Search Your Knowledge Base

```bash
# Semantic search (finds similar content by meaning)
logsqueak search "python testing tips"

# Force rebuild search index
logsqueak search "docker best practices" --reindex
```

Results show clickable `logseq://` links (works in modern terminals).

---

## Understanding the 3 Phases

### Phase 1: Block Selection

**What's happening:** AI reads your journal and classifies each block as "knowledge" (worth saving) or "activity log" (daily noise).

**Your job:** Review the selections. The AI is pretty good, but you know best.

**Keyboard shortcuts:**
- `j`/`k` or arrows: Navigate blocks
- `Space`: Select/deselect current block
- `a`: Accept all AI suggestions
- `c`: Clear all selections
- `Shift+j`/`Shift+k`: Jump to next/previous knowledge block
- `n`: Proceed to Phase 2
- `q`: Quit

### Phase 2: Content Editing

**What's happening:** AI rewrites selected blocks to remove temporal context ("today I learned...") and improve clarity.

**Your job:** Accept AI suggestions, edit them, or keep the original.

**Keyboard shortcuts:**
- `j`/`k`: Navigate blocks (auto-saves changes)
- `a`: Accept AI reworded version
- `r`: Revert to original
- `Tab`: Focus/unfocus editor for manual editing
- `n`: Proceed to Phase 3 (waits for semantic search to complete)
- `q`: Go back to Phase 1

**Three panels:**
- Left: Original journal content
- Middle: AI's suggested rewrite
- Right: Current version (editable)

### Phase 3: Integration Review

**What's happening:** AI suggests where to save each knowledge block in your graph (which page, which section).

**Your job:** Review each suggestion and approve or skip.

**Keyboard shortcuts:**
- `j`/`k`: Navigate decisions
- `y`: Accept decision (writes to file immediately)
- `s`: Skip this decision
- `a`: Accept all decisions for current block
- `n`: Move to next knowledge block
- `q`: Go back to Phase 2

**What you see:**
- Target page preview with green bar showing insertion point
- Integration action (add new section, add under existing, etc.)
- Provenance: Journal gets `extracted-to::` markers after successful writes

---

## Keyboard Shortcuts Cheat Sheet

All phases use vim-style navigation:

| Key | Action |
|-----|--------|
| `j` / `k` | Navigate down/up |
| `Space` | Select/deselect (Phase 1) |
| `a` | Accept AI suggestion / Accept all |
| `r` | Revert to original (Phase 2) |
| `y` | Yes, accept decision (Phase 3) |
| `s` | Skip decision (Phase 3) |
| `Tab` | Focus/unfocus editor (Phase 2) |
| `n` | Next phase / Next block |
| `q` | Quit / Go back |

**No mouse needed!** Everything is keyboard-driven.

---

## How It Works (Under the Hood)

**For the curious:**

1. **Parsing**: Logsqueak uses a custom Logseq markdown parser that preserves exact structure (round-trip tested)

2. **Classification**: AI analyzes each journal block to identify knowledge vs. activity logs

3. **Rewording**: AI removes temporal context and improves clarity while preserving meaning

4. **Semantic Search (RAG)**:
   - Builds a searchable index of your entire graph
   - Finds similar content by *meaning*, not just keywords
   - Uses hierarchical chunks for context-aware search

5. **Integration Planning**:
   - AI searches for relevant pages in your graph
   - Analyzes page structure to suggest insertion points
   - Optimized prompts

6. **Atomic Writes**:
   - Writes to target pages happen immediately on approval
   - Journal gets `extracted-to::` markers only after successful write
   - Every integrated block gets a unique `id::` property for traceability

**Non-destructive guarantee:** All operations are traceable. Nothing gets deleted. You can always find where content came from.

---

## Configuration Reference

### Complete Config File

```yaml
llm:
  # LLM API endpoint
  endpoint: http://localhost:11434/v1  # Ollama local
  # endpoint: https://api.openai.com/v1  # OpenAI cloud

  # API key (any string for Ollama, real key for OpenAI)
  api_key: ollama

  # Model name
  model: mistral:7b-instruct  # Ollama model (recommended)
  # model: your-chosen-model  # OpenAI model

  # Context window size (Ollama only, optional)
  # Controls VRAM usage - smaller = less memory, smaller context
  num_ctx: 32768

logseq:
  # Path to your Logseq graph directory
  # Must contain journals/ and logseq/ subdirectories
  graph_path: ~/Documents/logseq-graph

rag:
  # Number of similar blocks to retrieve per search
  # Higher = more context but slower (default: 20)
  top_k: 20
```

**Note on semantic search:** Logsqueak uses the `all-mpnet-base-v2` embedding model for semantic search. This is not currently configurable but provides excellent quality for finding similar content in your knowledge base.

---

## Advanced Topics

### Understanding Semantic Search

Logsqueak builds a searchable index of your entire Logseq graph:

```bash
# First run: Builds index (takes a minute)
logsqueak search "python tips"

# Subsequent runs: Uses cached index (instant)
logsqueak search "docker containers"

# Force rebuild (if you've added lots of new pages)
logsqueak search "test" --reindex
```

**How it works:**
- Converts your pages into "embeddings" (AI representations of meaning)
- Searches by semantic similarity, not just keyword matching
- Boosts results that have explicit links to relevant pages
- Shows hierarchical context (parent blocks) for better understanding

**When to rebuild:**
- After adding many new pages manually
- If search results seem stale
- If you changed your graph structure significantly

### Provenance Tracking

Every integration is traceable:

**In your journal:**
```markdown
- Learned about TDD best practices
  extracted-to:: [[TDD]]#65b1c1f0-1234-5678-89ab-cdef01234567
```

**In the target page:**
```markdown
## Best Practices
- Test-Driven Development emphasizes writing tests first
  id:: 65b1c1f0-1234-5678-89ab-cdef01234567
```

The `id::` property links back to the journal entry. The `extracted-to::` marker shows where it went.

### File Safety

Logsqueak uses "atomic two-phase writes":

1. **Read target page** and verify it hasn't changed
2. **Write new content** to a temporary file
3. **Move temp file** to final location (atomic operation)
4. **Mark journal** with `extracted-to::` marker

If any step fails, the operation is rolled back. You never get partial writes or corrupted files.

**Concurrent modification detection:** If you edit a target page in Logseq while Logsqueak is running, the write will fail with an error instead of overwriting your changes.

### Worker Dependencies

Background tasks run in a specific order:

```
Phase 1:
  - LLM Classification (immediate)
  - Embedding Model Loading (immediate)
    └─→ Page Indexing (waits for model)

Phase 2:
  - LLM Rewording (immediate)
  - RAG Search (waits for indexing)
    └─→ Integration Planning (waits for RAG)

Phase 3:
  - Decision Review (uses results from Phase 2)
```

The UI shows progress for all background tasks. You can navigate while workers run in the background.

---

## Development

Want to contribute or customize Logsqueak? See [CLAUDE.md](CLAUDE.md) for developer documentation.

### Quick Dev Commands

```bash
# Activate virtual environment (REQUIRED for all commands below)
source venv/bin/activate

# Run tests
pytest -v

# Run specific test suite
pytest tests/unit/ -v           # Unit tests only
pytest tests/integration/ -v    # Integration tests only
pytest tests/ui/ -v             # UI tests only

# Code quality
black src/ tests/               # Format code
ruff check src/ tests/          # Lint code
mypy src/                       # Type checking

# Coverage report
pytest --cov=logsqueak --cov=logseq_outline --cov-report=html -v
```

### Project Structure

```
logsqueak/
├── src/
│   ├── logsqueak/              # Main application
│   │   ├── models/             # Data models (Pydantic)
│   │   ├── services/           # LLM, RAG, file operations
│   │   ├── tui/                # Interactive UI (Textual)
│   │   ├── wizard/             # Setup wizard
│   │   ├── cli.py              # CLI commands
│   │   └── config.py           # Configuration management
│   └── logseq-outline-parser/  # Logseq markdown parser library
├── tests/                      # Test suite (376 tests)
│   ├── unit/                   # Unit tests (241 tests)
│   ├── integration/            # Integration tests (97 tests)
│   └── ui/                     # UI tests (38 tests)
├── specs/                      # Feature specifications
│   ├── 002-logsqueak-spec/     # Interactive TUI spec (complete)
│   └── 003-setup-wizard/       # Setup wizard spec (complete)
├── test-graph/                 # Sample Logseq graph for testing
└── pyproject.toml              # Dependencies and configuration
```

### Key Resources

- **[CLAUDE.md](CLAUDE.md)** - Developer guide, architecture, API docs

---

## Getting Help

- **Bugs**: [GitHub Issues](https://github.com/twaugh/logsqueak/issues)
- **Questions**: [GitHub Discussions](https://github.com/twaugh/logsqueak/discussions)
- **Documentation**: See [CLAUDE.md](CLAUDE.md) for developer docs

---

## Acknowledgments

Built with:
- [Textual](https://textual.textualize.io/) - Modern TUI framework
- [Ollama](https://ollama.com/) - Local LLM runtime
- [ChromaDB](https://www.trychroma.com/) - Vector database for semantic search
- [sentence-transformers](https://www.sbert.net/) - Embedding models

Developed with assistance from [Claude Code](https://claude.com/claude-code).
