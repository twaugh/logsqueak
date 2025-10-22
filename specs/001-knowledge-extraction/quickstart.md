# Quickstart: Knowledge Extraction from Journals

**Purpose**: Get developers set up to build and test the knowledge extraction feature.
**Audience**: Contributors to the Logsqueak POC.

## Prerequisites

- Python 3.11 or later
- A Logseq graph with journal entries (for testing)
- Access to an LLM API (OpenAI, Anthropic, or local Ollama)

## Installation

### 1. Clone and Setup Environment

```bash
# Clone repository
git checkout 001-knowledge-extraction

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

### 2. Install Dependencies

Create `requirements.txt` in repository root:

```txt
# Core dependencies
httpx>=0.27.0
markdown-it-py>=3.0.0
PyYAML>=6.0.1
pydantic>=2.0.0
click>=8.1.0
python-dateutil>=2.8.0

# Development dependencies
pytest>=7.4.0
pytest-cov>=4.1.0
black>=23.0.0
ruff>=0.1.0
mypy>=1.7.0

```

Install:

```bash
pip install -r requirements.txt

```

## Configuration

### 3. Create Config File

Create `~/.config/logsqueak/config.yaml`:

```bash
mkdir -p ~/.config/logsqueak
cat > ~/.config/logsqueak/config.yaml <<'EOF'
llm:
  endpoint: https://api.openai.com/v1
  api_key: sk-your-api-key-here
  model: gpt-4-turbo-preview

logseq:
  graph_path: ~/Documents/logseq-graph
EOF

```

**For Ollama (local)**:

```yaml
llm:
  endpoint: http://localhost:11434/v1
  api_key: ollama  # Any non-empty value works
  model: llama2

logseq:
  graph_path: ~/Documents/logseq-graph

```

**For Anthropic**:

```yaml
llm:
  endpoint: https://api.anthropic.com/v1
  api_key: sk-ant-...
  model: claude-3-5-sonnet-20241022

logseq:
  graph_path: ~/Documents/logseq-graph

```

### 4. Environment Variables (Alternative)

Instead of config file, use environment variables:

```bash
export LOGSQUEAK_LLM_ENDPOINT=https://api.openai.com/v1
export LOGSQUEAK_LLM_API_KEY=sk-...
export LOGSQUEAK_LLM_MODEL=gpt-4-turbo-preview
export LOGSQUEAK_LOGSEQ_GRAPH_PATH=~/Documents/logseq-graph

```

## Project Structure

After setup, your directory should look like:

```
logsqueak/
├── src/
│   └── logsqueak/
│       ├── __init__.py
│       ├── cli/            # Command-line interface
│       ├── config/         # Configuration loading
│       ├── extraction/     # Knowledge extraction logic
│       ├── integration/    # Page integration logic
│       ├── llm/            # LLM client abstraction
│       ├── logseq/         # Logseq file format handling
│       └── models/         # Data models
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
│       ├── journals/       # Sample journal entries for testing
│       └── pages/          # Sample Logseq pages for testing
├── specs/
│   └── 001-knowledge-extraction/
│       ├── spec.md
│       ├── plan.md
│       ├── research.md
│       ├── data-model.md
│       └── quickstart.md   # This file
├── requirements.txt
├── pyproject.toml          # Python project metadata
└── README.md

```

## Running the Tool

### Basic Usage

```bash
# Extract knowledge from today's journal (dry-run mode, default)
python -m logsqueak.cli.main extract

# Extract from specific date
python -m logsqueak.cli.main extract 2025-01-15

# Extract from date range
python -m logsqueak.cli.main extract 2025-01-10..2025-01-15

# Relative dates
python -m logsqueak.cli.main extract yesterday
python -m logsqueak.cli.main extract last-week

# Apply changes (skip dry-run preview)
python -m logsqueak.cli.main extract --apply 2025-01-15

# Override model
python -m logsqueak.cli.main extract --model gpt-4 2025-01-15

```

### Interactive Workflow

1. **Run extraction** (dry-run mode shows preview):

   ```bash
   python -m logsqueak.cli.main extract 2025-01-15

   ```

2. **Review preview**:

   ```
   Found 3 knowledge blocks in journals/2025_01_15.md:

     1. "Project X deadline moved to May"
        → Target: Project X
        → Section: Timeline
        → Action: Add child block

   Apply changes? [y/N/e]

   ```

3. **Choose action**:

   - `y` - Apply all changes
   - `n` - Cancel (default)
   - `e` - Edit preview before applying (future enhancement)

## Development Workflow

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=logsqueak --cov-report=html

# Run specific test file
pytest tests/unit/test_parser.py

# Run with verbose output
pytest -v

# Run integration tests only
pytest tests/integration/

```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type checking
mypy src/

```

### Creating Test Fixtures

Create sample journal entries in `tests/fixtures/journals/`:

**tests/fixtures/journals/2025_01_15.md**:

```markdown

- worked on [[Project X]]
- met with team
- Discovered the deadline for [[Project X]] slipping to May
  - Original deadline was March
  - Vendor delays caused the slip
- attended standup meeting
- Main competitor is [[Product Y]]
  - They use pricing model Z

```

Create sample pages in `tests/fixtures/pages/`:

**tests/fixtures/pages/Project X.md**:

```markdown

- ## Timeline
  - Kickoff: January 2025
  - MVP: March 2025 (original)
- ## Team
  - Alice (PM)
  - Bob (Eng)
- ## Status
  - In progress

```

### Manual Testing

1. **Create test graph**:

   ```bash
   mkdir -p ~/test-logseq-graph/journals
   mkdir -p ~/test-logseq-graph/pages

   ```

2. **Add sample journal** (`~/test-logseq-graph/journals/2025_01_15.md`):

   ```markdown

   - worked on [[Test Project]]
   - The [[Test Project]] deadline moved to June
   - attended meeting

   ```

3. **Add target page** (`~/test-logseq-graph/pages/Test Project.md`):

   ```markdown

   - ## Timeline
     - Start: January 2025
   - ## Status
     - Active

   ```

4. **Update config** to point to test graph:

   ```yaml
   logseq:
     graph_path: ~/test-logseq-graph

   ```

5. **Run extraction**:

   ```bash
   python -m logsqueak.cli.main extract 2025-01-15

   ```

6. **Verify output**:

   - Review preview showing extracted knowledge
   - Apply changes
   - Check `Test Project.md` for new bullet with provenance link

## Debugging

### Enable Verbose Logging

```bash
# Set log level
export LOGSQUEAK_LOG_LEVEL=DEBUG

# Run with debug output
python -m logsqueak.cli.main extract --verbose 2025-01-15

```

### Common Issues

**Issue**: `FileNotFoundError: Config file not found`

- **Solution**: Create `~/.config/logsqueak/config.yaml` or set environment variables

**Issue**: `ValidationError: graph_path does not exist`

- **Solution**: Verify Logseq graph path in config points to existing directory

**Issue**: `httpx.ConnectError: Connection refused`

- **Solution**: Check LLM endpoint URL, ensure Ollama is running if using local

**Issue**: `PermissionError: Cannot write to page`

- **Solution**: Ensure write permissions on Logseq graph directory

**Issue**: `JSONDecodeError: Invalid LLM response`

- **Solution**: Check model supports JSON mode (`response_format={"type": "json_object"}`)

### Inspecting LLM Requests/Responses

Enable request logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Logs will show:
# - Request payload sent to LLM
# - Response JSON received
# - Parsing errors if any

```

## Next Steps

1. **Implement User Story 1** (P1): Extract and preview knowledge

   - See `specs/001-knowledge-extraction/tasks.md` (generated by `/speckit.tasks`)
   - Start with `src/logsqueak/extraction/extractor.py`

2. **Write tests first** (TDD per constitution):

   - Create `tests/unit/test_extractor.py`
   - Write failing tests for knowledge extraction
   - Implement until tests pass

3. **Iterate on feedback**:

   - Test with real journal entries
   - Tune LLM prompts for better accuracy
   - Refine matching logic for target pages

## Resources

- **Spec**: [spec.md](./spec.md) - Feature requirements and user stories
- **Plan**: [plan.md](./plan.md) - Implementation plan and architecture
- **Research**: [research.md](./research.md) - Library choices and rationale
- **Data Model**: [data-model.md](./data-model.md) - Entity definitions
- **Constitution**: `../../.specify/memory/constitution.md` - Project principles

## Support

- GitHub Issues: https://github.com/twaugh/logsqueak/issues
- Discussions: https://github.com/twaugh/logsqueak/discussions

## License

GPLv3 - See [LICENSE](../../LICENSE) file in the repository root.

This project uses AI assistance (Claude Code) in development. All code is licensed under GPLv3 regardless of authorship method.
