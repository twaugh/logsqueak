# Implementation Plan: Interactive Setup Wizard

**Branch**: `003-setup-wizard` | **Date**: 2025-11-19 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/003-setup-wizard/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Implement an interactive CLI setup wizard (`logsqueak init`) that guides users through initial configuration, validates all settings (LLM connection and embedding model), and creates a properly configured `config.yaml` file with mode 600 permissions. The wizard supports multiple LLM providers (Ollama, OpenAI, custom OpenAI-compatible), handles existing configs gracefully, and provides clear error messages throughout the process.

## Technical Context

**Language/Version**: Python 3.11+
**Primary Dependencies**: Click (CLI framework), Rich (formatted CLI output), httpx (HTTP client for API testing), PyYAML (config serialization), sentence-transformers (embedding model), existing Pydantic models (Config validation)
**Storage**: File-based YAML config at `~/.config/logsqueak/config.yaml` with mode 600 permissions
**Testing**: pytest with mock-based unit tests for wizard logic, integration tests for config file operations and API validation
**Target Platform**: Linux/macOS/Windows CLI (cross-platform)
**Project Type**: Single project (CLI extension to existing Logsqueak application)
**Performance Goals**: Setup completion <3 minutes for new users, config updates <30 seconds (skipping cached embedding test), LLM connection test <30 seconds, embedding download <5 minutes
**Constraints**: Must preserve all provider settings when switching providers, must validate both LLM and embedding model before saving config, must handle network timeouts gracefully (prompt user instead of auto-abort)
**Scale/Scope**: Single `init` command with ~15-20 prompts, supports 3 provider types, validates 2 external systems (LLM + embedding model)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### I. Proof-of-Concept First ✅
- **Alignment**: Wizard provides immediate value by simplifying first-run experience for POC users
- **Iteration**: Simple prompt-based flow, can iterate on UX based on user feedback
- **No premature optimization**: Using Rich for formatting (already in deps), basic validation logic

### II. Non-Destructive Operations ✅
- **Safe config handling**: Wizard prompts for confirmation before overwriting existing config
- **Preserves provider settings**: When switching providers, preserves credentials for all providers in config
- **Graceful failures**: Network errors, validation failures show clear messages without corrupting config
- **User control**: Can abort at any prompt without creating/modifying files

### III. Simplicity and Transparency ✅
- **File-based config**: YAML config at known location (`~/.config/logsqueak/config.yaml`)
- **Show validation results**: Clear success/failure messages for LLM and embedding tests
- **Helpful errors**: All error scenarios provide actionable guidance
- **No hidden magic**: Each step explains what it's doing (testing connection, downloading model, etc.)

**GATE STATUS**: ✅ PASS - Feature aligns with all constitution principles

## Project Structure

### Documentation (this feature)

```text
specs/[###-feature]/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
src/logsqueak/
├── models/
│   └── config.py              # Existing: Config, LLMConfig, LogseqConfig, RAGConfig (Pydantic models)
├── services/
│   ├── llm_client.py          # Existing: LLMClient for connection testing
│   └── page_indexer.py        # Existing: PageIndexer for embedding model validation
├── wizard/                     # NEW: Setup wizard module
│   ├── __init__.py
│   ├── wizard.py              # Main wizard orchestration logic
│   ├── prompts.py             # Interactive prompt helpers (using Rich)
│   ├── validators.py          # Path, network, and config validation functions
│   └── providers.py           # Provider-specific logic (Ollama, OpenAI, Custom)
└── cli.py                      # Existing: Add new `init` command

tests/
├── unit/
│   └── wizard/                # NEW: Unit tests for wizard components
│       ├── test_prompts.py    # Test prompt logic with mocked user input
│       ├── test_validators.py # Test validation functions
│       └── test_providers.py  # Test provider-specific logic
└── integration/
    └── test_wizard.py         # NEW: End-to-end wizard tests with mocked API calls
```

**Structure Decision**: Single project structure (Option 1). This feature extends the existing CLI application with a new `wizard/` module under `src/logsqueak/`. The wizard reuses existing services (`LLMClient` for connection testing, `PageIndexer` for embedding validation) and models (`Config` for final validation). New `init` command added to existing `cli.py` Click application.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

**No violations** - All design decisions align with constitution principles.

---

## Post-Design Constitution Re-Check

### I. Proof-of-Concept First ✅
- **Implementation approach**: Straightforward prompt-based CLI wizard
- **Iteration path**: Rich library provides quick visual feedback for UX iteration
- **No over-engineering**: Simple ValidationResult pattern, no complex abstractions
- **Deliverable**: Functional wizard that unblocks new users immediately

### II. Non-Destructive Operations ✅
- **Config preservation**: All provider settings preserved in `llm_providers` dict
- **Confirmation gates**: Prompts before overwriting existing config
- **Abort handling**: User can exit at any step without side effects
- **Atomic writes**: Temp file + rename ensures config integrity
- **Safe failures**: Network errors, validation failures don't corrupt state

### III. Simplicity and Transparency ✅
- **File-based storage**: YAML config at known location
- **Clear validation steps**: User sees each validation result (LLM connection, embedding model)
- **Helpful error messages**: All errors include actionable guidance
- **No hidden magic**: Wizard explicitly shows what it's doing at each step
- **Minimal abstractions**: ValidationResult is only abstraction, keeps code readable

**GATE STATUS**: ✅ PASS - Post-design review confirms full alignment with all principles

---
