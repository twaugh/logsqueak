<!--
Sync Impact Report:
- Version change: none → 1.0.0
- New constitution created from template
- Principles defined: 3 core principles for POC
- Templates status:
  ✅ plan-template.md - reviewed, compatible (Constitution Check section references this file)
  ✅ spec-template.md - reviewed, compatible (requirements-driven approach aligns)
  ✅ tasks-template.md - reviewed, compatible (task structure supports principles)
  ⚠ commands/*.md - no command files exist yet
- Follow-up TODOs: None
-->

# Logsqueak Constitution

## Core Principles

### I. Proof-of-Concept First

This is a POC project - prioritize working software over perfection. Features should demonstrate feasibility, not production readiness. Code quality matters, but shipping iteratively matters more. No backwards compatibility guarantees until v1.0 is declared production-ready.

**Rationale**: POC stage requires rapid iteration and learning. Premature optimization or over-engineering would slow discovery of what actually works for LLM-driven knowledge extraction.

### II. Non-Destructive Operations (NON-NEGOTIABLE)

The system MUST NOT modify or delete existing Logseq content except by explicit user approval. All changes MUST be additive (new blocks, new sections) or reviewable in dry-run mode before application. Provenance links to source journals are MANDATORY for all extracted knowledge.

**Rationale**: Users trust their knowledge bases. A tool that corrupts or loses information will never be adopted, regardless of how clever the LLM integration is. Safety first enables experimentation.

### III. Simplicity and Transparency

Keep it simple. Prefer file-based I/O over databases. Use JSON for structured LLM outputs. Show the user what the LLM is doing (what it extracted, where it plans to place it). Avoid abstraction layers until complexity demands them.

**Rationale**: Transparency builds trust and aids debugging. Simple architectures are easier to reason about when LLM behavior surprises us (which it will).

## Development Workflow

### Commit Messages

All git commit messages created by AI coding assistants MUST include a trailing line:

```
Assisted-by: Claude Code
```

This ensures clear attribution of AI-assisted work.

### User Approval Gates

- **Dry-run by default**: Never write to Logseq files without showing the user proposed changes
- **Graceful failures**: If the LLM produces invalid output or cannot determine placement, inform the user and skip (don't guess or force)
- **Logging**: Log LLM requests/responses for debugging when things go wrong

### Iteration Strategy

Build in this order:
1. Core extraction (identify knowledge blocks in journals)
2. Simple additive integration (new child blocks only)
3. Dry-run mode and user review
4. Expand to section creation and more sophisticated placement

Defer semantic merging, contradiction detection, and template-based page creation to future versions.

## Governance

This constitution defines the architectural and operational principles for Logsqueak.

**Amendments**: This document may be updated as the project evolves. Since this is a POC with no backwards compatibility guarantees, amendments do not require migration plans but SHOULD be documented with rationale.

**Compliance**: All feature specifications and implementation plans should reference these principles. Constitution violations must be explicitly justified in the plan's Complexity Tracking section.

**Version**: 1.0.0 | **Ratified**: 2025-10-22 | **Last Amended**: 2025-10-22
