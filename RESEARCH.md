# Logsqueak Research Notes

## Project Overview

Logsqueak is a CLI tool with a text-based interface for Logseq, designed to use LLMs for intelligent knowledge management.

## Core Features

### 1. Knowledge Extraction and Integration (v1 Priority)

Extract lasting knowledge from journal entries and intelligently integrate it into relevant pages.

#### Knowledge vs. Activity Distinction

- **Activity logs** (stays in journal): "worked on Project X"
- **Knowledge blocks** (extracted to pages):
  - "Discovered the deadline for Project X slipping by 2 months"
  - Meeting notes with lasting information about projects, people, processes

#### Integration Strategy

- **Integrate, not append**: New information should be merged intelligently into existing content
- **Provenance**: Link back to source journal page (initially via page links, block references on roadmap)
- **Conflict handling**: Newer journal entries supersede older ones
  - Track changes: "Deadline is May (changed from March)"

#### Merge Operations

For each knowledge block, the system should:
1. Identify target page(s) (existing or potentially new)
2. Locate the most relevant section/block within the target page
3. Choose merge strategy

**V1: Simple Additive Strategy**

Focus on non-destructive operations only:
- **(a) Add child block** - Append knowledge as a new block under the most relevant section
  - Include provenance link to source journal: `- Knowledge content [[2025-01-15]]`
  - Primary operation for v1
- **(c) Create new section** - When no relevant section exists for the knowledge
  - May require user confirmation in dry-run mode
  - Example: Add "## Competitors" section if adding competitor information

**Roadmap: Semantic Merging**

Deferred to future versions:
- **(b) Update existing block** - Modify existing content based on new information
  - Requires semantic understanding of relationships (update, contradiction, elaboration)
  - Track changes: "Deadline is May (changed from March)"
- **Information type awareness**:
  - Factual updates (supersedes old info)
  - Elaborations (adds context to existing)
  - Contradictions (conflict resolution)
- **Inline merging** - Weave information into existing prose
- **Deduplication** - Detect and consolidate similar blocks
- **Confidence-based strategies** - Different merge approaches based on LLM confidence

### 2. Journal Summarization (Roadmap)

Summarize historical journal entries similar to how Claude conversations are summarized when token context runs low.

- Details TBD (hierarchical summarization? date range selection?)

## Technical Architecture

### API Support

Support multiple LLM API types:

1. **OpenAI-compatible API** (de facto standard):
   - GET `/v1/models`
   - POST `/v1/completions`
   - POST `/v1/chat/completions`
   - Bearer token authentication
   - Supported by: OpenAI, Anthropic, Azure OpenAI, many local servers

2. **Ollama**:
   - Native Ollama API
   - Also supports OpenAI-compatible endpoint
   - Can use network-based ollama server

### LLM Processing Pipeline

Multi-step process for each journal entry:

1. **Extract knowledge blocks** - identify blocks with lasting information
2. **Identify target pages** - determine relevant page(s) for each knowledge block
3. **Locate insertion point** - find most relevant section/block in target page
4. **Determine merge strategy** - update existing, add new, or create section
5. **Execute with provenance** - make changes and link back to journal

Use structured output (JSON) for reliable parsing between steps.

### Logseq Integration

- **Format**: Markdown (primary), org-mode on roadmap
- **Filesystem-based**: Read/write directly to graph files
- **Conventions**:
  - Page links: `[[page name]]`
  - Properties: `property:: value`
  - Templates: `template:: name` property
  - Block structure: Outline-style with indentation

### CLI Design

Planned commands:
- `logsqueak extract [date/range]` - process journal entries
- `logsqueak summarize [range]` - summarize journals (future)

Configuration file for:
- API endpoints and credentials
- Model selection
- Logseq graph path

### Dry-Run Mode

Essential feature: show proposed changes before applying them.

Example output:
```
Found 3 knowledge blocks in journals/2025_01_15.md:
  1. "Project X deadline slipped to May"
     → Update pages/Project X.md :: Timeline section
     → Replace "Deadline: March" with "Deadline: May (changed from March)"

  2. "Main competitor is Product Y"
     → Add to pages/Project X.md :: Market Analysis section

Apply changes? [y/N/edit]
```

## Implementation Details

### Language

Python - rich ecosystem for LLM work and markdown parsing

### Logseq-Specific Considerations

- **Templates**: Identified via `template:: name` property on pages
  - Could guide LLM on page structure and where to place information
  - On roadmap for new page creation

- **Page structure**: Flexible
  - No assumption of standard sections
  - Tool should work for any user's organizational system
  - Templates can provide hints but aren't required

## Roadmap

### v1 (Initial Implementation)

- [x] Research and design
- [ ] Basic API client + config
- [ ] Markdown/Logseq parser
- [ ] Knowledge extraction (identify knowledge blocks)
- [ ] Page integration (merge into existing pages)
- [ ] Dry-run mode
- [ ] Page link provenance

### Future Versions

- Block references with `id::` properties
- New page creation from journal knowledge
- Template application for new pages
- Journal summarization (hierarchical: daily → weekly → monthly)
- Batch processing of multiple journal entries
- Interactive TUI mode (vs simple CLI)

## Open Questions

- Scope for initial processing: one entry at a time or batch?
- Should it track last-run state to only process new entries?
- Test graph: use existing or create minimal example?
- How to handle pages without clear section structure?
- Confidence thresholds for knowledge extraction?

## Design Principles

1. **User control**: Always show changes before applying (dry-run default)
2. **Flexibility**: Work with any page structure, not just prescribed templates
3. **Provenance**: Always link back to source journal entries
4. **Generality**: Useful for any Logseq user, not just one person's workflow
