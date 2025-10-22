# Feature Specification: Knowledge Extraction from Journals

**Feature Branch**: `001-knowledge-extraction`
**Created**: 2025-10-22
**Status**: Draft
**Input**: User description: "Extract lasting knowledge from Logseq journal entries and integrate into relevant pages using LLM-powered analysis"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Extract and Preview Knowledge (Priority: P1)

A Logseq user has been journaling daily about their work projects, meetings, and research. They want to identify which journal entries contain lasting knowledge (vs. temporary activity logs) and see what would be extracted before any changes are made to their knowledge base.

**Why this priority**: This is the foundation of the entire feature. Without reliable knowledge extraction and safe preview, users won't trust the system with their valuable knowledge base. Dry-run mode is non-negotiable per the constitution.

**Independent Test**: Can be fully tested by running extraction on a journal entry and viewing the preview output. Delivers immediate value by showing users what knowledge they've captured without modifying any files.

**Acceptance Scenarios**:

1. **Given** a journal entry with mixed activity logs and knowledge ("worked on Project X" and "Project X deadline moved to May"), **When** user runs extraction on that entry, **Then** system identifies only the knowledge block ("deadline moved to May") and displays it in preview with the target page identified
2. **Given** a journal entry with only activity logs ("attended meeting", "wrote email"), **When** user runs extraction, **Then** system reports no knowledge blocks found
3. **Given** a journal entry with knowledge about multiple topics, **When** user runs extraction, **Then** system identifies all knowledge blocks and shows the target page for each one
4. **Given** an invalid journal file path, **When** user runs extraction, **Then** system shows clear error message without crashing

---

### User Story 2 - Add Knowledge to Existing Pages (Priority: P2)

After reviewing the extraction preview, the user wants to integrate the identified knowledge blocks into their existing Logseq pages. The system should add new content under the most relevant section of each target page, always linking back to the source journal entry.

**Why this priority**: This completes the core value loop - extracting knowledge is only useful if it gets properly integrated into the knowledge base. However, it depends on successful extraction (P1).

**Independent Test**: Can be tested by approving an extraction and verifying the knowledge appears in the correct page location with proper provenance link. Delivers value by actually organizing extracted knowledge.

**Acceptance Scenarios**:

1. **Given** a knowledge block about "Project X deadline change" and an existing "Project X" page with a "Timeline" section, **When** user approves integration, **Then** system adds the knowledge as a child block under the Timeline section with a link to the source journal date
2. **Given** a knowledge block and a target page with no clear matching section, **When** user approves integration, **Then** system adds the knowledge at the end of the page with provenance link
3. **Given** multiple knowledge blocks targeting the same page, **When** user approves integration, **Then** system adds each block to its most relevant section without duplicating or overwriting existing content
4. **Given** a knowledge block targeting a non-existent page, **When** user attempts integration, **Then** system reports the page doesn't exist and skips integration for that block
5. **Given** an extraction preview is displayed, **When** user responds with "y" at the prompt, **Then** system applies all proposed changes; **When** user responds with "n", **Then** system aborts without changes; **When** user responds with "e", **Then** system allows editing the preview before applying

---

### User Story 3 - Create New Sections When Needed (Priority: P3)

Sometimes extracted knowledge belongs to a page but doesn't fit any existing outline structure (e.g., competitor information when the page has no "Competitors" bullet). The user wants the system to create appropriate new top-level bullets or nested structure when needed, matching the page's existing outline conventions.

**Why this priority**: Enhances the integration intelligence but isn't strictly necessary for MVP - knowledge can still be added at page end. This is a quality-of-life improvement.

**Independent Test**: Can be tested by extracting knowledge that clearly needs a new organizational bullet and verifying the structure is created matching the page's outline format. Delivers value by better organizing knowledge.

**Acceptance Scenarios**:

1. **Given** a knowledge block about competitors and a "Project X" page with no competitors section, **When** user approves integration, **Then** system creates an appropriate organizational bullet (e.g., "- Competitors" or "- ## Competitors" matching the page's existing structure) and adds the knowledge as a child with provenance link
2. **Given** a knowledge block that needs new outline structure, **When** user reviews in dry-run mode, **Then** system shows "Will create new bullet: [Structure]" in the preview
3. **Given** multiple knowledge blocks that need the same new organizational structure, **When** user approves integration, **Then** system creates the structure once and adds all relevant blocks under it
4. **Given** a page using markdown headings as bullets (e.g., "- ## Section"), **When** creating new structure, **Then** system matches this convention; **Given** a page using plain bullets, **Then** system uses plain bullets

---

### Edge Cases

- **Resolved:** What happens when a journal entry is malformed or has invalid markdown? → System logs warning, skips extraction for that entry, continues processing remaining entries
- **Resolved:** How does the system handle very large journal entries (1000+ lines)? → System processes up to 2000 lines maximum, logs truncation warning if entry exceeds limit
- What if the LLM API is unavailable or returns an error?
- What if a target page exists but is empty (no sections at all)?
- What if the LLM extracts invalid knowledge (hallucinations or misinterpretations)?
- How does the system handle special characters or formatting in knowledge blocks?
- What if the user's Logseq graph path is invalid or inaccessible?
- **Resolved:** What happens when the same knowledge appears in multiple journal entries on different dates? → System detects duplicates, skips integration, and logs the duplicate for user awareness

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST distinguish between activity logs (stays in journal) and knowledge blocks (extracted to pages)
- **FR-002**: System MUST present all proposed changes in dry-run mode before modifying any files
- **FR-003**: System MUST add a provenance link to the source journal date for every extracted knowledge block
- **FR-004**: System MUST identify the most relevant target page for each knowledge block
- **FR-005**: System MUST identify the most relevant bullet/section within the target page's outline for placement
- **FR-006**: System MUST add knowledge as new child bullets in the outline hierarchy (non-destructive, additive only)
- **FR-007**: System MUST support creating new organizational bullets when no relevant structure exists, matching the page's existing outline conventions
- **FR-008**: System MUST preserve existing page content and outline structure without modification or deletion
- **FR-009**: System MUST handle cases where target pages don't exist by reporting the issue
- **FR-010**: System MUST maintain Logseq's outline format (bullets with indentation, not flat markdown) when reading and writing pages
- **FR-011**: System MUST handle LLM API failures gracefully without corrupting files
- **FR-012**: System MUST allow users to approve, reject, or edit proposed changes before applying via interactive CLI prompt (y/n/e options)
- **FR-013**: System MUST process journal entries specified by date or date range
- **FR-014**: System MUST respect Logseq's filesystem-based storage (direct file I/O, no database)
- **FR-015**: System MUST use structured output formats for reliable LLM response parsing
- **FR-016**: System MUST store configuration (API endpoints, credentials, Logseq graph path, model selection) in ~/.config/logsqueak/config.yaml following XDG Base Directory specification
- **FR-017**: System MUST detect when extracted knowledge already exists on the target page and skip integration, logging the duplicate detection to inform the user
- **FR-018**: System MUST handle malformed or invalid markdown in journal entries by logging a warning, skipping extraction for that entry, and continuing to process remaining entries
- **FR-019**: System MUST enforce a maximum journal entry size of 2000 lines, processing only the first 2000 lines and logging a truncation warning if the entry exceeds this limit

### Key Entities

- **Journal Entry**: A daily Logseq journal file containing a mix of activity logs and knowledge blocks in outline format. Attributes: date, file path, raw content, parsed outline structure.
- **Knowledge Block**: A piece of information extracted from a journal that has lasting value. Attributes: content text, source journal date, confidence level, target page, target bullet/location in outline.
- **Target Page**: An existing Logseq page where knowledge should be integrated. Attributes: page name, file path, outline structure (bullets and hierarchy), existing organizational conventions.
- **Provenance Link**: A reference back to the source journal entry, formatted as a Logseq page link (e.g., `[[2025-01-15]]`). Maintains traceability.
- **Extraction Preview**: A summary of what will be changed, shown before applying. Attributes: knowledge blocks found, target locations in outline, proposed actions (add child bullet, create organizational bullet).
- **Configuration**: User settings stored at ~/.config/logsqueak/config.yaml. Attributes: LLM API endpoint, API credentials, Logseq graph path, model name/selection.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users receive clear progress feedback during extraction (status messages showing each processing step)
- **SC-002**: System correctly distinguishes between activity logs and knowledge in at least 80% of test cases (based on human review)
- **SC-003**: 100% of integrated knowledge blocks include a valid provenance link to the source journal
- **SC-004**: Users can complete the full workflow (extract, preview, approve, integrate) without any file corruption or data loss in 100% of operations
- **SC-005**: System handles LLM API errors gracefully in 100% of cases (no crashes, clear error messages)
- **SC-006**: Users can successfully process and integrate knowledge from typical journal entries (50-200 lines) and large entries (up to 2000 lines) regardless of LLM response time
- **SC-007**: Dry-run mode shows all proposed changes before any file modifications in 100% of operations

## Clarifications

### Session 2025-10-22

- Q: Where should the configuration file be stored (API endpoints, credentials, Logseq graph path, model selection)? → A: User's home directory (~/.config/logsqueak/config.yaml)
- Q: How should users approve, reject, or edit proposed changes (FR-012)? → A: Interactive CLI prompt with y/n/e (yes/no/edit) options
- Q: What happens when the same knowledge appears in multiple journal entries on different dates? → A: Skip integration and log duplicate detection
- Q: What happens when a journal entry is malformed or has invalid markdown? → A: Log warning, skip extraction, continue processing
- Q: How does the system handle very large journal entries (1000+ lines)? → A: Process up to 2000 lines, log truncation warning

## Assumptions

- Users have a Logseq graph with existing pages and journal entries in markdown-based outline format
- All Logseq pages use outline structure (bullets with indentation), never flat markdown
- Users have access to an OpenAI-compatible LLM API or Ollama instance
- Journal entries follow standard Logseq conventions (outline structure with bullets, page links, properties)
- Target pages exist for most knowledge (handling new page creation is deferred to roadmap)
- Pages may use different organizational conventions (plain bullets, bullets with markdown headings, mixed) - system must detect and match these
- Users are comfortable with command-line tools for this POC phase
- Processing happens one journal entry at a time (batch processing on roadmap)
