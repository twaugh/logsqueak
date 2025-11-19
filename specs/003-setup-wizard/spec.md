# Feature Specification: Interactive Setup Wizard

**Feature Branch**: `003-setup-wizard`
**Created**: 2025-11-19
**Status**: Draft
**Input**: User description: "Interactive setup wizard that guides users through initial configuration, validates settings, and creates config file with proper permissions"

## Clarifications

### Session 2025-11-19

- Q: Should the wizard test the embedding model (sentence-transformers) during setup, which is a ~500MB download needed for RAG search? → A: Test embedding model during setup (download if needed, validate it loads, show progress)
- Q: What timeout values should be used for network operations (LLM connection test, embedding model download)? → A: Moderate timeouts (30s LLM, 5min embedding) but prompt user instead of auto-aborting on timeout
- Q: When updating existing config, should wizard re-test embedding model if already cached? → A: Skip test if model cached and loadable (faster config updates, assume cached model is valid)
- Q: Should wizard check available disk space before downloading ~500MB embedding model? → A: Check disk space and warn if low but allow proceeding (user makes final call)
- Q: What are the exact validation criteria for Logseq graph path? → A: Must have both journals/ AND logseq/ directories (strict validation for valid graphs)
- Q: When user switches from one working provider to another (e.g., OpenAI to Ollama), should the config file preserve the previous provider's settings? → A: Yes, preserve all provider settings in config file even when switching active provider (so users don't re-enter API keys when switching back)
- Q: How should API keys be displayed on screen? → A: Existing keys from config shown masked only (first 8 + "..." + last 4), but NEW key entry shown visible for verification
- Q: What does "Custom endpoint" provider option mean? → A: OpenAI-compatible API (Azure OpenAI, LocalAI, LM Studio, vLLM, etc.)

## User Scenarios & Testing *(mandatory)*

### User Story 1 - First-Time Setup (Priority: P1)

A new user has just installed Logsqueak and wants to start extracting knowledge from their Logseq journals. They run `logsqueak init` and are guided through configuring their Logseq graph location, choosing an LLM provider (Ollama or OpenAI), and testing the connection. The wizard also validates the embedding model (downloading if needed with progress indicator). The wizard creates a properly configured `config.yaml` file with correct permissions, and the system is ready for immediate use.

**Why this priority**: This is the primary barrier for new users. Without an easy setup process, users cannot use Logsqueak at all. This delivers immediate value by transforming a manual, error-prone configuration process into a guided experience.

**Independent Test**: Can be fully tested by running `logsqueak init` on a fresh system without any existing config, completing the wizard prompts, and verifying that `logsqueak extract` works afterward.

**Acceptance Scenarios**:

1. **Given** no existing config file, **When** user runs `logsqueak init`, **Then** wizard displays welcome message and guides through all configuration steps
2. **Given** user completes all wizard prompts with valid values, **When** wizard finishes, **Then** config file exists at `~/.config/logsqueak/config.yaml` with mode 600 permissions
3. **Given** user provides Logseq graph path, **When** wizard validates the path, **Then** user sees confirmation if path is valid or error message with guidance if invalid
4. **Given** user selects Ollama provider, **When** Ollama is running locally, **Then** wizard automatically detects it and shows available models
5. **Given** user selects OpenAI provider, **When** user enters API key, **Then** wizard tests the connection before saving config
6. **Given** LLM connection succeeds, **When** wizard tests embedding model, **Then** user sees progress indicator for download (if needed) and confirmation when model loads successfully
7. **Given** wizard completes successfully, **When** user sees success message, **Then** message includes next steps (extract, search commands)

---

### User Story 2 - Fixing Broken Configuration (Priority: P2)

A user has an existing config file but it has problems (wrong permissions, invalid YAML, outdated settings, moved Logseq graph). They run `logsqueak init` to fix the issues. The wizard detects the existing config, loads what it can, uses those values as defaults, and lets the user correct specific problems without re-entering everything. If the embedding model is already cached, the wizard skips the lengthy download/test for faster completion.

**Why this priority**: Users will encounter config problems (permissions after git clone, moved graph location, expired API keys). Being able to fix config easily without manual YAML editing reduces friction and support burden.

**Independent Test**: Can be tested by creating a broken config file (wrong permissions or invalid values), running `logsqueak init`, and verifying that the wizard offers to fix it while preserving valid settings.

**Acceptance Scenarios**:

1. **Given** existing config with wrong permissions, **When** user runs `logsqueak init`, **Then** wizard warns about permissions and offers to fix them
2. **Given** existing valid config, **When** user runs `logsqueak init`, **Then** wizard loads current values and shows them as defaults for each prompt
3. **Given** existing config with valid graph path, **When** wizard reaches graph path step, **Then** user can press Enter to keep current path or type new one
4. **Given** existing config with valid LLM settings, **When** wizard reaches LLM step, **Then** user can choose to keep current configuration or reconfigure
5. **Given** partially invalid config (some fields missing), **When** wizard starts, **Then** wizard extracts valid fields and prompts only for missing/invalid ones
6. **Given** user confirms overwriting existing config, **When** wizard completes, **Then** new config file has all valid settings and correct permissions

---

### User Story 3 - Remote Ollama and Custom Endpoints (Priority: P2)

A user wants to use either Ollama running on a remote server (different machine, Docker container, cloud instance) or a custom OpenAI-compatible endpoint (Azure OpenAI, LocalAI, LM Studio). They run `logsqueak init`, select their provider type, and enter the appropriate endpoint details. The wizard connects to the instance, shows available models (for Ollama), and validates the connection.

**Why this priority**: Many users run LLMs on separate machines (GPU servers, Docker, cloud) or use alternative OpenAI-compatible services. Supporting these deployment patterns expands the user base beyond local Ollama and OpenAI.

**Independent Test**: Can be tested by running `logsqueak init` when Ollama is not running locally, entering a remote Ollama URL, and verifying the wizard successfully connects and retrieves models.

**Acceptance Scenarios**:

1. **Given** Ollama not running on localhost, **When** user selects Ollama provider, **Then** wizard displays suggestion to start local Ollama or connect to remote instance
2. **Given** wizard prompts for Ollama URL, **When** user enters custom URL, **Then** wizard attempts to connect and shows success or error message
3. **Given** successful connection to remote Ollama, **When** wizard retrieves models, **Then** available models are displayed in a table
4. **Given** remote Ollama has Mistral 7B Instruct model, **When** wizard shows model list, **Then** Mistral model is highlighted as recommended
5. **Given** connection to remote Ollama fails, **When** wizard detects failure, **Then** user sees helpful error message and can retry with different URL or abort
6. **Given** user selects Custom provider, **When** wizard prompts for endpoint, **Then** user can enter OpenAI-compatible URL (e.g., Azure OpenAI, LocalAI, LM Studio)
7. **Given** user enters custom endpoint details, **When** wizard tests connection, **Then** connection is validated before saving config

---

### User Story 4 - Updating Existing Configuration (Priority: P3)

A user has a working config but wants to change one thing (switch from OpenAI to Ollama, update API key, change graph path). They run `logsqueak init` and the wizard loads their current settings as defaults. They can quickly navigate to the setting they want to change, update it, and keep everything else the same. The wizard skips the embedding model test if already cached, making updates fast.

**Why this priority**: Configuration needs change over time (API key rotation, switching providers for cost, moving Logseq graph). Making updates easy encourages proper maintenance rather than manual YAML editing which is error-prone.

**Independent Test**: Can be tested by running `logsqueak init` with existing valid config, changing only one setting (e.g., API key), and verifying other settings remain unchanged.

**Acceptance Scenarios**:

1. **Given** existing valid config, **When** user runs `logsqueak init`, **Then** wizard shows current values for all settings
2. **Given** user wants to update API key only, **When** wizard reaches LLM configuration, **Then** user can change API key while keeping endpoint and model unchanged
3. **Given** user wants to switch from OpenAI to Ollama, **When** user selects different provider, **Then** wizard walks through new provider's settings
4. **Given** user wants to keep most settings, **When** wizard prompts for each setting, **Then** pressing Enter accepts the current value
5. **Given** user completes wizard with partial updates, **When** wizard writes config, **Then** unchanged settings retain their original values

---

### Edge Cases

- What happens when the Logseq graph path exists but is missing either `journals/` or `logseq/` directories (wizard rejects with clear error message stating which subdirectory is missing)?
- How does the wizard handle network timeouts when testing LLM connection (after 30 seconds, prompt user to continue waiting, retry, or skip)?
- What if the user provides an OpenAI API key but selects a model that doesn't exist or they don't have access to?
- What happens when Ollama is running but has no models installed?
- How does the wizard handle a config file that exists but is completely empty or has only whitespace?
- What if the config directory (`~/.config/logsqueak/`) doesn't exist and cannot be created (permission denied)?
- What happens when the user aborts the wizard mid-way (Ctrl+C or answers "no" to continue)?
- How does the wizard handle existing config with invalid YAML syntax?
- What if the Ollama endpoint returns an error for `/api/tags` but is actually running?
- What happens when the graph path contains spaces or special characters?
- What happens when embedding model download exceeds timeout (after 5 minutes, prompt user to continue waiting, retry, or skip)?
- What if embedding model download succeeds but model fails to load (corrupted download, insufficient memory)?
- How does the wizard handle partial embedding model downloads (interrupted connection)?
- What happens when available disk space is less than 1GB before embedding model download (warn user, allow proceeding if confirmed)?
- What if disk space runs out during embedding model download (download fails, show clear error with disk space info)?
- What happens when user switches from OpenAI to Ollama (wizard preserves OpenAI API key in config file for future use)?
- What if user has both OpenAI and Ollama configured and switches between them (both sets of credentials preserved in config)?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST provide a `logsqueak init` command that launches an interactive setup wizard
- **FR-002**: Wizard MUST detect if a config file already exists at `~/.config/logsqueak/config.yaml`
- **FR-003**: Wizard MUST prompt user for confirmation before overwriting an existing config file
- **FR-004**: Wizard MUST attempt to load and parse existing config file if it exists, gracefully handling errors
- **FR-005**: Wizard MUST use values from existing config as defaults for all prompts where applicable
- **FR-006**: Wizard MUST prompt user for Logseq graph directory path
- **FR-007**: Wizard MUST validate that the provided graph path exists and is a directory
- **FR-008**: Wizard MUST validate that the graph path contains both `journals/` and `logseq/` subdirectories
- **FR-009**: Wizard MUST reject graph path with clear error message if either `journals/` or `logseq/` subdirectory is missing
- **FR-010**: Wizard MUST offer choice of LLM providers: Ollama, OpenAI, or Custom OpenAI-compatible endpoint
- **FR-011**: For Ollama provider, wizard MUST attempt to connect to existing config's Ollama URL if available, otherwise try `http://localhost:11434` automatically
- **FR-012**: For Ollama provider, wizard MUST allow user to specify custom Ollama URL if both localhost and any pre-existing remote connection fail
- **FR-013**: For Ollama provider, wizard MUST retrieve list of installed models via `/api/tags` endpoint
- **FR-014**: For Ollama provider, wizard MUST display available models in a formatted table with model name and size
- **FR-015**: For Ollama provider, wizard MUST highlight Mistral 7B Instruct model as recommended if available
- **FR-016**: For Ollama provider, wizard MUST suggest installing Mistral 7B Instruct if no models are installed
- **FR-017**: For OpenAI provider, wizard MUST prompt for API key
- **FR-018**: For OpenAI provider, wizard MUST offer common models (gpt-4o, gpt-4-turbo, gpt-3.5-turbo) plus custom option
- **FR-019**: For Custom provider, wizard MUST prompt for OpenAI-compatible endpoint URL, API key, and model name (supports Azure OpenAI, LocalAI, LM Studio, vLLM, etc.)
- **FR-020**: Wizard MUST test LLM connection before saving configuration by sending a minimal test request
- **FR-021**: Wizard MUST display clear success or failure message after testing LLM connection
- **FR-022**: Wizard MUST allow user to retry configuration if LLM connection test fails
- **FR-023**: Wizard MUST allow user to continue without successful test if they explicitly confirm
- **FR-024**: Wizard MUST optionally prompt for advanced settings (num_ctx for Ollama, top_k for RAG)
- **FR-025**: Wizard MUST create `~/.config/logsqueak/` directory if it doesn't exist
- **FR-026**: Wizard MUST write configuration as valid YAML to `~/.config/logsqueak/config.yaml`
- **FR-027**: Wizard MUST set config file permissions to mode 600 (read/write for owner only)
- **FR-028**: Wizard MUST display success message with next steps after successful configuration
- **FR-029**: Wizard MUST use Rich library for formatted output (panels, tables, colored text, spinners)
- **FR-030**: Wizard MUST display current values from existing config when prompting for updates
- **FR-031**: For existing API keys from config, wizard MUST display masked version only (first 8 chars + "..." + last 4 chars), never full key on screen
- **FR-032**: When prompting for NEW API key entry, wizard MUST show input visible (unmasked) so user can visually verify correctness
- **FR-033**: When user changes provider, wizard MUST preserve previous provider's settings in config file (not delete them)
- **FR-034**: Wizard MUST write all previously configured provider settings to config file along with newly selected provider
- **FR-035**: Wizard MUST handle partial config extraction when existing config has validation errors
- **FR-036**: Wizard MUST provide helpful error messages for all failure scenarios (network errors, permission errors, invalid paths)
- **FR-037**: Wizard MUST allow user to abort at any prompt without creating/modifying config file
- **FR-038**: Wizard MUST check if embedding model (sentence-transformers) is already cached and loadable
- **FR-039**: Wizard MUST skip embedding model test if model is already cached and loads successfully (quick validation)
- **FR-040**: Wizard MUST test embedding model after successful LLM connection test only if model is not cached or fails quick validation
- **FR-041**: Wizard MUST check available disk space before attempting embedding model download
- **FR-042**: Wizard MUST warn user if available disk space is less than 1GB and embedding model download is needed
- **FR-043**: Wizard MUST allow user to proceed with download despite low disk space warning if they explicitly confirm
- **FR-044**: Wizard MUST display progress indicator during embedding model download showing download status
- **FR-045**: Wizard MUST validate that embedding model loads successfully by attempting to initialize it
- **FR-046**: Wizard MUST display clear error message if embedding model download or initialization fails
- **FR-047**: Wizard MUST allow user to retry embedding model download if it fails
- **FR-048**: Wizard MUST allow user to continue without successful embedding test if they explicitly confirm (warn that RAG search will fail)
- **FR-049**: Wizard MUST use 30-second timeout for LLM connection test operations
- **FR-050**: Wizard MUST use 5-minute timeout for embedding model download operations
- **FR-051**: When timeout is reached, wizard MUST prompt user with options to continue waiting, retry, or skip (not auto-abort)

### Key Entities *(include if feature involves data)*

- **Configuration File**: YAML file at `~/.config/logsqueak/config.yaml` containing three sections (llm, logseq, rag) with proper permissions (mode 600), can store settings for multiple LLM providers simultaneously
- **LLM Provider**: One of three types (Ollama, OpenAI, Custom OpenAI-compatible) with provider-specific settings (endpoint, API key, model, optional num_ctx), config preserves all provider settings even when switching active provider, Custom type includes Azure OpenAI, LocalAI, LM Studio, vLLM, and other OpenAI-compatible services
- **Ollama Model**: Model installed in Ollama instance with name and size attributes, retrieved from `/api/tags` endpoint
- **Embedding Model**: Sentence-transformers model for RAG semantic search, approximately 500MB download, cached locally after first successful load
- **Validation Result**: Success/failure status from LLM connection test and embedding model test, includes error details if failed

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: New users can complete initial setup in under 3 minutes from running `logsqueak init` to successful `logsqueak extract`
- **SC-002**: Users with broken configs can fix issues in under 2 minutes without manually editing YAML
- **SC-003**: LLM connection validation catches 100% of invalid configurations before saving config file
- **SC-004**: Zero manual YAML editing required for common configuration tasks (initial setup, provider change, API key update)
- **SC-005**: All config files created by wizard pass permission check (mode 600) on first attempt
- **SC-006**: Users can switch between LLM providers (OpenAI ↔ Ollama) in under 1 minute using wizard
- **SC-007**: Wizard successfully detects and offers to fix 100% of permission errors in existing configs
- **SC-008**: Remote Ollama configuration (non-localhost) works for 100% of valid remote endpoints
- **SC-009**: Config updates with cached embedding model complete in under 30 seconds (skips ~500MB download)
- **SC-010**: Wizard detects and warns about insufficient disk space before starting 500MB download in 100% of cases

## Assumptions *(mandatory)*

- Users have basic command-line familiarity (can run commands, understand file paths)
- Users have write permissions to `~/.config/` directory
- For Ollama users, Ollama is either already installed or user knows where remote instance is running
- For OpenAI users, users have already created an API key and have it available
- Network connectivity is available for LLM connection testing
- Users understand the difference between local and remote Ollama (or wizard guidance is sufficient)
- Existing Rich library dependency is acceptable (no new dependencies added)
- YAML is appropriate format for human-editable config files
- Mode 600 permissions are sufficient security for API keys in config file
- Cached embedding models remain valid and don't require re-download unless corrupted or deleted
- Quick model load validation is sufficient to verify cached embedding model integrity

## Dependencies *(optional)*

### External Systems

- **Ollama API**: Wizard connects to `/api/tags` endpoint to retrieve installed models, expects JSON response with models array
- **OpenAI API**: Wizard sends test request to `/v1/chat/completions` or similar endpoint to validate API key and model access
- **Logseq Graph**: Wizard validates that specified path contains Logseq directory structure (journals/ or logseq/ subdirectories)

### Internal Dependencies

- **Existing Config Module** (`src/logsqueak/models/config.py`): Wizard uses Config.load() to validate final configuration
- **LLMClient Service** (`src/logsqueak/services/llm_client.py`): Wizard uses for LLM connection testing
- **PageIndexer Service** (`src/logsqueak/services/page_indexer.py`): Wizard uses to test embedding model loading (sentence-transformers)
- **Rich Library**: Already in dependencies for TUI, wizard reuses for formatted CLI output
- **Sentence-Transformers**: Embedding model library (~500MB) required for RAG semantic search, wizard validates it downloads and loads successfully

## Out of Scope *(optional)*

The following are explicitly NOT included in this feature:

- **Automatic Ollama Installation**: Wizard will not install Ollama itself, only detect and configure existing instances
- **Automatic Model Downloads**: Wizard will not run `ollama pull` to download models, only suggest which to install
- **Logseq Graph Auto-Discovery**: Wizard will not search filesystem for Logseq graphs, user must provide path
- **Config Migration**: Wizard will not migrate configs from old formats or other tools
- **Multi-Config Management**: Wizard creates/updates only the default config location, no support for multiple configs
- **Encryption of API Keys**: Config file uses mode 600 permissions but keys are not encrypted at rest
- **Advanced LLM Settings**: Only num_ctx and top_k are configurable, other LLM parameters use defaults
- **Provider Auto-Detection**: Wizard will not try to detect which provider user should use, user must choose
- **Logseq Version Compatibility Checks**: Wizard validates directory structure but not Logseq version
- **Doctor/Diagnostic Command**: Separate from wizard, out of scope for this feature (future enhancement)

## Risks *(optional)*

- **LLM Connection Testing Delays**: Testing connection might be slow (5-10+ seconds), could frustrate users during setup. Mitigation: Show spinner/progress indicator, 30-second timeout with user prompt (continue/retry/skip), allow skipping test.
- **Embedding Model Download Delays**: First-time download is ~500MB and may take several minutes on slow connections. Users might abort thinking wizard is frozen. Mitigation: Show clear progress bar with download percentage and estimated time, 5-minute timeout with user prompt (continue/retry/skip), allow skipping test with warning.
- **Embedding Model Download Failures**: Large download increases chance of network interruptions, timeouts, or firewall blocks. Mitigation: Implement retry logic, user-controlled timeout handling (prompt instead of auto-abort), allow continuing without embedding test (warn RAG search won't work), provide clear error messages with troubleshooting steps.
- **Insufficient Disk Space**: Users might not have 1GB+ free for embedding model download. Download could fail mid-way if disk fills. Mitigation: Check disk space before download, warn if less than 1GB but allow proceeding if user confirms, provide clear error with disk space info if download fails.
- **Permission Errors on Config Directory**: User might not have write access to `~/.config/`. Mitigation: Catch exception, suggest alternative locations or manual creation with sudo.
- **Ollama Version Compatibility**: Different Ollama versions might have different `/api/tags` response formats. Mitigation: Handle JSON parsing errors gracefully, fall back to manual model entry.
- **Network Firewall Issues**: Corporate networks might block access to OpenAI, remote Ollama, or HuggingFace model downloads. Mitigation: Clear error messages, allow continuing without test.
- **Breaking Changes to Config Format**: If config schema changes, wizard-created configs might break. Mitigation: Version the config format, plan migration strategy.
- **User Confusion on Provider Choice**: Users might not know which provider to choose. Mitigation: Provide descriptions in table, default to Ollama for simplicity.
