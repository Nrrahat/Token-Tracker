# Development Log

## Phase 1: Foundation

Status: implemented and validated.

- Added uv project metadata, runtime dependencies, optional future dependency groups, and pytest configuration.
- Added Pydantic settings loaded from `AOP_` environment variables and centralized console logging.
- Added shared provider/execution/event enums and validated agent, task, and token usage schemas.
- Added SQLAlchemy models for agents, tasks, execution runs, and ordered execution events. No API secrets are stored.
- Added SQLite engine/session setup, automatic parent-directory creation for file databases, and SQLite foreign-key enforcement.
- Added a database connectivity/schema/persistence test.
- Verified the focused pytest passes and initialized the default file-backed SQLite database.

Architectural decision: `pyproject.toml` is the canonical dependency configuration; optional groups keep UI, provider SDK, and tokenizer dependencies out of the foundation install until their implementation phases.

Known limitations: schema changes are currently managed through SQLAlchemy `create_all`; migrations will be introduced when schema evolution requires them. No application UI or provider integration exists yet.

Next: Phase 1 is complete; proceed with provider integration.

## Phase 2: Provider Integration

Status: first provider adapter implemented and validated; broader provider coverage remains open.

- Added a provider-neutral request/response contract and synchronous adapter protocol.
- Added OpenAI Chat Completions request, response, usage, latency, and function tool-call normalization.
- API credentials are injected as `SecretStr` or a transient string; no credential is persisted or placed in normalized error text.
- Added mocked provider tests for usage normalization, tool calls, malformed arguments, safe request errors, and provider factory behavior.
- Added an opt-in interactive key prompt example for a live OpenAI request; it is not run automatically.
- `uv run pytest` passes (6 tests).

Known limitations: Gemini and Anthropic adapters, streaming, and real-key integration checks are not implemented. Tool execution remains Phase 5/6 work; this adapter only returns normalized tool-call requests.

## Phase 3: Token Engine

Status: implemented and validated.

- Added model-aware tiktoken estimation with an explicit `cl100k_base` fallback for unknown model names.
- Added separate estimates for system/developer instructions, user prompts, skills, tool schemas, tool results, and conversation history.
- Added normalized provider usage extraction that preserves raw metadata and leaves unavailable counts unset.
- Added signed attribution gap calculation only when provider input usage is reported; the gap is not interpreted as evidence of hidden prompts.
- Added tests for tokenizer selection, component counts, provider usage preservation, and attribution gap behavior.
- Added tiktoken to pip-based runtime requirements and documented fallback and attribution interpretation.
- `uv run pytest` passes (13 tests); workspace diagnostics report no errors.

Known limitations: prompt text must be supplied as explicit components by the caller; this phase does not infer a provider's full serialized request shape. Per-request usage persistence and analytics integration are later phases.

Next: Phase 3 is complete; continue with Phase 4 instrumentation.

## Phase 4: Instrumentation

Status: implemented and validated.

- Added ordered event persistence and request start/success/failure event types.
- Added LLM call persistence for normalized usage, raw provider usage, component estimates, attribution gap, provider and measured latency, status, and safe error type.
- Added configurable model pricing loaded from JSON; the catalog starts empty, so cost remains unset unless pricing is supplied.
- Added trace retrieval for ordered events and LLM-call summaries.
- Request/response text, credentials, and raw exception messages are deliberately excluded from LLM trace records.
- Added mocked success/failure tests, cost arithmetic coverage, and event ordering checks.
- `uv run pytest` passes (18 tests); workspace diagnostics report no errors.

Known limitations: pricing starts empty and must be populated with verified rates; instrumentation currently wraps synchronous provider calls. Event sequencing assumes serial event writes within a run. Schema evolution still uses `create_all`; migrations will be needed before deployment upgrades.

Next: Phase 4 is complete; continue with Phase 5 tools and skills.

## Phase 5: Tools and Skills

Status: implemented and validated.

- Added an allowlist tool registry, Pydantic input validation, execution status/results, timeout handling, output sizing, duplicate-call detection, failure/empty metrics, and optional output-token estimates.
- Added a calculator that evaluates only bounded numeric AST nodes and a root-confined UTF-8 text document reader with file-size and extension limits.
- Added dynamically discovered, versioned skills using `SKILL.md` and `metadata.json`, including a document-analysis skill and optional instruction-token estimates.
- Tool-result estimates remain separate from LLM usage; successful results must be explicitly added as prompt components to count toward a subsequent request.
- Tool manager deadlines stop waiting and cancel pending work; an already-running Python thread cannot be forcibly terminated. Built-in tools are bounded, and custom tools must be trusted and bounded.
- Agent attachment persistence and runtime integration were completed in Phase 6.
- Tests cover arithmetic restrictions, schema and file-size limits, path confinement, timeout reporting, duplicate calls, skill versions, token estimates, and disabled skills.
- `uv run pytest` passes (27 tests); workspace diagnostics report no errors.

Known limitations: timed-out threads cannot be forcibly killed, so the registry is for trusted built-ins rather than untrusted plugins. Skill attachment to agents and persistent tool/skill assignment are not implemented until Phase 6.

Next: Phase 5 is complete; continue with Phase 6 agent runtime and configuration.

## Phase 6: Agent Runtime

Status: implemented and validated.

- Added persisted agent configuration CRUD, activation, deletion guards, and validated tool/skill attachment snapshots including skill versions.
- Added per-agent maximum iterations, tool calls, and provider request timeout configuration; the OpenAI adapter receives the timeout value.
- Added a transparent synchronous execution loop that creates task/run records, assembles instructions and skills, requests attached tool schemas, executes only attached registered tools, and returns tool results with matching call IDs.
- Added cancellation checks between provider/tool operations, stop conditions, and ordered lifecycle, iteration, tool, and LLM trace events.
- Integrated prompt-component attribution with each LLM request and persists the final run state/result.
- Added mocked end-to-end tests for tool use, limits, cancellation, attachment validation, and persisted traces.
- Added a frozen Alembic migration that creates a fresh schema or upgrades the existing Phase 1 database, including agent request timeouts and attachment tables.
- Verified migration behavior against an empty database and a legacy database containing existing agent data.
- `uv run pytest` passes (36 tests); `uv run alembic current` reports `20261030_0001 (head)` for the workspace database; diagnostics report no errors.

Known limitations: execution is synchronous; cancellation is checked between requests and cannot stop an in-flight provider call. The current OpenAI adapter receives the per-agent timeout; future adapters must honor the same request contract. Provider credentials remain injected by the caller and are not part of agent persistence. The initial adoption migration is deliberately forward-only to protect pre-existing data; restore a backup to roll back.

Next: Phase 6 is complete; continue with Phase 7 Streamlit interface.

## Phase 7: Streamlit Interface

Status: core workflows implemented and validated.

- Added the Streamlit application shell, shared navigation/styling, and cached service composition with Alembic startup.
- Added Overview, Provider Settings, Agent Builder, Skill Manager, Tool Manager, Task Execution, and Execution Traces pages.
- Provider key entry remains in Streamlit session state and is injected into the provider per task; it is not stored in the database.
- Added skill enable/disable controls and a confined document upload/list workflow for the safe reader tool.
- Added runtime composition and upload-boundary tests covering migrations, available tools/skills, filenames, file types, size limits, and overwrite prevention.
- Browser-smoke-tested Overview, Agent Builder, Provider Settings, Skill Manager, Tool Manager, Task Execution, and Execution Traces; verified responsive layout and empty states.
- Started the local app at `http://localhost:8501` for interactive use.
- `uv run pytest` passes (39 tests); workspace diagnostics report no errors.

Known limitations: the first UI increment includes task execution and trace inspection but not the remaining token/tool analytics, evaluation, experiment, export, and additional-provider pages. Provider key is ephemeral Streamlit session state; it is not intended as deployed multi-user secret management. Tool metrics shown in the UI are process-local.

Next: Phase 7 core workflows are complete; continue with token/tool analytics, evaluation, experiments, exports, and additional providers.

## Phase 8: Analytics and Evaluation

Status: implemented and validated.

- Added persisted token analytics grouped by model/provider/agent/day and prompt component; reported usage stays separate from estimated counts and costs remain grouped by currency.
- Added durable tool analytics for duration, failure, empty results, duplicate argument fingerprints, output size, estimated output tokens, and tool-result estimates included in later LLM calls.
- Added per-agent execution, call, token, tool-use, and currency-separated estimated-cost aggregates.
- Added migration `20261030_0002` and typed evaluation records for automatic execution/tool outcomes and human quality ratings.
- Automatic evaluation is idempotent per run and does not manufacture relevance/completeness/quality scores; only finished runs can be evaluated.
- Added Token Analytics, Tool Analytics, Agent Analytics, and Evaluation pages.
- Browser-smoke-tested all four Phase 8 pages against the migrated workspace database.
- `uv run pytest` passes (44 tests); `uv run alembic current` reports `20261030_0002 (head)`; workspace diagnostics report no errors.

Known limitations: tool duplicate detection is scoped to identical argument fingerprints within the same run and does not store raw arguments. Tool metrics summarize persisted events; estimated tool-result token contribution is counted from the component estimates recorded in later LLM calls. Human scores are subjective inputs and are not combined with automatic execution outcomes.

Next: Phase 8 analytics and evaluation are complete; continue with experiment management, exports, and additional provider adapters.

## Phase 9: Experiment Management

Status: implemented and validated.

- Added validated repeatable JSON datasets with stable unique case IDs and bounded task counts/input sizes.
- Added experiment persistence with baseline/optimized agent configuration snapshots, skill versions, dataset snapshots, paired run records, and resumable status.
- Added a sequential paired runner that rejects changed agent snapshots, reuses persisted pairs when resuming, and records automatic execution evaluations.
- Added comparisons for reported input/output/reasoning/cached tokens, LLM/tool calls, latency, currency-separated estimated cost, success rate, and human ratings.
- Token/cost reduction is reported with success and human-quality deltas; no single metric declares an optimization successful.
- Added Experiment Manager UI for creating datasets, running/resuming paired trials, and inspecting comparison results.
- Browser-smoke-tested the Experiment Manager setup state; paired trial execution and comparison are covered with mocked runtime integration tests.
- `uv run pytest` passes (48 tests); `uv run alembic current` reports `20261030_0003 (head)`; workspace diagnostics report no errors.

Known limitations: experiment arms use existing saved agents; changing an arm's config or skill-version snapshot blocks execution. Dataset expected outcomes are retained but are not automatically graded. Human quality scoring remains a separate evaluation action, and pricing must be configured for cost comparisons.

Next: Phase 9 experiment management is complete; continue with final documentation and verification.

## Phase 10: Documentation and Final Verification

Status: implemented and validated.

- Completed source-checkout setup instructions for uv and pip, migration/startup commands, page descriptions, dataset JSON shape, data interpretation, and security limitations.
- Aligned `requirements.txt` with the Streamlit/Pandas/Plotly application runtime; uv remains the locked canonical setup.
- Added trace JSON, token usage CSV, tool analytics CSV, experiment comparison CSV/JSON, and Markdown summary/experiment reports with contextual download controls.
- Added synchronous Gemini, OpenRouter, and Ollama adapters; OpenRouter and Ollama use the OpenAI-compatible chat-completions shape, with Ollama key-optional and endpoint configurable.
- Added per-session provider credentials/endpoints and provider selection to Provider Settings, Agent Builder, Task Execution, and Experiment Manager.
- Verified the installed google-genai GenerateContentConfig supports http_options timeout configuration.
- Mocked provider/session-routing tests pass for Gemini, OpenRouter, Ollama, custom OpenAI-compatible endpoints, and existing OpenAI behavior.
- Installed the locked full app and test dependencies into a newly created temporary environment using `uv sync --locked`.
- Ran the complete test suite in that clean environment: 51 passed.
- Applied all migrations to a fresh temporary SQLite database; Alembic reported `20261030_0003 (head)`.
- Started Streamlit from the clean environment on port 8502 and browser-smoke-tested Overview/download controls and Experiment Manager without an API key or synthetic database records.
- Workspace diagnostics report no errors.

Known limitations: no live provider request was performed; local Ollama availability depends on the user's own Ollama service and pulled models. Anthropic, streaming, web/PDF tools, automatic grading against expected outcomes, and deployment-grade secret management remain out of scope.

Project phases 1–10 are complete for the implemented scope described in README. Suggested follow-on work is additional provider adapters, broader tools/evaluation, then deployment/security hardening.