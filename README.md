# Agent Observability & Token Optimization Platform

Research-oriented Streamlit platform for inspecting AI-agent executions, token use, estimated costs, and task quality. Phases 1–9 provide the runtime, OpenAI adapter, instrumentation, bounded tools and skills, analytics, evaluation, and paired experiments. Phase 10 completes setup documentation and final verification.

## Requirements

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)

## Setup

Run these commands from the repository root in PowerShell:

```powershell
uv sync --group dev --extra providers --extra tokenization --extra ui
Copy-Item .env.example .env
uv run alembic upgrade head
uv run streamlit run app.py
```

Then open `http://localhost:8501`. The default SQLite database is `storage/agent_observability.db`; startup applies pending Alembic migrations. The initial schema-adoption and subsequent history migrations are forward-only to protect existing data; back up before upgrades.

`.env.example` contains `AOP_DATABASE_URL` and `AOP_LOG_LEVEL`. Provider credentials are not read from `.env`: enter the API key for your chosen provider in Provider Settings. Keys are kept in Streamlit session state for the current browser session and passed to the adapter at request time.

### Pip setup

For a source-checkout installation without uv, use PowerShell from the repository root:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install pytest
alembic upgrade head
streamlit run app.py
```

`pyproject.toml` remains the canonical dependency configuration; uv is recommended because `uv.lock` pins the tested environment.

For a development checkout, run the tests with:

```powershell
uv run pytest
```

The OpenAI and Gemini SDKs are included in the `providers` extra. OpenRouter and Groq use OpenAI's SDK against their compatible endpoints. Ollama uses the same protocol at its local endpoint and does not require a key by default. Anthropic remains unimplemented; its SDK is isolated in the optional `additional-providers` extra.

## Foundation architecture

- `config/` owns environment-backed settings, constants, and logging configuration.
- `core/` holds validated boundary schemas and shared enums.
- `database/` owns SQLAlchemy entities and SQLite engine/session setup.
- `providers/` defines the common adapter protocol and adapters for OpenAI, Gemini, OpenRouter, Groq, and Ollama, plus custom OpenAI-compatible endpoints.
- `token_engine/` estimates explicit prompt components with tiktoken and compares them with normalized provider usage.
- `instrumentation/` persists ordered events and LLM-call usage, measures latency, estimates configured costs, and loads trace summaries.
- `tools/` contains an allowlisted tool registry, schema validation, bounded calculator and document reader, execution metrics, and estimated output token counts.
- `skills/` discovers versioned `SKILL.md` + `metadata.json` directories and estimates loaded instruction tokens.
- `agents/` manages isolated agent configurations and attachments, then runs bounded, instrumented tool-calling tasks.
- `analytics/` aggregates persisted reported token usage, component estimates, tool events, and per-agent execution outcomes.
- `evaluation/` stores execution-derived automatic checks separately from human response-quality ratings.
- `experiments/` snapshots paired agent configurations and repeatable datasets, runs baseline/optimized cases, and compares persisted outcomes.
- `tests/` verifies persistence, provider normalization, token attribution, instrumentation, tools, skills, and the mocked agent runtime.

Provider credentials are not part of the database schema or agent configuration. Supply a key to `create_provider(ProviderName.OPENAI, SecretStr(key))` or `create_provider(ProviderName.GEMINI, SecretStr(key))`; OpenRouter and Groq require a key, while Ollama is key-optional by default. Credentials are held in memory only and excluded from normalized error messages. The Streamlit Provider Settings page keeps credentials and endpoints in the current browser session only.

To make an optional live request from a Python shell, prompt for the key without echoing or saving it:

```python
from getpass import getpass

from pydantic import SecretStr

from core.enums import ProviderName
from core.schemas import ChatMessage, ProviderRequest
from providers.provider_factory import create_provider

provider = create_provider(ProviderName.OPENAI, SecretStr(getpass("OpenAI API key: ")))
response = provider.complete(
	ProviderRequest(
		model="gpt-4o-mini",
		messages=[ChatMessage(role="user", content="Reply with a short greeting.")],
	)
)
print(response.content)
print(response.usage)
```

This sends a billable request. Automated tests use mocked responses and do not require a key.

Token estimates can be generated without an API call. `TiktokenTokenizer` selects an encoding for known models and falls back to `cl100k_base` for unknown model names; inspect `encoding_name` and `used_fallback` when interpreting estimates. Prompt components are counted independently based on the inputs explicitly provided to `PromptAnalyzer`. Provider-reported usage stays separate, and the signed attribution gap is provider input tokens minus summed component estimates. A nonzero gap is a measurement difference, not evidence of hidden system prompts.

`config/model_pricing.json` starts with an empty `models` list. Add verified entries using `provider`, exact `model`, `input_price_per_million`, `output_price_per_million`, optional `cached_input_price_per_million`, `currency`, and `effective_date`. Until a matching entry and complete reported input/output usage are available, estimated cost remains unset. Cost figures are estimates, not provider billing records.

Phase 5 includes an AST-allowlisted arithmetic calculator and a UTF-8 reader restricted to a configured root directory, a 1 MB default file limit, and `.txt`, `.md`, `.csv`, and `.json` files. The Tool Manager page uploads documents into that confined directory and prevents overwrite. No unrestricted Python or shell execution is registered. Tool outputs are not counted as provider token usage automatically; the tool result can be explicitly converted to a prompt component when the runtime includes it in a later request. Tool timeouts bound how long the caller waits, but Python threads cannot forcibly stop a tool that is already running, so only trusted, bounded implementations should be registered.

The Tool Builder stores custom tool source in `custom_tools/<name>.py` and metadata in `custom_tools/tools.json`. Saved tools are loaded into the explicit tool registry at application startup and can be selected in Agent Builder. Agent execution runs the same restricted Python subset in a separate process with bounded inputs and output, range and sequence limits, and a two-second timeout. Imports, attribute access, and file/network APIs are unavailable to custom code. This is not an OS-level sandbox for hostile code; only run custom tools you trust.

## Agent runtime

Create an `AgentConfig` with explicit provider/model, instructions, skill/tool names, and iteration/tool/request-timeout limits. `AgentManager` validates attachments against the current skill and tool registries and snapshots skill versions. `AgentExecutor.execute(agent_id, prompt, provider)` creates a task/run, calls the injected provider through `LLMTracker`, executes only attached tools, passes results back using provider tool-call IDs, and records the final run and event trace. No credentials are part of `AgentConfig`; build the provider separately with the transient key as shown above.

Execution is synchronous. Cancellation is checked between provider/tool operations and cannot interrupt a request already in flight; request timeout is passed through to the OpenAI SDK. Run `uv run alembic upgrade head` when setting up or upgrading the database; the initial migration supports the Phase 1 schema and creates current tables.

## Streamlit pages

- Overview summarizes persisted executions, LLM calls, provider-reported token counts, success rate, and configured estimated costs.
- Agent Builder creates and edits model/instruction/limit settings and attaches discovered skills and registered tools.
- Skill Manager displays versioned skill instructions and toggles enabled state.
- Tool Manager displays schemas and in-process metrics, and uploads bounded documents for the confined reader.
- Tool Builder creates, tests, edits, and deletes persisted custom tools, which are available to agents through the tool allowlist.
- Task Execution submits a task to an active agent; the key for its configured provider must first be set in Provider Settings.
- Execution Traces displays ordered events and normalized LLM-call records.
- Token Analytics separates reported tokens, component estimates, and attribution gaps.
- Tool Analytics summarizes persisted tool events and subsequent-request result estimates.
- Agent Analytics compares execution, token, call, tool, and currency-separated cost metrics.
- Evaluation stores automatic execution outcomes separately from human quality ratings.
- Experiment Manager creates and runs paired baseline/optimized trials.
- Downloads include summary Markdown, trace JSON, per-call token CSV, tool analytics CSV, and experiment CSV/JSON/Markdown reports.

The experiment dataset editor expects JSON with a dataset name and cases. Each `case_id` must be unique; `expected_outcome` is optional metadata and is not automatically graded.

```json
{
	"name": "Support task sample",
	"description": "A small repeatable task set",
	"cases": [
		{
			"case_id": "case-001",
			"name": "Summarize a request",
			"prompt": "Summarize the request in one sentence.",
			"expected_outcome": "A concise summary of the request"
		}
	]
}
```

## Analytics and evaluation

Token Analytics separates provider-reported input/output/reasoning/cached counts from local component estimates and reports attribution gaps as measurement differences. Tool Analytics derives latency, failures, duplicate-argument fingerprints, output size, and result estimates from persisted events; it separately totals estimated tool-result components included in later LLM calls. Agent Analytics reports run success, calls, reported tokens, tool calls, and cost grouped by currency.

The Evaluation page can record automatic task/tool outcomes from finished run state and human 1–5 relevance, completeness, and quality scores. Automatic evaluations do not generate response-quality ratings. The `evaluations` table is added by Alembic revision `20261030_0002`; schema-adoption migrations are forward-only, so back up before upgrades.

Experiment Manager accepts a JSON dataset, freezes both agent configurations and skill-version snapshots, and records each baseline/optimized run against its case. Before running, it verifies the saved agent configurations have not changed. Comparisons use provider-reported input plus output tokens for reduction, keep reasoning/cached counts separate, group estimated costs by currency, and show task-success and human-quality changes alongside token reduction. No experiment is labeled successful from token savings alone. Experiment definitions/results are added by Alembic revision `20261030_0003`; migrations are forward-only.

## Provider support

- OpenAI: official SDK, synchronous Chat Completions, tools, normalized usage, request timeout.
- Gemini: official `google-genai` SDK, content/tool calls, prompt/candidate/reasoning/cached usage when reported.
- OpenRouter: OpenAI-compatible API; select the provider model identifier expected by OpenRouter, such as `openai/gpt-4o-mini`.
- Ollama: local OpenAI-compatible API at `http://localhost:11434/v1` by default; enter a model already available in the local Ollama installation.
- Custom OpenAI-compatible: provide a base URL and key at runtime through the factory; the Streamlit session UI also exposes those settings. Provider keys and custom endpoints remain in session state only.

## Current scope

Persisted entities establish the execution spine: agents, submitted tasks, execution runs, ordered trace events, LLM-call usage, evaluations, paired experiment definitions/results, and agent skill/tool attachments. Streamlit includes the pages and contextual downloads listed above. Gemini, OpenRouter, Ollama, and custom OpenAI-compatible calls are synchronous; Anthropic, streaming, web search/page reading, PDF reading, unrestricted Python analysis, and automatic grading against `expected_outcome` are not implemented. Tool execution intentionally has no unrestricted shell or Python capability.

Automated tests use mocked provider responses and do not require an API key or incur provider charges. Live OpenAI, Gemini, and OpenRouter actions require the corresponding session key; Ollama connects to its configured local endpoint and is key-optional by default. Estimated cost appears only when verified model rates are configured in `config/model_pricing.json`.

Phase 10 final verification used a fresh temporary uv environment with the locked full dependency set, ran the complete test suite, migrated a new temporary SQLite database to head, and launched the app against that database. This verifies source-checkout setup; deployment, multi-user secrets, and a live paid provider request are not covered.