# ADS Agent

![Python](https://img.shields.io/badge/Python-3.14+-3776AB?logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-1.2+-000000)
![FastAPI](https://img.shields.io/badge/FastAPI-0.139+-009688?logo=fastapi&logoColor=white)
![MCP](https://img.shields.io/badge/MCP-FastMCP-000000)
![Gemini](https://img.shields.io/badge/Gemini-Google%20AI-8E75B2?logo=google&logoColor=white)
![pgvector](https://img.shields.io/badge/pgvector-PostgreSQL%2016-336791?logo=postgresql&logoColor=white)
![Langfuse](https://img.shields.io/badge/Langfuse-v4-6366F1)
![License](https://img.shields.io/badge/License-MIT-green)

**ADS Agent** (Architecture Decision Support) is a production-grade multi-agent system that helps engineering teams evaluate technical alternatives with structured evidence, trade-off analysis, and auditable execution metadata. A **Supervisor** node orchestrates specialized **Research**, **Analysis**, and **Writer** agents through a LangGraph state machine, producing a structured **DecisionReport** and an **ExecutionReceipt** for every run. Full OpenTelemetry-native observability via **Langfuse v4** and native token/cost tracking make each decision traceable, comparable, and FinOps-ready.

---

## Key Features

- **Multi-agent orchestration** — LangGraph `StateGraph` with a hybrid Supervisor router: deterministic Python rules for clear states, LLM-assisted routing for ambiguous ones, and a circuit breaker to prevent runaway loops.
- **Fail-safe OpenTelemetry observability** — Langfuse v4 spans and generations wrapped in `tracer.py`; tracing failures never crash the pipeline.
- **Precise token and cost tracking** — `ExecutionReceipt` aggregates per-agent metrics; LiteLLM `completion_cost()` records real USD estimates on every LLM call.
- **Automated post-execution quality scoring** — RAGAS evaluation (faithfulness, answer relevancy, context precision) runs off the critical path and pushes scores to Langfuse.
- **Modular agent topology** — Clean Architecture layers (`core` → `application` → `infrastructure` → `agents` → `api`) with swappable checkpointers, vector store, and observability adapters.

---

## Architecture & Workflow

### System Architecture

The system exposes a **FastAPI** gateway and CLI entry point. Both invoke the same LangGraph pipeline, which routes LLM calls through **LiteLLM** to provider models (default: Google Gemini). A sidecar **Langfuse v4** layer captures traces without blocking execution. **PostgreSQL + pgvector** backs RAG retrieval and LangGraph checkpoint persistence.

```mermaid
flowchart TB
    subgraph Clients
        CLI["CLI<br/>(ads-agent run)"]
        API["FastAPI Gateway<br/>POST /api/v1/decisions"]
    end

    subgraph Pipeline["LangGraph Engine"]
        SG["StateGraph<br/>build_graph()"]
        SUP["Supervisor Node<br/>supervisor_node()"]
        RES["Research Node<br/>research_node()"]
        ANA["Analysis Node<br/>analysis_node()"]
        WRT["Writer Node<br/>writer_node()"]
        SG --> SUP
        SUP --> RES & ANA & WRT
        RES & ANA & WRT --> SUP
    end

    subgraph LLM["LiteLLM Wrapper"]
        LC["complete()<br/>infrastructure/llm/client.py"]
    end

    subgraph Providers["LLM Providers"]
        GEM_PRO["gemini/gemini-2.5-pro<br/>(Supervisor routing)"]
        GEM_FLASH["gemini/gemini-2.5-flash<br/>(Workers)"]
    end

    subgraph Tools["Tool & Data Layer"]
        MCP["FastMCP Server<br/>web_search · fetch_url · hybrid_search"]
        PG["PostgreSQL + pgvector<br/>RAG · Checkpoints"]
    end

    subgraph Observability["Observability Sidecar"]
        TR["tracer.py<br/>pipeline_trace · agent_span · llm_generation"]
        LF["Langfuse v4<br/>(OpenTelemetry-native)"]
        TR --> LF
    end

    CLI --> SG
    API --> SG
    RES & ANA & WRT & SUP --> LC
    LC --> GEM_PRO & GEM_FLASH
    RES --> MCP
    MCP --> PG
    SG --> PG
    Pipeline --> TR
```

> **Note:** Observability is opt-in. When `LANGFUSE_PUBLIC_KEY` is unset or uses a test key (`pk-lf-test*`), `get_langfuse_client()` returns `None` and all tracer context managers become no-ops — the pipeline continues normally.

### Pipeline Workflow (Graph Topology)

Every run starts at the **Supervisor**, which sets `next_agent` in shared `AgentState`. The conditional edge function `should_continue()` maps that value to the next node. All worker nodes unconditionally loop back to the Supervisor until `FINISH` or the circuit breaker triggers.

```mermaid
stateDiagram-v2
    [*] --> Supervisor: START

    state Supervisor {
        [*] --> CheckCircuitBreaker
        CheckCircuitBreaker --> FINISH: iterations >= MAX_ITERATIONS
        CheckCircuitBreaker --> RouteOnError: error present
        CheckCircuitBreaker --> DeterministicRoute: clear state
        CheckCircuitBreaker --> LLMRoute: ambiguous state
        LLMRoute --> ValidateRoute: SupervisorDecision.next_agent
        ValidateRoute --> SetNextAgent
        DeterministicRoute --> SetNextAgent
        RouteOnError --> SetNextAgent: next_agent = FINISH
    }

    Supervisor --> Research: next_agent = research
    Supervisor --> Analysis: next_agent = analysis
    Supervisor --> Writer: next_agent = writer
    Supervisor --> [*]: next_agent = FINISH

    Research --> Supervisor: research_output populated
    Analysis --> Supervisor: analysis_output populated
    Writer --> Supervisor: final_report populated

    note right of Supervisor
        deterministic_route():
        no research → research
        no analysis → analysis
        no report → writer
        else → FINISH
    end note

    note right of Writer
        Produces DecisionReport
        + ExecutionReceipt finalized
    end note
```

**Routing logic summary**

| Condition | Route |
| --- | --- |
| `iterations >= MAX_ITERATIONS` | `FINISH` (circuit breaker) |
| `error` is set | `FINISH` |
| No `research_output` | `research` |
| No `analysis_output` | `analysis` |
| No `final_report` | `writer` |
| Ambiguous outputs (short text, insufficiency markers, &lt; 3 trade-offs) | LLM consult via `llm_route()` → validated by `validate_llm_route()` |

---

## Tech Stack & Tools

| Category | Component | Version / Notes |
| --- | --- | --- |
| **Runtime** | Python | ≥ 3.14 |
| **Package manager** | uv | Lockfile: `uv.lock` |
| **Agent framework** | LangGraph | ≥ 1.2.2 |
| **Agent prebuilt** | langgraph-prebuilt | ≥ 1.1.0 |
| **Checkpointing** | langgraph-checkpoint-postgres | ≥ 2.0.0 |
| **LLM orchestration** | LangChain Core / Community | ≥ 1.4.0 / ≥ 0.3.0 |
| **LLM abstraction** | LiteLLM + langchain-litellm | ≥ 1.86.2 / ≥ 0.3.0 |
| **LLM providers** | Google Gemini (default) | Via `GEMINI_API_KEY` |
| **API gateway** | FastAPI + Uvicorn | ≥ 0.139.0 / ≥ 0.48.0 |
| **Validation / settings** | Pydantic v2 + pydantic-settings | ≥ 2.13.4 / ≥ 2.14.1 |
| **Tool protocol** | MCP SDK + FastMCP + langchain-mcp-adapters | ≥ 1.27.2 / ≥ 3.3.1 |
| **Web search** | Tavily (via MCP) | `TAVILY_API_KEY` |
| **Vector store** | PostgreSQL 16 + pgvector | Docker: `pgvector/pgvector:pg16` |
| **DB driver** | psycopg + psycopg-pool | ≥ 3.2.0 |
| **Observability** | Langfuse v4 (OpenTelemetry-native) | ≥ 4.7.1 |
| **Evaluation** | RAGAS + DeepEval (dev) | ≥ 0.3 / ≥ 2.0.0 |
| **HTTP client** | httpx | ≥ 0.28.1 |
| **Content extraction** | trafilatura | ≥ 2.0.0 |
| **Resilience** | tenacity | ≥ 9.0.0 |
| **Logging** | structlog | ≥ 25.5.0 |
| **Numerics** | numpy | ≥ 2.0.0 |
| **Datasets** | HuggingFace datasets | ≥ 3.0.0 |
| **Lint / format** | Ruff | ≥ 0.15.15 (dev) |
| **Testing** | pytest + pytest-asyncio + pytest-cov | ≥ 9.0.3 (dev) |

---

## Prerequisites & Requirements

### System dependencies

| Requirement | Purpose |
| --- | --- |
| **Python 3.14+** | Runtime (see `requires-python` in `pyproject.toml`) |
| **[uv](https://docs.astral.sh/uv/)** | Dependency installation and script execution |
| **Docker + Docker Compose** | Local PostgreSQL with pgvector (`make docker-up`) |
| **GEMINI_API_KEY** | LLM inference via LiteLLM |
| **TAVILY_API_KEY** | Web search in the Research agent |
| **LANGFUSE_* keys** (optional) | Production observability and trace dashboards |

### Environment variables

Copy the template and fill in your values:

```bash
cp .env.example .env
```

```dotenv
# --- PostgreSQL (docker-compose) ---
POSTGRES_DB=adsagent
POSTGRES_USER=adsagent
POSTGRES_PASSWORD=adsagent
POSTGRES_PORT=5432
ADS_DATABASE_URL=postgresql://adsagent:adsagent@localhost:5432/adsagent

# --- LLM: Google Gemini (via LiteLLM) ---
GEMINI_API_KEY=your-gemini-key

# --- Web search: Tavily ---
TAVILY_API_KEY=your-tavily-key

# --- Observability: Langfuse Cloud (recommended) ---
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com

# --- ADS Agent settings ---
ADS_RESEARCH_MODEL=gemini/gemini-2.5-flash
ADS_LLM_SUPERVISOR_MODEL=gemini/gemini-2.5-pro
ADS_LLM_WORKER_MODEL=gemini/gemini-2.5-flash
ADS_MAX_ITERATIONS=5
ADS_LOG_LEVEL=INFO

# --- Evaluation (opt-in) ---
ADS_EVAL_ENABLED=false
ADS_EVAL_SAMPLE_RATE=1.0

# --- FastAPI Gateway ---
APP_ENV=development
ADS_API_HOST=0.0.0.0
ADS_API_PORT=8080
ADS_API_PIPELINE_TIMEOUT=120
ADS_CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

> **Note:** Langfuse runs on **Cloud** by default. Self-hosted Langfuse is supported by setting `LANGFUSE_HOST` to your instance URL.

---

## API & Data Structures

### `run_pipeline()` — pipeline entry point

Defined in `ads_agent.agents.supervisor.graph`. This is the single orchestration function used by both the CLI and FastAPI service layer.

```python
async def run_pipeline(
    request: DecisionRequest,
    thread_id: str | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> tuple[AgentState, ExecutionReceipt]:
```

**Input expectations**

| Parameter | Type | Description |
| --- | --- | --- |
| `request` | `DecisionRequest` | Validated domain entity with `id` (UUID), `query` (10–2000 chars), optional `context` |
| `thread_id` | `str \| None` | LangGraph checkpoint thread ID; defaults to `request.id` for resumable runs |
| `checkpointer` | `BaseCheckpointSaver \| None` | `MemorySaver` (tests), `AsyncPostgresSaver` (production), or `None` |

**Initial `AgentState`**

```python
{
    "request": request,
    "messages": [HumanMessage(content=request.query)],
    "next_agent": "",
    "research_output": None,
    "retrieved_contexts": [],
    "analysis_output": None,
    "final_report": None,
    "receipt": ExecutionReceipt(request_id=request.id),
    "iterations": 0,
    "error": None,
}
```

**Returns** — `(final_state, receipt)` where `final_state["final_report"]` is a `DecisionReport` on success.

### `ExecutionReceipt` — cost and token mapping

Domain entity: `ads_agent.core.entities.execution_receipt.ExecutionReceipt`

| Field | Description |
| --- | --- |
| `request_id` | Links to `DecisionRequest.id` |
| `trace_id` | Langfuse trace ID captured at pipeline start |
| `agents` | List of `AgentMetrics` — one entry per node execution |
| `estimated_cost_usd` | Accumulated via LiteLLM `completion_cost()` in `complete()` |
| `source_urls` / `sources_consulted` | Unique URLs from Research |
| `iterations` | Supervisor loop count |
| `circuit_breaker_triggered` | `True` when `MAX_ITERATIONS` exceeded |

Each `AgentMetrics` record captures `agent_name`, `status`, `started_at`, `completed_at`, `input_tokens`, `output_tokens`, and computed `duration_seconds` / `total_tokens`.

### Fail-safe tracing (`tracer.py`)

All Langfuse SDK calls are isolated in `ads_agent.infrastructure.observability.tracer`. The rest of the codebase never imports Langfuse directly.

| Function | Purpose |
| --- | --- |
| `is_tracing_enabled()` | Returns `True` when valid `LANGFUSE_PUBLIC_KEY` is configured |
| `get_langfuse_client()` | Returns client or `None` — never raises |
| `pipeline_trace(request_id, session_id=...)` | Root span `ads-agent-pipeline` via `start_as_current_observation()` |
| `agent_span(name, iteration=...)` | Per-node span (`supervisor`, `research`, `analysis`, `writer`) |
| `llm_generation(name, model, input_messages)` | Nested generation observation; yields observation for `update_generation()` |
| `capture_trace_id()` | Stores trace ID on `ExecutionReceipt.trace_id` |
| `submit_pipeline_scores()` | Post-run scores: `has_sources`, `trade_offs_count` |
| `submit_evaluation_scores()` | Async RAGAS scores pushed to Langfuse |
| `flush_traces()` | Ensures pending spans are exported before process exit |

**Resilience guarantees**

- `_safe_call()` wraps every SDK mutation; errors are logged and swallowed.
- Context managers (`pipeline_trace`, `agent_span`, `llm_generation`) yield immediately when the client is `None` or on SDK failure.
- Test keys prefixed with `pk-lf-test` are treated as unconfigured.

### REST API endpoints

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/api/v1/decisions` | Run full pipeline synchronously (~10–30 s) |
| `GET` | `/api/v1/decisions/{request_id}` | Retrieve report from LangGraph checkpoints |
| `GET` | `/api/v1/decisions/{request_id}/receipt` | Full `ExecutionReceipt` with per-agent breakdown |
| `GET` | `/health` | PostgreSQL (required) + Langfuse (optional) health |

OpenAPI docs: [http://localhost:8080/docs](http://localhost:8080/docs)

### API screenshots

![POST /api/v1/decisions — Swagger UI response showing a structured DecisionReport with recommendation, trade-offs, and receipt summary](docs/images/decisions-api.png)

*Figure 1 — **Decision creation via REST API.** A `POST /api/v1/decisions` request with query and context returns HTTP 200 with a full `DecisionReport`: recommendation, strength, summary, structured `trade_offs` (dimension, option_a, option_b, winner), and an embedded receipt summary including `request_id` and `trace_id` for downstream observability.*

![GET /api/v1/decisions/{request_id}/receipt — Full ExecutionReceipt with per-agent token usage and Langfuse trace URL](docs/images/receipe-request.png)

*Figure 2 — **Execution receipt for FinOps and AgentOps.** The `/receipt` endpoint exposes the complete operational record: per-agent timing (`supervisor`, `research`, …), token counts (`input_tokens`, `output_tokens`, `total_tokens`), and a direct `langfuse_trace_url` linking to the OpenTelemetry trace in Langfuse Cloud for deep-dive debugging.*

---

## Installation & Quickstart

### 1. Clone and install

```bash
git clone https://github.com/lasmcode/ads-agent.git
cd ads-agent

# Install runtime + dev dependencies and pre-commit hooks
make dev
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — at minimum: GEMINI_API_KEY, TAVILY_API_KEY
```

### 3. Start local services

```bash
make docker-up
# PostgreSQL → localhost:5432
```

### 4. Run tests

```bash
make test-unit       # Fast — no external services required
make test            # All tests
make test-cov        # With HTML coverage report
```

### 5. Run the pipeline (CLI)

```bash
# Text output (default)
uv run ads-agent run "Should I use pgvector or Qdrant for my RAG system?"

# JSON output
uv run ads-agent run "Should I use pgvector or Qdrant?" --output json

# Resumable run with checkpoint thread ID
uv run ads-agent run "Should I use Redis or Memcached?" --thread-id my-session-1
```

### 6. Start the API server

```bash
make api-dev            # Development (auto-reload, port 8080)
make api-run            # Production mode (no reload)
```

**End-to-end curl flow**

```bash
curl -s http://localhost:8080/health | jq

curl -s -X POST http://localhost:8080/api/v1/decisions \
  -H "Content-Type: application/json" \
  -d '{"query": "Should I use Redis or Memcached for session storage?", "context": "Team of 5, 10k concurrent users"}' \
  | tee response.json | jq

REQUEST_ID=$(jq -r '.request_id' response.json)
curl -s "http://localhost:8080/api/v1/decisions/${REQUEST_ID}" | jq
curl -s "http://localhost:8080/api/v1/decisions/${REQUEST_ID}/receipt" | jq
```

---

## Development

```bash
make help            # Show all available commands
make lint            # Ruff linter
make format          # Ruff formatter
make check           # Lint + format check (CI mode)
make docker-down     # Stop services
make clean           # Remove caches and coverage artifacts
```

### Project structure

```
ads-agent/
├── src/ads_agent/
│   ├── core/              # Domain entities, ports, settings
│   ├── application/       # Decision service, evaluation runner
│   ├── infrastructure/    # LLM, MCP, vector store, observability, persistence
│   ├── agents/            # LangGraph nodes (supervisor, research, analysis, writer)
│   └── api/               # FastAPI gateway (v1 routes, schemas, middleware)
├── tests/
│   ├── unit/              # Pure unit tests (no I/O)
│   └── integration/       # Tests requiring Postgres / LLM APIs
├── docs/images/           # API and observability screenshots
└── docker-compose.yml     # PostgreSQL + pgvector
```

### Tiered LLM strategy

| Role | Default model | Environment variable |
| --- | --- | --- |
| Supervisor (ambiguous routing) | `gemini/gemini-2.5-pro` | `ADS_LLM_SUPERVISOR_MODEL` |
| Workers (research, analysis, writer) | `gemini/gemini-2.5-flash` | `ADS_LLM_WORKER_MODEL` / `ADS_RESEARCH_MODEL` |

Deterministic Python rules remain the circuit breaker. The supervisor LLM is consulted only when outputs appear insufficient (short text, insufficiency markers, invalid analysis JSON).

**Langfuse trace hierarchy (per run)**

- Root trace `ads-agent-pipeline` per run, linked via `ExecutionReceipt.trace_id`
- Nested spans per graph node (`supervisor`, `research`, `analysis`, `writer`)
- LLM generations with model, input/output, and token usage
- Post-execution scores: `has_sources`, `trade_offs_count`
- Async RAGAS scores (fire-and-forget): `faithfulness`, `answer_relevancy`, `context_precision`, `quality_score`

### Observability verification

1. Configure `LANGFUSE_*` in `.env`
2. Run: `uv run ads-agent run "Should I use Redis or Memcached for session storage?"`
3. Copy the **Trace ID** from the execution receipt
4. Open [cloud.langfuse.com](https://cloud.langfuse.com) → **Tracing** → search by trace ID
5. Confirm hierarchy: `ads-agent-pipeline` → `supervisor` / `research` / `analysis` / `writer` spans with nested generations
6. Check **Scores** for `has_sources`, `trade_offs_count`, and (when `ADS_EVAL_ENABLED=true`) RAGAS metrics — RAGAS scores appear after ~1–2s

---

## Roadmap

- [x] Phase 0 — Project bootstrap
- [x] Phase 1 — Supervisor Agent with LangGraph
- [x] Phase 2 — MCP Tool Layer
- [x] Phase 3 — RAG Pipeline with pgvector
- [x] Phase 4 — Full Multi-Agent System (LiteLLM tiered models)
- [x] Phase 5 — Langfuse Observability
- [x] Phase 6 — Evaluation Engine (RAGAS + golden dataset gate)
- [x] Phase 7 — FastAPI Gateway
- [ ] Phase 8 — Docker + CI/CD deployment

---

## Phase 6 — Evaluation Engine

RAGAS scores pipeline output quality off the critical path; DeepEval gates regressions on the golden dataset in nightly CI.

| Metric | Weight | Production threshold | Langfuse score name |
| --- | --- | --- | --- |
| Faithfulness | 40% | ≥ 0.85 (`EVAL_FAITHFULNESS_THRESHOLD`) | `faithfulness` |
| Answer Relevancy | 35% | ≥ 0.80 (`EVAL_ANSWER_RELEVANCY_THRESHOLD`) | `answer_relevancy` |
| Context Precision | 25% | ≥ 0.75 (`EVAL_CONTEXT_PRECISION_THRESHOLD`) | `context_precision` |
| **Quality score** | weighted avg | ≥ 0.75 batch (`EVAL_QUALITY_THRESHOLD`) | `quality_score` |

**Weighted formula:** `quality_score = 0.40×faithfulness + 0.35×answer_relevancy + 0.25×context_precision` (weights renormalized when context precision is unavailable).

| Variable | Default | Description |
| --- | --- | --- |
| `ADS_EVAL_ENABLED` | `false` | Enable fire-and-forget RAGAS evaluation (opt-in) |
| `RUN_QUALITY_GATE` | _(unset)_ | Set to `1` to run golden dataset quality gate tests |
| `ADS_EVAL_SAMPLE_RATE` | `1.0` | Fraction of runs to evaluate (use `0.05–0.15` in production) |
| `ADS_EVAL_TIMEOUT_SECONDS` | `60` | Max seconds per RAGAS evaluation |
| `ADS_EVAL_MODEL` | `gemini/gemini-2.5-flash` | LiteLLM model for RAGAS metrics |
| `EVAL_QUALITY_THRESHOLD` | `0.75` | Nightly batch gate threshold |

Golden dataset: [`tests/fixtures/golden_dataset.json`](tests/fixtures/golden_dataset.json) (11 architecture questions).

### Interpreting low Faithfulness (< 0.85)

**Scenario:** Query "pgvector vs Qdrant" → Faithfulness = 0.62

**What it means:** ~38% of claims in the report are not supported by retrieved chunks. Example: the report states "Qdrant supports native ACID transactions" but no chunk mentions it.

**What to check first:**

1. **Retrieval** (if `context_precision` is also low < 0.75): Did `hybrid_search` return relevant chunks? In Langfuse, open the `research` span and compare `receipt.source_urls` to cited facts. Tune `ADS_RAG_SCORE_THRESHOLD`, re-ingest docs, or improve chunking.
2. **Writer prompt** (if `context_precision` is OK but faithfulness is low): Chunks were correct but the writer distorted them. Review `WRITER_SYSTEM_PROMPT` — reinforce "only cite facts from research_output".
3. **Research agent** (if both metrics are low): ReAct agent failed to extract evidence from MCP/RAG. Review `RESEARCH_SYSTEM_PROMPT` and tool usage.

### Verification commands

```bash
make test-unit                                          # includes eval formula + fire-and-forget smoke
uv run pytest tests/unit/application/test_evaluation_service.py -m unit -v
uv run pytest tests/unit/test_golden_smoke.py -m unit -v
make test-eval                                          # skipped by default (no Gemini calls)
RUN_QUALITY_GATE=1 make test-eval                       # full golden gate — requires GEMINI_API_KEY
ADS_EVAL_ENABLED=true uv run ads-agent run "..."        # enable runtime RAGAS scoring
```

Nightly workflow: [`.github/workflows/nightly-eval.yml`](.github/workflows/nightly-eval.yml) — `workflow_dispatch` only (cron disabled). Add `RUN_QUALITY_GATE: "1"` to re-enable. Does not block PRs.

## License

[MIT](LICENSE)
