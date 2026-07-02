# ADS Agent — Architecture Decision Support

> A multi-agent system that assists engineering teams in evaluating technical alternatives with structured evidence, trade-off analysis, and full execution traceability.

## Overview

ADS Agent orchestrates specialized AI agents to answer questions like:

- _"Should we migrate from Jenkins to GitHub Actions?"_
- _"pgvector vs Qdrant — which is right for our use case?"_
- _"LangGraph or PydanticAI for our new agent system?"_
- _"Redis or Valkey — what are the real trade-offs in 2026?"_

Each query produces two outputs:

1. **Decision Report** — structured analysis with evidence, trade-offs, and a recommendation
2. **Execution Receipt** — operational metadata: time per agent, tokens consumed, cost estimate, sources consulted, quality scores

## Architecture

```
FastAPI Gateway
     │
LangGraph Supervisor Agent
     ├── Research Agent   (web search + documentation retrieval)
     ├── Analysis Agent   (trade-off evaluation + structured reasoning)
     └── Writer Agent     (structured report generation)
          │
     MCP Tool Layer (FastMCP)
          │
     PostgreSQL + pgvector  │  Langfuse v4 (Observability)
```

## Tech Stack

| Layer            | Technology            | Version |
| ---------------- | --------------------- | ------- |
| Agent Framework  | LangGraph             | 1.2.2   |
| Tool Protocol    | MCP SDK + FastMCP     | 1.27.2  |
| LLM Abstraction  | LiteLLM               | ≥1.70   |
| API              | FastAPI + Pydantic v2 | 0.136.3 |
| Vector Store     | PostgreSQL + pgvector | pg16    |
| Observability    | Langfuse v4           | 4.7.1   |
| Package Manager  | uv                    | 0.11.16 |
| Linter/Formatter | Ruff                  | 0.15.15 |

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) installed
- Docker and Docker Compose

### 1. Clone and setup

```bash
git clone https://github.com/your-username/ads-agent.git
cd ads-agent

# Install dependencies and pre-commit hooks
make dev
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your API keys and Postgres credentials
```

### 3. Start local services

```bash
make docker-up
# PostgreSQL → localhost:5432
```

Langfuse observability uses **Langfuse Cloud** by default. Set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_HOST=https://cloud.langfuse.com` in `.env` (see `.env.example`).

### 4. Run tests

```bash
make test-unit       # Fast — no services required
make test            # All tests
make test-cov        # With coverage report
```

### 5. Run the pipeline locally

```bash
# Text output (default)
uv run ads-agent run "Should I use pgvector or Qdrant for my RAG system?"

# JSON output
uv run ads-agent run "Should I use pgvector or Qdrant for my RAG system?" --output json

# Resumable run with checkpoint thread ID
uv run ads-agent run "Should I use pgvector or Qdrant?" --thread-id my-session-1
```

Optional environment variables:

| Variable                  | Default                                              | Description                                |
| ------------------------- | ---------------------------------------------------- | ------------------------------------------ |
| `POSTGRES_DB`             | `adsagent`                                           | Docker Postgres database name              |
| `POSTGRES_USER`           | `adsagent`                                           | Docker Postgres user                       |
| `POSTGRES_PASSWORD`       | `adsagent`                                           | Docker Postgres password                   |
| `POSTGRES_PORT`           | `5432`                                               | Host port mapped to Postgres               |
| `ADS_DATABASE_URL`        | `postgresql://adsagent:adsagent@localhost:5432/adsagent` | App connection string (must match Postgres) |
| `ADS_MAX_ITERATIONS`      | `5`                                                  | Circuit breaker limit for supervisor loops |
| `ADS_LOG_LEVEL`           | `INFO`                                               | Logging level                              |
| `LANGFUSE_PUBLIC_KEY`     | —                                                    | Langfuse Cloud public key (optional)       |
| `LANGFUSE_SECRET_KEY`     | —                                                    | Langfuse Cloud secret key (optional)       |
| `LANGFUSE_HOST`           | `https://cloud.langfuse.com`                         | Langfuse API host                          |

## Development Commands

```bash
make help            # Show all available commands
make lint            # Run Ruff linter
make format          # Run Ruff formatter
make check           # Lint + format check (CI mode)
make docker-up       # Start services
make docker-down     # Stop services
make clean           # Remove caches
```

## Project Structure

```
ads-agent/
├── src/ads_agent/
│   ├── core/              # Domain layer — no external dependencies
│   │   ├── entities/      # Pure domain models
│   │   ├── ports/         # Interface contracts (ABCs)
│   │   └── use_cases/     # Business logic
│   ├── application/       # Orchestration layer
│   ├── infrastructure/    # Adapters (LLM, DB, MCP, Observability)
│   ├── agents/            # LangGraph agent graphs
│   └── api/               # FastAPI gateway
├── tests/
│   ├── unit/              # Pure unit tests (no I/O)
│   └── integration/       # Tests requiring services
└── docker/                # Container configuration
```

## Observability

Every agent execution produces a Langfuse trace when credentials are configured:

- Root trace `ads-agent-pipeline` per run, linked via `ExecutionReceipt.trace_id`
- Nested spans per graph node (`supervisor`, `research`, `analysis`, `writer`)
- LLM generations with model, input/output, and token usage
- Post-execution scores: `has_sources`, `trade_offs_count`

Configure Langfuse Cloud in `.env` and open traces at [cloud.langfuse.com](https://cloud.langfuse.com). The CLI prints the trace ID when tracing is active.

### Visual verification

1. Run: `uv run ads-agent run "Should I use Redis or Memcached for session storage?"`
2. Copy the **Trace ID** from the execution receipt output
3. In Langfuse → **Tracing**, search by trace ID
4. Confirm hierarchy: `ads-agent-pipeline` → repeated `supervisor` spans → `research`/`analysis`/`writer` each with a nested generation
5. Check **Scores** tab for `has_sources` and `trade_offs_count`

## Roadmap

- [x] Phase 0 — Project bootstrap
- [x] Phase 1 — Supervisor Agent with LangGraph
- [x] Phase 2 — MCP Tool Layer
- [x] Phase 3 — RAG Pipeline with pgvector
- [x] Phase 4 — Full Multi-Agent System (LiteLLM tiered models)
- [x] Phase 5 — Langfuse Observability
- [ ] Phase 6 — Evaluation Engine
- [ ] Phase 7 — FastAPI Gateway
- [ ] Phase 8 — Docker + CI/CD deployment

## Phase 4 — Tiered LLM Strategy

| Role | Default model | Env var |
| --- | --- | --- |
| Supervisor (ambiguous routing) | `gemini/gemini-2.5-pro` | `ADS_LLM_SUPERVISOR_MODEL` |
| Workers (research, analysis, writer) | `gemini/gemini-2.5-flash` | `ADS_LLM_WORKER_MODEL` / `ADS_RESEARCH_MODEL` |

Deterministic Python rules remain the circuit breaker. The supervisor LLM is consulted only when outputs look insufficient (short text, insufficiency markers, invalid analysis JSON).

### Estimated cost per full pipeline run

Assumptions: research ReAct ~3 turns, one ambiguous supervisor call, one analysis call, one writer call.

| Agent | Calls | Input tokens | Output tokens |
| --- | --- | --- | --- |
| Research (ReAct) | 3 | ~6,000 | ~1,500 |
| Supervisor LLM | 1 | ~800 | ~50 |
| Analysis | 1 | ~2,500 | ~800 |
| Writer | 1 | ~3,000 | ~600 |

| Strategy | Models | Est. cost / run |
| --- | --- | --- |
| **Tiered (recommended)** | Supervisor: `gemini-2.5-pro`; workers: `gemini-2.5-flash` | ~$0.008–$0.012 |
| **Single premium** | All `gemini-2.5-pro` | ~$0.025–$0.035 |
| **Savings (tiered vs premium)** | — | **~60–70%** |

Tiered breakdown (approximate): research flash ~$0.002, supervisor pro ~$0.003, analysis flash ~$0.001, writer flash ~$0.001 → **~$0.007 total**.

### Verification commands

```bash
uv sync
uv run pytest tests/unit -m unit -v
uv run pytest tests/unit/test_graph_routing.py tests/unit/agents/test_supervisor_llm.py -v
uv run pytest tests/integration/test_llm_pipeline.py -m integration -v  # requires GEMINI_API_KEY
uv run ruff check src tests
```

```bash
uv run ads-agent run "Should I use pgvector or Qdrant for my RAG system?"
```

## License

MIT
