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

| Layer | Technology | Version |
|---|---|---|
| Agent Framework | LangGraph | 1.2.2 |
| Tool Protocol | MCP SDK + FastMCP | 1.27.2 |
| LLM Abstraction | LiteLLM | ≥1.70 |
| API | FastAPI + Pydantic v2 | 0.136.3 |
| Vector Store | PostgreSQL + pgvector | pg16 |
| Observability | Langfuse v4 | 4.7.1 |
| Package Manager | uv | 0.11.16 |
| Linter/Formatter | Ruff | 0.15.15 |

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
cp docker/.env.example .env
# Edit .env with your API keys
```

### 3. Start local services

```bash
make docker-up
# PostgreSQL → localhost:5432
# Langfuse   → http://localhost:3000
```

### 4. Run tests

```bash
make test-unit       # Fast — no services required
make test            # All tests
make test-cov        # With coverage report
```

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

Every agent execution is fully traced in Langfuse:

- Trace per query with nested spans per agent
- Token consumption and cost per span
- Execution time per agent
- Quality scores (faithfulness, relevance)

Access the Langfuse UI at `http://localhost:3000` after `make docker-up`.

## Roadmap

- [x] Phase 0 — Project bootstrap
- [ ] Phase 1 — Supervisor Agent with LangGraph
- [ ] Phase 2 — MCP Tool Layer
- [ ] Phase 3 — RAG Pipeline with pgvector
- [ ] Phase 4 — Full Multi-Agent System
- [ ] Phase 5 — Langfuse Observability
- [ ] Phase 6 — Evaluation Engine
- [ ] Phase 7 — FastAPI Gateway
- [ ] Phase 8 — Docker + CI/CD deployment

## License

MIT
