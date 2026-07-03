# =============================================================================
# ADS Agent — Makefile
# Usage: make <target>
# =============================================================================

.PHONY: help install dev lint lint-fix format format-check check test \
        test-unit test-integration test-cov test-watch docker-up \
        docker-down docker-reset docker-logs docker-logs-pg \
        docker-logs-langfuse clean

# Default target: show help
.DEFAULT_GOAL := help

# Colors for terminal output
CYAN  := \033[36m
RESET := \033[0m

help: ## Show this help message
	@echo ""
	@echo "  ADS Agent — Available commands"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-22s$(RESET) %s\n", $$1, $$2}'
	@echo ""

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------

install: ## Install all dependencies (runtime + dev)
	uv sync --all-extras

dev: install ## Full dev setup: install deps + pre-commit hooks
	uv run pre-commit install
	@echo ""
	@echo "  ✅ Dev environment ready."
	@echo "  Next: copy .env.example to .env and fill in your values."
	@echo ""

# -----------------------------------------------------------------------------
# Code quality
# -----------------------------------------------------------------------------

lint: ## Run ruff linter
	uv run ruff check src tests

lint-fix: ## Run ruff linter with auto-fix
	uv run ruff check src tests --fix

format: ## Run ruff formatter
	uv run ruff format src tests

format-check: ## Check formatting without modifying files (CI mode)
	uv run ruff format src tests --check

check: lint format-check ## Run all checks (lint + format) — CI mode

# -----------------------------------------------------------------------------
# Testing
# -----------------------------------------------------------------------------

test: ## Run all tests
	uv run pytest

test-unit: ## Run only unit tests (no external services required)
	uv run pytest -m unit -v

test-integration: ## Run only integration tests (requires Docker services)
	uv run pytest -m integration -v

test-cov: ## Run all tests with coverage report on the src package
	uv run pytest --cov=src/ads_agent --cov-report=term-missing --cov-report=html --no-cov-on-fail

test-watch: ## Run tests in watch mode (requires pytest-watch plugin)
	uv run pytest-watch -- --tb=short -q --no-header

# -----------------------------------------------------------------------------
# Docker
# -----------------------------------------------------------------------------

docker-up: ## Start local development services (PostgreSQL)
	docker compose up -d
	@echo ""
	@echo "  ✅ Services started:"
	@echo "  PostgreSQL   → localhost:5432"
	@echo ""
	@echo "  Langfuse: configure LANGFUSE_* in .env for Cloud tracing"
	@echo ""

docker-down: ## Stop all local services
	docker compose down

docker-reset: ## Stop services and delete all volumes (fresh start)
	docker compose down -v
	@echo "  ⚠️  All data volumes deleted."

docker-logs: ## Follow logs from all services
	docker compose logs -f

docker-logs-pg: ## Follow PostgreSQL logs only
	docker compose logs -f postgres

docker-logs-langfuse: ## (Reserved) Langfuse is Cloud-hosted; use cloud.langfuse.com
	@echo "Langfuse runs on Cloud — set LANGFUSE_HOST=https://cloud.langfuse.com in .env"

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

clean: ## Remove caches, compiled files, and coverage artifacts
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".coverage" -delete 2>/dev/null || true
	@echo "  ✅ Cache and build artifacts removed."
