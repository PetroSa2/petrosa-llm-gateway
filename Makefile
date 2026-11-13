#!/usr/bin/env make
# Standardized Makefile for Petrosa services: petrosa-llm-gateway

PYTHON ?= $(shell command -v python3.11 || command -v python3.12 || command -v python3)
VENV := .venv
BIN := $(VENV)/bin
COVERAGE_THRESHOLD := 40
IMAGE_NAME := petrosa-llm-gateway

.PHONY: help setup clean format lint type-check test security smoke load-test pipeline validate-python
.DEFAULT_GOAL := help

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "%-16s %s\n", $$1, $$2}'

validate-python: ## Check that Python is 3.11 or newer
	@$(PYTHON) -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else "Python >= 3.11 required, found " + sys.version.split()[0])'

$(BIN)/pytest: requirements.txt requirements-dev.txt
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install -q --upgrade pip
	$(BIN)/python -m pip install -q -r requirements.txt -r requirements-dev.txt
	touch $(BIN)/pytest

setup: validate-python $(BIN)/pytest ## Install development dependencies and the pre-commit hooks
	$(BIN)/pre-commit install

clean: ## Remove caches and coverage output
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml bandit-report.json
	find . -path ./$(VENV) -prune -o -type d -name __pycache__ -exec rm -rf {} +

format: $(BIN)/pytest ## Format code with ruff
	$(BIN)/ruff format .
	$(BIN)/ruff check . --select I --fix

lint: $(BIN)/pytest ## Run linting checks with ruff
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

type-check: $(BIN)/pytest ## Run type checking with mypy
	$(BIN)/mypy . || echo "Type checking found issues (non-blocking, as in CI)"

test: validate-python $(BIN)/pytest ## Run all tests with coverage
	$(BIN)/pytest --cov=. --cov-report=term-missing --cov-fail-under=$(COVERAGE_THRESHOLD)

security: $(BIN)/pytest ## Run security scans (Bandit)
	$(BIN)/bandit -q -ll -r gateway

smoke: ## Build the image and check it against the mock provider
	IMAGE_NAME=$(IMAGE_NAME) scripts/smoke.sh

load-test: validate-python $(BIN)/pytest ## Replay CIO and Holmes burst against the mock provider
	$(BIN)/python scripts/load_test.py --calls 18 --max-p95-wait 30

pipeline: validate-python ## Run the complete CI pipeline locally
	$(MAKE) clean
	$(MAKE) lint
	$(MAKE) type-check
	$(MAKE) test
	$(MAKE) security
	@echo "Pipeline completed successfully"
