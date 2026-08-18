SHELL := /bin/sh
PYTHON ?= python3.11
VENV ?= .venv
BIN := $(VENV)/bin

.DEFAULT_GOAL := help

.PHONY: help setup install install-visual install-ui install-web install-evaluation clean test lint format typecheck verify run-api run-ui ingest build-index build-visual-index evaluate-baseline evaluate-visual evaluate-system experiments orchestrate validate-data validate-data-strict

help:
	@echo "ADASPilot development commands"
	@echo "  make setup                Create .venv and install development dependencies"
	@echo "  make run-api              Run FastAPI locally with reload"
	@echo "  make run-ui               Run the minimal Streamlit UI"
	@echo "  make ingest               Extract pages, tables, figures, OCR, and chunks"
	@echo "  make build-index          Build the local Qdrant dense index"
	@echo "  make evaluate-baseline    Evaluate E0 dense retrieval and cited answers"
	@echo "  make experiments          Run controlled E0-E4 text experiments"
	@echo "  make build-visual-index   Index rendered pages and figures"
	@echo "  make evaluate-visual      Evaluate visual Recall@k"
	@echo "  make evaluate-system      Evaluate direct, ReAct, and plan modes"
	@echo "  make orchestrate QUERY='...' MODE=direct|react|plan"
	@echo "  make install-visual       Install optional ColPali dependencies"
	@echo "  make install-ui           Install the optional Streamlit UI"
	@echo "  make install-web          Install optional bounded web-search support"
	@echo "  make install-evaluation   Install optional RAGAS judge dependencies"
	@echo "  make validate-data        Validate corpus and current benchmark drafts"
	@echo "  make validate-data-strict Require the final verified benchmark target"
	@echo "  make test                 Run tests with coverage"
	@echo "  make lint                 Check formatting and lint rules"
	@echo "  make typecheck            Run mypy"
	@echo "  make verify               Run lint, types, tests, and data validation"

setup:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip
	$(BIN)/python -m pip install -e ".[dev]"

install:
	$(BIN)/python -m pip install -e ".[dev]"

install-visual:
	$(BIN)/python -m pip install -e ".[dev,visual]"

install-ui:
	$(BIN)/python -m pip install -e ".[dev,ui]"

install-web:
	$(BIN)/python -m pip install -e ".[dev,web]"

install-evaluation:
	$(BIN)/python -m pip install -e ".[dev,evaluation]"

clean:
	$(PYTHON) -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ['.pytest_cache', '.mypy_cache', '.ruff_cache', 'htmlcov', 'build', 'dist']]"

test:
	$(BIN)/pytest

lint:
	$(BIN)/ruff format --check src tests apps
	$(BIN)/ruff check src tests apps

format:
	$(BIN)/ruff format src tests apps
	$(BIN)/ruff check --fix src tests apps

typecheck:
	$(BIN)/mypy src

verify: lint typecheck test validate-data-strict

run-api:
	$(BIN)/adaspilot-api

run-ui:
	$(BIN)/streamlit run apps/streamlit_app.py --browser.gatherUsageStats=false

ingest:
	$(BIN)/adaspilot-ingest

build-index:
	$(BIN)/adaspilot-build-index

build-visual-index:
	$(BIN)/adaspilot-build-visual-index

evaluate-baseline:
	$(BIN)/adaspilot-evaluate --mode baseline --split development

evaluate-visual:
	$(BIN)/adaspilot-evaluate --mode visual --split development

evaluate-system:
	$(BIN)/adaspilot-evaluate-system --split development

orchestrate:
	$(BIN)/adaspilot-orchestrate "$(QUERY)" --mode "$(MODE)"

experiments:
	$(BIN)/adaspilot-run-experiments

validate-data:
	$(BIN)/adaspilot-validate-data

validate-data-strict:
	$(BIN)/adaspilot-validate-data --require-target
