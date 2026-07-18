.PHONY: help install ensure-env validate lint format test download-data repro train evaluate \
        mlflow-ui promote-staging promote-production docker-build docker-up clean

PYTHON := python
# Python compatível com o projeto (>=3.10,<3.13). Fixa o venv do Poetry para
# evitar que um `python` de outra versão no shell crie um venv vazio.
PROJECT_PYTHON := 3.10.0
PYENV_PYTHON := $(shell pyenv root 2>/dev/null)/versions/$(PROJECT_PYTHON)/bin/python

help: ## Mostra esta ajuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

ensure-env:
	@test -x "$(PYENV_PYTHON)" && poetry env use "$(PYENV_PYTHON)" >/dev/null 2>&1 || true

install: ensure-env ## Instala dependências (prod + dev) com Poetry
	poetry install
	poetry run pre-commit install || true

validate: ensure-env ## Valida o ambiente
	poetry run $(PYTHON) -m scripts.validate_env

lint: ensure-env ## Linting com ruff
	poetry run ruff check src tests scripts
	poetry run ruff format --check src tests scripts

format: ensure-env ## Formata o código com ruff
	poetry run ruff format src tests scripts
	poetry run ruff check --fix src tests scripts

download-data: ensure-env ## Baixa o dataset do Kaggle
	poetry run python -m scripts.download_data

train: ensure-env ## Roda o pipeline completo (sem DVC)
	poetry run python -m scripts.run_pipeline

mlflow-ui: ensure-env ## Abre a UI local do MLflow
	poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db

repro: ## Roda o pipeline completo via DVC
	poetry run dvc repro

test: ensure-env ## Executa os testes
	poetry run pytest

clean: ## Remove caches e artefatos temporários
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
