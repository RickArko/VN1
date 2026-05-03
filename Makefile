# VN1 forecasting experiments — uv-managed.
# Windows users: run under WSL.

SHELL := /bin/bash
KERNEL_NAME := vn1
KERNEL_DISPLAY := VN1 (uv)

.DEFAULT_GOAL := help

.PHONY: help install sync lock upgrade kernel lab notebook fmt lint clean nuke

help:  ## Show available targets.
	@awk 'BEGIN{FS":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "  %-12s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: sync kernel  ## Create venv, install deps, register Jupyter kernel.

sync:  ## Resolve and install dependencies into .venv.
	uv sync

lock:  ## Refresh uv.lock without installing.
	uv lock

upgrade:  ## Upgrade all dependencies to latest compatible versions.
	uv lock --upgrade
	uv sync

kernel:  ## Register the project venv as a Jupyter kernel.
	uv run python -m ipykernel install --user --name $(KERNEL_NAME) --display-name "$(KERNEL_DISPLAY)"

lab:  ## Launch JupyterLab using the project venv.
	uv run jupyter lab

notebook:  ## Launch classic Jupyter Notebook using the project venv.
	uv run jupyter notebook

fmt:  ## Format code with black + isort.
	uv run isort src
	uv run black src

lint:  ## Check formatting without modifying files.
	uv run isort --check-only src
	uv run black --check src

clean:  ## Remove caches and build artifacts.
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .ipynb_checkpoints .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info

nuke: clean  ## Also remove the virtual environment and lockfile state.
	rm -rf .venv
