# VN1 forecasting experiments — uv-managed.
# Windows users: run under WSL.

SHELL := /bin/bash
KERNEL_NAME := vn1
KERNEL_DISPLAY := VN1 (uv)

.DEFAULT_GOAL := help

.PHONY: help install sync lock upgrade kernel lab notebook \
        fmt lint typecheck \
        test test-smoke test-unit test-integration test-fast cov check \
        build verify-build clean nuke

help:  ## Show available targets.
	@awk 'BEGIN{FS=":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ---- Setup ---------------------------------------------------------

install: sync kernel  ## Create venv, install deps + dev + notebook groups, register Jupyter kernel.

sync:  ## Resolve and install dependencies (incl. dev + notebook groups) into .venv.
	uv sync --all-groups

lock:  ## Refresh uv.lock without installing.
	uv lock

upgrade:  ## Upgrade all dependencies to latest compatible versions.
	uv lock --upgrade
	uv sync --all-groups

kernel:  ## Register the project venv as a Jupyter kernel.
	uv run python -m ipykernel install --user --name $(KERNEL_NAME) --display-name "$(KERNEL_DISPLAY)"

lab:  ## Launch JupyterLab using the project venv.
	uv run jupyter lab

notebook:  ## Launch classic Jupyter Notebook using the project venv.
	uv run jupyter notebook

# ---- Quality -------------------------------------------------------

fmt:  ## Format with ruff (replaces black + isort).
	uv run ruff format .
	uv run ruff check --fix .

lint:  ## Lint without modifying files.
	uv run ruff check .
	uv run ruff format --check .

typecheck:  ## mypy on src/vn1.
	uv run mypy

# ---- Tests (smoke / unit / integration / slow) ---------------------

test:  ## Run the full pytest suite (smoke + unit + integration; skips `slow`).
	uv run pytest -m "not slow"

test-smoke:  ## Smoke tier — every public surface imports.
	uv run pytest -m smoke

test-unit:  ## Unit tier — pure-function tests on metrics, data, ensemble, validation.
	uv run pytest -m unit

test-integration:  ## Integration tier — model fit/predict + CV pipeline on toy data.
	uv run pytest -m integration

test-fast:  ## Smoke + unit only (skip integration and `slow`).
	uv run pytest -m "smoke or unit" --no-cov -q

cov:  ## Run the suite with coverage (terminal + htmlcov/).
	uv run pytest --cov --cov-report=term-missing --cov-report=html

check: lint typecheck test  ## Lint + types + tests (CI entry point).

# ---- Package build -------------------------------------------------

build:  ## Build sdist + wheel into dist/.
	uv build

verify-build: build  ## Build, install into a clean venv, smoke-test imports.
	@rm -rf /tmp/vn1-verify
	@uv venv /tmp/vn1-verify
	@uv pip install --python /tmp/vn1-verify/bin/python --quiet dist/vn1-*.whl
	@/tmp/vn1-verify/bin/python -c "import vn1; from vn1.metrics import comp_loss; from vn1.ensemble import optimize_weights; print(f'vn1 {vn1.__version__} — wheel OK')"
	@rm -rf /tmp/vn1-verify

# ---- Cleanup -------------------------------------------------------

clean:  ## Remove caches and build artifacts.
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .ipynb_checkpoints .pytest_cache .mypy_cache .ruff_cache build dist htmlcov .coverage *.egg-info

nuke: clean  ## Also remove the virtual environment and lockfile state.
	rm -rf .venv
