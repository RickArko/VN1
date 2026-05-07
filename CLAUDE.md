# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repo.

## Project Overview

VN1 is a forecasting-experiments repo for the [VN1 Challenge](https://www.datasource.ai/en/home/data-science-competitions-for-startups/phase-2-vn1-forecasting-accuracy-challenge),
a weekly retail-demand competition (Client × Warehouse × Product time
series, ~13-week horizon). The repo is a notebook-first experimentation
sandbox — there is no service, CLI, or library surface. Two starter
models live at the root: a LightGBM (`mlforecast`) recursive forecaster
and an AutoETS (`statsforecast`) baseline. Shared data loading lives in
`src/data.py`.

## Key Commands

```bash
make install   # uv sync + register `vn1` Jupyter kernel (one-time setup)
make lab       # JupyterLab using the project venv
make notebook  # classic Jupyter Notebook
make fmt       # black + isort on src/
make lint      # check formatting without modifying
make help      # full target list
```

All commands run via `uv run` under the hood — you do **not** need to
activate the venv manually. The registered kernel display name is
**VN1 (uv)**.

## Architecture

- **`src/data.py`** — `load_full_data()` fetches phase 1 + phase 2 CSVs
  from `datasource.ai` (URLs are baked in) and concatenates them into a
  wide polars DataFrame. `process_wide_df()` unpivots to the long
  `(unique_id, ds, y)` schema that `mlforecast` / `statsforecast`
  expect, where `unique_id = "{Client}-{Warehouse}-{Product}"`.
- **`notebooks/LightGbmStarter.ipynb`** — full pipeline: leading-zero trimming →
  `LightGBMCV` (4 windows, h=13, lags `[13, 52]`, expanding/rolling-mean
  + rolling-std lag transforms, `LocalRobustScaler` IQR target
  transform) → `MLForecast.from_cv` refit → zero-variance naive
  override → wide submission DataFrame.
- **`notebooks/AutoETS.ipynb`** — `statsforecast` AutoETS baseline.
- **`src/style.py`** — matplotlib rcParams shared by notebooks.
- **No tests, no linter beyond black/isort, no CI.** This is research
  code; treat new abstractions skeptically.

## Conventions

- **Python 3.10** (pinned in `.python-version`); managed by `uv`.
- **Formatter:** black (line length 120) + isort. Both run on save in
  VS Code via `.vscode/settings.json`. `make fmt` for batch.
- **DataFrames:** prefer `polars` for transforms; convert to pandas
  only when a downstream library (LightGBM with full validation,
  matplotlib) requires it.
- **Notebooks are the experimentation surface.** When code stabilizes,
  promote it to `src/` as a function — don't grow long notebook helper
  cells. Mirror the existing `data.py` style (typed args, polars
  in/out, terse docstring).
- **Pinned dependencies** (intentional workarounds, do not bump
  blindly):
  - `lightgbm<4.5` — 4.5+ defines `feature_names_in_` as a read-only
    property; clashes with sklearn's `validate_data` when fitting on a
    polars DataFrame (the path mlforecast uses).
  - `ipykernel>=6.29,<7` — ipykernel 7 is incompatible with the
    bundled `jupyter_client` / `notebook` versions and causes the
    browser kernel to hang.
  - Both pins live in `pyproject.toml` with comments. Verify upstream
    fixes before relaxing.
- **`make install` is the entry point for new contributors.** Don't
  paper over environment problems with one-off `pip install` — fix
  `pyproject.toml`.

## Things to avoid

- Do not commit notebook outputs casually — kernel restarts produce
  noisy diffs. If you need to share state, save to a file under a
  gitignored directory.
- Do not edit `.venv/`, `uv.lock` directly (regenerate via `make
  upgrade` / `make lock`).
- Do not add a `requirements.txt` — the source of truth is
  `pyproject.toml` + `uv.lock`.

## AI layer

This repo is enriched with `ai-enrich` (see `.ai-enrich.toml`). Available
helpers from any Claude Code session:

- `/ai-condense <path>` — produce / refresh `AI-CONTEXT.md` at repo root
  (single token-optimized digest, conventional location)
- `/wiki ingest <path>` — index a source into `.claude/wiki/` (good for
  competition rules, papers, competitor write-ups)
- `/wiki query <q>` — answer a question using the wiki
- `.claude/.prompts/` — numbered focus briefs for tracked work

Wiki schema: `.claude/wiki/AGENTS.md`. Catalog: `.claude/wiki/index.md`.
