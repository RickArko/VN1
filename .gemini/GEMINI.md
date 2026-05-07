# GEMINI.md — VN1

Guidance for the Gemini CLI when working in this repo. Mirrors the rules
in `CLAUDE.md` and `.claude/AGENTS.md` so behavior is consistent across
providers.

## Project Overview

VN1 is a forecasting-experiments repo for the VN1 Challenge — weekly
retail demand at the Client × Warehouse × Product grain, ~13-week
horizon. Notebook-first sandbox; no service or library surface. Two
starter models live at the root: `LightGbmStarter.ipynb` (mlforecast +
LightGBM) and `AutoETS.ipynb` (statsforecast baseline). Shared loaders
in `src/data.py`.

## Key Commands

```bash
make install   # uv sync + register `vn1` Jupyter kernel
make lab       # JupyterLab
make notebook  # classic Notebook
make fmt       # black + isort
make help      # full target list
```

`uv run` runs everything inside `.venv/` — no manual activation needed.
The registered kernel is **VN1 (uv)**.

## Default Working Mode

- Implementation > speculation.
- Promote stable notebook code into `src/` as typed functions.
- Call out blockers, risks, and assumptions explicitly.

## Conventions

- Python 3.10 (`.python-version`); deps via `uv` + `pyproject.toml` + `uv.lock`.
- black (line 120) + isort.
- Polars for transforms; pandas only at library boundaries.
- **Pinned for upstream-bug reasons — do not bump blindly:**
  `lightgbm<4.5`, `ipykernel>=6.29,<7`. Comments in `pyproject.toml`.

## Output Preferences

- Concise summaries first, then details.
- Include exact file paths and commands.
- Show verification status and any gaps.

## AI-enrichment

This repo is enriched with `ai-enrich`. The Gemini CLI loads context
from `.gemini/settings.json` → `contextFiles`, which by default includes
`GEMINI.md` and `.claude/wiki/AGENTS.md`. The wiki schema and conventions
apply equally to Gemini-driven work. The prompts layer at
`.claude/.prompts/` is also provider-agnostic and readable from Gemini.
