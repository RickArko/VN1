# AGENTS.md — VN1

Durable rules for any AI agent working in this repo. Loaded by Claude
Code, Codex, Gemini CLI, and other agents that read AGENTS-style files.

## Default working mode

- Prefer implementation over speculation.
- Decompose substantial work into atomic tasks with clear file ownership.
- Call out blockers, risks, and assumptions explicitly.
- Run narrow relevant tests and lint where practical.
- Say clearly when verification could not be run.

## Architecture preferences

- Notebooks (`*.ipynb` at repo root) are the experimentation surface;
  `src/` holds promoted, typed, polars-first helpers (mirror `src/data.py`).
- `pyproject.toml` + `uv.lock` are the dependency source of truth — no
  `requirements.txt`. Use `make sync` / `make upgrade`, never `pip install`.
- Pinned versions exist for upstream-bug reasons (`lightgbm<4.5`,
  `ipykernel>=6.29,<7`). Read the comments in `pyproject.toml` before
  bumping.
- Polars for transforms; pandas only at library boundaries that demand it
  (LightGBM full validation, matplotlib, sklearn).

## Output preferences

- Concise summaries first, then details on request.
- Include exact file paths and commands.
- Show verification status and any gaps.

## High-value paths

- `LightGbmStarter.ipynb` — primary mlforecast + LightGBM pipeline.
- `AutoETS.ipynb` — statsforecast AutoETS baseline.
- `src/data.py` — shared loaders (`load_full_data`, `process_wide_df`).
- `pyproject.toml` — dependency pins (read the comments before editing).
- `Makefile` — developer entry points.
- `.claude/wiki/pages/` — knowledge graph (papers, competition rules,
  competitor write-ups).
- `.claude/.prompts/` — numbered focus briefs for tracked work.

## AI-enrichment

- Wiki: `.claude/wiki/` — see `.claude/wiki/AGENTS.md` for schema.
- Condensed context: `AI-CONTEXT.md` at repo root — produced by
  `/ai-condense`. The file starts with the marker
  `# ** ~ AI-Condensed Context File ~ TOKEN-CONDENSED!`. Do not edit
  by hand. Re-run `/ai-condense <path>` to refresh.
- Provider config: `.ai-enrich.toml`.
