# Wiki Log

Append-only chronological record. Newest entries at the bottom. Never edit
historical entries — corrections go in a new entry referencing the original
date.

---

## [2026-05-07] init | bootstrap empty wiki via ai-enrich

Created tree at `.claude/wiki/`:

- `AGENTS.md` — canonical schema
- `README.md` — human entry point
- `index.md` — empty catalog
- `log.md` — this file
- `raw/`, `pages/{entities,concepts,topics,sources}/` — empty page directories

No sources ingested yet. Run `/wiki ingest <path>` to add one.
