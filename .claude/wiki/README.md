# Wiki — VN1 knowledge base

Persistent, LLM-curated knowledge under `.claude/wiki/`. Sources go in
`raw/`, summaries and analysis go in `pages/`, and the chronological
record lives in `log.md`.

## Quick navigation

- [[index.md]] — catalog of what's here
- [[log.md]] — chronological record
- [[AGENTS.md]] — schema and rules

## How it works

You drop sources into `raw/`. You ask Claude `/wiki ingest <path>` and
it summarizes the source as a `pages/sources/<slug>.md` page, then
updates the entity / concept / topic graph. You ask `/wiki query <q>`
and it cites pages with `[[wikilinks]]`.

The wiki is the codebase. Obsidian (or any markdown viewer with
backlinks) is the IDE. The agent is the programmer.

## Conventions

- `raw/` is immutable. Corrections go in `pages/`, not source files.
- Every page under `pages/` has YAML frontmatter — see [[AGENTS.md]].
- Wikilinks include the directory: `[[entities/foo]]`, not `[[foo]]`.
- No secrets — use `bw://vault/<item>` placeholders if needed.
- The log is append-only.

## Commands

```
/wiki init                 # bootstrap or repair tree (idempotent)
/wiki ingest <path-or-url> # add a new source, link it into the graph
/wiki query <question>     # answer a question using existing pages
/wiki lint                 # health check (broken links, orphans, drift)
/wiki log [N]              # show last N log entries
/wiki index                # rebuild index.md from the page tree
```

Pattern: Karpathy's llm-wiki —
https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f
