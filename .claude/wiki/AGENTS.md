# AGENTS.md — Wiki Schema (canonical)

This is the operational schema for `.claude/wiki/`. The `/wiki` skill reads
this file every session. If this disagrees with the skill description, this
file wins. Edit deliberately.

Pattern: Karpathy's llm-wiki (https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f).

---

## Directory layout

```
.claude/wiki/
├── AGENTS.md          # this file — schema (canonical)
├── README.md          # human entry point
├── index.md           # content catalog (auto-maintained)
├── log.md             # append-only chronological record
├── raw/               # immutable source dumps (do NOT edit)
└── pages/
    ├── entities/      # named things: people, projects, devices, services
    ├── concepts/      # ideas, patterns, conventions
    ├── topics/        # cross-cutting analyses, comparisons, FAQs
    └── sources/       # one summary page per file in raw/
```

`raw/` is immutable. `pages/` is everything else. Curation lives in pages,
not raw.

---

## Page types

| Type       | Directory             | Purpose |
|------------|-----------------------|---------|
| `entity`   | `pages/entities/`     | A specific named thing — a project, device, person, service, repo |
| `concept`  | `pages/concepts/`     | An idea, pattern, convention, or technique |
| `topic`    | `pages/topics/`       | Cross-cutting analysis or comparison |
| `source`   | `pages/sources/`      | One-page summary of a file under `raw/`. Always cites the `raw/` path |

---

## Frontmatter spec (mandatory)

Every page under `pages/` MUST start with YAML frontmatter:

```yaml
---
type: source | entity | concept | topic
created: YYYY-MM-DD
updated: YYYY-MM-DD
tags: [tag1, tag2]
source: raw/<filename>      # required for type: source only
---
```

Optional: `aliases: [other-name]`, `status: draft | stable | deprecated`.

A page without frontmatter is invalid. Refuse to write it.

---

## Page conventions

- **Filenames** are kebab-case slugs.
- **First H1** matches the page title. Wikilinks resolve against the slug.
- **Wikilinks** use the form `[[<dir>/<slug>]]` — `[[entities/foo]]`,
  `[[concepts/bar]]`. The dir prefix is required.
- **External links** use markdown `[text](url)`. Reserve `[[wikilinks]]`
  for intra-wiki references.
- Every page leads with a one-line TL;DR (after frontmatter, before any
  `##`).

---

## index.md structure

```markdown
# Wiki Index

<optional human-edited prose at the top — preserved by `/wiki index`>

## Sources
## Entities
## Concepts
## Topics
```

---

## log.md format

Append-only, newest at the bottom. Each entry: `## [date] verb | summary`
followed by 1–3 lines of detail. Verbs: `init`, `ingest`, `query`, `lint`,
`index`, `note`, `correction`. Never edit historical entries.

---

## Hard rules

1. **Never modify `raw/`.** Sources are immutable. Corrections go in pages.
2. **Never silently overwrite a page on contradiction.** Flag with a
   `## Contested` section and ask the user.
3. **No secrets in the wiki.** Use `bw://vault/<item>` placeholders.
4. **Frontmatter on every page** under `pages/`. No exceptions.
5. **Wikilinks include the directory.** `[[entities/foo]]`, not `[[foo]]`.
6. **The log is append-only.**
