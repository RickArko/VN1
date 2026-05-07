# `.claude/.prompts/` — Operational Schema

This file is the schema for the prompts layer. Every LLM session that
touches `.claude/.prompts/` reads this file first. Keep it short,
opinionated, and current. Co-evolve it with the user as conventions
solidify.

Skill entry point: `.claude/skills/ai-enrich/SKILL.md`
(`/ai-enrich prompts ...`).

## What a prompt is

A **prompt** is a self-contained focus brief for one piece of work. It
exists so an LLM can re-enter the task cold and get precise framing
without re-deriving context from scratch.

A good prompt:

- States the **goal** in 1-3 sentences and says *why now*.
- **Links** to existing artifacts (beads, sessions, wiki pages, files,
  ideas, plans). It does **not** inline them — links stay fresh; copies
  rot.
- Names the **focal files** with `path:line` ranges when known.
- Lists **constraints** that aren't obvious from reading the code.
- Has a **plan** — short, ordered, revisable.
- Has **acceptance** — checkboxes that an LLM can verify.
- Names what is **out of scope**.

A prompt is *not* a plan document, *not* a design doc, *not* a status
report. If it grows past ~150 lines, it is doing too much — split it.

## Lifecycle

```
draft  →  active  →  shipped  → archived
                  ↘  abandoned → archived
```

When a prompt ships, append `## Outcome` (PR/commit refs) and move to
`archive/`.

## Directory layout

```
.claude/.prompts/
├── AGENTS.md                  # this file
├── README.md                  # human entry point
├── INDEX.md                   # live-prompt catalog
├── <NNN>-<slug>.md            # one prompt per file
└── archive/
    └── <NNN>-<slug>.md
```

`NNN` is a zero-padded sequence assigned by `ai-enrich prompts new`.
Slugs are kebab-case, ≤ 40 chars, regex `^[a-z][a-z0-9-]{1,39}$`.

## Frontmatter spec

```yaml
---
id: 001
slug: kebab-case-slug
title: Human-readable title
status: draft           # draft | active | shipped | abandoned | archived
created: __DATE__
updated: __DATE__
intent: |
  One paragraph. Read this first.

beads:    []            # bead IDs
sessions: []            # sys://claude/session/<uuid> refs
files:    []            # focal source files (path or path:line ranges)
wiki:     []            # wiki page slugs
ideas:    []            # documentation/.ideas slugs
plans:    []            # documentation/.planned slugs
tags:     []

pr:      null
commit:  null
---
```

## Body sections (in this order)

1. **Goal** — what + why now.
2. **Context** — pointers; never inline.
3. **Constraints** — must / must-not.
4. **Plan** — short, ordered.
5. **Acceptance** — verifiable checkboxes.
6. **Out of scope** — bullets.
7. **Outcome** (filled on ship) — PR + commit refs.

## When to create a prompt

- A focused edit will span more than one session.
- Another agent / teammate may pick it up cold.
- The work has tricky constraints `git log` won't preserve.

## Hygiene rules

- One prompt = one goal. If you write "and also …", split.
- Frontmatter stays accurate. If files move, update or archive.
- Never paste secrets. Use `bw://vault/<item>` references.
- `INDEX.md` is rebuildable cache; not source of truth.
