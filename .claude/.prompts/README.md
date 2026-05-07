# `.claude/.prompts/` — focus briefs for one edit at a time

Each file in this directory is a **prompt**: a self-contained brief that
gives an LLM precise framing, constraints, and links so it can pick up a
focused piece of work cold.

Schema: [AGENTS.md](AGENTS.md). Index: [INDEX.md](INDEX.md). Tooling:
`/ai-enrich prompts {init,new,list,status}`.

## Quick start

```bash
ai-enrich prompts new <slug>            # create new prompt
ai-enrich prompts list                  # show active prompts
ai-enrich prompts status                # validate schema

/ai-enrich prompts new fix-auth --bead foo-123 \
  --files src/auth.py,tests/test_auth.py
```

## When to create a prompt

- A focused edit will span more than one session.
- A teammate / another agent may pick it up cold.
- The work has tricky constraints that `git log` won't preserve.

## When not

- One-shot edits you'll finish in this turn.
- Long-lived knowledge — use `.claude/wiki/`.
- Undecided ideas — use the idea funnel.

## Lifecycle

`draft → active → shipped → archived` (or `→ abandoned → archived`).

When work ships, append `## Outcome` (PR + commit refs) and move to
`archive/`.

## Conventions

- File names: `NNN-slug.md` (zero-padded sequence).
- Slugs: kebab-case, ≤ 40 chars.
- Body links **out** to beads / sessions / wiki / files. Never inlines.
- Secrets only as `bw://vault/<item>` references.

See [AGENTS.md](AGENTS.md) for the full frontmatter schema.
