# Agent Orchestration — Harness Configuration

This project uses a planner/implementer/explore/reviewer harness to delegate work to cheaper
models where appropriate. The config lives in `.opencode/`; this doc is the reproducible setup.

## Roles & models

| Role | Model | Responsibility | Spawned when |
|------|-------|----------------|--------------|
| **Planner** | deepseek-v4-pro (the main agent) | architecture, specs, safety/ML review, editing planner-owned files, commits, deploy orchestration | always (you) |
| **Implementer** (`general`) | deepseek-v4.1-flash | write pipeline/studio code in `src/**`, `tests/**`, `requirements.txt` | a concrete, well-specified code task |
| **Explore** (`explore`) | deepseek-v4.1-flash | read-only research (models, datasets, hardware, prior art) with web access | research/survey questions |
| **Reviewer** (`reviewer`) | gpt-6-luna | read-only second opinion on safety/ML/eval decisions | high-risk or consequential changes |

## How delegation works

The **build** agent may spawn subagents via task delegation. `.opencode/opencode.jsonc` whitelists
exactly three subagent profiles (`general`, `explore`, `reviewer`) and denies everything else, so a
subagent can never escape its permission boundary.

Each profile in `.opencode/agents/*.md` hard-caps permissions:

- **general** — edit only `src/**`, `tests/**`, `requirements.txt`; never `.git`, `.env`, `data/`,
  planner-owned docs (`SPEC/STATUS/AGENTS/README/docs/config`), or `.opencode/`. Shell is `ask`.
- **explore** — read-only; websearch/webfetch allowed; no edit/bash.
- **reviewer** — read-only; no web; findings only.

This is a *permission* boundary, not an OS sandbox: it prevents accidental edits, it does not stop a
determined malicious process.

## Cost-effective delegation rules

1. **Read-only research → `explore`.** Surveys of datasets, model comparisons, hardware options, and
   prior art are delegated, not done by the planner.
2. **Well-specified code → `general`.** Once the planner has written the spec/handoff, implementation
   goes to the flash model. The planner reviews the diff, not the typing.
3. **Consequential review → `reviewer`.** Safety changes (anything touching `docs/safety.md`'s
   requirements), model/eval decisions, and tax/risk-adjacent choices get a read-only second opinion.
4. **Specs, architecture, commits, and planner-owned files → planner only.**
5. **One implementer at a time**; explore/read tasks may run in parallel.

## Verifying the harness

```powershell
opencode debug agent general
opencode debug agent explore
opencode debug agent reviewer
```

Start opencode from the project root so `.opencode/` profiles load, and restart after changing them.
