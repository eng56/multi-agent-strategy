# Local Codex loop

This directory contains a deliberately small local orchestration loop. It selects
one markdown prompt, creates a task branch, runs Codex CLI, runs configured checks,
captures artifacts, then commits, pushes, opens or reuses the PR, merges it, and
switches back to the base branch before moving to the next queued prompt.

## Setup

The loop uses `.codex-loop/config.example.yaml` when no local config exists. To
customize the Codex command, checks, or publish behavior, create the ignored local
config:

```sh
cp .codex-loop/config.example.yaml .codex-loop/config.yaml
```

The parser uses PyYAML when it is installed and otherwise supports the mapping,
list, scalar, and inline JSON-list forms shown in the example. No new project
dependency is required.

The worktree must be clean before a real run. This prevents existing local changes
from leaking into a new task branch.

## Add and run prompts

Add normal markdown files to `.codex-loop/prompts/`. The loop uses
`.codex-loop/QUEUE_ORDER.md` when present, so keep that file in sync with the
prompt bundle. A numbered filename still helps as a fallback:

```text
.codex-loop/prompts/001-fix-planner.md
.codex-loop/prompts/002-add-observability.md
```

Preview the first pending prompt without changing files or git state:

```sh
python .codex-loop/codex_loop.py --dry-run
```

Run one prompt:

```sh
python .codex-loop/codex_loop.py
```

Select a prompt explicitly:

```sh
python .codex-loop/codex_loop.py \
  --prompt .codex-loop/prompts/001-example.md
```

`--continue-on-success` permits selection of another prompt after a successful
merge. The loop switches back to the base branch first, so the next prompt starts
from a clean branch.

## Optional frontmatter

Prompts do not require frontmatter. A prompt may override checks and explicitly
deny secret-enabled modes:

```markdown
---
title: Fix planner fallback
checks:
  - ["ruff", "check", "."]
  - ["pytest", "tests/test_workflow.py"]
allow_secrets: false
---

Prompt body here.
```

Without `checks`, the config defaults apply. `allow_secrets: false` rejects
`staging` and `prod` runs for that prompt.

## Inspect and finish a run

Each real run writes:

```text
.codex-loop/runs/<prompt-stem>-<UTC-timestamp>/
  prompt.md
  codex.log
  checks.log
  git.diff
  git.status
  result.json
```

Inspect `result.json`, `git.diff`, `git.status`, and `checks.log`. Then inspect the
live worktree with `git diff` and run any additional checks you need. If the result
is acceptable, the loop will commit, push, and merge automatically when the git
publish toggles are enabled.

If you want to manage the PR yourself, disable the publish toggles in
`.codex-loop/config.yaml`. In that mode the loop still leaves you with the diff and
checks artifacts, but it will stop at `READY_FOR_REVIEW`.

After a manual review, mark the prompt complete only if you are archiving it
instead of merging it:

```sh
python .codex-loop/codex_loop.py \
  --mark-completed .codex-loop/prompts/001-example.md
```

This records `COMPLETED_MANUALLY` in the ignored `.codex-loop/state.json`. The
`prompts-done/` directory is available if you also want to archive reviewed prompt
files manually; the loop never moves or deletes them.

## Retry after failure

`CODEX_FAILED` and `FAILED_CHECKS` prompts remain eligible. First resolve or commit
any worktree changes left by the failed attempt, then retry explicitly:

```sh
python .codex-loop/codex_loop.py \
  --prompt .codex-loop/prompts/001-example.md
```

An interrupted `RUNNING_CODEX`, `TESTING`, or `MERGING` state is a review gate.
Inspect the last run, clean up deliberately, and use `--prompt` to retry it.

## Modes and credential safety

- `--mode code` loads no env file.
- `--mode staging` loads only the configured staging env file.
- `--mode prod --allow-prod` loads only the configured production env file.
- `--mode prod` without `--allow-prod` is rejected.

Real credentials belong only in local files such as
`.codex-loop/secrets/staging.env` and `.codex-loop/secrets/prod.env`. These files
are ignored by git. Never add credentials to prompts, config, source files, or the
committed `runtime.env.example`.

Loaded env values are not printed. Codex and check output is captured in memory,
redacted, and only then written to logs. Environment variable names may appear;
their loaded values are replaced with `[REDACTED]`.

The subprocesses inherit the caller's normal environment so tools such as `git`
and `codex` continue to work. The mode controls only the additional env file loaded
by the orchestrator. Review your shell environment separately if it contains
credentials that a code-mode subprocess should not receive.

## Recommended workflow

1. Add one focused prompt and update `QUEUE_ORDER.md`.
2. Run the loop.
3. Inspect `git.diff`, `git.status`, and the live `git diff`.
4. Inspect `checks.log`.
5. Let the loop commit, push, and merge, or disable the publish toggles and do
   those steps manually.
6. Mark the prompt completed manually only when you are intentionally archiving it
   instead of merging it.

The recorded statuses are `PENDING`, `RUNNING_CODEX`, `CODEX_FAILED`, `TESTING`,
`FAILED_CHECKS`, `READY_FOR_REVIEW`, `MERGING`, `MERGED`, and
`COMPLETED_MANUALLY`.
