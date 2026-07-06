# Local Codex loop

This directory contains a deliberately small local orchestration loop. It selects
one markdown prompt, creates a task branch, runs Codex CLI, runs configured checks,
and saves everything needed for human review. It never pushes, merges, deploys, or
commits.

## Setup

The loop uses `.codex-loop/config.example.yaml` when no local config exists. To
customize the Codex command or checks, create the ignored local config:

```sh
cp .codex-loop/config.example.yaml .codex-loop/config.yaml
```

The parser uses PyYAML when it is installed and otherwise supports the mapping,
list, scalar, and inline JSON-list forms shown in the example. No new project
dependency is required.

The worktree must be clean before a real run. This prevents existing local changes
from leaking into a new task branch.

## Add and run a prompt

Add normal markdown files to `.codex-loop/prompts/`. Names determine queue order,
so use numeric prefixes:

```text
.codex-loop/prompts/001-fix-planner.md
.codex-loop/prompts/002-add-observability.md
```

Preview the first pending prompt without changing files or git state:

```sh
python .codex-loop/codex_loop.py --dry-run
```

Run one prompt (the default always stops after one):

```sh
python .codex-loop/codex_loop.py
```

Select a prompt explicitly:

```sh
python .codex-loop/codex_loop.py \
  --prompt .codex-loop/prompts/001-example.md
```

`--continue-on-success` permits selection of another prompt, but only if the first
task leaves the worktree clean. Since v1 does not auto-commit, normal code-changing
runs stop for review even with this flag.

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
is acceptable, commit and push manually:

```sh
git add <reviewed-files>
git commit -m "Describe the reviewed change"
git push -u origin "$(git branch --show-current)"
```

Open a pull request manually after pushing. Nothing in this loop opens one.
Before starting the next independent prompt, switch back to the intended base
branch and update it; the loop always branches from the currently checked-out
commit.

After review, mark the prompt complete:

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

An interrupted `RUNNING_CODEX` or `TESTING` state is a review gate. Inspect the
last run, clean up deliberately, and use `--prompt` to retry it.

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

1. Add one focused prompt.
2. Run the loop.
3. Inspect `git.diff` and the live `git diff`.
4. Inspect `checks.log`.
5. Manually commit, push, and open a PR.
6. Mark the prompt completed manually.
7. Return to the intended base branch before starting the next prompt.

The recorded statuses are `PENDING`, `RUNNING_CODEX`, `CODEX_FAILED`, `TESTING`,
`FAILED_CHECKS`, `READY_FOR_REVIEW`, and `COMPLETED_MANUALLY`.
