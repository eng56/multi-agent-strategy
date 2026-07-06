---
title: Create local Codex CLI orchestration loop
mode: code
allow_secrets: false
checks:
  - ["python", ".codex-loop/codex_loop.py", "--dry-run"]
---

# Task

Create a local Codex CLI orchestration loop for this repository.

# Context

This repo is a managed-services multi-agent investment research prototype. We want a safe local loop that consumes structured prompts one by one, gives each to Codex CLI, captures logs/diffs/checks, and stops for human review.

# Required files

Create:

```text
.codex-loop/
  codex_loop.py
  README.md
  config.example.yaml
  prompts/
    001-example.md
  prompts-done/
  runs/
  secrets/
    runtime.env.example
```

Create `.codex-loop/state.json` automatically at runtime, not as a committed static file unless needed.

# Core behavior

The script must:

1. Read `.codex-loop/prompts/*.md` in sorted order.
2. Select the first prompt not marked complete in `.codex-loop/state.json`.
3. Support `--prompt` to run a specific prompt.
4. Create a run directory under `.codex-loop/runs/<prompt-stem>-<timestamp>/`.
5. Copy the prompt into that run directory.
6. Create a git branch named `codex-loop/<prompt-stem>-<short-timestamp>` unless disabled in config.
7. Run Codex CLI using a configurable command from config.
8. Capture:
   - `codex.log`
   - `git.diff`
   - `git.status`
   - `checks.log`
   - `result.json`
9. Run configured checks.
10. Stop after one prompt by default.

# Required statuses

Use these statuses in `state.json` and `result.json`:

```text
PENDING
RUNNING_CODEX
CODEX_FAILED
TESTING
FAILED_CHECKS
READY_FOR_REVIEW
COMPLETED_MANUALLY
```

# CLI requirements

Support:

```bash
python .codex-loop/codex_loop.py
python .codex-loop/codex_loop.py --dry-run
python .codex-loop/codex_loop.py --prompt .codex-loop/prompts/001-example.md
python .codex-loop/codex_loop.py --mode code
python .codex-loop/codex_loop.py --mode staging
python .codex-loop/codex_loop.py --mode prod --allow-prod
python .codex-loop/codex_loop.py --continue-on-success
```

# Config

Create `.codex-loop/config.example.yaml` with:

```yaml
codex:
  command:
    - codex
    - exec
  args:
    - "--full-auto"
  pass_prompt_as: "stdin"
checks:
  - ["ruff", "check", "."]
  - ["pytest"]
git:
  create_branch: true
  commit_on_success: false
  push_on_success: false
security:
  env_file_by_mode:
    code: null
    staging: ".codex-loop/secrets/staging.env"
    prod: ".codex-loop/secrets/prod.env"
  redact_env_values_in_logs: true
```

Use standard library where possible. If PyYAML is unavailable, support JSON-compatible config or document the limitation.

# Security requirements

- Never print secret values.
- Redact values loaded from env files in logs.
- Do not commit `.codex-loop/secrets/*.env`.
- Add `.codex-loop/secrets/*.env` to `.gitignore`.
- `prod` mode must require `--allow-prod`.
- Do not auto-push, auto-merge, or deploy.

# Dry-run behavior

`--dry-run` must show:

- selected prompt
- branch name
- Codex command
- checks
- mode
- whether an env file would be loaded

It must not modify git, run Codex, or run checks.

# Acceptance criteria

- `python .codex-loop/codex_loop.py --dry-run` works.
- `python .codex-loop/codex_loop.py --prompt .codex-loop/prompts/001-example.md --dry-run` works.
- Running the loop creates a run directory and result file.
- Logs and diffs are saved.
- Secrets are not printed.
- `.gitignore` protects `.codex-loop/secrets/*.env`.
- README explains how to add prompts, run one prompt, inspect output, retry failures, and mark completion manually.

# Commands to run

```bash
python .codex-loop/codex_loop.py --dry-run
```

Do not push or deploy.
