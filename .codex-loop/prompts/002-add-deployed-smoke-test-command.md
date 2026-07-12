---
title: Add deployed run smoke-test script
mode: code
allow_secrets: false
checks:
  - ["ruff", "check", "."]
  - ["pytest"]
  - ["bash", "scripts/smoke-run-fed-cut.sh", "--help"]
---

# Task

Add a repeatable deployed smoke-test command for the multi-agent research runtime.

# Context

We need a fast way to validate that the deployed backend can progress beyond planner stage and create tasks/artifacts for the canonical Fed/gold/USD/equities question.

# Required changes

Create:

```text
scripts/smoke-run-fed-cut.sh
```

The script should:

1. Require:
   - `ORCHESTRATOR_API_URL`
   - `ORCHESTRATOR_API_TOKEN`
2. Submit the canonical Fed/gold/USD/equities question to the API.
3. Poll run detail until terminal status or timeout.
4. Print a compact jq summary:
   - run status
   - failure reason
   - active branches
   - tasks count
   - artifacts count
   - principal actions count
   - final present / absent
5. Exit non-zero if:
   - planner-stage failure occurs
   - `tasks_count == 0`
   - `market/crypto` appears without crypto being explicitly mentioned in the question
6. Store raw run detail in:
   - `.codex-loop/latest-smoke-run-detail.json` if `.codex-loop/` exists
   - otherwise `smoke-run-detail.json`

# Constraints

- Bash script should use `curl` and `jq`.
- Do not require Python.
- Do not print API token.
- Include clear error messages.
- Add executable bit if possible.

# Acceptance criteria

```bash
bash scripts/smoke-run-fed-cut.sh --help
```

works and documents required env vars.

The script should fail fast when env vars are missing.

# Commands to run

```bash
ruff check .
pytest
bash scripts/smoke-run-fed-cut.sh --help
```
