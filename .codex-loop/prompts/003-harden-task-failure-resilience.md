---
title: Harden workflow when one task fails
mode: code
allow_secrets: false
checks:
  - ["ruff", "check", "."]
  - ["pytest", "tests/test_workflow.py", "tests/test_run_state.py"]
---

# Task

Make the agent workflow resilient when one research task fails.

# Context

The runtime should not collapse the whole run because one tool call, one LLM summary, or one claim generation fails. It should record the failed task, persist an auditable action/artifact when possible, and allow other tasks to continue.

# Required behavior

When a task execution fails:

1. Mark only that task as failed.
2. Persist a `PrincipalAction` or equivalent audit record explaining the failure.
3. Do not record a successful executed tool action if the side effect failed.
4. Emit an event or state update that makes the failure visible in run detail.
5. Continue processing other tasks.
6. If enough verified evidence exists, still allow aggregation.
7. If no usable evidence exists after all tasks fail, produce a partial/failure final state with a useful reason.

# Required tests

Add or update tests for:

- one failed task does not prevent another task from completing
- failed tool execution does not create a successful observation artifact
- failed task is visible in run state / run detail
- aggregation can proceed with partial evidence
- all tasks failing produces a clear terminal status

# Constraints

- Keep changes minimal and local to workflow/state models where possible.
- Preserve existing public API shape unless a small additive field is needed.
- Do not introduce new managed services.
- Do not hide exceptions silently; record concise failure reasons.

# Acceptance criteria

```bash
ruff check .
pytest tests/test_workflow.py tests/test_run_state.py
```

passes.

# Design preference

Use explicit statuses and auditable records rather than implicit logs only.
