---
title: Wire PrincipalPolicy in shadow mode
mode: code
allow_secrets: false
checks:
  - ["ruff", "check", "."]
  - ["pytest", "tests/test_principal_policy.py", "tests/test_workflow.py", "tests/test_run_state.py"]
---

# Task

Wire the deterministic PrincipalPolicy into runtime in safe shadow mode.

# Context

Prompt 013 added a deterministic PrincipalPolicy. Now we need to run it during state transitions without letting it break the current event pipeline.

# Required behavior

Add a setting:

```text
PRINCIPAL_POLICY_MODE=off|shadow|active
```

Default must be `shadow`.

Modes:

```text
off:
  do nothing

shadow:
  build PrincipalSnapshot after important state changes
  propose/select PrincipalAction
  persist candidate or shadow decision as PrincipalAction with status=PROPOSED
  do not dispatch side effects

active:
  only for tests / later use
  selected actions may be dispatched through a small deterministic executor
```

# Important

Do not turn active mode on by default.

# State changes to evaluate in shadow mode

Call policy evaluation after:

```text
run.created / planner finished
task completed
task failed
observation created
claim created
claim verified/rejected
skeptic artifact created
final created
judge completed
dead letter created
```

If this is too invasive, start with the lowest-risk points:

```text
after plan()
after verify_claim()
after aggregate()
after judge()
```

# Persistence rule

A shadow-selected action should be auditable but not misleading.

Set:

```text
status = PROPOSED
reason includes "shadow policy proposed:"
producer = "principal-policy-shadow"
```

Do not mark shadow actions as EXECUTED.

# Idempotency

Avoid writing the same shadow candidate repeatedly.

Use a simple key derived from:

```text
run_id
action_type
target_branch
required_role
current_phase
wave_number if available
```

If an equivalent PROPOSED action already exists, skip.

# Tests

Add tests for:

- shadow mode persists PROPOSED action but does not emit side-effect event
- off mode persists nothing
- active mode disabled by default
- no duplicate shadow proposals
- terminal run produces no proposals

# Acceptance criteria

```bash
ruff check .
pytest tests/test_principal_policy.py tests/test_workflow.py tests/test_run_state.py
```

passes.

# Constraints

- Keep existing hard-coded pipeline working.
- No broad refactor.
- No new external services.
