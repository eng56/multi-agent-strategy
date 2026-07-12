---
title: Add optional PrincipalPolicy active executor
mode: code
allow_secrets: false
checks:
  - ["ruff", "check", "."]
  - ["pytest", "tests/test_principal_policy.py", "tests/test_workflow.py", "tests/test_run_state.py"]
---

# Task

Add an optional PrincipalPolicy active executor.

# Context

The PrincipalPolicy exists and can run in shadow mode. Now add an executor that can dispatch selected actions, but keep it behind `PRINCIPAL_POLICY_MODE=active`.

# Required behavior

Create a small executor, for example:

```python
execute_principal_decision(runtime, snapshot, decision)
```

It should support only a safe subset:

```text
ASSIGN_TASK
REQUEST_TOOL_CALL
REQUEST_VERIFICATION
REQUEST_SKEPTIC_REVIEW
REQUEST_AGGREGATION
REQUEST_FOLLOWUP
STOP_RUN
```

# Action execution mapping

ASSIGN_TASK:
- create deterministic tasks if no tasks exist
- persist action as EXECUTED
- emit TASK_CREATED events

REQUEST_TOOL_CALL:
- emit TASK_CREATED or direct tool event for the selected pending task
- avoid duplicate events for same task/action

REQUEST_VERIFICATION:
- emit CLAIM_CREATED for one unverified claim

REQUEST_SKEPTIC_REVIEW:
- emit SKEPTIC_REVIEW_REQUESTED or call existing skeptic handler path if already implemented

REQUEST_AGGREGATION:
- emit CLAIM_VERIFIED or FINAL aggregation event in whatever way current workflow expects
- do not aggregate if already aggregating/final exists unless follow-up requires it

REQUEST_FOLLOWUP:
- create follow-up tasks using existing logic
- persist action

STOP_RUN:
- mark run terminal with clear reason only when no useful action remains

# Idempotency

Every dispatched action must have an idempotency guard. Do not emit duplicate tool/verify/aggregate events for the same target.

# Active mode tests

Use `PRINCIPAL_POLICY_MODE=active` in tests only.

Add tests:

- active ASSIGN_TASK creates tasks and events
- active REQUEST_VERIFICATION emits correct event
- active REQUEST_AGGREGATION does not double-aggregate
- active REQUEST_FOLLOWUP creates at most one wave
- idempotency prevents duplicate dispatch

# Constraints

- Do not make active mode default.
- Do not remove existing event-chain behavior.
- Do not create dynamic Kubernetes pods.
- Keep this minimal.

# Acceptance criteria

```bash
ruff check .
pytest tests/test_principal_policy.py tests/test_workflow.py tests/test_run_state.py
```

passes.
