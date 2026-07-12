---
title: Add run-level retry and dead-letter handling
mode: code
allow_secrets: false
checks:
  - ["ruff", "check", "."]
  - ["pytest", "tests/test_workflow.py", "tests/test_run_state.py"]
---

# Task

Add basic retry and dead-letter handling for runtime events.

# Context

The worker consumes events and executes handlers. We need to prevent transient provider/tool failures from permanently losing events, while avoiding infinite retry loops.

# Required behavior

For handler failures:

1. Retry transient failures up to a small max count.
2. Store retry count in event payload or an equivalent metadata field.
3. Use exponential or simple increasing backoff if the event bus allows it; otherwise requeue with metadata and document the limitation.
4. After max retries, write a dead-letter record to the blackboard or artifact store.
5. Mark the affected task/run state accordingly.
6. Expose dead-letter count or records in run detail / RunState if feasible.

# Classification

Treat as transient:
- provider HTTP 429/5xx
- network timeout
- temporary tool failure

Treat as permanent:
- validation error
- impossible missing task id
- malformed internal event payload

# Required tests

- transient failure is retried
- permanent failure is not retried infinitely
- after max retries, a dead-letter record exists
- run state exposes dead-letter/failure signal
- no successful action is recorded for a dead-lettered failed side effect

# Constraints

- Keep implementation simple.
- Do not add a new external queue.
- Do not require self-hosted infrastructure.
- Preserve current managed-services architecture.

# Acceptance criteria

```bash
ruff check .
pytest tests/test_workflow.py tests/test_run_state.py
```

passes.
