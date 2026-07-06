---
title: Add cost and budget observability
mode: code
allow_secrets: false
checks:
  - ["ruff", "check", "."]
  - ["pytest", "tests/test_budget.py", "tests/test_run_state.py", "tests/test_workflow.py"]
---

# Task

Improve cost and budget observability in run detail and logs.

# Context

The system has budget policies by role and tool limits. For decision-oriented multi-agent research, we need to understand where budget was spent, reserved, exhausted, or protected.

# Required behavior

Expose a clear budget summary in run detail / RunState:

- total limit
- spent
- reserved
- remaining
- role spent by planner/research/verifier/aggregator/judge
- role reserved
- tool usage:
  - Tavily credits used / max
  - market data requests used / max
- protected budget if present
- stop reason when budget blocks an action

# Required additions

1. Add or improve serializer/state code for budget.
2. Ensure budget stop reasons are explicit and visible.
3. Add concise logs when an action is skipped due to budget.
4. Add tests for budget summary and budget-exhausted state.

# Constraints

- Do not change pricing model unless necessary.
- Do not add external observability services.
- Do not expose secrets.

# Acceptance criteria

```bash
ruff check .
pytest tests/test_budget.py tests/test_run_state.py tests/test_workflow.py
```

passes.
