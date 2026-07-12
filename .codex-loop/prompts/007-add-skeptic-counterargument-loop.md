---
title: Add skeptic and counterargument loop
mode: code
allow_secrets: false
checks:
  - ["ruff", "check", "."]
  - ["pytest", "tests/test_workflow.py", "tests/test_run_state.py"]
---

# Task

Add a minimal skeptic/counterargument loop.

# Context

Decision-oriented investment research needs explicit counterarguments and risks. The system should not only collect supportive evidence. It should ask a skeptic agent to challenge the emerging thesis before final aggregation.

# Required behavior

Trigger a skeptic/counterargument step when there are enough verified claims to form a preliminary thesis.

Suggested condition:
- at least 2 verified knowledge artifacts or verified claims
- no skeptic artifact already exists for the run

# Skeptic output

Create an artifact with type or metadata representing skeptic feedback / counterargument.

It should include:

- strongest counterargument
- evidence that would contradict the main thesis
- risks / regime changes
- what would change conclusion
- confidence/calibration comments

# Integration

Aggregation should include skeptic/counterargument artifacts when creating the final answer.

# Required tests

- skeptic action is proposed when enough verified evidence exists
- skeptic artifact is created
- aggregation consumes skeptic artifact
- skeptic is not triggered repeatedly forever
- if there is insufficient verified evidence, skeptic is not triggered

# Constraints

- Keep this deterministic and simple.
- No new external services.
- Reuse existing LLM interface.
- Do not block aggregation forever if skeptic fails; record failure and allow partial aggregation.

# Acceptance criteria

```bash
ruff check .
pytest tests/test_workflow.py tests/test_run_state.py
```

passes.
