---
title: Add judge-triggered follow-up wave
mode: code
allow_secrets: false
checks:
  - ["ruff", "check", "."]
  - ["pytest", "tests/test_workflow.py", "tests/test_run_state.py"]
---

# Task

Add a minimal judge-triggered follow-up wave.

# Context

The judge already scores the final answer. If the answer is weak, the system should be able to request one targeted follow-up research wave instead of simply ending.

# Required behavior

When judge score is below a threshold, for example `< 0.75`, and no follow-up wave has already been run:

1. Persist judge feedback.
2. Create 1-3 targeted follow-up tasks based on judge feedback.
3. Mark them as a follow-up wave.
4. Continue the workflow.
5. Avoid infinite loops by allowing only one follow-up wave in v1.

If judge score is high enough, complete normally.

# Required model/state additions

Add minimal metadata to identify follow-up tasks, such as:
- task wave number
- task reason
- or encoded in title/reason if avoiding schema change

Prefer explicit fields only if clean and low-risk.

# Required tests

- low judge score creates follow-up tasks
- high judge score completes run
- second low score does not create infinite follow-up waves
- follow-up tasks are auditable through PrincipalAction
- final status is still deterministic

# Constraints

- Keep v1 simple.
- Do not create recursive autonomous planning.
- Do not deploy.
- Do not change external services.

# Acceptance criteria

```bash
ruff check .
pytest tests/test_workflow.py tests/test_run_state.py
```

passes.
