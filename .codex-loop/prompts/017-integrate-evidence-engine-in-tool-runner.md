---
title: Integrate EvidenceEngine into tool-runner
mode: code
allow_secrets: false
checks:
  - ["ruff", "check", "."]
  - ["pytest", "tests/test_evidence_engine.py", "tests/test_workflow.py", "tests/test_run_state.py"]
---

# Task

Integrate EvidenceEngine v1 into the tool-runner path while preserving legacy behavior.

# Context

EvidenceEngine v1 exists. Now use it for web_search tasks so observations and artifacts become more evidence-aware.

# Required behavior

When `execute_tool()` handles a task with `tool == "web_search"`:

1. Build an EvidenceRequest:
   - run_id
   - task_id
   - branch from `branch_for_task`
   - objective from task.question
   - search_mode = exploratory by default
   - max_sources from config or small default

2. Call EvidenceEngine.

3. Persist raw provider output as before if needed.

4. Observation summary should mention:
   - number of sources
   - strongest evidence snippets
   - source limitations
   - branch

5. Observation artifact should include:
   - tags
   - source refs
   - branch
   - legacy_object_type / id
   - a compact text summary

6. Keep old `web_search` behavior as fallback if EvidenceEngine fails.

# Important

Do not break market_data tasks.

# Failure behavior

If EvidenceEngine fails:
- mark task failed or fall back to old web_search depending on failure type
- do not record a successful tool action unless an observation was created
- do not hide the error

# Tests

- web_search task creates EvidenceBundle-backed observation
- empty EvidenceBundle still creates useful observation or explicit failure
- source refs are propagated
- market_data path still works
- EvidenceEngine failure does not create successful tool action

# Acceptance criteria

```bash
ruff check .
pytest tests/test_evidence_engine.py tests/test_workflow.py tests/test_run_state.py
```

passes.
