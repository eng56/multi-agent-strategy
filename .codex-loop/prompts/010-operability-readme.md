---
title: Write operator README for demo
mode: code
allow_secrets: false
checks:
  - ["ruff", "check", "."]
  - ["pytest"]
---

# Task

Write or update an operator-focused README for running the managed-services demo.

# Context

The repo has multiple moving pieces: FastAPI backend, worker, frontend, managed services, deployments, smoke tests, budgets, run detail, and failure modes. We need a practical operating guide, not a marketing overview.

# Required documentation

Create or update an operator doc, for example:

```text
docs/OPERATING_DEMO.md
```

It should explain:

1. Architecture in one page.
2. Required managed services.
3. Required environment variables.
4. Local development mode.
5. Staging deployment.
6. Production/demo deployment.
7. How to run the Fed/gold/USD/equities smoke test.
8. How to inspect a run:
   - run status
   - run state
   - tasks
   - observations
   - claims
   - verifications
   - artifacts
   - principal actions
   - final report
9. Common failure modes:
   - planner JSON failure
   - tool provider failure
   - LLM provider failure
   - event stuck / worker not consuming
   - budget exhausted
   - no final answer
10. Recovery playbook.
11. Cost safety and budget controls.
12. What not to do with credentials.

# Required additions

Add links from root README if appropriate.

# Constraints

- Do not invent credentials.
- Do not expose secrets.
- Keep examples copy-pasteable.
- Use concise commands.

# Acceptance criteria

- Document exists.
- It includes smoke-test and run-detail commands.
- It includes failure recovery steps.
- Root README links to it if docs folder exists.

# Commands to run

```bash
ruff check .
pytest
```
