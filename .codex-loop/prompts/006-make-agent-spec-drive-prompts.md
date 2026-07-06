---
title: Make AgentSpec drive prompts
mode: code
allow_secrets: false
checks:
  - ["ruff", "check", "."]
  - ["pytest", "tests/test_workflow.py", "tests/test_run_state.py"]
---

# Task

Make `AgentSpec` materially affect LLM prompts.

# Context

The repo now creates AgentSpecs and semantic branches, but runtime behavior should increasingly use those specs. The goal is to make role_template, branch, domain, objective, allowed_tools, retrieval_tags, visibility_scope, and local_budget_usd influence the prompts and decisions.

# Required changes

1. Add a helper that builds an agent-specific system or instruction block from `AgentSpec`.
2. Use it in at least:
   - tool-summary prompt, or
   - claim creation prompt, or
   - aggregation prompt
3. Include:
   - branch
   - domain
   - objective
   - allowed tools
   - visibility / evidence rules
   - budget awareness if relevant
4. Keep prompt output schemas strict JSON.

# Example behavior

A `market/gold` agent should be explicitly told it is responsible for gold evidence and should not overreach into equities except to compare when relevant.

An `fx` agent should focus on USD/DXY/rate differentials.

An aggregator should synthesize across verified claims only and expose trade-offs.

# Required tests

- branch-specific AgentSpec text appears in prompt sent to fake LLM
- gold task uses gold branch context
- aggregator prompt includes synthesis role and verified evidence requirement
- output parsing still works

# Constraints

- Do not make a huge prompt framework.
- Keep helper small and testable.
- Do not break existing tests.

# Acceptance criteria

```bash
ruff check .
pytest tests/test_workflow.py tests/test_run_state.py
```

passes.
