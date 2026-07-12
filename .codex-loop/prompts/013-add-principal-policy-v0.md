---
title: Add deterministic PrincipalPolicy v0
mode: code
allow_secrets: false
checks:
  - ["ruff", "check", "."]
  - ["pytest", "tests/test_principal_policy.py", "tests/test_run_state.py", "tests/test_workflow.py"]
---

# Task

Add a deterministic `PrincipalPolicy` v0.

# Context

The project has `PrincipalAction`, `RunState`, `AgentSpec`, `Artifact`, `BudgetSummary`, dead letters, a KnowledgeRouter, and an existing event-driven workflow. But `PrincipalAction` is still mostly audit/logging. We need to start turning it into a real policy output.

Do not rewrite the whole runtime. This PR should add the policy module and tests, then integrate it only lightly if safe.

# Required files

Create:

```text
src/agents/principal_policy.py
tests/test_principal_policy.py
```

# Required concepts

Add small internal/dataclass-style objects or Pydantic models if appropriate:

```text
PrincipalSnapshot
PrincipalActionCandidate
PrincipalDecision
PolicyDecisionReason
```

Keep this minimal. It is acceptable if these are plain dataclasses rather than API models.

`PrincipalSnapshot` should include:

```text
run
run_state
organization_plan
agent_specs
tasks
observations
claims
verifications
artifacts
principal_actions
dead_letters
final
```

# Deterministic candidate rules

Generate candidates from current state:

1. If no organization exists:
   - propose SPAWN_AGENT / create organization.

2. If no tasks exist:
   - propose ASSIGN_TASK.

3. If tasks are pending:
   - propose REQUEST_TOOL_CALL for the highest-priority pending task.

4. If observations exist without corresponding claims:
   - propose ASSIGN_TASK or REQUEST_TOOL_CALL-like internal research action for claim extraction.
   - If no better action type exists, use ASSIGN_TASK with clear reason.

5. If claims exist without verification:
   - propose REQUEST_VERIFICATION.

6. If there are >= 2 verified knowledge items and no counterargument artifact:
   - propose REQUEST_SKEPTIC_REVIEW.

7. If verified knowledge exists and skeptic/counterargument exists:
   - propose REQUEST_AGGREGATION.

8. If final exists and judge is enabled but no judge score:
   - propose STOP_RUN? No. Prefer a candidate that says judge should run.
   - If no explicit action type exists for judge, use STOP_RUN only after judge score exists.
   - Do not invent new action types unless truly needed.

9. If final exists and judge score is low and follow-up wave remains:
   - propose REQUEST_FOLLOWUP.

10. If budget is too low for useful action:
   - propose REQUEST_AGGREGATION for partial synthesis, or STOP_RUN if no evidence exists.

11. If dead letters exist:
   - propose STOP_RUN or REQUEST_FOLLOWUP only if there is useful recovery context.

# Scoring

Each candidate should have:

```text
priority
expected_information_gain
estimated_cost
target_branch
required_role
reason
```

Use deterministic scoring. Prefer high information gain per estimated cost.

# Validation rules

Add a validator that rejects candidates when:

```text
run is completed/failed/partial_budget_exhausted
budget_remaining <= 0 and action is not STOP_RUN or partial aggregation
candidate duplicates a recently executed equivalent PrincipalAction
required branch already has a pending task/action of same type
follow-up wave already used
```

# Integration v1

Do not take over the full workflow yet. Add a helper:

```python
build_principal_snapshot(...)
propose_principal_actions(snapshot) -> list[PrincipalAction]
select_principal_action(snapshot) -> PrincipalAction | None
```

Then use it in `build_run_state` or run-detail only if easy, so `next_action_candidates` can be produced by the policy module.

Do not break existing `RunState.next_action_candidates`.

# Tests

Add tests for:

- no tasks -> ASSIGN_TASK
- pending tasks -> REQUEST_TOOL_CALL
- unverified claims -> REQUEST_VERIFICATION
- verified claims without skeptic -> REQUEST_SKEPTIC_REVIEW
- verified claims with skeptic -> REQUEST_AGGREGATION
- low judge score and no follow-up -> REQUEST_FOLLOWUP
- terminal run -> no actions
- duplicate action is filtered
- budget exhausted -> no expensive action

# Acceptance criteria

```bash
ruff check .
pytest tests/test_principal_policy.py tests/test_run_state.py tests/test_workflow.py
```

passes.

# Constraints

- No LLM Principal yet.
- No new external services.
- Do not remove existing workflow handlers.
- Do not rename existing public API fields.
