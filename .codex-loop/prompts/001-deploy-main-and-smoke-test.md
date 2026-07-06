---
title: Deploy main and rerun planner smoke test
mode: staging
allow_secrets: true
checks:
  - ["ruff", "check", "."]
  - ["pytest", "tests/test_workflow.py", "tests/test_run_state.py"]
---

# Task

Deploy the correct `main` version and rerun the Fed/gold/USD/equities smoke test.

# Context

PR #28 is the real code change that merged into `main`.
PR #29 only synced `main` back into the Codex branch and had no changed files.

Deploy target:

```text
main @ 3210e82ead5c4c716c5641aad08ddf498effd3c9
```

This version should include:
- planner `LLMOutputError` retry handling
- `compact_planner_prompt`
- deterministic planner fallback
- token-aware semantic branch matching
- tests preventing irrelevant `market/crypto` from words like “whether” or “method”

# Required actions

1. Verify local checkout is on `main` and at or after `3210e82ead5c4c716c5641aad08ddf498effd3c9`.
2. Verify the code contains:
   - `compact_planner_prompt`
   - `deterministic_task_items`
   - `text_matches_any`
   - planner retry on `LLMOutputError`
3. Deploy the backend using the repo’s existing deployment path.
4. Run the exact smoke-test question through the deployed API:

```text
Evaluate the next 3-month impact of a surprise Fed rate cut on gold, the US dollar, and US equities.

I want a decision-oriented investment research report, not generic commentary.

Please separate:
1. the main thesis,
2. verified evidence,
3. risks and counterarguments,
4. what would change the conclusion,
5. confidence level and time horizon.

Assume the objective is to understand whether gold is a better tactical opportunity than USD or equities after the rate cut.
```

5. Fetch run detail and save it to a local file under `.codex-loop/runs/<run>/deployed-smoke-run-detail.json`.

# Acceptance criteria

The deployed smoke-test result must show:

```text
run.status must not fail at planner stage
tasks_count > 0
principal_actions_count > 1
market/crypto absent unless the question explicitly mentions crypto/ETH/BTC
```

It is acceptable if later research/tool execution fails for a separate reason, but planner-stage failure with zero tasks is not acceptable.

# Required output

Write a concise report to the Codex log including:

- deployed commit SHA
- image tag or deployment identifier
- health-check output
- smoke-test run id
- jq summary of run detail
- whether the planner issue is fixed in deployment

# Suggested jq summary

```bash
curl -sS "$ORCHESTRATOR_API_URL/v1/runs/<RUN_ID>/detail" \
  -H "x-api-key: $ORCHESTRATOR_API_TOKEN" \
  | jq '{
    status: .run.status,
    failure_reason: .run.failure_reason,
    active_branches: .run_state.active_branches,
    tasks_count: (.tasks | length),
    artifacts_count: (.artifacts | length),
    principal_actions_count: (.principal_actions | length),
    final: .final
  }'
```

# Safety

Use staging credentials unless explicitly running in prod mode.
Do not print secrets.
Do not rotate or modify credentials.
