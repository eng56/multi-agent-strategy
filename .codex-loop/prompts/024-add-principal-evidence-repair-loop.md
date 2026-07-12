---
title: Add PrincipalPolicy evidence-repair loop
checks:
  - ["ruff", "check", "."]
  - ["pytest", "tests/test_principal_policy.py", "tests/test_run_state.py", "tests/test_workflow.py"]
allow_secrets: false
---

Task: Implement a PrincipalPolicy Evidence-Repair Loop.

Paper alignment:

This is paper-aligned because it implements observation/payoff-driven next action selection. The Principal should observe failed verification, disputed claims, unsupported parts, required caveats, source quality issues, budget, and remaining search capacity; then it should select targeted repair actions before stopping. Do not say EvidenceEngine is in the papers. EvidenceEngine is this repo's engineering adaptation of the observation layer.

Required behavior:

If all of the following are true:

- no claims are verified,
- at least one claim is disputed or uncertain,
- budget_remaining is above a repair threshold,
- Tavily or another search budget remains,
- a repair wave has not already been attempted,

then:

- do not write a terminal evidence-limited final yet;
- create targeted evidence-repair tasks from verifier `unsupported_parts`, `contradictions`, `required_caveats`, and `source_quality_summary`;
- keep existing verification strictness; do not relax verifier thresholds just to produce a confident final answer.

Repair task types:

- split broad claim into atomic claims
- search for dated primary sources
- search for higher-quality secondary/news sources
- run contradiction search
- retry failed branch
- reverify after repair observations

Only write an evidence-limited final after:

- the repair wave fails,
- the repair wave produces no verified claims,
- budget or tool budget is low,
- or max repair waves is reached.

Implementation guidance:

- Prefer deterministic PrincipalPolicy logic over ad hoc aggregator-side behavior where possible.
- Persist auditable `PrincipalAction` records for repair decisions.
- Preserve artifact graph links between disputed claims, verifier artifacts, repair observations, and the final report.
- The UI should continue to show strict verification outcomes clearly.
- Do not change production credentials, deployment configuration, or provider keys.

Acceptance:

- Tests cover the no-verified/disputed-claim repair decision.
- Tests cover repair-wave idempotency.
- Tests cover budget/tool-budget exhaustion falling back to evidence-limited final.
- Tests confirm broad disputed claims are split into repair tasks using verifier caveats and unsupported parts.
- Tests confirm strict verification remains unchanged.
- The prompt and implementation must not say EvidenceEngine is in the papers.
