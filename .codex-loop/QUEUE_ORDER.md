# Suggested queue order

Run the prompts in this order:

1. `000-create-codex-cli-loop.md` — only needed if the loop does not already exist.
2. `001-deploy-main-and-smoke-test.md` — verify deployed main includes PR #28 fixes.
3. `002-add-deployed-smoke-test-command.md` — make the deployed smoke test repeatable.
4. `003-harden-task-failure-resilience.md` — avoid full-run collapse when one task fails.
5. `004-add-run-retry-and-dead-letter-handling.md` — avoid lost events and infinite retries.
6. `005-add-knowledge-router-skeleton.md` — introduce deterministic evidence routing.
7. `006-make-agent-spec-drive-prompts.md` — make AgentSpec materially affect behavior.
8. `007-add-skeptic-counterargument-loop.md` — force counterarguments before synthesis.
9. `008-add-judge-triggered-followup-wave.md` — let weak final reports trigger one targeted retry wave.
10. `009-harden-deployment-and-health-checks.md` — make deployment state verifiable.
11. `010-operability-readme.md` — document how to operate the demo.
12. `011-add-cost-and-budget-observability.md` — improve budget/cost traceability.
13. `012-add-artifact-graph-ui-improvements.md` — improve frontend inspection of runs.

Rule:

```text
One prompt = one branch = one diff = one test report.
```

Do not run all prompts blindly in production mode.
