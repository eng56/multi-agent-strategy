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
14. `013-add-principal-policy-v0.md` — add deterministic PrincipalPolicy v0.
15. `014-wire-principal-policy-shadow-mode.md` — wire PrincipalPolicy in shadow mode.
16. `015-add-principal-policy-active-executor.md` — add active executor scaffolding without enabling active mode by default.
17. `016-add-evidence-engine-v1-tavily.md` — add Tavily-backed EvidenceEngine v1.
18. `017-integrate-evidence-engine-in-tool-runner.md` — integrate EvidenceEngine in the tool runner.
19. `018-add-source-quality-scoring.md` — add source quality scoring.
20. `019-strengthen-verifier-with-evidence-bundles.md` — strengthen verifier with evidence bundles.
21. `020-add-search-provider-interface-and-brave.md` — add optional Brave search provider interface.
22. `021-add-exa-firecrawl-optional-fetching.md` — add optional Exa/Firecrawl retrieval hooks.
23. `022-add-evidence-quality-eval-harness.md` — add evidence quality eval harness.
24. `023-principal-policy-and-evidence-engine-docs.md` — document PrincipalPolicy and EvidenceEngine.

Rule:

```text
One prompt = one branch = one diff = one test report.
```

Do not run all prompts blindly in production mode.
Do not run optional-provider prompts with prod secrets until staging smoke passes.
Do not enable PrincipalPolicy active mode by default.
