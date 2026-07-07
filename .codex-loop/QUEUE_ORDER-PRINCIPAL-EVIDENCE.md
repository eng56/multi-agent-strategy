# PrincipalPolicy + EvidenceEngine prompt queue

Recommended order:

1. `013-add-principal-policy-v0.md`
2. `014-wire-principal-policy-shadow-mode.md`
3. `015-add-principal-policy-active-executor.md`
4. `016-add-evidence-engine-v1-tavily.md`
5. `017-integrate-evidence-engine-in-tool-runner.md`
6. `018-add-source-quality-scoring.md`
7. `019-strengthen-verifier-with-evidence-bundles.md`
8. `020-add-search-provider-interface-and-brave.md`
9. `021-add-exa-firecrawl-optional-fetching.md`
10. `022-add-evidence-quality-eval-harness.md`
11. `023-principal-policy-and-evidence-engine-docs.md`

Rules:

```text
One prompt = one branch = one diff = one test report.
Do not run optional-provider prompts with prod secrets until staging smoke passes.
Do not make PrincipalPolicy active by default.
Do not remove the existing stable event pipeline until active policy is proven.
```
