# Managed-services-only architecture

```text
Browser -> Vercel password gate + BFF -> GKE Autopilot application containers
                                     |-> OpenRouter (user-selected models, deployment-owned key)
                                     |-> Tavily + Polygon (deployment-owned keys)
                                     |-> Confluent Cloud (lightweight coordination events)
                                     |-> Upstash Redis (structured state and budgets)
                                     |-> Google Cloud Storage (large raw artifacts)
                                     |-> Langfuse Cloud (sanitized LLM/tool traces)
Google Secret Manager -> deployment-level credentials only
```

## Hard boundary

No Kafka brokers, Redis servers, Kubernetes nodes, models, search, market data, tracing backend,
secret store, or object storage are self-hosted. GKE must be Autopilot. The frontend runs on Vercel.

## Private demo access

This first demo is not public. Vercel middleware redirects every route to `/login` unless the visitor
has entered `DEMO_PASSWORD`. The Vercel URL should remain unlisted, and the backend still requires
`ORCHESTRATOR_API_TOKEN` on every proxied request. Users never see deployment provider keys.

## Provider credentials

OpenRouter, Tavily, and market-data credentials are deployment-owned secrets stored in Google Secret
Manager for GKE. They are not entered by users, not stored in browser storage, not sent through Kafka,
not written to GCS, and not included in Langfuse traces. Google Secret Manager is for deployment
credentials only.

## Budgets and finalization

A logged-in demo user selects any OpenRouter model per role, one shared LLM budget, editable role caps,
and maximum cost per call. Aggregator and optional judge protected amounts are removed from the budget
visible to other roles before work begins. Reservations are serialized with an Upstash lock and
reconciled with OpenRouter-reported actual cost. Tavily uses a simple credit limit; market data uses a
request limit. Validation consumes one unit from both tool limits.

If a call cannot be funded it is not started. Runs that cannot fund synthesis produce a deterministic
partial report from verified structured evidence without another LLM call.

## Event flow

`run.created` → planner → `task.created` → tool-runner → `observation.created` → research worker →
`claim.created` → verifier → `claim.verified` → aggregator → `final.created` → optional judge.
Events contain identifiers and routing metadata only; current state lives in Upstash and raw artifacts
live in GCS.

## Control-loop architecture target

The runtime is evolving from a linear agent pipeline into a budgeted epistemic control loop:

```text
Principal policy -> action space -> blackboard environment -> typed agents -> observations
                 -> belief/trust update -> payoff/judge -> next action or stop
```

The first architectural seam is explicit state. `RunState` gives the Principal a compact view of the environment instead of forcing it to inspect every raw artifact. `PrincipalAction` makes the action space typed and auditable. `AgentSpec` and `OrganizationPlan` describe the generated research organization. `Artifact` is the generic graph shape used to move from flat claims toward artifact publication, lineage, contradiction tracking, and trust promotion.

The API returns these structures in run detail responses so the UI can show the system, not just the answer: current state, candidate principal decisions, organization, artifacts, evidence, and payoff.

## Runtime activation slice

The previous iteration introduced the epistemic-control vocabulary. This iteration makes it partially active in runtime: the planner creates a logical organization, runtime persists executed Principal decisions, legacy workflow outputs are dual-written into the generic artifact graph, and RunState summarizes semantic branches, artifacts, budget, and candidate next actions.

The system still does not implement a full recurrent Principal policy, dynamic agent spawning, AgentSpec-driven prompt composition, a complete KnowledgeRouter, or judge-triggered follow-up loops. Those are next steps.

Next planned iteration:

- Principal policy consumes RunState and selects actions.
- AgentSpec drives worker prompt composition.
- KnowledgeRouter filters artifacts by branch, tags, status, and visibility.
- Skeptic agent creates counterargument artifacts.
- Judge feedback can trigger one follow-up wave.
