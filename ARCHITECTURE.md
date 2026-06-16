# Managed-services-only architecture

```text
Browser -> Vercel BFF -> GKE Autopilot application containers
                       |-> OpenRouter (user-selected models, user-provided key)
                       |-> Tavily + Polygon (user-provided keys)
                       |-> Confluent Cloud (lightweight coordination events)
                       |-> Upstash Redis (structured state + KMS ciphertext with TTL)
                       |-> Cloud KMS (encrypt/decrypt ephemeral user credentials)
                       |-> Google Cloud Storage (large raw artifacts)
                       |-> Langfuse Cloud (sanitized LLM/tool traces)
Google Secret Manager -> deployment-level credentials only
```

## Hard boundary

No Kafka brokers, Redis servers, Kubernetes nodes, models, search, market data, tracing backend,
secret store, or object storage are self-hosted. GKE must be Autopilot. The frontend runs on Vercel.

## User-provided credentials

OpenRouter, Tavily, and market-data keys arrive through the non-cached Vercel BFF, are validated
before run creation, encrypted with Cloud KMS, and stored only as ciphertext in Upstash with a hard
six-hour TTL. Plaintext keys never enter Kafka, GCS, Langfuse, logs, environment variables, or API
responses. Terminal runs delete ciphertext immediately; TTL is the crash-safety fallback. Kubernetes
Secrets and Google Secret Manager contain deployment credentials only, never per-run user keys.

## Budgets and finalization

The user selects any OpenRouter model per role, one shared LLM budget, editable role caps, and maximum
cost per call. Aggregator and optional judge protected amounts are removed from the budget visible to
other roles before work begins. Reservations are serialized with an Upstash lock and reconciled with
OpenRouter-reported actual cost. Tavily uses a simple credit limit; market data uses a request limit.
Validation consumes one unit from both tool limits.

If a call cannot be funded it is not started. Expired runs and runs that cannot fund synthesis produce
a deterministic partial report from verified structured evidence without another LLM call.

## Event flow

`run.created` → planner → `task.created` → tool-runner → `observation.created` → research worker →
`claim.created` → verifier → `claim.verified` → aggregator → `final.created` → optional judge.
Events contain identifiers and routing metadata only; current state lives in Upstash and raw artifacts
live in GCS.
