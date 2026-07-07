# Managed Multi-Agent Investment Research

A password-locked managed-services-only demo. GKE Autopilot runs the application containers, Vercel
hosts the private UI, and all infrastructure is consumed as managed APIs. See [ARCHITECTURE.md](ARCHITECTURE.md).
For day-to-day runbooks, smoke tests, run inspection, and recovery steps, see
[docs/OPERATING_DEMO.md](docs/OPERATING_DEMO.md).
For the current control-loop and research-quality internals, see
[docs/PRINCIPAL_POLICY.md](docs/PRINCIPAL_POLICY.md) and
[docs/EVIDENCE_ENGINE.md](docs/EVIDENCE_ENGINE.md).

## Demo access and model policy

This first version does **not** use bring-your-own provider keys. The deployment owner stores the
OpenRouter, Tavily, and market-data API keys as backend deployment secrets. Anyone with the private
Vercel URL and `DEMO_PASSWORD` can access the UI, choose any OpenRouter model for planner, utility,
research, verifier, aggregator, and optional judge roles, and run a small demo that is charged to the
deployment owner's provider accounts.

Aggregator and judge budgets are protected before other agents spend. Calls that cannot be funded do
not start. If aggregation cannot be funded, the system creates a deterministic partial report from
verified evidence without another LLM call.

## Local development

1. Provision managed Confluent Cloud, Upstash, Langfuse Cloud, GCS, and provider accounts.
2. Copy `.env.example` to `.env`; add deployment credentials only.
3. Run `docker compose up --build` for application processes only.
4. Run `cd frontend && npm install && ORCHESTRATOR_API_URL=http://localhost:8080 ORCHESTRATOR_API_TOKEN=... DEMO_PASSWORD=... npm run dev`.

There are intentionally no local Kafka, Redis, object-storage, Langfuse, model, search, or market-data containers.

## Cloud demo

1. Create a GKE Autopilot cluster, Artifact Registry repository, and GCS bucket.
2. Enable Workload Identity and the GKE Secret Manager add-on. For an existing cluster, run `gcloud container clusters update "$GKE_CLUSTER" --enable-secret-manager --location="$GCP_REGION" --project="$GCP_PROJECT_ID"`.
3. Give the runtime Google service account least-privilege GCS access. Do not grant pods permission to create Kubernetes Secrets.
4. Store deployment variables from `.env.example` as the dotenv-formatted `runtime-env` Google Secret Manager secret.
5. Configure GitHub variables `GCP_WORKLOAD_IDENTITY_PROVIDER`, `GCP_DEPLOY_SERVICE_ACCOUNT`,
   `GKE_CLUSTER`, `GCP_REGION`, `GCP_PROJECT_ID`, and `ARTIFACT_REGISTRY_REPO`.
6. Configure Vercel server-side `ORCHESTRATOR_API_URL`, `ORCHESTRATOR_API_TOKEN`, and `DEMO_PASSWORD`,
   plus deployment credentials `VERCEL_TOKEN`, `VERCEL_ORG_ID`, and `VERCEL_PROJECT_ID` in GitHub.

### Runtime service account and secret access

The Kubernetes service account `agent-runtime` is annotated to use the Google service account
`agent-runtime@$GCP_PROJECT_ID.iam.gserviceaccount.com`. Create that Google service account before
deploying, allow the Kubernetes service account to impersonate it, and grant it access to the managed
runtime secret:

```bash
gcloud iam service-accounts create agent-runtime \
  --project="$GCP_PROJECT_ID" \
  --display-name="Agent runtime"

gcloud iam service-accounts add-iam-policy-binding \
  "agent-runtime@$GCP_PROJECT_ID.iam.gserviceaccount.com" \
  --project="$GCP_PROJECT_ID" \
  --role="roles/iam.workloadIdentityUser" \
  --member="serviceAccount:$GCP_PROJECT_ID.svc.id.goog[default/agent-runtime]"

gcloud secrets add-iam-policy-binding runtime-env \
  --project="$GCP_PROJECT_ID" \
  --member="serviceAccount:agent-runtime@$GCP_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

Grant the same runtime service account least-privilege access to the configured GCS artifact bucket,
for example `roles/storage.objectAdmin` on that bucket for the first demo.

### GitHub deploy service account IAM

The GitHub Actions deploy service account referenced by `GCP_DEPLOY_SERVICE_ACCOUNT` must be allowed
to fetch GKE credentials, push images, and apply Kubernetes manifests. For the first demo, grant these
roles at the project level:

```bash
gcloud projects add-iam-policy-binding "$GCP_PROJECT_ID" \
  --member="serviceAccount:$GCP_DEPLOY_SERVICE_ACCOUNT" \
  --role="roles/container.developer"

gcloud projects add-iam-policy-binding "$GCP_PROJECT_ID" \
  --member="serviceAccount:$GCP_DEPLOY_SERVICE_ACCOUNT" \
  --role="roles/artifactregistry.writer"
```

`roles/container.developer` includes the `container.clusters.get` permission required by
`google-github-actions/get-gke-credentials`. If you still see Kubernetes `forbidden` errors after
credentials are fetched, bind the same deploy identity to an appropriate Kubernetes RBAC role in the
target cluster or temporarily use `roles/container.admin` for the demo deploy service account.


## GKE demo cost controls

The deployed demo runs one public orchestrator `LoadBalancer` service and eight always-on Kubernetes
deployments by default: the orchestrator API plus seven role-specific worker deployments. The
orchestrator deployment requests a small pod and exposes a public load balancer, while each
role-specific worker deployment also keeps one replica ready for a demo run.

For short idle windows, scale the demo to zero pods without changing secrets, topics, or Vercel settings:

```bash
scripts/gke-demo-cost.sh sleep
```

Wake the same deployment back up before a demo:

```bash
scripts/gke-demo-cost.sh wake
```

Scaling to zero reduces Autopilot pod usage, but it does **not** delete the GKE cluster or the external
LoadBalancer. For the lowest idle cost between demos, delete the Autopilot cluster and recreate/redeploy
it when needed; expect the load-balancer IP to change and update Vercel `ORCHESTRATOR_API_URL` after
the next deploy.

```bash
gcloud container clusters delete "$GKE_CLUSTER" \
  --location="$GCP_REGION" \
  --project="$GCP_PROJECT_ID"
```

## Tool limits

The first version intentionally uses simple provider-native limits: the UI defaults to a $1 shared
OpenRouter budget, 2 Tavily credits, and 1 market-data request for each run. Tavily basic searches
consume one credit and market-data calls consume one request. Credential validation checks provider
keys before the run starts, but these per-run tool limits are reserved for actual agent tool calls.
Exact USD accounting is enforced for OpenRouter using reservations and returned usage cost; tool USD
cost is not estimated because it depends on the deployment owner's provider plans.

The OpenRouter budget is a hard cap, not a spend target. Low-cost models such as `openai/gpt-4.1-mini`
can complete an end-to-end run while spending only a small fraction of a $4 cap. For deeper and more
expensive runs, choose higher-cost OpenRouter models in the UI, increase role caps, and increase
`max_output_tokens`; the runtime will still reconcile reservations to the actual usage cost returned
by OpenRouter.

Langfuse note: traces use the run UUID converted to a 32-character hex trace ID, and each generation
ingest also creates the parent trace. If traces are empty, verify `LANGFUSE_HOST`, `LANGFUSE_PUBLIC_KEY`,
and `LANGFUSE_SECRET_KEY` in the `runtime-env` secret and inspect worker logs for `langfuse ... ingest failed`.

OpenRouter note: if provider validation returns `HTTP 401` with `User not found`, rotate the
`OPENROUTER_API_KEY` value in Google Secret Manager `runtime-env`, restart the GKE deployments, and
rerun `curl -H "x-api-key: $ORCHESTRATOR_API_TOKEN" "$ORCHESTRATOR_API_URL/v1/models/openrouter"`
before trying the UI again.

Market-data note: Polygon.io rebranded to Massive.com, so `MARKET_DATA_PROVIDER` must currently stay
`massive` and `MARKET_DATA_BASE_URL` defaults to `https://api.massive.com`. Tavily and Massive are
separate providers with separate keys: `TAVILY_API_KEY` should look like a Tavily `tvly-...` key,
while `MARKET_DATA_API_KEY` must be a Massive REST API key from the Massive dashboard. Massive Flat
Files Access Key ID / Secret Access Key credentials are not supported by this prototype. If provider
validation returns Massive `HTTP 401` with `Unknown API Key`, rotate the `MARKET_DATA_API_KEY` value
in the same `runtime-env` secret, restart the GKE deployments, and retry run creation.

Confluent note: use a cluster-scoped Kafka API key/secret for CONFLUENT_API_KEY and CONFLUENT_API_SECRET.

## Epistemic control-loop direction

This project explores the systems layer behind LLM-driven multi-agent decision making: a Principal generates a temporary organization of agents, agents publish structured artifacts instead of chatting freely, a trust layer promotes or rejects knowledge, and a payoff loop decides whether to continue under budget.

This is a managed-services prototype for a budgeted, hierarchical, artifact-based multi-agent decision loop. It is **not** a full Social Environment Design implementation, a POMG simulator, an RL-trained policymaker, a trading system, or a production 1,000-agent swarm.

The current implementation now exposes the foundation for the next control-loop iteration:

- `RunState` summarizes the Principal's environment view: phase, coverage, trusted facts, open questions, budget remaining, tool remaining, payoff feedback, and candidate next actions.
- `PrincipalAction` records typed decisions such as tool calls, verification, aggregation, follow-up, and stopping rather than free-form agent chat.
- `AgentSpec` and `OrganizationPlan` model the temporary research organization that can later drive hierarchy, branch budgets, and scoped memory.
- `Artifact` provides a generic graph shape for claims, observations, counterarguments, verifications, judge feedback, and final reports with support/contradiction/dependency links.

Paper mapping:

- Social Environment Design: borrows the Principal/follower/environment/observation abstraction.
- Large Legislative Models: borrows an LLM Principal conditioned on context, history, observations, and payoff.
- Cooperative AI Policymaking Platform: borrows transparent decision-support UI and forecasting orientation.

Current limitation: payoff is approximated with verifier and judge signals rather than realized real-world outcomes.

### Active control-loop runtime slice

The previous iteration introduced the epistemic-control vocabulary. This iteration makes it partially active in runtime: the planner creates a logical organization, runtime persists executed Principal decisions, legacy workflow outputs are dual-written into the generic artifact graph, and RunState summarizes semantic branches, artifacts, budget, and candidate next actions.

The system still does not implement a full recurrent Principal policy, dynamic agent spawning, AgentSpec-driven prompt composition, a complete KnowledgeRouter, or judge-triggered follow-up loops. Those are next steps.

Next planned iteration:

- Principal policy consumes RunState and selects actions.
- AgentSpec drives worker prompt composition.
- KnowledgeRouter filters artifacts by branch, tags, status, and visibility.
- Skeptic agent creates counterargument artifacts.
- Judge feedback can trigger one follow-up wave.
