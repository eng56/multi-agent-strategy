# Managed Multi-Agent Investment Research

A password-locked managed-services-only demo. GKE Autopilot runs the application containers, Vercel
hosts the private UI, and all infrastructure is consumed as managed APIs. See [ARCHITECTURE.md](ARCHITECTURE.md).

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
2. Enable Workload Identity and the GKE Secret Manager add-on.
3. Give the runtime Google service account least-privilege GCS access. Do not grant pods permission to create Kubernetes Secrets.
4. Store deployment variables from `.env.example` as the dotenv-formatted `runtime-env` Google Secret Manager secret.
5. Configure GitHub variables `GCP_WORKLOAD_IDENTITY_PROVIDER`, `GCP_DEPLOY_SERVICE_ACCOUNT`,
   `GKE_CLUSTER`, `GCP_REGION`, `GCP_PROJECT_ID`, and `ARTIFACT_REGISTRY_REPO`.
6. Configure Vercel server-side `ORCHESTRATOR_API_URL`, `ORCHESTRATOR_API_TOKEN`, and `DEMO_PASSWORD`,
   plus deployment credentials `VERCEL_TOKEN`, `VERCEL_ORG_ID`, and `VERCEL_PROJECT_ID` in GitHub.

## Tool limits

The first version intentionally uses simple provider-native limits: Tavily basic searches consume one
credit and market-data calls consume one request. Credential validation consumes one of each. Exact
USD accounting is enforced for OpenRouter using reservations and returned usage cost; tool USD cost
is not estimated because it depends on the deployment owner's provider plans.
