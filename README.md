# Managed Multi-Agent Investment Research

A managed-services-only prototype. GKE Autopilot runs application containers, Vercel hosts the UI,
and all infrastructure is consumed as a managed API. See [ARCHITECTURE.md](ARCHITECTURE.md).

## User-controlled run policy

For every run, the user pastes OpenRouter, Tavily, and market-data keys; chooses any OpenRouter model
for planner, utility, research, verifier, aggregator, and optional judge; sets a shared LLM budget and
editable role caps; and sets simple Tavily-credit and market-data-request limits. Policies are not
reusable. User credentials are validated first, KMS-encrypted, retained for at most six hours, and
deleted immediately at terminal states.

Aggregator and judge budgets are protected before other agents spend. Calls that cannot be funded do
not start. If the six-hour limit expires or aggregation cannot be funded, the system creates a
deterministic partial report from verified evidence without another LLM call.

## Local development

1. Provision managed Confluent Cloud, Upstash, Langfuse Cloud, GCS, and Cloud KMS resources.
2. Grant application-default credentials permission to use the configured GCS bucket and KMS key.
3. Copy `.env.example` to `.env`; add deployment credentials only. Never add user provider keys.
4. Run `docker compose up --build` for application processes only.
5. Run `cd frontend && npm install && ORCHESTRATOR_API_URL=http://localhost:8080 ORCHESTRATOR_API_TOKEN=... npm run dev`.

There are intentionally no local Kafka, Redis, object-storage, Langfuse, model, search, or market-data containers.

## Cloud demo

1. Create a GKE Autopilot cluster, Artifact Registry repository, GCS bucket, and Cloud KMS key.
2. Enable Workload Identity and the GKE Secret Manager add-on.
3. Give the runtime Google service account least-privilege GCS access plus KMS encrypt/decrypt access;
   enable KMS audit logs. Do not grant pods permission to create Kubernetes Secrets.
4. Store deployment variables from `.env.example` as the dotenv-formatted `runtime-env` Google Secret
   Manager secret. User provider keys never belong there.
5. Configure GitHub variables `GCP_WORKLOAD_IDENTITY_PROVIDER`, `GCP_DEPLOY_SERVICE_ACCOUNT`,
   `GKE_CLUSTER`, `GCP_REGION`, `GCP_PROJECT_ID`, and `ARTIFACT_REGISTRY_REPO`.
6. Configure Vercel server-side `ORCHESTRATOR_API_URL` and `ORCHESTRATOR_API_TOKEN`, plus deployment
   credentials `VERCEL_TOKEN`, `VERCEL_ORG_ID`, and `VERCEL_PROJECT_ID` in GitHub.

## Tool limits

The first version intentionally uses simple provider-native limits: Tavily basic searches consume one
credit and market-data calls consume one request. Credential validation consumes one of each. Exact
USD accounting is enforced for OpenRouter using reservations and returned usage cost; tool USD cost
is not estimated because it depends on each user's provider plan.
