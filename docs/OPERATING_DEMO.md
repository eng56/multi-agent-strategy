# Operating the Managed-Services Demo

This is the operator runbook for the managed multi-agent investment research demo. It focuses on
starting the system, validating it, inspecting runs, and recovering from common failures without
exposing credentials.

## Architecture in One Page

```text
Browser
  -> Vercel private frontend and BFF
    -> GKE Autopilot orchestrator API
      -> Confluent Cloud topic: coordination events only
      -> Upstash Redis REST: runs, tasks, state, budgets, dead letters
      -> GCS: large raw artifacts
      -> OpenRouter: deployment-owned LLM key
      -> Tavily: deployment-owned search key
      -> Massive: deployment-owned market-data key
      -> Langfuse Cloud: sanitized traces

Google Secret Manager -> runtime-env dotenv file mounted into GKE pods
Artifact Registry -> backend image consumed by GKE
GitHub Actions -> GKE deploy, Vercel deploy, production smoke workflow
```

The runtime has one orchestrator API deployment plus role-specific workers:

- `planner-agent`
- `worker-agents`
- `verifier-agent`
- `skeptic-agent`
- `aggregator-agent`
- `judge-agent`
- `tool-runner`

Events contain identifiers and routing metadata. Current run state lives in Upstash, raw tool output
lives in GCS, and `/v1/runs/{run_id}/detail` joins the pieces for operators and the UI.

## Required Managed Services

Provision separate resources for staging and production/demo.

| Service | Used for |
| --- | --- |
| GKE Autopilot | Backend API and worker containers |
| Artifact Registry | Backend container images |
| Google Secret Manager | Dotenv-formatted `runtime-env` runtime secret |
| Google Cloud Storage | Raw run artifacts |
| Confluent Cloud | `agent-runtime` coordination topic |
| Upstash Redis REST | Blackboard state, budgets, worker heartbeats, dead letters |
| OpenRouter | LLM calls with deployment-owned key |
| Tavily | Web search tool |
| Massive | Market-data tool |
| Langfuse Cloud | Sanitized LLM and tool traces |
| Vercel | Password-gated frontend and API proxy |

Do not run local Kafka, Redis, object storage, model, search, market-data, tracing, or secret-store
containers for this demo.

## Required Environment Variables

Backend runtime variables come from `.env.example` locally and from the Google Secret Manager
`runtime-env` secret in GKE.

```text
CONFLUENT_BOOTSTRAP_SERVERS
CONFLUENT_API_KEY
CONFLUENT_API_SECRET
CONFLUENT_SECURITY_PROTOCOL
CONFLUENT_SASL_MECHANISM
UPSTASH_REDIS_REST_URL
UPSTASH_REDIS_REST_TOKEN
OPENROUTER_API_KEY
TAVILY_API_KEY
MARKET_DATA_API_KEY
MARKET_DATA_PROVIDER
LANGFUSE_PUBLIC_KEY
LANGFUSE_SECRET_KEY
LANGFUSE_HOST
GCP_PROJECT_ID
GCP_REGION
GCS_BUCKET_NAME
ARTIFACT_REGISTRY_REPO
ORCHESTRATOR_API_TOKEN
ENVIRONMENT
DEFAULT_MAX_PARALLEL_AGENTS
OPENROUTER_BASE_URL
RUNTIME_TOPIC
TAVILY_BASE_URL
MARKET_DATA_BASE_URL
```

Frontend runtime variables:

```text
ORCHESTRATOR_API_URL
ORCHESTRATOR_API_TOKEN
DEMO_PASSWORD
```

GitHub Actions deployment variables:

```text
GCP_WORKLOAD_IDENTITY_PROVIDER
GCP_DEPLOY_SERVICE_ACCOUNT
GKE_CLUSTER
GCP_REGION
GCP_PROJECT_ID
ARTIFACT_REGISTRY_REPO
VERCEL_TOKEN
VERCEL_ORG_ID
VERCEL_PROJECT_ID
```

`OPENROUTER_API_KEY`, `TAVILY_API_KEY`, `MARKET_DATA_API_KEY`, `LANGFUSE_SECRET_KEY`,
`UPSTASH_REDIS_REST_TOKEN`, `CONFLUENT_API_SECRET`, `ORCHESTRATOR_API_TOKEN`, `DEMO_PASSWORD`, and
`VERCEL_TOKEN` are secrets. Never print them.

## Local Development Mode

Local mode still uses managed services.

```bash
cp .env.example .env
```

Fill `.env` with development managed-service credentials. Then start backend application processes:

```bash
docker compose up --build
```

Start the frontend in another shell:

```bash
cd frontend
npm install
ORCHESTRATOR_API_URL=http://localhost:8080 \
ORCHESTRATOR_API_TOKEN="$ORCHESTRATOR_API_TOKEN" \
DEMO_PASSWORD="$DEMO_PASSWORD" \
npm run dev
```

Check local health:

```bash
curl -sS http://localhost:8080/health | jq
curl -sS http://localhost:8080/ready | jq
```

## Staging Deployment

There is no dedicated staging workflow checked in. Treat staging as a separate managed stack using
the same manifests and scripts, not as a shared production namespace.

Minimum staging rules:

- Use separate GKE, Confluent, Upstash, GCS, Langfuse, OpenRouter/Tavily/Massive provider budget,
  Secret Manager `runtime-env`, and Vercel settings.
- Set `ENVIRONMENT=staging` in the staging `runtime-env` secret.
- Point `kubectl` at the staging cluster before applying manifests.
- Do not reuse production `ORCHESTRATOR_API_TOKEN` or `DEMO_PASSWORD`.

Manual staging deploy from a local checkout:

```bash
gcloud auth configure-docker "$GCP_REGION-docker.pkg.dev" --quiet
IMAGE="$GCP_REGION-docker.pkg.dev/$GCP_PROJECT_ID/$ARTIFACT_REGISTRY_REPO/runtime:staging-$(git rev-parse --short HEAD)"
BUILD_TIME="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"

docker build \
  --build-arg GIT_SHA="$(git rev-parse HEAD)" \
  --build-arg IMAGE_TAG="$IMAGE" \
  --build-arg BUILD_TIME="$BUILD_TIME" \
  -t "$IMAGE" .
docker push "$IMAGE"

tmpdir="$(mktemp -d)"
cp -R deploy/gke/base "$tmpdir/base"
sed -i.bak "s|APP_IMAGE|$IMAGE|g; s|GCP_PROJECT_ID|$GCP_PROJECT_ID|g" "$tmpdir"/base/*.yaml
kubectl apply -k "$tmpdir/base"

for d in orchestrator-api planner-agent worker-agents verifier-agent skeptic-agent aggregator-agent judge-agent tool-runner; do
  kubectl rollout status "deployment/$d" --timeout=10m
done
```

Validate the staging endpoint:

```bash
ORCHESTRATOR_API_URL=http://STAGING_LOAD_BALANCER_IP \
ORCHESTRATOR_API_TOKEN="$ORCHESTRATOR_API_TOKEN" \
bash scripts/check-deployment.sh
```

## Production/Demo Deployment

Production GKE deploys on push to `main` through `.github/workflows/deploy-gke.yml`. The frontend
deploys through `.github/workflows/deploy-vercel.yml` when `frontend/**` changes and Vercel settings
are configured.

After the GKE deploy, validate the backend:

```bash
ORCHESTRATOR_API_URL=http://PRODUCTION_LOAD_BALANCER_IP \
ORCHESTRATOR_API_TOKEN="$ORCHESTRATOR_API_TOKEN" \
EXPECTED_GIT_SHA="$(git rev-parse HEAD)" \
bash scripts/check-deployment.sh
```

Run the GitHub Production Smoke Test workflow after merge:

```bash
gh workflow run "Production Smoke Test" --ref main
```

The workflow reads the deployed LoadBalancer IP and API token without printing the token, runs
`scripts/check-deployment.sh`, runs the Fed/gold/USD/equities smoke test, and uploads the raw run
detail JSON as an artifact when available.

## Fed/Gold/USD/Equities Smoke Test

Run against staging or production:

```bash
ORCHESTRATOR_API_URL=http://LOAD_BALANCER_IP \
ORCHESTRATOR_API_TOKEN="$ORCHESTRATOR_API_TOKEN" \
SMOKE_OUTPUT_PATH=.codex-loop/latest-smoke-run-detail.json \
bash scripts/smoke-run-fed-cut.sh
```

The script creates the canonical Fed surprise-rate-cut run, polls `/v1/runs/{run_id}/detail`, writes
the raw detail JSON, and fails if the run never gets past planner-stage task creation.

Useful knobs:

```bash
SMOKE_TIMEOUT_SECONDS=900
SMOKE_POLL_SECONDS=15
SMOKE_OUTPUT_PATH=smoke-run-detail.json
```

## Inspecting a Run

Set the endpoint and run ID:

```bash
export ORCHESTRATOR_API_URL=http://LOAD_BALANCER_IP
export RUN_ID=00000000-0000-0000-0000-000000000000
export DETAIL_PATH=run-detail.json
```

Run status:

```bash
curl -sS -H "x-api-key: $ORCHESTRATOR_API_TOKEN" \
  "$ORCHESTRATOR_API_URL/v1/runs/$RUN_ID" \
  | jq '{id, status, failure_reason, budget, final_answer_present: (.final_answer != null)}'
```

Full run detail:

```bash
curl -sS -H "x-api-key: $ORCHESTRATOR_API_TOKEN" \
  "$ORCHESTRATOR_API_URL/v1/runs/$RUN_ID/detail" \
  > "$DETAIL_PATH"
```

Run state:

```bash
jq '.run_state | {
  current_phase,
  active_branches,
  budget_remaining,
  tool_budget_remaining,
  verified_claim_count,
  rejected_claim_count,
  disputed_claim_count,
  dead_letter_count,
  stop_reasons,
  next_action_candidates
}' "$DETAIL_PATH"
```

Tasks:

```bash
jq '.tasks[] | {id, title, tool, status, wave_number, reason}' "$DETAIL_PATH"
```

Observations:

```bash
jq '.observations[] | {id, task_id, tool, summary, artifact_uri: .artifact.uri, sources}' "$DETAIL_PATH"
```

Claims:

```bash
jq '.claims[] | {id, task_id, statement, confidence, evidence_observation_ids, sources}' "$DETAIL_PATH"
```

Verifications:

```bash
jq '.verifications[] | {id, claim_id, verdict, confidence, rationale, sources}' "$DETAIL_PATH"
```

Artifacts:

```bash
jq '.artifacts[] | {
  id,
  artifact_type,
  branch,
  status,
  visibility,
  confidence,
  text_or_summary,
  source_refs,
  supports_artifact_ids,
  contradicts_artifact_ids
}' "$DETAIL_PATH"
```

Principal actions:

```bash
jq '.principal_actions[] | {
  id,
  action_type,
  status,
  target_branch,
  required_role,
  estimated_cost,
  reason
}' "$DETAIL_PATH"
```

Final report:

```bash
jq -r '.final.answer // .run.final_answer // "no final report"' "$DETAIL_PATH"
```

Dead letters:

```bash
jq '.dead_letters[] | {
  event_type,
  worker_role,
  classification,
  error_type,
  error_message,
  retry_count,
  event_payload
}' "$DETAIL_PATH"
```

Read a raw GCS artifact when the observation has a `gs://` URI and your Google identity has access:

```bash
ARTIFACT_URI="$(jq -r '.observations[0].artifact.uri' "$DETAIL_PATH")"
gcloud storage cat "$ARTIFACT_URI" | jq
```

## Common Failure Modes

Planner JSON failure:

- Symptom: smoke test reports planner-stage failure, or run detail has `LLMOutputError` with invalid
  JSON in `failure_reason` or `dead_letters`.
- Inspect: `jq '.run.failure_reason, .dead_letters' "$DETAIL_PATH"`.
- Recover: choose a model that follows JSON mode better, raise `max_output_tokens` if responses are
  being truncated, reduce task scope if the prompt is too large, or rerun. The planner has
  deterministic fallback logic, so repeated failures usually point to provider/model behavior or
  malformed provider responses.

Tool provider failure:

- Symptom: run creation returns provider validation `400`, tasks fail, or dead letters mention
  Tavily/Massive HTTP `401`, `429`, or `5xx`.
- Inspect: `jq '.tasks, .dead_letters' "$DETAIL_PATH"` and `kubectl logs deployment/tool-runner --tail=200`.
- Recover: rotate the affected provider key in `runtime-env` for `401`; wait or reduce load for
  `429`; retry later for `5xx`; increase `tool_budget` only when the task legitimately needs more
  Tavily credits or market-data requests.

LLM provider failure:

- Symptom: run creation fails while validating models, workers dead-letter with `LLMProviderError`,
  or OpenRouter returns `401`, `429`, `400`, or `5xx`.
- Inspect:

```bash
curl -sS -H "x-api-key: $ORCHESTRATOR_API_TOKEN" \
  "$ORCHESTRATOR_API_URL/v1/models/openrouter" \
  | jq '.models[:20]'
```

- Recover: rotate `OPENROUTER_API_KEY` for auth errors, pick model IDs returned by the models
  endpoint, choose cheaper models or raise role caps for budget errors, and rerun the smoke test.

Event stuck or worker not consuming:

- Symptom: run remains `created` or `running`, `/ready` fails worker checks, tasks exist but no
  observations/claims appear, or worker heartbeats are stale.
- Inspect:

```bash
bash scripts/check-deployment.sh
kubectl get pods
kubectl logs deployment/planner-agent --tail=120
kubectl logs deployment/worker-agents --tail=120
kubectl logs deployment/tool-runner --tail=120
```

- Recover: wake scaled-down deployments, restart workers, then check readiness again.

```bash
scripts/gke-demo-cost.sh wake
kubectl rollout restart deployment/planner-agent deployment/worker-agents deployment/verifier-agent deployment/skeptic-agent deployment/aggregator-agent deployment/judge-agent deployment/tool-runner
kubectl rollout status deployment/planner-agent --timeout=10m
bash scripts/check-deployment.sh
```

Budget exhausted:

- Symptom: run status is `partial_budget_exhausted`, tasks are `skipped_budget`, or final report says
  it was assembled without another LLM call.
- Inspect:

```bash
jq '{
  budget: .run.budget,
  stop_reasons: .run_state.stop_reasons,
  skipped_tasks: [.tasks[] | select(.status == "skipped_budget")]
}' "$DETAIL_PATH"
```
- Recover: raise `llm_budget_usd`, role `cap_usd`, and `max_call_cost_usd`; keep aggregator and judge
  `protected_usd` at least equal to their `max_call_cost_usd`; choose cheaper models for exploratory
  roles; rerun the run.

No final answer:

- Symptom: `jq '.final == null and .run.final_answer == null' "$DETAIL_PATH"` returns `true`.
- Inspect: check run status, aggregator dead letters, failed tasks, and budget.

```bash
jq '{status: .run.status, failure_reason: .run.failure_reason, final: .final, dead_letters: .dead_letters, budget: .run.budget}' "$DETAIL_PATH"
kubectl logs deployment/aggregator-agent --tail=200
```

- Recover: if the run is still active, fix the stuck worker path first. If terminal with no final,
  preserve the detail JSON, fix the underlying dead letter or budget issue, and rerun.

## Recovery Playbook

1. Preserve the current detail JSON before making changes.

```bash
curl -sS -H "x-api-key: $ORCHESTRATOR_API_TOKEN" \
  "$ORCHESTRATOR_API_URL/v1/runs/$RUN_ID/detail" \
  > "run-$RUN_ID-detail.json"
```

2. Check service health and worker readiness.

```bash
bash scripts/check-deployment.sh
kubectl get deployments
kubectl get pods
```

3. Inspect run state and dead letters.

```bash
jq '{status: .run.status, failure_reason: .run.failure_reason, run_state: .run_state, dead_letters: .dead_letters}' "run-$RUN_ID-detail.json"
```

4. Inspect the worker role named by the failed event.

```bash
kubectl logs deployment/ROLE_FROM_DEAD_LETTER --tail=200
```

5. Apply the smallest fix:

- Credentials: add a new `runtime-env` secret version and restart deployments.
- Worker stuck: wake or restart worker deployments.
- Provider 429/5xx: wait, reduce concurrency or budget exposure, and retry.
- Model or JSON issue: use a model ID returned by `/v1/models/openrouter` and rerun.
- Budget issue: raise per-run budget and protected finalization budgets, or choose cheaper models.

6. If credentials changed, add a new Secret Manager version from a secured dotenv file.

```bash
gcloud secrets versions add runtime-env \
  --project="$GCP_PROJECT_ID" \
  --data-file="$RUNTIME_ENV_FILE"
```

7. Restart deployments after changing mounted runtime secrets.

```bash
kubectl rollout restart deployment/orchestrator-api deployment/planner-agent deployment/worker-agents deployment/verifier-agent deployment/skeptic-agent deployment/aggregator-agent deployment/judge-agent deployment/tool-runner
for d in orchestrator-api planner-agent worker-agents verifier-agent skeptic-agent aggregator-agent judge-agent tool-runner; do
  kubectl rollout status "deployment/$d" --timeout=10m
done
```

8. Revalidate and rerun the smoke test.

```bash
bash scripts/check-deployment.sh
bash scripts/smoke-run-fed-cut.sh
```

## Cost Safety and Budget Controls

Per run:

- `llm_budget_usd` is the hard shared OpenRouter cap.
- Each role can set `cap_usd`, `max_call_cost_usd`, and `max_output_tokens`.
- Aggregator and judge `protected_usd` are reserved before research roles spend.
- `tool_budget.tavily_max_credits` limits Tavily basic searches.
- `tool_budget.market_data_max_requests` limits Massive market-data calls.
- `DEFAULT_MAX_PARALLEL_AGENTS` limits active agents for a run.

Idle GKE cost controls:

```bash
scripts/gke-demo-cost.sh status
scripts/gke-demo-cost.sh sleep
scripts/gke-demo-cost.sh wake
```

Scaling to zero stops demo pods but does not delete the cluster or external LoadBalancer. For the
lowest idle cost between demos, delete the Autopilot cluster and redeploy later. Expect the
LoadBalancer IP to change.

```bash
gcloud container clusters delete "$GKE_CLUSTER" \
  --location="$GCP_REGION" \
  --project="$GCP_PROJECT_ID"
```

## What Not to Do With Credentials

- Do not commit `.env`, `.codex-loop/secrets/*.env`, downloaded smoke artifacts containing internal
  payloads, or provider keys.
- Do not print, paste, or screenshot `ORCHESTRATOR_API_TOKEN`, provider API keys, Redis tokens,
  Confluent secrets, Langfuse secrets, Vercel tokens, or demo passwords.
- Do not put provider keys in the browser, Vercel client-side variables, Kafka event payloads, GCS
  artifacts, Langfuse metadata, or run questions.
- Do not grant pods permission to create Kubernetes Secrets.
- Do not reuse production secrets in staging or local development.
- Do not rotate credentials unless you own the provider account or are executing an incident response
  step. Add a new Secret Manager version and restart pods instead of editing manifests.
