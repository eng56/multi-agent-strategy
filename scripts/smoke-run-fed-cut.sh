#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ORCHESTRATOR_API_URL=... ORCHESTRATOR_API_TOKEN=... bash scripts/smoke-run-fed-cut.sh

Submit the canonical Fed/gold/USD/equities deployed smoke test, poll run detail,
print a compact jq summary, and fail if the deployed run does not progress beyond
planner-stage task creation.

Required environment:
  ORCHESTRATOR_API_URL      Orchestrator base URL, for example http://34.27.225.48
  ORCHESTRATOR_API_TOKEN    API token for x-api-key; never printed

Optional environment:
  SMOKE_TIMEOUT_SECONDS     Poll timeout in seconds; default 600
  SMOKE_POLL_SECONDS        Poll interval in seconds; default 10
  SMOKE_OUTPUT_PATH         Raw detail path; default .codex-loop/latest-smoke-run-detail.json
                            when .codex-loop exists, otherwise smoke-run-detail.json

Exit codes:
  0   Smoke acceptance criteria passed
  1   Acceptance criteria failed or API returned an error
  2   Missing command/env/configuration
  124 Timed out before terminal run status
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "error: required command not found: $1" >&2
    exit 2
  fi
}

require_env() {
  if [[ -z "${!1:-}" ]]; then
    echo "error: missing required environment variable: $1" >&2
    exit 2
  fi
}

require_command curl
require_command jq
require_env ORCHESTRATOR_API_URL
require_env ORCHESTRATOR_API_TOKEN

timeout_seconds="${SMOKE_TIMEOUT_SECONDS:-600}"
poll_seconds="${SMOKE_POLL_SECONDS:-10}"
if ! [[ "$timeout_seconds" =~ ^[0-9]+$ && "$poll_seconds" =~ ^[0-9]+$ ]]; then
  echo "error: SMOKE_TIMEOUT_SECONDS and SMOKE_POLL_SECONDS must be positive integers" >&2
  exit 2
fi

if [[ -n "${SMOKE_OUTPUT_PATH:-}" ]]; then
  output_path="$SMOKE_OUTPUT_PATH"
elif [[ -d ".codex-loop" ]]; then
  output_path=".codex-loop/latest-smoke-run-detail.json"
else
  output_path="smoke-run-detail.json"
fi
mkdir -p "$(dirname "$output_path")"
tmp_output="${output_path}.tmp"

base_url="${ORCHESTRATOR_API_URL%/}"
base_url="${base_url%/healthz}"
base_url="${base_url%/v1/runs}"

question='Evaluate the next 3-month impact of a surprise Fed rate cut on gold, the US dollar, and US equities.

I want a decision-oriented investment research report, not generic commentary.

Please separate:
1. the main thesis,
2. verified evidence,
3. risks and counterarguments,
4. what would change the conclusion,
5. confidence level and time horizon.

Assume the objective is to understand whether gold is a better tactical opportunity than USD or equities after the rate cut.'

payload="$(
  jq -n --arg question "$question" '{
    question: $question,
    llm_budget_usd: 1,
    models: {
      planner: {model: "anthropic/claude-sonnet-5", cap_usd: 0.1, protected_usd: 0, max_output_tokens: 2000, max_call_cost_usd: 0.05},
      utility: {model: "openai/gpt-5.4-nano", cap_usd: 0.05, protected_usd: 0, max_output_tokens: 2000, max_call_cost_usd: 0.05},
      research: {model: "qwen/qwen3.6-flash", cap_usd: 0.3, protected_usd: 0, max_output_tokens: 2000, max_call_cost_usd: 0.08},
      verifier: {model: "openai/gpt-5.4-mini", cap_usd: 0.2, protected_usd: 0, max_output_tokens: 2000, max_call_cost_usd: 0.05},
      aggregator: {model: "anthropic/claude-sonnet-5", cap_usd: 0.25, protected_usd: 0.25, max_output_tokens: 4000, max_call_cost_usd: 0.25},
      judge: {model: "openai/gpt-5.4-mini", cap_usd: 0.1, protected_usd: 0.1, max_output_tokens: 2000, max_call_cost_usd: 0.1}
    },
    tool_budget: {tavily_max_credits: 2, market_data_max_requests: 1}
  }'
)"

curl_json() {
  local method="$1"
  local url="$2"
  local data="${3:-}"
  local response http_code body
  if [[ -n "$data" ]]; then
    response="$(
      curl -sS --max-time 90 -w '\n%{http_code}' -X "$method" "$url" \
        -H "x-api-key: ${ORCHESTRATOR_API_TOKEN}" \
        -H "content-type: application/json" \
        --data-binary "$data"
    )"
  else
    response="$(
      curl -sS --max-time 90 -w '\n%{http_code}' -X "$method" "$url" \
        -H "x-api-key: ${ORCHESTRATOR_API_TOKEN}"
    )"
  fi
  http_code="$(printf '%s' "$response" | tail -n 1)"
  body="$(printf '%s' "$response" | sed '$d')"
  if [[ "$http_code" -lt 200 || "$http_code" -ge 300 ]]; then
    echo "error: $method $url returned HTTP $http_code" >&2
    printf '%s\n' "$body" >&2
    return 1
  fi
  printf '%s' "$body"
}

echo "Submitting Fed/gold/USD/equities smoke run to ${base_url}"
create_body="$(curl_json POST "${base_url}/v1/runs" "$payload")"
run_id="$(printf '%s' "$create_body" | jq -r '.id // empty')"
if [[ -z "$run_id" ]]; then
  echo "error: create-run response did not contain .id" >&2
  printf '%s\n' "$create_body" >&2
  exit 1
fi
echo "run_id=${run_id}"

deadline=$((SECONDS + timeout_seconds))
terminal_status=""
while :; do
  detail_body="$(curl_json GET "${base_url}/v1/runs/${run_id}/detail")"
  printf '%s' "$detail_body" > "$tmp_output"

  jq '{
    status: .run.status,
    failure_reason: .run.failure_reason,
    active_branches: .run_state.active_branches,
    tasks_count: (.tasks | length),
    artifacts_count: (.artifacts | length),
    principal_actions_count: (.principal_actions | length),
    final_present: (.final != null)
  }' "$tmp_output"

  status_value="$(jq -r '.run.status // ""' "$tmp_output")"
  case "$status_value" in
    completed|partial_budget_exhausted|failed)
      terminal_status="$status_value"
      break
      ;;
  esac

  if (( SECONDS >= deadline )); then
    mv "$tmp_output" "$output_path"
    echo "error: timed out waiting for terminal run status; saved latest detail to $output_path" >&2
    exit 124
  fi
  sleep "$poll_seconds"
done

mv "$tmp_output" "$output_path"
echo "Saved raw run detail to $output_path"

failure_reason="$(jq -r '.run.failure_reason // ""' "$output_path")"
tasks_count="$(jq '.tasks | length' "$output_path")"
principal_actions_count="$(jq '.principal_actions | length' "$output_path")"
crypto_branch_count="$(jq '[.run_state.active_branches[]? | select(. == "market/crypto")] | length' "$output_path")"

if [[ "$terminal_status" == "failed" ]] && printf '%s' "$failure_reason" | grep -qi 'planner'; then
  echo "error: run failed at planner stage: $failure_reason" >&2
  exit 1
fi
if (( tasks_count == 0 )); then
  echo "error: tasks_count is 0" >&2
  exit 1
fi
if (( principal_actions_count <= 1 )); then
  echo "error: principal_actions_count must be > 1; got $principal_actions_count" >&2
  exit 1
fi
if (( crypto_branch_count > 0 )); then
  echo "error: market/crypto branch appeared for a non-crypto question" >&2
  exit 1
fi

echo "Smoke acceptance passed for run ${run_id}"
