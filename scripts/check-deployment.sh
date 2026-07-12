#!/usr/bin/env bash
set -uo pipefail

usage() {
  cat <<'EOF'
Usage: ORCHESTRATOR_API_URL=... bash scripts/check-deployment.sh [options]

Call /health, /version, and /ready on a deployed orchestrator API and print
clear pass/fail output. Secret values are never printed.

Options:
  --url URL           Orchestrator base URL. Defaults to ORCHESTRATOR_API_URL.
  --token TOKEN       Optional x-api-key token. Defaults to ORCHESTRATOR_API_TOKEN.
  --expect-sha SHA    Require /version.git_sha to match SHA. Defaults to EXPECTED_GIT_SHA.
  --timeout SECONDS   curl timeout per request. Defaults to CHECK_DEPLOYMENT_TIMEOUT_SECONDS or 10.
  -h, --help          Show this help text.

Exit codes:
  0   All deployment checks passed
  1   A health, version, readiness, or expected-SHA check failed
  2   Missing command/env/configuration
EOF
}

base_url="${ORCHESTRATOR_API_URL:-}"
api_token="${ORCHESTRATOR_API_TOKEN:-}"
expected_sha="${EXPECTED_GIT_SHA:-}"
timeout_seconds="${CHECK_DEPLOYMENT_TIMEOUT_SECONDS:-10}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --url)
      if [[ $# -lt 2 || -z "${2:-}" ]]; then
        echo "error: --url requires a value" >&2
        exit 2
      fi
      base_url="${2:-}"
      shift 2
      ;;
    --token)
      if [[ $# -lt 2 || -z "${2:-}" ]]; then
        echo "error: --token requires a value" >&2
        exit 2
      fi
      api_token="${2:-}"
      shift 2
      ;;
    --expect-sha)
      if [[ $# -lt 2 || -z "${2:-}" ]]; then
        echo "error: --expect-sha requires a value" >&2
        exit 2
      fi
      expected_sha="${2:-}"
      shift 2
      ;;
    --timeout)
      if [[ $# -lt 2 || -z "${2:-}" ]]; then
        echo "error: --timeout requires a value" >&2
        exit 2
      fi
      timeout_seconds="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "error: required command not found: $1" >&2
    exit 2
  fi
}

require_command curl
require_command python3

if [[ -z "$base_url" ]]; then
  echo "error: missing ORCHESTRATOR_API_URL or --url" >&2
  exit 2
fi
if ! [[ "$timeout_seconds" =~ ^[0-9]+$ ]] || [[ "$timeout_seconds" -le 0 ]]; then
  echo "error: timeout must be a positive integer" >&2
  exit 2
fi

base_url="${base_url%/}"
base_url="${base_url%/health}"
base_url="${base_url%/healthz}"
base_url="${base_url%/version}"
base_url="${base_url%/ready}"

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

curl_headers=()
if [[ -n "$api_token" ]]; then
  curl_headers=(-H "x-api-key: ${api_token}")
fi

failures=0

fetch_endpoint() {
  local name="$1"
  local path="$2"
  local output_file="$3"
  local http_code

  http_code="$(
    curl -sS --max-time "$timeout_seconds" -w '%{http_code}' \
      -o "$output_file" "${curl_headers[@]}" "${base_url}${path}"
  )"
  if [[ "$http_code" -lt 200 || "$http_code" -ge 300 ]]; then
    echo "FAIL ${name}: HTTP ${http_code}"
    failures=$((failures + 1))
    return 1
  fi
  return 0
}

validate_json() {
  local name="$1"
  local file="$2"
  shift 2
  if python3 - "$name" "$file" "$expected_sha" "$@" <<'PY'
import json
import sys

name, path, expected_sha, *extra = sys.argv[1:]
with open(path, encoding="utf-8") as handle:
    data = json.load(handle)

if name == "health":
    if data.get("status") != "ok" or data.get("service") is None:
        raise SystemExit("health response missing status=ok or service")
    print(f"PASS health: service={data['service']}")
elif name == "version":
    version = data.get("version")
    git_sha = data.get("git_sha")
    image_tag = data.get("image_tag")
    if not version:
        raise SystemExit("version response missing backend version")
    if expected_sha and git_sha != expected_sha:
        raise SystemExit(f"expected git_sha {expected_sha}, got {git_sha or '<missing>'}")
    print(f"PASS version: version={version} git_sha={git_sha or '<missing>'} image_tag={image_tag or '<missing>'}")
elif name == "ready":
    checks = data.get("checks") or {}
    if data.get("status") != "ok":
        failing = sorted(key for key, value in checks.items() if value.get("status") != "ok")
        raise SystemExit(f"readiness status={data.get('status')} failing_checks={','.join(failing) or '<unknown>'}")
    print(f"PASS ready: checks={','.join(sorted(checks))}")
else:
    raise SystemExit(f"unknown validator {name}")
PY
  then
    return 0
  fi
  echo "FAIL ${name}: validation failed"
  failures=$((failures + 1))
  return 1
}

run_check() {
  local name="$1"
  local path="$2"
  local file="$tmp_dir/${name}.json"
  if fetch_endpoint "$name" "$path" "$file"; then
    validate_json "$name" "$file"
  fi
}

echo "Checking deployment at ${base_url}"
run_check health /health
run_check version /version
run_check ready /ready

if [[ "$failures" -gt 0 ]]; then
  echo "Deployment checks failed: ${failures}"
  exit 1
fi

echo "Deployment checks passed"
