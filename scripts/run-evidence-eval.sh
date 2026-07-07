#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ORCHESTRATOR_API_URL=... ORCHESTRATOR_API_TOKEN=... bash scripts/run-evidence-eval.sh [options]

Run offline evidence-quality evaluation questions against an orchestrator API, poll
each run to terminal status or timeout, save raw run details, and write summary
CSV/JSON files.

Required environment, except for --help:
  ORCHESTRATOR_API_URL      Orchestrator base URL, for example http://34.27.225.48
  ORCHESTRATOR_API_TOKEN    API token for x-api-key; never printed

Options:
  --questions PATH          Questions JSON file. Default: evals/research_questions.json
  --output-dir DIR          Output directory. Default: evals/output/<timestamp>
  --budget USD             LLM budget per run. Default: 1.0
  --tavily-credits N       Tavily credits per run. Default: 2
  --market-data-requests N Market-data requests per run. Default: 1
  --timeout-seconds N      Poll timeout per run. Default: 900
  --poll-seconds N         Poll interval. Default: 10
  --max-failures N         Fail if more than N runs fail before final. Default: 0
  --limit N                Run only the first N questions.
  -h, --help               Show this help.

Exit codes:
  0   Evaluation completed within failure threshold
  1   API/evaluation failure threshold exceeded
  2   Missing command/env/configuration
  124 One or more runs timed out and exceeded failure threshold
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "error: required command not found: python3" >&2
  exit 2
fi

python3 - "$@" <<'PY'
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TERMINAL_STATUSES = {"completed", "partial_budget_exhausted", "failed"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run evidence-quality evaluation questions against an orchestrator API."
    )
    parser.add_argument("--questions", default="evals/research_questions.json")
    parser.add_argument("--output-dir")
    parser.add_argument("--budget", type=float, default=1.0)
    parser.add_argument("--tavily-credits", type=int, default=2)
    parser.add_argument("--market-data-requests", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument("--max-failures", type=int, default=0)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        print(f"error: missing required environment variable: {name}", file=sys.stderr)
        sys.exit(2)
    return value


def normalize_base_url(value: str) -> str:
    base = value.rstrip("/")
    for suffix in ("/healthz", "/health", "/ready", "/v1/runs"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    return base.rstrip("/")


def http_json(method: str, url: str, token: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={
            "x-api-key": token,
            "content-type": "application/json",
            "accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} returned HTTP {exc.code}: {error_body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{method} {url} failed: {exc.reason}") from exc
    return json.loads(raw or "{}")


def load_questions(path: Path) -> list[dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    questions = data.get("questions") if isinstance(data, dict) else data
    if not isinstance(questions, list) or not questions:
        raise ValueError(f"{path} must contain a non-empty questions list")
    normalized: list[dict[str, str]] = []
    for index, item in enumerate(questions, start=1):
        if isinstance(item, str):
            normalized.append({"id": f"question-{index:02d}", "topic": "", "question": item})
            continue
        if not isinstance(item, dict) or not isinstance(item.get("question"), str):
            raise ValueError(f"question {index} must be a string or object with a question")
        question_id = str(item.get("id") or f"question-{index:02d}")
        normalized.append(
            {
                "id": question_id,
                "topic": str(item.get("topic") or ""),
                "question": item["question"],
            }
        )
    return normalized


def model_policy(budget: float) -> dict[str, Any]:
    return {
        "planner": role("anthropic/claude-sonnet-5", budget, 0.10, 0.00, 0.05, 2000),
        "utility": role("openai/gpt-5.4-nano", budget, 0.05, 0.00, 0.05, 2000),
        "research": role("qwen/qwen3.6-flash", budget, 0.30, 0.00, 0.08, 2000),
        "verifier": role("openai/gpt-5.4-mini", budget, 0.20, 0.00, 0.05, 2000),
        "aggregator": role("anthropic/claude-sonnet-5", budget, 0.25, 0.25, 0.25, 4000),
        "judge": role("openai/gpt-5.4-mini", budget, 0.10, 0.10, 0.10, 2000),
    }


def role(
    model: str,
    budget: float,
    cap_fraction: float,
    protected_fraction: float,
    max_call_fraction: float,
    max_output_tokens: int,
) -> dict[str, Any]:
    return {
        "model": model,
        "cap_usd": budget * cap_fraction,
        "protected_usd": budget * protected_fraction,
        "max_call_cost_usd": budget * max_call_fraction,
        "max_output_tokens": max_output_tokens,
    }


def payload_for(question: str, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "question": question,
        "llm_budget_usd": args.budget,
        "models": model_policy(args.budget),
        "tool_budget": {
            "tavily_max_credits": args.tavily_credits,
            "market_data_max_requests": args.market_data_requests,
        },
    }


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip(".-_").lower()
    return slug[:64] or "question"


def count_unique_sources(detail: dict[str, Any]) -> int:
    sources: set[str] = set()
    final = detail.get("final") or {}
    for value in final.get("sources") or []:
        if isinstance(value, str) and value:
            sources.add(value)
    for collection_name in ("observations", "claims", "verifications", "artifacts"):
        collection = detail.get(collection_name) or []
        if not isinstance(collection, list):
            continue
        for item in collection:
            if not isinstance(item, dict):
                continue
            for key in ("sources", "source_refs", "evidence_item_refs"):
                values = item.get(key) or []
                if isinstance(values, list):
                    sources.update(value for value in values if isinstance(value, str) and value)
    return len(sources)


def summarize_detail(question_id: str, detail: dict[str, Any], detail_path: Path) -> dict[str, Any]:
    run = detail.get("run") or {}
    final = detail.get("final")
    run_state = detail.get("run_state") or {}
    budget = run.get("budget") or {}
    budget_summary = run_state.get("budget_summary") or {}
    verifications = detail.get("verifications") or []
    artifacts = detail.get("artifacts") or []
    stop_reasons = list(run_state.get("stop_reasons") or [])
    stop_reasons.extend(budget_summary.get("stop_reasons") or [])
    return {
        "question_id": question_id,
        "run_id": run.get("id", ""),
        "status": run.get("status", ""),
        "final_present": final is not None,
        "judge_score": (final or {}).get("judge_score") or run_state.get("last_judge_score"),
        "budget_spent": budget.get("spent_usd", 0),
        "artifact_count": len(artifacts) if isinstance(artifacts, list) else 0,
        "verified_claim_count": run_state.get("verified_claim_count")
        or sum(1 for item in verifications if item.get("verdict") == "verified"),
        "rejected_claim_count": run_state.get("rejected_claim_count")
        or sum(1 for item in verifications if item.get("verdict") == "rejected"),
        "disputed_claim_count": run_state.get("disputed_claim_count")
        or sum(1 for item in artifacts if item.get("status") == "disputed"),
        "dead_letter_count": run_state.get("dead_letter_count")
        or len(detail.get("dead_letters") or []),
        "source_count": count_unique_sources(detail),
        "stop_reasons": "; ".join(dict.fromkeys(str(value) for value in stop_reasons if value)),
        "detail_path": str(detail_path),
    }


def write_summaries(rows: list[dict[str, Any]], output_dir: Path) -> None:
    (output_dir / "summary.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fieldnames = [
        "question_id",
        "run_id",
        "status",
        "final_present",
        "judge_score",
        "budget_spent",
        "artifact_count",
        "verified_claim_count",
        "rejected_claim_count",
        "disputed_claim_count",
        "dead_letter_count",
        "source_count",
        "stop_reasons",
        "detail_path",
    ]
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_one(
    question: dict[str, str],
    index: int,
    args: argparse.Namespace,
    base_url: str,
    token: str,
    output_dir: Path,
) -> dict[str, Any]:
    question_id = question["id"]
    print(f"[{index}] submitting {question_id}: {question['topic'] or question_id}")
    create_response = http_json("POST", f"{base_url}/v1/runs", token, payload_for(question["question"], args))
    run_id = str(create_response.get("id") or "")
    if not run_id:
        raise RuntimeError(f"create-run response for {question_id} did not contain id")

    run_dir = output_dir / f"{index:02d}-{slugify(question_id)}-{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.timeout_seconds
    latest_detail: dict[str, Any] | None = None
    detail_path = run_dir / "run-detail.json"

    while True:
        latest_detail = http_json("GET", f"{base_url}/v1/runs/{run_id}/detail", token)
        detail_path.write_text(json.dumps(latest_detail, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        status = ((latest_detail.get("run") or {}).get("status")) or ""
        final_present = latest_detail.get("final") is not None
        tasks_count = len(latest_detail.get("tasks") or [])
        artifacts_count = len(latest_detail.get("artifacts") or [])
        print(
            f"[{index}] run_id={run_id} status={status} final={final_present} "
            f"tasks={tasks_count} artifacts={artifacts_count}"
        )
        if status in TERMINAL_STATUSES:
            return summarize_detail(question_id, latest_detail, detail_path)
        if time.monotonic() >= deadline:
            row = summarize_detail(question_id, latest_detail, detail_path)
            row["status"] = "timeout"
            return row
        time.sleep(args.poll_seconds)


def main() -> int:
    args = parse_args()
    if args.budget <= 0:
        print("error: --budget must be > 0", file=sys.stderr)
        return 2
    if args.timeout_seconds <= 0 or args.poll_seconds <= 0:
        print("error: timeout and poll seconds must be positive", file=sys.stderr)
        return 2
    base_url = normalize_base_url(require_env("ORCHESTRATOR_API_URL"))
    token = require_env("ORCHESTRATOR_API_TOKEN")
    questions = load_questions(Path(args.questions))
    if args.limit is not None:
        questions = questions[: args.limit]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir or f"evals/output/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    timeout_seen = False
    for index, question in enumerate(questions, start=1):
        try:
            rows.append(run_one(question, index, args, base_url, token, output_dir))
        except Exception as exc:
            print(f"error: {question['id']} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            rows.append(
                {
                    "question_id": question["id"],
                    "run_id": "",
                    "status": "submit_or_poll_failed",
                    "final_present": False,
                    "judge_score": "",
                    "budget_spent": 0,
                    "artifact_count": 0,
                    "verified_claim_count": 0,
                    "rejected_claim_count": 0,
                    "disputed_claim_count": 0,
                    "dead_letter_count": 0,
                    "source_count": 0,
                    "stop_reasons": f"{type(exc).__name__}: {exc}",
                    "detail_path": "",
                }
            )

    write_summaries(rows, output_dir)
    print(f"wrote {output_dir / 'summary.json'}")
    print(f"wrote {output_dir / 'summary.csv'}")

    failed_before_final = [
        row
        for row in rows
        if row["status"] in {"failed", "timeout", "submit_or_poll_failed"} and not row["final_present"]
    ]
    timeout_seen = any(row["status"] == "timeout" for row in failed_before_final)
    if len(failed_before_final) > args.max_failures:
        print(
            f"error: {len(failed_before_final)} run(s) failed before final answer; "
            f"max allowed is {args.max_failures}",
            file=sys.stderr,
        )
        return 124 if timeout_seen else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY
