from datetime import UTC, datetime
import json
from json import JSONDecodeError
import logging
import re
from typing import Any
from uuid import UUID, uuid4

import httpx

from src.common.budget import PersistentBudget
from src.common.models import AgentRole

logger = logging.getLogger(__name__)


class LLMOutputError(RuntimeError):
    """Raised when a model response cannot be parsed as the expected JSON object."""


class LLMProviderError(RuntimeError):
    """Raised when the model provider rejects or fails a request."""


FENCED_JSON_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)


def parse_json_output(output: str, name: str) -> dict[str, Any]:
    """Parse model JSON, accepting common markdown-fenced JSON wrappers."""
    candidates = [output]
    fenced = FENCED_JSON_RE.match(output)
    if fenced:
        candidates.insert(0, fenced.group(1))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            raise LLMOutputError(f"model returned non-object JSON for {name}: {json.dumps(parsed)[:500]}")
        return parsed
    snippet = output[:500] if output else "<empty>"
    raise LLMOutputError(f"model returned invalid JSON for {name}: {snippet}")


def provider_error_message(response: httpx.Response, name: str) -> str:
    body = response.text[:1000] if response.text else "<empty>"
    return f"model provider returned HTTP {response.status_code} for {name}: {body}"


def completion_content(response_data: dict[str, Any], name: str) -> str:
    try:
        output = response_data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        snippet = json.dumps(response_data)[:1000]
        raise LLMProviderError(f"model provider returned malformed completion for {name}: {snippet}") from exc
    if not isinstance(output, str):
        snippet = json.dumps(response_data)[:1000]
        raise LLMProviderError(f"model provider returned non-text completion for {name}: {snippet}")
    return output


def langfuse_trace_id(run_id: UUID) -> str:
    """Langfuse trace IDs must be 32 lowercase hex characters."""
    return run_id.hex


class LangfuseRecorder:
    def __init__(self, host: str, public_key: str, secret_key: str) -> None:
        self.url = f"{host.rstrip('/')}/api/public/ingestion"
        self.auth = (public_key, secret_key)

    async def generation(self, event_id: str, body: dict[str, Any]) -> None:
        trace_id = body.get("traceId")
        payload = {
            "batch": [
                {
                    "id": str(uuid4()),
                    "timestamp": datetime.now(UTC).isoformat(),
                    "type": "trace-create",
                    "body": {
                        "id": trace_id,
                        "name": "multi-agent-investment-run",
                        "metadata": {"runId": body.get("metadata", {}).get("run_id")},
                    },
                },
                {
                    "id": event_id,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "type": "generation-create",
                    "body": body,
                }
            ]
        }
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(self.url, auth=self.auth, json=payload)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError:
                logger.warning("langfuse generation ingest failed status=%s body=%s", response.status_code, response.text[:500])

    async def span(self, event_id: str, body: dict[str, Any]) -> None:
        payload = {
            "batch": [
                {
                    "id": event_id,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "type": "span-create",
                    "body": body,
                }
            ]
        }
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(self.url, auth=self.auth, json=payload)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError:
                logger.warning("langfuse span ingest failed status=%s body=%s", response.status_code, response.text[:500])


class OpenRouterLLM:
    def __init__(
        self, base_url: str, api_key: str, budget: PersistentBudget, recorder: LangfuseRecorder
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.budget = budget
        self.recorder = recorder

    async def json(
        self, run_id: UUID, role: AgentRole, name: str, system: str, prompt: str
    ) -> dict[str, Any]:
        run = await self.budget.blackboard.get_run(run_id)
        if not run:
            raise RuntimeError("run not found")
        policy = getattr(run.models, role.value)
        if not policy:
            raise RuntimeError(f"role {role} is disabled")
        estimate = policy.max_call_cost_usd
        role_used = run.budget.role_spent_usd.get(role, 0) + run.budget.role_reserved_usd.get(role, 0)
        role_remaining = max(0, (policy.cap_usd or run.budget.limit_usd) - role_used)
        system = (
            f"{system}\nHard budget context: role={role.value}; "
            f"role remaining before this call=${role_remaining:.4f}; "
            f"this call maximum reservation=${estimate:.4f}. "
            "Use the available output budget for decision-grade detail; avoid terse answers when evidence, "
            "risks, and caveats are needed, while still returning valid JSON."
        )
        await self.budget.reserve_llm(run_id, role, estimate)
        actual = estimate
        try:
            payload = {
                "model": policy.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "response_format": {"type": "json_object"},
                "max_tokens": policy.max_output_tokens,
                "usage": {"include": True},
            }
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                if response.status_code == 400:
                    payload_without_json_mode = {key: value for key, value in payload.items() if key != "response_format"}
                    response = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        json=payload_without_json_mode,
                    )
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    raise LLMProviderError(provider_error_message(response, name)) from exc
                response_data = response.json()
            usage = response_data.get("usage", {})
            actual = float(usage.get("cost") or estimate)
            output = completion_content(response_data, name)
            await self.recorder.generation(
                str(response_data["id"]),
                {
                    "id": str(response_data["id"]),
                    "traceId": langfuse_trace_id(run_id),
                    "name": name,
                    "model": response_data.get("model", policy.model),
                    "input": prompt,
                    "output": output,
                    "usage": usage,
                    "metadata": {"role": role.value, "actual_cost_usd": actual, "run_id": str(run_id)},
                },
            )
            return parse_json_output(output, name)
        finally:
            await self.budget.reconcile_llm(run_id, role, estimate, actual)
