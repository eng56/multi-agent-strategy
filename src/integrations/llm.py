import json
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import httpx

from src.common.budget import PersistentBudget
from src.common.models import AgentRole
from src.integrations.credentials import EphemeralCredentialStore


class LangfuseRecorder:
    def __init__(self, host: str, public_key: str, secret_key: str) -> None:
        self.url = f"{host.rstrip('/')}/api/public/ingestion"
        self.auth = (public_key, secret_key)

    async def generation(self, event_id: str, body: dict[str, Any]) -> None:
        payload = {"batch": [{"id": event_id, "timestamp": datetime.now(UTC).isoformat(), "type": "generation-create", "body": body}]}
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(self.url, auth=self.auth, json=payload)
            response.raise_for_status()

    async def span(self, event_id: str, body: dict[str, Any]) -> None:
        payload = {"batch": [{"id": event_id, "timestamp": datetime.now(UTC).isoformat(), "type": "span-create", "body": body}]}
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(self.url, auth=self.auth, json=payload)
            response.raise_for_status()


class OpenRouterLLM:
    def __init__(self, base_url: str, budget: PersistentBudget, credentials: EphemeralCredentialStore, recorder: LangfuseRecorder) -> None:
        self.base_url = base_url.rstrip("/")
        self.budget = budget
        self.credentials = credentials
        self.recorder = recorder

    async def json(self, run_id: UUID, role: AgentRole, name: str, system: str, prompt: str) -> dict[str, Any]:
        run = await self.budget.blackboard.get_run(run_id)
        if not run:
            raise RuntimeError("run not found")
        policy = getattr(run.models, role.value)
        if not policy:
            raise RuntimeError(f"role {role} is disabled")
        estimate = policy.max_call_cost_usd
        role_used = run.budget.role_spent_usd.get(role, 0) + run.budget.role_reserved_usd.get(role, 0)
        role_remaining = max(0, (policy.cap_usd or run.budget.limit_usd) - role_used)
        system = f"{system}\nHard budget context: role={role.value}; role remaining before this call=${role_remaining:.4f}; this call maximum reservation=${estimate:.4f}."
        await self.budget.reserve_llm(run_id, role, estimate)
        actual = estimate
        response_data: dict[str, Any] = {}
        try:
            key = await self.credentials.get_key(run_id, "openrouter")
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json={"model": policy.model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}], "response_format": {"type": "json_object"}, "max_tokens": policy.max_output_tokens, "usage": {"include": True}},
                )
                response.raise_for_status()
                response_data = response.json()
            usage = response_data.get("usage", {})
            actual = float(usage.get("cost") or estimate)
            output = response_data["choices"][0]["message"]["content"]
            await self.recorder.generation(str(response_data["id"]), {"id": str(response_data["id"]), "traceId": str(run_id), "name": name, "model": response_data.get("model", policy.model), "input": prompt, "output": output, "usage": usage, "metadata": {"role": role.value, "actual_cost_usd": actual}})
            return json.loads(output)
        finally:
            await self.budget.reconcile_llm(run_id, role, estimate, actual)
