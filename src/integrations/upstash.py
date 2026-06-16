import json
from typing import TypeVar
from uuid import UUID

import httpx
from pydantic import BaseModel

from src.common.models import Claim, FinalReport, Observation, ResearchTask, Run, Verification

T = TypeVar("T", bound=BaseModel)


class UpstashBlackboard:
    """Shared operational state through Upstash's managed Redis REST API."""

    def __init__(self, url: str, token: str) -> None:
        self.url = url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    async def command(self, *parts: object) -> object:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.post(self.url, headers=self.headers, json=list(parts))
            response.raise_for_status()
        payload = response.json()
        if payload.get("error"):
            raise RuntimeError(f"Upstash command failed: {payload['error']}")
        return payload.get("result")

    async def put_model(self, key: str, value: BaseModel) -> None:
        await self.command("SET", key, value.model_dump_json())

    async def get_model(self, key: str, model: type[T]) -> T | None:
        value = await self.command("GET", key)
        return model.model_validate_json(value) if value else None

    async def append_model(self, run_id: UUID, kind: str, value: BaseModel) -> None:
        key = f"run:{run_id}:{kind}:{getattr(value, 'id')}"
        await self.put_model(key, value)
        await self.command("SADD", f"run:{run_id}:{kind}:ids", str(getattr(value, "id")))

    async def list_models(self, run_id: UUID | str, kind: str, model: type[T]) -> list[T]:
        ids = await self.command("SMEMBERS", f"run:{run_id}:{kind}:ids") or []
        values: list[T] = []
        for item_id in ids:
            value = await self.get_model(f"run:{run_id}:{kind}:{item_id}", model)
            if value:
                values.append(value)
        return values

    async def put_run(self, run: Run) -> None:
        await self.put_model(f"run:{run.id}", run)

    async def get_run(self, run_id: UUID | str) -> Run | None:
        return await self.get_model(f"run:{run_id}", Run)

    async def put_task(self, task: ResearchTask) -> None:
        await self.append_model(task.run_id, "tasks", task)

    async def put_observation(self, value: Observation) -> None:
        await self.append_model(value.run_id, "observations", value)

    async def put_claim(self, value: Claim) -> None:
        await self.append_model(value.run_id, "claims", value)

    async def put_verification(self, value: Verification) -> None:
        await self.append_model(value.run_id, "verifications", value)

    async def put_final(self, value: FinalReport) -> None:
        await self.put_model(f"run:{value.run_id}:final", value)

    async def get_final(self, run_id: UUID | str) -> FinalReport | None:
        return await self.get_model(f"run:{run_id}:final", FinalReport)

    async def get_json(self, key: str) -> dict[str, object] | None:
        value = await self.command("GET", key)
        return json.loads(value) if value else None
