import asyncio
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest

from src.common.models import AgentRole, Budget, ModelPolicy, RoleModelPolicy, Run, ToolBudget
from src.integrations.llm import LLMOutputError, OpenRouterLLM


def role_policy() -> RoleModelPolicy:
    return RoleModelPolicy(model="provider/model", cap_usd=1, max_call_cost_usd=0.1)


def model_policy() -> ModelPolicy:
    return ModelPolicy(
        planner=role_policy(),
        utility=role_policy(),
        research=role_policy(),
        verifier=role_policy(),
        aggregator=role_policy(),
        judge=role_policy(),
    )


class BudgetStub:
    def __init__(self, run: Run) -> None:
        async def get_run(_run_id):
            return run

        self.blackboard = SimpleNamespace(get_run=get_run)

    async def reserve_llm(self, *_args) -> None:
        return None

    async def reconcile_llm(self, *_args) -> None:
        return None


class RecorderStub:
    async def generation(self, *_args, **_kwargs) -> None:
        return None


def test_openrouter_llm_raises_diagnostic_error_for_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    run = Run(question="q", models=model_policy(), budget=Budget(limit_usd=1, tools=ToolBudget()))

    async def post(self, *_args, **_kwargs):
        return httpx.Response(
            200,
            json={
                "id": str(uuid4()),
                "choices": [{"message": {"content": "not json"}}],
                "usage": {"cost": 0.001},
            },
            request=httpx.Request("POST", "https://openrouter.example/chat/completions"),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    llm = OpenRouterLLM("https://openrouter.example", "key", BudgetStub(run), RecorderStub())  # type: ignore[arg-type]

    with pytest.raises(LLMOutputError, match="model returned invalid JSON for planner: not json"):
        asyncio.run(llm.json(run.id, AgentRole.PLANNER, "planner", "system", "prompt"))
