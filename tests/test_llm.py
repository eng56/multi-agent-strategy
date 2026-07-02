import asyncio
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest

from src.common.models import AgentRole, Budget, ModelPolicy, RoleModelPolicy, Run, ToolBudget
from src.integrations.llm import (
    LLMOutputError,
    LLMProviderError,
    LangfuseRecorder,
    OpenRouterLLM,
    completion_content,
    langfuse_trace_id,
    parse_json_output,
)


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


def test_parse_json_output_accepts_markdown_fenced_json() -> None:
    output = """```json
{"tasks":[{"title":"SAP","question":"SAP ticker data","tool":"market_data"}]}
```"""

    assert parse_json_output(output, "planner") == {
        "tasks": [{"title": "SAP", "question": "SAP ticker data", "tool": "market_data"}]
    }


def test_openrouter_llm_retries_without_response_format_on_bad_request(monkeypatch: pytest.MonkeyPatch) -> None:
    run = Run(question="q", models=model_policy(), budget=Budget(limit_usd=1, tools=ToolBudget()))
    seen_payloads = []

    async def post(self, *_args, **kwargs):
        seen_payloads.append(kwargs["json"])
        if len(seen_payloads) == 1:
            return httpx.Response(
                400,
                json={"error": {"message": "response_format is not supported"}},
                request=httpx.Request("POST", "https://openrouter.example/chat/completions"),
            )
        return httpx.Response(
            200,
            json={
                "id": str(uuid4()),
                "choices": [{"message": {"content": '{"claims":[]}'}}],
                "usage": {"cost": 0.001},
            },
            request=httpx.Request("POST", "https://openrouter.example/chat/completions"),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    llm = OpenRouterLLM("https://openrouter.example", "key", BudgetStub(run), RecorderStub())  # type: ignore[arg-type]

    assert asyncio.run(llm.json(run.id, AgentRole.RESEARCH, "claim-extractor", "system", "prompt")) == {
        "claims": []
    }
    assert "response_format" in seen_payloads[0]
    assert "response_format" not in seen_payloads[1]


def test_openrouter_llm_reports_provider_error_body(monkeypatch: pytest.MonkeyPatch) -> None:
    run = Run(question="q", models=model_policy(), budget=Budget(limit_usd=1, tools=ToolBudget()))

    async def post(self, *_args, **_kwargs):
        return httpx.Response(
            400,
            json={"error": {"message": "bad model request"}},
            request=httpx.Request("POST", "https://openrouter.example/chat/completions"),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    llm = OpenRouterLLM("https://openrouter.example", "key", BudgetStub(run), RecorderStub())  # type: ignore[arg-type]

    with pytest.raises(LLMProviderError, match="bad model request"):
        asyncio.run(llm.json(run.id, AgentRole.RESEARCH, "claim-extractor", "system", "prompt"))


def test_completion_content_reports_malformed_provider_response() -> None:
    with pytest.raises(LLMProviderError, match="malformed completion.*quota exceeded"):
        completion_content({"error": {"message": "quota exceeded"}}, "tool-summary")


def test_langfuse_trace_id_uses_uuid_hex() -> None:
    run_id = uuid4()

    assert langfuse_trace_id(run_id) == run_id.hex
    assert "-" not in langfuse_trace_id(run_id)
    assert len(langfuse_trace_id(run_id)) == 32


def test_langfuse_generation_creates_trace_and_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    payloads = []

    async def post(self, *_args, **kwargs):
        payloads.append(kwargs["json"])
        return httpx.Response(207, json={"successes": []}, request=httpx.Request("POST", "https://langfuse.example/api/public/ingestion"))

    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    recorder = LangfuseRecorder("https://langfuse.example", "pk", "sk")

    asyncio.run(
        recorder.generation(
            "generation-id",
            {
                "traceId": "a" * 32,
                "name": "planner",
                "input": "prompt",
                "output": "{}",
                "metadata": {"run_id": "run-id"},
            },
        )
    )

    assert [item["type"] for item in payloads[0]["batch"]] == ["trace-create", "generation-create"]
    assert payloads[0]["batch"][0]["body"]["id"] == "a" * 32
    assert payloads[0]["batch"][1]["body"]["traceId"] == "a" * 32
