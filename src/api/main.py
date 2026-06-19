from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException

from src.common.config import get_settings
from src.common.models import (
    Budget,
    Claim,
    EventEnvelope,
    EventType,
    Observation,
    ResearchTask,
    Run,
    RunDetail,
    RunRequest,
    Verification,
)
from src.integrations.validation import validate_demo_configuration
from src.runtime import build_runtime


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.runtime = build_runtime()
    yield


app = FastAPI(title="Managed Multi-Agent Strategy API", lifespan=lifespan)


def require_api_token(x_api_key: str = Header(default="")) -> None:
    if x_api_key != get_settings().orchestrator_api_token:
        raise HTTPException(status_code=401, detail="invalid API token")


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models/openrouter")
async def list_openrouter_models(_: None = Depends(require_api_token)) -> dict[str, list[str]]:
    settings = get_settings()
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(
            f"{settings.openrouter_base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
        )
        response.raise_for_status()
    models = sorted(item["id"] for item in response.json().get("data", []) if item.get("id"))
    return {"models": models}


@app.post("/v1/runs", response_model=Run, status_code=202)
async def create_run(request: RunRequest, _: None = Depends(require_api_token)) -> Run:
    runtime = app.state.runtime
    try:
        await validate_demo_configuration(
            request,
            runtime.settings.openrouter_api_key,
            runtime.settings.tavily_api_key,
            runtime.settings.market_data_api_key,
            runtime.settings.openrouter_base_url,
            runtime.settings.tavily_base_url,
            runtime.settings.polygon_base_url,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"provider or model validation failed: {type(exc).__name__}"
        ) from exc
    request.tool_budget.tavily_credits_used = 1
    request.tool_budget.market_data_requests_used = 1
    run = Run(
        question=request.question,
        models=request.models,
        budget=Budget(
            limit_usd=request.llm_budget_usd,
            tools=request.tool_budget,
            max_parallel_agents=runtime.settings.default_max_parallel_agents,
        ),
    )
    await runtime.blackboard.put_run(run)
    runtime.events.publish(
        runtime.settings.runtime_topic,
        EventEnvelope(type=EventType.RUN_CREATED, run_id=run.id, producer="orchestrator-api"),
    )
    return run


@app.get("/v1/runs/{run_id}", response_model=Run)
async def get_run(run_id: str, _: None = Depends(require_api_token)) -> Run:
    run = await app.state.runtime.blackboard.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")
    return run


@app.get("/v1/runs/{run_id}/detail", response_model=RunDetail)
async def get_run_detail(run_id: str, _: None = Depends(require_api_token)) -> RunDetail:
    blackboard = app.state.runtime.blackboard
    run = await blackboard.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")
    return RunDetail(
        run=run,
        tasks=await blackboard.list_models(run.id, "tasks", ResearchTask),
        observations=await blackboard.list_models(run.id, "observations", Observation),
        claims=await blackboard.list_models(run.id, "claims", Claim),
        verifications=await blackboard.list_models(run.id, "verifications", Verification),
        final=await blackboard.get_final(run.id),
    )
