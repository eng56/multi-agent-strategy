from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import AsyncIterator

from fastapi import Depends, FastAPI, Header, HTTPException

from src.common.config import get_settings
from src.agents.workflow import deterministic_partial
from src.common.models import Budget, Claim, EventEnvelope, EventType, Observation, ResearchTask, Run, RunDetail, RunRequest, Verification
from src.integrations.validation import validate_user_configuration
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


@app.post("/v1/runs", response_model=Run, status_code=202)
async def create_run(request: RunRequest, _: None = Depends(require_api_token)) -> Run:
    runtime = app.state.runtime
    try:
        await validate_user_configuration(request, runtime.settings.openrouter_base_url, runtime.settings.tavily_base_url, runtime.settings.polygon_base_url)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"credential or model validation failed: {type(exc).__name__}") from exc
    request.tool_budget.tavily_credits_used = 1
    request.tool_budget.market_data_requests_used = 1
    run = Run(question=request.question, models=request.models, budget=Budget(limit_usd=request.llm_budget_usd, tools=request.tool_budget, max_parallel_agents=runtime.settings.default_max_parallel_agents))
    await runtime.credentials.put(run.id, request.credentials)
    await runtime.blackboard.put_run(run)
    runtime.events.publish(runtime.settings.runtime_topic, EventEnvelope(type=EventType.RUN_CREATED, run_id=run.id, producer="orchestrator-api"))
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
    if datetime.now(UTC) >= run.expires_at:
        await deterministic_partial(app.state.runtime, run.id, "The six-hour credential window expired.")
        run = await blackboard.get_run(run.id) or run
    return RunDetail(run=run, tasks=await blackboard.list_models(run.id, "tasks", ResearchTask), observations=await blackboard.list_models(run.id, "observations", Observation), claims=await blackboard.list_models(run.id, "claims", Claim), verifications=await blackboard.list_models(run.id, "verifications", Verification), final=await blackboard.get_final(run.id))
