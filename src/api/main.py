import logging
import re
from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse

from src.common.config import get_settings
from src.agents.state import build_run_state
from src.common.models import (
    AgentSpec,
    Artifact,
    Budget,
    Claim,
    EventEnvelope,
    EventType,
    Observation,
    PrincipalAction,
    ResearchTask,
    Run,
    RunDetail,
    RunRequest,
    Verification,
)
from src.integrations.validation import validate_demo_configuration
from src.runtime import build_runtime

logger = logging.getLogger(__name__)

CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def scrub_jsonable(value):
    if isinstance(value, str):
        return CONTROL_CHAR_RE.sub(" ", value)
    if isinstance(value, list):
        return [scrub_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: scrub_jsonable(item) for key, item in value.items()}
    return value


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
            runtime.settings.market_data_base_url,
        )
    except Exception as exc:
        logger.exception("run provider validation failed question=%s", request.question[:120])
        raise HTTPException(
            status_code=400,
            detail=f"provider or model validation failed: {type(exc).__name__}: {exc}",
        ) from exc
    run = Run(
        question=request.question,
        models=request.models,
        budget=Budget(
            limit_usd=request.llm_budget_usd,
            tools=request.tool_budget,
            max_parallel_agents=runtime.settings.default_max_parallel_agents,
        ),
    )
    try:
        await runtime.blackboard.put_run(run)
    except Exception as exc:
        logger.exception("run blackboard write failed run_id=%s", run.id)
        raise HTTPException(
            status_code=502, detail=f"blackboard write failed: {type(exc).__name__}: {exc}"
        ) from exc
    try:
        runtime.events.publish(
            runtime.settings.runtime_topic,
            EventEnvelope(type=EventType.RUN_CREATED, run_id=run.id, producer="orchestrator-api"),
        )
    except Exception as exc:
        logger.exception("run event publish failed run_id=%s", run.id)
        raise HTTPException(
            status_code=502, detail=f"event publish failed: {type(exc).__name__}: {exc}"
        ) from exc
    logger.info("run created run_id=%s", run.id)
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
    try:
        tasks = await blackboard.list_models(run.id, "tasks", ResearchTask)
        observations = await blackboard.list_models(run.id, "observations", Observation)
        claims = await blackboard.list_models(run.id, "claims", Claim)
        verifications = await blackboard.list_models(run.id, "verifications", Verification)
        final = await blackboard.get_final(run.id)
        principal_actions = await blackboard.list_models(
            run.id, "principal_actions", PrincipalAction
        )
        agent_specs = await blackboard.list_models(run.id, "agent_specs", AgentSpec)
        artifacts = await blackboard.list_models(run.id, "artifacts", Artifact)
        organization_plan = await blackboard.get_organization_plan(run.id)
        detail = RunDetail(
            run=run,
            tasks=tasks,
            observations=observations,
            claims=claims,
            verifications=verifications,
            final=final,
            run_state=build_run_state(
                run, tasks, claims, verifications, final, agent_specs, artifacts, organization_plan
            ),
            principal_actions=principal_actions,
            agent_specs=agent_specs,
            artifacts=artifacts,
            organization_plan=organization_plan,
        )
        return JSONResponse(content=scrub_jsonable(detail.model_dump(mode="json")))
    except Exception as exc:
        logger.exception("run detail read failed run_id=%s", run_id)
        raise HTTPException(
            status_code=502, detail=f"run detail read failed: {type(exc).__name__}: {exc}"
        ) from exc
