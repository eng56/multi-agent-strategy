import json
import logging
from typing import Any
from uuid import UUID

from src.common.models import (
    AgentRole, Claim,
    EventEnvelope,
    EventType,
    FinalReport,
    Observation,
    ResearchTask,
    RunStatus,
    Verification,
)
from src.runtime import Runtime

SYSTEM = "You are a rigorous investment research agent. Return valid JSON only. Never invent sources."
logger = logging.getLogger(__name__)


def emit(runtime: Runtime, event_type: EventType, run_id: UUID, producer: str, **payload: Any) -> None:
    runtime.events.publish(
        runtime.settings.runtime_topic,
        EventEnvelope(type=event_type, run_id=run_id, producer=producer, payload=payload),
    )


async def plan(runtime: Runtime, event: EventEnvelope) -> None:
    run = await runtime.blackboard.get_run(event.run_id)
    if not run:
        return
    run.status = RunStatus.RUNNING
    await runtime.blackboard.put_run(run)
    result = await runtime.llm.json(
        run.id,
        AgentRole.PLANNER,
        "planner",
        SYSTEM,
        f"Create 3-5 independent research tasks for this investment question: {run.question}. "
        'Return {"tasks":[{"title":"...","question":"...","tool":"web_search|market_data"}]}. '
        "For market_data questions include a ticker symbol in the question.",
    )
    task_items = result.get("tasks") if isinstance(result.get("tasks"), list) else []
    created = 0
    for item in task_items:
        try:
            task = ResearchTask(run_id=run.id, **item)
        except Exception as exc:
            logger.warning("planner produced invalid task run_id=%s error=%s item=%s", run.id, exc, item)
            continue
        await runtime.blackboard.put_task(task)
        emit(runtime, EventType.TASK_CREATED, run.id, "planner-agent", task_id=str(task.id))
        created += 1
    if created == 0:
        reason = f"planner produced no usable tasks from output: {json.dumps(result, sort_keys=True)[:1000]}"
        logger.warning("%s run_id=%s", reason, run.id)
        run.status = RunStatus.FAILED
        run.failure_reason = reason
        await runtime.blackboard.put_run(run)


async def execute_tool(runtime: Runtime, event: EventEnvelope) -> None:
    tasks = await runtime.blackboard.list_models(event.run_id, "tasks", ResearchTask)
    task = next((value for value in tasks if str(value.id) == event.payload.get("task_id")), None)
    if not task:
        return
    if task.tool == "market_data":
        ticker_result = await runtime.llm.json(
            task.run_id,
            AgentRole.UTILITY,
            "ticker-extractor",
            SYSTEM,
            f'Extract the primary ticker from: {task.question}. Return {{"ticker":"..."}}.',
        )
        raw = await runtime.tools.market_data(task.run_id, ticker_result["ticker"])
        sources = [f"https://massive.com/stocks/{ticker_result['ticker']}"]
    else:
        raw = await runtime.tools.web_search(task.run_id, task.question)
        sources = [item["url"] for item in raw.get("results", []) if item.get("url")]
    artifact = runtime.artifacts.put_json(task.run_id, task.tool, raw)
    summary_result = await runtime.llm.json(
        task.run_id,
        AgentRole.RESEARCH,
        "tool-summary",
        SYSTEM,
        f"Summarize the most decision-relevant facts from this tool output. Return "
        f'{{"summary":"..."}}. Output: {json.dumps(raw)[:30000]}',
    )
    observation = Observation(
        run_id=task.run_id,
        task_id=task.id,
        tool=task.tool,
        summary=summary_result["summary"],
        artifact=artifact,
        sources=sources,
    )
    await runtime.blackboard.put_observation(observation)
    task.status = "completed"
    await runtime.blackboard.put_task(task)
    emit(
        runtime,
        EventType.OBSERVATION_CREATED,
        task.run_id,
        "tool-runner",
        observation_id=str(observation.id),
    )


async def create_claim(runtime: Runtime, event: EventEnvelope) -> None:
    observations = await runtime.blackboard.list_models(event.run_id, "observations", Observation)
    observation = next(
        (value for value in observations if str(value.id) == event.payload.get("observation_id")), None
    )
    if not observation:
        return
    result = await runtime.llm.json(
        event.run_id,
        AgentRole.RESEARCH,
        "claim-extractor",
        SYSTEM,
        f"Create one precise, decision-relevant claim supported only by this observation. "
        f'Return {{"statement":"...","confidence":0.0}}. Observation: {observation.summary}',
    )
    claim = Claim(
        run_id=event.run_id,
        task_id=observation.task_id,
        statement=result["statement"],
        confidence=result["confidence"],
        evidence_observation_ids=[observation.id],
        sources=observation.sources,
    )
    await runtime.blackboard.put_claim(claim)
    emit(runtime, EventType.CLAIM_CREATED, event.run_id, "worker-agents", claim_id=str(claim.id))


async def verify_claim(runtime: Runtime, event: EventEnvelope) -> None:
    claims = await runtime.blackboard.list_models(event.run_id, "claims", Claim)
    claim = next((value for value in claims if str(value.id) == event.payload.get("claim_id")), None)
    if not claim:
        return
    corroboration = await runtime.tools.web_search(event.run_id, claim.statement)
    result = await runtime.llm.json(
        event.run_id,
        AgentRole.VERIFIER,
        "claim-verifier",
        SYSTEM,
        f"Verify this claim using the corroborating results. Return "
        f'{{"verdict":"verified|rejected|uncertain","rationale":"...","confidence":0.0}}. '
        f"Claim: {claim.statement}. Results: {json.dumps(corroboration)[:30000]}",
    )
    sources = [item["url"] for item in corroboration.get("results", []) if item.get("url")]
    artifact = runtime.artifacts.put_json(event.run_id, "verification-search", corroboration)
    observation = Observation(
        run_id=event.run_id,
        task_id=claim.task_id,
        tool="verification_web_search",
        summary=result["rationale"],
        artifact=artifact,
        sources=sources,
    )
    await runtime.blackboard.put_observation(observation)
    verification = Verification(run_id=event.run_id, claim_id=claim.id, sources=sources, **result)
    await runtime.blackboard.put_verification(verification)
    emit(
        runtime,
        EventType.CLAIM_VERIFIED,
        event.run_id,
        "verifier-agent",
        verification_id=str(verification.id),
    )


async def aggregate(runtime: Runtime, event: EventEnvelope) -> None:
    tasks = await runtime.blackboard.list_models(event.run_id, "tasks", ResearchTask)
    verifications = await runtime.blackboard.list_models(event.run_id, "verifications", Verification)
    if not tasks or (len(verifications) < len(tasks) and not event.payload.get("force")) or await runtime.blackboard.get_final(event.run_id):
        return
    claims = await runtime.blackboard.list_models(event.run_id, "claims", Claim)
    verified_ids = {value.claim_id for value in verifications if value.verdict == "verified"}
    verified = [claim for claim in claims if claim.id in verified_ids]
    run = await runtime.blackboard.get_run(event.run_id)
    if not run:
        return
    result = await runtime.llm.json(
        event.run_id,
        AgentRole.AGGREGATOR,
        "aggregator",
        SYSTEM,
        f"Answer the investment question using only verified claims. Include risks, opportunities, "
        f"and limitations. Return {{\"answer\":\"...\"}}. Question: {run.question}. Claims: "
        f"{json.dumps([value.model_dump(mode='json') for value in verified])}",
    )
    final = FinalReport(
        run_id=event.run_id,
        answer=result["answer"],
        verified_claim_ids=[value.id for value in verified],
        sources=sorted({source for value in verified for source in value.sources}),
    )
    await runtime.blackboard.put_final(final)
    if run.models.judge:
        emit(runtime, EventType.FINAL_CREATED, event.run_id, "aggregator-agent")
    else:
        run.final_answer = final.answer
        run.status = RunStatus.COMPLETED
        await runtime.blackboard.put_run(run)


async def judge(runtime: Runtime, event: EventEnvelope) -> None:
    final = await runtime.blackboard.get_final(event.run_id)
    run = await runtime.blackboard.get_run(event.run_id)
    if not final or not run:
        return
    result = await runtime.llm.json(
        event.run_id,
        AgentRole.JUDGE,
        "judge",
        SYSTEM,
        f"Score this answer for evidence, completeness, and usefulness. Return "
        f'{{"score":0.0,"feedback":"..."}}. Question: {run.question}. Answer: {final.answer}',
    )
    final.judge_score = result["score"]
    final.judge_feedback = result["feedback"]
    run.final_answer = final.answer
    run.status = RunStatus.COMPLETED
    await runtime.blackboard.put_final(final)
    await runtime.blackboard.put_run(run)


HANDLERS = {
    "planner-agent": {EventType.RUN_CREATED: plan},
    "tool-runner": {EventType.TASK_CREATED: execute_tool},
    "worker-agents": {EventType.OBSERVATION_CREATED: create_claim},
    "verifier-agent": {EventType.CLAIM_CREATED: verify_claim},
    "aggregator-agent": {EventType.CLAIM_VERIFIED: aggregate},
    "judge-agent": {EventType.FINAL_CREATED: judge},
}


async def deterministic_partial(runtime: Runtime, run_id: UUID, reason: str) -> None:
    """Complete without another LLM call when expiry or protected budgets prevent synthesis."""
    run = await runtime.blackboard.get_run(run_id)
    if not run or run.status in {RunStatus.COMPLETED, RunStatus.PARTIAL_BUDGET_EXHAUSTED}:
        return
    claims = await runtime.blackboard.list_models(run_id, "claims", Claim)
    verifications = await runtime.blackboard.list_models(run_id, "verifications", Verification)
    verified_ids = {item.claim_id for item in verifications if item.verdict == "verified"}
    verified = [claim for claim in claims if claim.id in verified_ids]
    findings = "\n".join(f"- {claim.statement}" for claim in verified) or "- No claims were verified before the run stopped."
    answer = f"# Partial Investment Research Report\n\nReason: {reason}\n\n## Verified findings\n{findings}\n\nThis deterministic report was assembled without an additional LLM call."
    final = FinalReport(run_id=run_id, answer=answer, verified_claim_ids=[claim.id for claim in verified], sources=sorted({source for claim in verified for source in claim.sources}), partial=True)
    run.final_answer = answer
    run.status = RunStatus.PARTIAL_BUDGET_EXHAUSTED
    await runtime.blackboard.put_final(final)
    await runtime.blackboard.put_run(run)
