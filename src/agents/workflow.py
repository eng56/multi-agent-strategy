import json
import logging
from typing import Any
from uuid import UUID

from src.agents.state import (
    branch_for_task,
    infer_semantic_branch,
    tags_for_text,
    text_matches_any,
    title_from_branch,
)
from src.common.budget import BudgetExceeded
from src.common.models import (
    ActionStatus,
    AgentRole,
    AgentSpec,
    AgentStatus,
    Artifact,
    ArtifactStatus,
    ArtifactType,
    Claim,
    EventEnvelope,
    EventType,
    FinalReport,
    InformationGain,
    Observation,
    OrganizationPlan,
    PrincipalAction,
    PrincipalActionType,
    ResearchTask,
    Run,
    RunStatus,
    Verification,
    VisibilityScope,
)
from src.integrations.llm import LLMOutputError
from src.runtime import Runtime

SYSTEM = (
    "You are a rigorous investment research agent. Return valid JSON only. Never invent sources."
)
logger = logging.getLogger(__name__)


def emit(
    runtime: Runtime, event_type: EventType, run_id: UUID, producer: str, **payload: Any
) -> None:
    runtime.events.publish(
        runtime.settings.runtime_topic,
        EventEnvelope(type=event_type, run_id=run_id, producer=producer, payload=payload),
    )


async def persist_action(
    runtime: Runtime,
    run_id: UUID,
    action_type: PrincipalActionType,
    reason: str,
    *,
    required_role: str | None = None,
    target_branch: str | None = None,
    expected_information_gain: InformationGain = InformationGain.MEDIUM,
    estimated_cost: float = 0,
    priority: int = 5,
    status: ActionStatus = ActionStatus.EXECUTED,
    producer: str = "principal-policy",
) -> PrincipalAction:
    action = PrincipalAction(
        run_id=run_id,
        action_type=action_type,
        reason=reason,
        required_role=required_role,
        target_branch=target_branch,
        expected_information_gain=expected_information_gain,
        estimated_cost=estimated_cost,
        priority=priority,
        status=status,
    )
    await runtime.blackboard.put_principal_action(action)
    emit(runtime, EventType.PRINCIPAL_ACTION_CREATED, run_id, producer, action_id=str(action.id))
    return action


async def persist_artifact(runtime: Runtime, artifact: Artifact, producer: str) -> Artifact:
    await runtime.blackboard.put_artifact(artifact)
    emit(
        runtime, EventType.ARTIFACT_CREATED, artifact.run_id, producer, artifact_id=str(artifact.id)
    )
    return artifact


def concise_exception(exc: Exception, limit: int = 240) -> str:
    return f"{type(exc).__name__}: {str(exc)[:limit]}"


async def fail_task(
    runtime: Runtime,
    task: ResearchTask,
    branch: str,
    stage: str,
    exc: Exception,
    *,
    producer: str,
) -> None:
    reason = f"Task failed during {stage}: {task.title} ({concise_exception(exc)})"
    logger.warning(
        "task failed run_id=%s task_id=%s stage=%s error=%s",
        task.run_id,
        task.id,
        stage,
        concise_exception(exc),
    )
    task.status = "failed"
    await runtime.blackboard.put_task(task)
    await persist_action(
        runtime,
        task.run_id,
        PrincipalActionType.REQUEST_TOOL_CALL,
        reason,
        required_role="tool_runner" if producer == "tool-runner" else "research_agent",
        target_branch=branch,
        expected_information_gain=InformationGain.LOW,
        priority=6,
        status=ActionStatus.FAILED,
        producer=producer,
    )
    emit(
        runtime,
        EventType.TASK_FAILED,
        task.run_id,
        producer,
        task_id=str(task.id),
        stage=stage,
        reason=reason,
    )
    await fail_run_if_all_tasks_failed(runtime, task.run_id)


async def fail_run_if_all_tasks_failed(runtime: Runtime, run_id: UUID) -> None:
    tasks = await runtime.blackboard.list_models(run_id, "tasks", ResearchTask)
    if not tasks or any(task.status != "failed" for task in tasks):
        return
    if await runtime.blackboard.get_final(run_id):
        return
    verifications = await runtime.blackboard.list_models(run_id, "verifications", Verification)
    if any(verification.verdict == "verified" for verification in verifications):
        return
    run = await runtime.blackboard.get_run(run_id)
    if not run or run.status in {
        RunStatus.COMPLETED,
        RunStatus.PARTIAL_BUDGET_EXHAUSTED,
        RunStatus.FAILED,
    }:
        return
    reason = (
        f"All {len(tasks)} research task(s) failed before any verified evidence was produced."
    )
    run.status = RunStatus.FAILED
    run.failure_reason = reason
    run.final_answer = None
    await persist_action(
        runtime,
        run_id,
        PrincipalActionType.STOP_RUN,
        reason,
        required_role="principal_policy",
        target_branch="root",
        expected_information_gain=InformationGain.LOW,
        priority=4,
        producer="principal-policy",
    )
    await runtime.blackboard.put_run(run)


def text_from_model_field(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def score_from_model_field(value: Any) -> float:
    score = float(value)
    if score > 1 and score <= 10:
        score /= 10
    return max(0, min(1, score))


def role_budget(run: Run, role: AgentRole) -> float:
    policy = getattr(run.models, role.value, None)
    if not policy or policy.cap_usd is None:
        return 0
    return policy.cap_usd


def semantic_branches(question: str) -> list[str]:
    branches: list[str] = []
    checks = [
        ("market/gold", ("gold", "xau")),
        ("market/fx", ("usd", "dxy", "fx", "eur/usd", "eurusd", "usdjpy", "usd/jpy")),
        ("market/oil", ("oil", "crude", "wti", "brent")),
        (
            "market/equities",
            ("equities", "equity", "stocks", "stock", "spx", "s&p", "nasdaq", "sap"),
        ),
        ("macro/rates", ("rates", "fed", "cut", "cuts", "hike", "hikes", "yield", "yields")),
        ("macro/inflation", ("inflation", "cpi", "pce")),
        ("market/crypto", ("crypto", "bitcoin", "btc", "ethereum", "eth")),
        ("market/bonds", ("bond", "bonds", "treasury")),
    ]
    for branch, needles in checks:
        if text_matches_any(question, needles):
            branches.append(branch)
    return branches or ["research/general"]


FALLBACK_TASKS: dict[str, tuple[str, str, str]] = {
    "market/gold": (
        "Gold reaction to surprise Fed cut",
        "Find evidence on 1-week to 3-month gold performance and drivers after surprise Fed cuts.",
        "web_search",
    ),
    "market/fx": (
        "US dollar reaction to surprise Fed cut",
        "Find evidence on DXY or broad USD performance after surprise Fed cuts and lower rate differentials.",
        "web_search",
    ),
    "market/equities": (
        "US equities reaction to surprise Fed cut",
        "Find evidence on S&P 500 performance after surprise Fed cuts, distinguishing growth scare vs easing support.",
        "web_search",
    ),
    "macro/rates": (
        "Macro context for surprise Fed cut",
        "Find evidence on real yields, inflation expectations, recession risk, and policy path after surprise Fed cuts.",
        "web_search",
    ),
    "macro/inflation": (
        "Inflation context for surprise Fed cut",
        "Find evidence on inflation expectations and CPI or PCE conditions around surprise Fed cuts.",
        "web_search",
    ),
    "market/oil": (
        "Oil reaction to surprise Fed cut",
        "Find evidence on WTI or Brent performance after surprise Fed cuts and changing growth expectations.",
        "web_search",
    ),
    "market/crypto": (
        "Crypto reaction to surprise Fed cut",
        "Find evidence on Bitcoin or Ethereum performance after surprise Fed cuts and liquidity expectations.",
        "web_search",
    ),
    "market/bonds": (
        "Bond-market reaction to surprise Fed cut",
        "Find evidence on Treasury yields and bond returns after surprise Fed cuts.",
        "web_search",
    ),
    "research/general": (
        "General evidence for investment question",
        "Find decision-relevant evidence, catalysts, risks, and historical context for the investment question.",
        "web_search",
    ),
}


def deterministic_task_items(
    question: str, organization: OrganizationPlan | None
) -> list[dict[str, str]]:
    branches = [
        branch
        for branch in (organization.branches if organization else semantic_branches(question))
        if not branch.startswith(("root", "trust/", "synthesis/"))
    ]
    seen: set[str] = set()
    task_items: list[dict[str, str]] = []
    for branch in branches or ["research/general"]:
        if branch in seen:
            continue
        seen.add(branch)
        title, task_question, tool = FALLBACK_TASKS.get(branch, FALLBACK_TASKS["research/general"])
        task_items.append({"title": title, "question": task_question, "tool": tool})
    return task_items


def compact_planner_prompt(question: str) -> str:
    return (
        "Return JSON only, no markdown fences. Create exactly 4 tasks for this investment question. "
        "Each task must have title <= 80 chars, question <= 220 chars, and tool exactly "
        "web_search or market_data. Prefer web_search unless a ticker price series is required. "
        'Schema: {"tasks":[{"title":"...","question":"...","tool":"web_search"}]}. '
        f"Question: {question}"
    )


def planner_failure_reason(
    error: Exception | None, *, retry_used: bool, fallback_used: bool
) -> str:
    snippet = str(error or "<no model output>")[:300]
    return (
        "planner failed to create usable tasks "
        f"type={type(error).__name__ if error else 'NoUsableTasks'} role=planner name=planner "
        f"retry_used={retry_used} deterministic_fallback_used={fallback_used} bad_output={snippet}"
    )


def role_for_branch(branch: str) -> tuple[str, str, list[str]]:
    if branch.startswith("macro/"):
        return f"{branch.split('/')[-1]} agent", "macro_research_agent", ["web_search"]
    if branch == "market/fx":
        return "fx agent", "fx_research_agent", ["web_search", "market_data"]
    if branch.startswith("market/"):
        return (
            f"{branch.split('/')[-1]} agent",
            "asset_research_agent",
            ["web_search", "market_data"],
        )
    return "research agent", "asset_research_agent", ["web_search", "market_data"]


def build_organization(run: Run) -> OrganizationPlan:
    root = AgentSpec(
        run_id=run.id,
        name="root principal",
        role_template="principal_policy",
        branch="root",
        domain="control",
        objective="Design and steer a budgeted research organization for the user's question.",
        allowed_tools=[],
        retrieval_tags=["principal", "control"],
        local_budget_usd=role_budget(run, AgentRole.PLANNER),
        visibility_scope=VisibilityScope.PUBLIC_VERIFIED,
        can_request_spawn=True,
        status=AgentStatus.ACTIVE,
    )
    specs = [root]
    worker_ids: list[UUID] = []
    branches = semantic_branches(run.question)
    per_worker_budget = role_budget(run, AgentRole.RESEARCH) / max(1, len(branches))
    for branch in branches:
        name, template, tools = role_for_branch(branch)
        spec = AgentSpec(
            run_id=run.id,
            parent_id=root.id,
            name=name,
            role_template=template,
            branch=branch,
            domain=branch.split("/")[0],
            objective=f"Research {title_from_branch(branch)} evidence relevant to: {run.question}",
            allowed_tools=tools,
            retrieval_tags=tags_for_text(f"{branch} {run.question}"),
            local_budget_usd=per_worker_budget,
            visibility_scope=VisibilityScope.TEAM,
            status=AgentStatus.ACTIVE,
        )
        specs.append(spec)
        worker_ids.append(spec.id)
    verifier = AgentSpec(
        run_id=run.id,
        parent_id=root.id,
        name="source verifier",
        role_template="source_verifier_agent",
        branch="trust/source_verifier",
        domain="trust",
        objective="Verify or dispute claims against independent sources before synthesis.",
        allowed_tools=["web_search"],
        retrieval_tags=["trust", "source"],
        local_budget_usd=role_budget(run, AgentRole.VERIFIER),
        visibility_scope=VisibilityScope.PUBLIC_UNVERIFIED,
        status=AgentStatus.ACTIVE,
    )
    aggregator = AgentSpec(
        run_id=run.id,
        parent_id=root.id,
        name="aggregator",
        role_template="aggregator_agent",
        branch="synthesis/aggregator",
        domain="synthesis",
        objective="Synthesize trusted claims into a decision-grade final report.",
        allowed_tools=[],
        retrieval_tags=["synthesis", "final"],
        local_budget_usd=role_budget(run, AgentRole.AGGREGATOR),
        visibility_scope=VisibilityScope.PUBLIC_VERIFIED,
        status=AgentStatus.ACTIVE,
    )
    specs.extend([verifier, aggregator])
    verifier_ids = [verifier.id]
    aggregator_ids = [aggregator.id]
    judge_ids: list[UUID] = []
    if run.models.judge:
        judge = AgentSpec(
            run_id=run.id,
            parent_id=root.id,
            name="judge",
            role_template="judge_agent",
            branch="synthesis/judge",
            domain="synthesis",
            objective="Score the final answer for evidence, usefulness, calibration, and traceability.",
            allowed_tools=[],
            retrieval_tags=["judge", "payoff"],
            local_budget_usd=role_budget(run, AgentRole.JUDGE),
            visibility_scope=VisibilityScope.PUBLIC_VERIFIED,
            status=AgentStatus.ACTIVE,
        )
        specs.append(judge)
        judge_ids.append(judge.id)
    return OrganizationPlan(
        run_id=run.id,
        root_agent_id=root.id,
        agent_specs=specs,
        branches=[spec.branch for spec in specs],
        manager_agents=[root.id],
        worker_agents=worker_ids,
        verifier_agents=verifier_ids,
        aggregator_agents=aggregator_ids,
        judge_agents=judge_ids,
        dependencies={spec.id: [root.id] for spec in specs if spec.id != root.id},
        budget_allocation={
            "planner": role_budget(run, AgentRole.PLANNER),
            "research": role_budget(run, AgentRole.RESEARCH),
            "verifier": role_budget(run, AgentRole.VERIFIER),
            "aggregator": role_budget(run, AgentRole.AGGREGATOR),
            "judge": role_budget(run, AgentRole.JUDGE) if run.models.judge else 0,
        },
        stop_conditions=[
            "budget exhausted",
            "all planned tasks verified and synthesized",
            "judge completed",
        ],
    )


async def persist_organization(runtime: Runtime, run: Run) -> OrganizationPlan:
    organization = build_organization(run)
    await runtime.blackboard.put_organization_plan(organization)
    emit(
        runtime,
        EventType.ORGANIZATION_PLAN_CREATED,
        run.id,
        "planner-agent",
        organization_plan_id=str(organization.root_agent_id),
    )
    for spec in organization.agent_specs:
        await runtime.blackboard.put_agent_spec(spec)
        emit(
            runtime,
            EventType.AGENT_SPEC_CREATED,
            run.id,
            "planner-agent",
            agent_spec_id=str(spec.id),
        )
    await persist_action(
        runtime,
        run.id,
        PrincipalActionType.SPAWN_AGENT,
        f"Created logical organization with {len(organization.agent_specs)} agent specs across semantic branches.",
        required_role="principal_policy",
        target_branch="root",
        expected_information_gain=InformationGain.HIGH,
        priority=9,
        producer="planner-agent",
    )
    return organization


async def stop_run(runtime: Runtime, run_id: UUID, reason: str) -> None:
    run = await runtime.blackboard.get_run(run_id)
    if not run or run.status in {
        RunStatus.COMPLETED,
        RunStatus.PARTIAL_BUDGET_EXHAUSTED,
        RunStatus.FAILED,
    }:
        return
    logger.warning("stopping run_id=%s reason=%s", run_id, reason)
    run.status = RunStatus.FAILED
    run.failure_reason = reason
    await runtime.blackboard.put_run(run)


async def plan(runtime: Runtime, event: EventEnvelope) -> None:
    run = await runtime.blackboard.get_run(event.run_id)
    if not run:
        return
    run.status = RunStatus.RUNNING
    await runtime.blackboard.put_run(run)
    organization = await persist_organization(runtime, run)

    retry_used = False
    fallback_used = False
    last_error: Exception | None = None
    result: dict[str, Any] | None = None
    for name, prompt in (
        (
            "planner",
            f"Create 3-5 independent research tasks for this investment question: {run.question}. "
            'Return {"tasks":[{"title":"...","question":"...","tool":"web_search|market_data"}]}. '
            "For market_data questions include a ticker symbol in the question.",
        ),
        ("planner-retry-compact", compact_planner_prompt(run.question)),
    ):
        try:
            result = await runtime.llm.json(run.id, AgentRole.PLANNER, name, SYSTEM, prompt)
            last_error = None
            break
        except LLMOutputError as exc:
            last_error = exc
            if name == "planner":
                retry_used = True
                logger.warning(
                    "planner JSON parse failed run_id=%s retrying compact prompt error=%s",
                    run.id,
                    str(exc)[:300],
                )
                continue
            logger.warning(
                "planner compact retry failed run_id=%s using deterministic fallback error=%s",
                run.id,
                str(exc)[:300],
            )

    task_items = result.get("tasks") if result and isinstance(result.get("tasks"), list) else []
    if not task_items:
        fallback_used = True
        task_items = deterministic_task_items(run.question, organization)

    created = 0
    for item in task_items:
        try:
            task = ResearchTask(run_id=run.id, **item)
        except Exception as exc:
            logger.warning(
                "planner produced invalid task run_id=%s error=%s item=%s", run.id, exc, item
            )
            continue
        await runtime.blackboard.put_task(task)
        await persist_action(
            runtime,
            run.id,
            PrincipalActionType.ASSIGN_TASK,
            f"Assigned research task: {task.title}",
            required_role="research_agent",
            target_branch=branch_for_task(task),
            expected_information_gain=InformationGain.HIGH,
            priority=8,
            producer="planner-agent",
        )
        emit(runtime, EventType.TASK_CREATED, run.id, "planner-agent", task_id=str(task.id))
        created += 1
    if created == 0:
        await stop_run(
            runtime,
            run.id,
            planner_failure_reason(last_error, retry_used=retry_used, fallback_used=fallback_used),
        )


async def execute_tool(runtime: Runtime, event: EventEnvelope) -> None:
    tasks = await runtime.blackboard.list_models(event.run_id, "tasks", ResearchTask)
    task = next((value for value in tasks if str(value.id) == event.payload.get("task_id")), None)
    if not task or task.status != "created":
        return
    branch = branch_for_task(
        task, await runtime.blackboard.list_models(event.run_id, "agent_specs", AgentSpec)
    )
    stage = "tool_execution"
    try:
        if task.tool == "market_data":
            stage = "ticker_extraction"
            ticker_result = await runtime.llm.json(
                task.run_id,
                AgentRole.UTILITY,
                "ticker-extractor",
                SYSTEM,
                f'Extract the primary ticker from: {task.question}. Return {{"ticker":"..."}}.',
            )
            stage = "tool_execution"
            raw = await runtime.tools.market_data(task.run_id, ticker_result["ticker"])
            sources = [f"https://massive.com/stocks/{ticker_result['ticker']}"]
        else:
            raw = await runtime.tools.web_search(task.run_id, task.question)
            sources = [item["url"] for item in raw.get("results", []) if item.get("url")]
        stage = "raw_artifact_persistence"
        artifact_pointer = runtime.artifacts.put_json(task.run_id, task.tool, raw)
        stage = "tool_summary"
        summary_result = await runtime.llm.json(
            task.run_id,
            AgentRole.RESEARCH,
            "tool-summary",
            SYSTEM,
            f"Summarize the most decision-relevant facts from this tool output. Return "
            f'{{"summary":"..."}}. Output: {json.dumps(raw)[:30000]}',
        )
    except BudgetExceeded:
        raise
    except Exception as exc:
        await fail_task(runtime, task, branch, stage, exc, producer="tool-runner")
        return
    observation = Observation(
        run_id=task.run_id,
        task_id=task.id,
        tool=task.tool,
        summary=summary_result["summary"],
        artifact=artifact_pointer,
        sources=sources,
    )
    await runtime.blackboard.put_observation(observation)
    await persist_artifact(
        runtime,
        Artifact(
            run_id=task.run_id,
            artifact_type=ArtifactType.OBSERVATION,
            branch=branch,
            text_or_summary=observation.summary,
            tags=tags_for_text(f"{task.title} {task.question} {observation.summary}"),
            visibility=VisibilityScope.PUBLIC_UNVERIFIED,
            status=ArtifactStatus.UNVERIFIED,
            source_refs=observation.sources,
            legacy_object_type="observation",
            legacy_object_id=observation.id,
        ),
        "tool-runner",
    )
    task.status = "completed"
    await runtime.blackboard.put_task(task)
    await persist_action(
        runtime,
        task.run_id,
        PrincipalActionType.REQUEST_TOOL_CALL,
        f"Executed {task.tool} for task: {task.title}",
        required_role="tool_runner",
        target_branch=branch,
        expected_information_gain=InformationGain.HIGH,
        priority=7,
        producer="tool-runner",
    )
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
        (value for value in observations if str(value.id) == event.payload.get("observation_id")),
        None,
    )
    if not observation:
        return
    tasks = await runtime.blackboard.list_models(event.run_id, "tasks", ResearchTask)
    task = next((value for value in tasks if value.id == observation.task_id), None)
    branch = branch_for_task(task) if task else infer_semantic_branch(observation.summary)
    try:
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
    except BudgetExceeded:
        raise
    except Exception as exc:
        if not task:
            raise
        await fail_task(runtime, task, branch, "claim_generation", exc, producer="worker-agents")
        return
    await runtime.blackboard.put_claim(claim)
    observation_artifacts = await runtime.blackboard.list_models(
        event.run_id, "artifacts", Artifact
    )
    depends_on = [
        value.id
        for value in observation_artifacts
        if value.artifact_type == ArtifactType.OBSERVATION
        and value.legacy_object_type == "observation"
        and value.legacy_object_id == observation.id
    ]
    await persist_artifact(
        runtime,
        Artifact(
            run_id=event.run_id,
            artifact_type=ArtifactType.CLAIM,
            branch=branch,
            text_or_summary=claim.statement,
            tags=tags_for_text(claim.statement),
            visibility=VisibilityScope.PUBLIC_UNVERIFIED,
            status=ArtifactStatus.UNVERIFIED,
            confidence=claim.confidence,
            source_refs=claim.sources,
            legacy_object_type="claim",
            legacy_object_id=claim.id,
            depends_on_artifact_ids=depends_on[:1],
        ),
        "worker-agents",
    )
    emit(runtime, EventType.CLAIM_CREATED, event.run_id, "worker-agents", claim_id=str(claim.id))


def verification_search_query(statement: str) -> str:
    normalized = " ".join(statement.split())
    return normalized[:280].rsplit(" ", 1)[0] or normalized[:280]


def failed_verification_result(exc: Exception) -> dict[str, Any]:
    return {
        "verdict": "uncertain",
        "rationale": (
            "Automated claim verification failed, so this claim remains disputed instead of "
            f"being treated as verified: {type(exc).__name__}: {str(exc)[:240]}"
        ),
        "confidence": 0.0,
    }


async def verify_claim(runtime: Runtime, event: EventEnvelope) -> None:
    claims = await runtime.blackboard.list_models(event.run_id, "claims", Claim)
    claim = next(
        (value for value in claims if str(value.id) == event.payload.get("claim_id")), None
    )
    if not claim:
        return
    try:
        corroboration = await runtime.tools.web_search(
            event.run_id, verification_search_query(claim.statement)
        )
    except BudgetExceeded:
        raise
    except Exception as exc:
        logger.warning(
            "verification search failed run_id=%s claim_id=%s error=%s",
            event.run_id,
            claim.id,
            exc,
        )
        corroboration = {"results": [], "error": f"{type(exc).__name__}: {str(exc)[:500]}"}
        result = failed_verification_result(exc)
    else:
        try:
            result = await runtime.llm.json(
                event.run_id,
                AgentRole.VERIFIER,
                "claim-verifier",
                SYSTEM,
                f"Verify this claim using the corroborating results. Return "
                f'{{"verdict":"verified|rejected|uncertain","rationale":"...","confidence":0.0}}. '
                f"Claim: {claim.statement}. Results: {json.dumps(corroboration)[:30000]}",
            )
        except BudgetExceeded:
            raise
        except Exception as exc:
            logger.warning(
                "verification model failed run_id=%s claim_id=%s error=%s",
                event.run_id,
                claim.id,
                concise_exception(exc),
            )
            result = failed_verification_result(exc)
    sources = [item["url"] for item in corroboration.get("results", []) if item.get("url")]
    artifact_pointer = runtime.artifacts.put_json(
        event.run_id, "verification-search", corroboration
    )
    observation = Observation(
        run_id=event.run_id,
        task_id=claim.task_id,
        tool="verification_web_search",
        summary=result["rationale"],
        artifact=artifact_pointer,
        sources=sources,
    )
    await runtime.blackboard.put_observation(observation)
    verification = Verification(run_id=event.run_id, claim_id=claim.id, sources=sources, **result)
    await runtime.blackboard.put_verification(verification)
    await persist_artifact(
        runtime,
        Artifact(
            run_id=event.run_id,
            artifact_type=ArtifactType.OBSERVATION,
            branch="trust/source_verifier",
            text_or_summary=observation.summary,
            tags=tags_for_text(observation.summary),
            visibility=VisibilityScope.PUBLIC_UNVERIFIED,
            status=ArtifactStatus.UNVERIFIED,
            source_refs=observation.sources,
            legacy_object_type="observation",
            legacy_object_id=observation.id,
        ),
        "verifier-agent",
    )
    artifacts = await runtime.blackboard.list_models(event.run_id, "artifacts", Artifact)
    claim_artifacts = [
        value
        for value in artifacts
        if value.artifact_type == ArtifactType.CLAIM
        and value.legacy_object_type == "claim"
        and value.legacy_object_id == claim.id
    ]
    status = {
        "verified": ArtifactStatus.VERIFIED,
        "rejected": ArtifactStatus.REJECTED,
        "uncertain": ArtifactStatus.DISPUTED,
    }[verification.verdict]
    for claim_artifact in claim_artifacts:
        claim_artifact.status = status
        if verification.verdict == "verified":
            claim_artifact.visibility = VisibilityScope.PUBLIC_VERIFIED
        await runtime.blackboard.put_artifact(claim_artifact)
    claim_artifact_ids = [value.id for value in claim_artifacts]
    await persist_artifact(
        runtime,
        Artifact(
            run_id=event.run_id,
            artifact_type=ArtifactType.VERIFICATION,
            branch="trust/source_verifier",
            text_or_summary=verification.rationale,
            tags=tags_for_text(verification.rationale),
            visibility=VisibilityScope.PUBLIC_VERIFIED
            if verification.verdict == "verified"
            else VisibilityScope.PUBLIC_UNVERIFIED,
            status=status,
            confidence=verification.confidence,
            source_refs=verification.sources,
            legacy_object_type="verification",
            legacy_object_id=verification.id,
            supports_artifact_ids=claim_artifact_ids[:1]
            if verification.verdict == "verified"
            else [],
            contradicts_artifact_ids=claim_artifact_ids[:1]
            if verification.verdict == "rejected"
            else [],
        ),
        "verifier-agent",
    )
    await persist_action(
        runtime,
        event.run_id,
        PrincipalActionType.REQUEST_VERIFICATION,
        f"Verified claim candidate: {claim.statement[:120]}",
        required_role="source_verifier_agent",
        target_branch="trust/source_verifier",
        expected_information_gain=InformationGain.MEDIUM,
        priority=6,
        producer="verifier-agent",
    )
    emit(
        runtime,
        EventType.CLAIM_VERIFIED,
        event.run_id,
        "verifier-agent",
        verification_id=str(verification.id),
    )


async def aggregate(runtime: Runtime, event: EventEnvelope) -> None:
    tasks = await runtime.blackboard.list_models(event.run_id, "tasks", ResearchTask)
    verifications = await runtime.blackboard.list_models(
        event.run_id, "verifications", Verification
    )
    completed_tasks = [task for task in tasks if task.status == "completed"]
    pending_tasks = [task for task in tasks if task.status == "created"]
    if (
        not tasks
        or (
            not event.payload.get("force")
            and (pending_tasks or len(verifications) < len(completed_tasks))
        )
        or await runtime.blackboard.get_final(event.run_id)
    ):
        return
    claims = await runtime.blackboard.list_models(event.run_id, "claims", Claim)
    verified_ids = {value.claim_id for value in verifications if value.verdict == "verified"}
    verified = [claim for claim in claims if claim.id in verified_ids]
    if not verified and all(task.status == "failed" for task in tasks):
        await fail_run_if_all_tasks_failed(runtime, event.run_id)
        return
    run = await runtime.blackboard.get_run(event.run_id)
    if not run:
        return
    result = await runtime.llm.json(
        event.run_id,
        AgentRole.AGGREGATOR,
        "aggregator",
        SYSTEM,
        f"Answer the investment question using only verified claims. Include risks, opportunities, "
        f'and limitations. Return {{"answer":"..."}}. Question: {run.question}. Claims: '
        f"{json.dumps([value.model_dump(mode='json') for value in verified])}",
    )
    final = FinalReport(
        run_id=event.run_id,
        answer=text_from_model_field(result["answer"]),
        verified_claim_ids=[value.id for value in verified],
        sources=sorted({source for value in verified for source in value.sources}),
    )
    await runtime.blackboard.put_final(final)
    await persist_artifact(
        runtime,
        Artifact(
            run_id=event.run_id,
            artifact_type=ArtifactType.FINAL_REPORT,
            branch="synthesis/aggregator",
            text_or_summary=final.answer,
            tags=["synthesis", "final"],
            visibility=VisibilityScope.PUBLIC_VERIFIED,
            status=ArtifactStatus.VERIFIED,
            source_refs=final.sources,
            legacy_object_type="final_report",
            legacy_object_id=event.run_id,
        ),
        "aggregator-agent",
    )
    await persist_action(
        runtime,
        event.run_id,
        PrincipalActionType.REQUEST_AGGREGATION,
        "Synthesized verified claims into a final report.",
        required_role="aggregator_agent",
        target_branch="synthesis/aggregator",
        expected_information_gain=InformationGain.MEDIUM,
        priority=5,
        producer="aggregator-agent",
    )
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
    final.judge_score = score_from_model_field(result["score"])
    final.judge_feedback = text_from_model_field(result["feedback"])
    run.final_answer = final.answer
    run.status = RunStatus.COMPLETED
    await runtime.blackboard.put_final(final)
    await persist_artifact(
        runtime,
        Artifact(
            run_id=event.run_id,
            artifact_type=ArtifactType.JUDGE_FEEDBACK,
            branch="synthesis/judge",
            text_or_summary=final.judge_feedback or "Judge completed without feedback.",
            tags=["judge", "payoff"],
            visibility=VisibilityScope.PUBLIC_VERIFIED,
            status=ArtifactStatus.VERIFIED,
            confidence=final.judge_score,
            source_refs=final.sources,
            legacy_object_type="judge_feedback",
            legacy_object_id=event.run_id,
        ),
        "judge-agent",
    )
    await persist_action(
        runtime,
        event.run_id,
        PrincipalActionType.STOP_RUN,
        "Completed run after judge payoff score was recorded.",
        required_role="judge_agent",
        target_branch="synthesis/judge",
        expected_information_gain=InformationGain.LOW,
        priority=3,
        producer="judge-agent",
    )
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
    findings = (
        "\n".join(f"- {claim.statement}" for claim in verified)
        or "- No claims were verified before the run stopped."
    )
    answer = f"# Partial Investment Research Report\n\nReason: {reason}\n\n## Verified findings\n{findings}\n\nThis deterministic report was assembled without an additional LLM call."
    final = FinalReport(
        run_id=run_id,
        answer=answer,
        verified_claim_ids=[claim.id for claim in verified],
        sources=sorted({source for claim in verified for source in claim.sources}),
        partial=True,
    )
    run.final_answer = answer
    run.status = RunStatus.PARTIAL_BUDGET_EXHAUSTED
    await runtime.blackboard.put_final(final)
    await persist_artifact(
        runtime,
        Artifact(
            run_id=run_id,
            artifact_type=ArtifactType.FINAL_REPORT,
            branch="synthesis/aggregator",
            text_or_summary=answer,
            tags=["synthesis", "partial"],
            visibility=VisibilityScope.PUBLIC_VERIFIED,
            status=ArtifactStatus.VERIFIED,
            source_refs=final.sources,
            legacy_object_type="final_report",
            legacy_object_id=run_id,
        ),
        "aggregator-agent",
    )
    await persist_action(
        runtime,
        run_id,
        PrincipalActionType.REQUEST_AGGREGATION,
        f"Produced deterministic partial final report: {reason}",
        required_role="aggregator_agent",
        target_branch="synthesis/aggregator",
        expected_information_gain=InformationGain.LOW,
        priority=4,
        producer="aggregator-agent",
    )
    await runtime.blackboard.put_run(run)
