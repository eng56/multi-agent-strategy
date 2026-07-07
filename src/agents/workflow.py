import json
import logging
import re
from typing import Any
from uuid import UUID

from src.agents.state import (
    FOLLOWUP_JUDGE_SCORE_THRESHOLD,
    MAX_FOLLOWUP_WAVES,
    build_run_state,
    branch_for_task,
    infer_semantic_branch,
    max_task_wave,
    should_reaggregate_after_followup,
    tags_for_text,
    text_matches_any,
    title_from_branch,
)
from src.agents.evidence_engine import (
    EvidenceBundle,
    EvidenceEngine,
    EvidenceItem,
    EvidenceRequest,
    SearchMode,
    SourceTier,
)
from src.agents.knowledge_router import select_context_for_agent
from src.agents.principal_runtime import evaluate_principal_policy
from src.agents.prompts import build_agent_instruction_block
from src.common.budget import BudgetExceeded
from src.common.failures import PermanentEventError, concise_exception, is_transient_failure
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
from src.markets.snapshot import MarketDataGap, MarketSnapshot, get_market_snapshot
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


async def evaluate_principal_policy_safely(
    runtime: Runtime, run_id: UUID, *, trigger: str
) -> None:
    try:
        await evaluate_principal_policy(runtime, run_id, trigger=trigger)
    except Exception:
        logger.exception(
            "principal policy evaluation failed run_id=%s trigger=%s",
            run_id,
            trigger,
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
        producer=producer,
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
    await evaluate_principal_policy_safely(
        runtime, task.run_id, trigger=f"task.failed:{stage}"
    )


async def skip_task_due_to_budget(
    runtime: Runtime,
    task: ResearchTask,
    branch: str,
    stage: str,
    exc: BudgetExceeded,
    *,
    producer: str,
) -> None:
    reason = f"Budget blocked {stage} for task: {task.title} ({concise_exception(exc)})"
    logger.warning(
        "budget skipped task run_id=%s task_id=%s branch=%s stage=%s reason=%s",
        task.run_id,
        task.id,
        branch,
        stage,
        concise_exception(exc),
    )
    task.status = "skipped_budget"
    task.reason = reason
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
        status=ActionStatus.REJECTED,
        producer=producer,
    )
    await evaluate_principal_policy_safely(
        runtime, task.run_id, trigger=f"task.skipped_budget:{stage}"
    )


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


def flatten_feedback_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        values: list[str] = []
        for item in value.values():
            values.extend(flatten_feedback_values(item))
        return values
    if isinstance(value, list):
        values = []
        for item in value:
            values.extend(flatten_feedback_values(item))
        return values
    return [str(value)]


def feedback_focus_items(feedback: str | None) -> list[str]:
    raw_values: list[str] = []
    if feedback:
        parsed: Any | None = None
        if feedback.lstrip().startswith(("{", "[")):
            try:
                parsed = json.loads(feedback)
            except json.JSONDecodeError:
                parsed = None
        raw_values = flatten_feedback_values(parsed) if parsed is not None else [feedback]

    segments: list[str] = []
    for value in raw_values:
        for segment in re.split(r"(?:\n+|(?<=[.!?;])\s+)", value):
            clean = re.sub(r"^[\s\-*\d.)]+", "", segment).strip()
            clean = re.sub(r"\s+", " ", clean)
            if clean:
                segments.append(clean)

    if not segments:
        segments.append("Strengthen weak evidence, coverage, or usefulness gaps.")

    deduped: list[str] = []
    seen: set[str] = set()
    for segment in segments:
        key = segment.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(segment)
        if len(deduped) == 3:
            break
    return deduped


def followup_title(focus: str) -> str:
    title_words = re.sub(r"[^A-Za-z0-9]+", " ", focus).strip().split()
    if not title_words:
        title_words = ["evidence", "gap"]
    return f"Follow-up: {' '.join(title_words[:8])}"[:80].rstrip()


def followup_task_items(run: Run, final: FinalReport, wave_number: int) -> list[dict[str, Any]]:
    score = final.judge_score if final.judge_score is not None else 0
    items: list[dict[str, Any]] = []
    for focus in feedback_focus_items(final.judge_feedback):
        short_focus = focus[:500].rstrip()
        reason = (
            f"Judge score {score:.2f} below {FOLLOWUP_JUDGE_SCORE_THRESHOLD:.2f}: "
            f"{short_focus}"
        )
        question = (
            "Find targeted evidence addressing this quality feedback: "
            f"{short_focus}. Original question: {run.question}"
        )
        items.append(
            {
                "title": followup_title(short_focus),
                "question": question[:1000].rstrip(),
                "tool": "web_search",
                "wave_number": wave_number,
                "reason": reason[:1000].rstrip(),
            }
        )
    return items


def followup_already_requested(
    tasks: list[ResearchTask], actions: list[PrincipalAction]
) -> bool:
    return max_task_wave(tasks) >= MAX_FOLLOWUP_WAVES or any(
        action.action_type == PrincipalActionType.REQUEST_FOLLOWUP for action in actions
    )


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


def agent_spec_for_branch(
    run_id: UUID,
    branch: str,
    agent_specs: list[AgentSpec],
    *,
    task: ResearchTask | None = None,
) -> AgentSpec:
    for spec in agent_specs:
        if spec.branch == branch:
            return spec
    name, role_template, tools = role_for_branch(branch)
    task_text = f"{task.title} {task.question}" if task else branch
    objective = (
        f"Research {title_from_branch(branch)} evidence for: {task.question}"
        if task
        else f"Research {title_from_branch(branch)} evidence."
    )
    return AgentSpec(
        run_id=run_id,
        name=name,
        role_template=role_template,
        branch=branch,
        domain=branch.split("/", 1)[0] if "/" in branch else None,
        objective=objective,
        allowed_tools=tools,
        retrieval_tags=tags_for_text(f"{branch} {task_text}"),
        visibility_scope=VisibilityScope.TEAM,
        status=AgentStatus.ACTIVE,
    )


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
    skeptic = AgentSpec(
        run_id=run.id,
        parent_id=root.id,
        name="skeptic",
        role_template="skeptic_agent",
        branch="trust/skeptic",
        domain="trust",
        objective="Challenge the emerging thesis with counterarguments, risks, and calibration checks.",
        allowed_tools=[],
        retrieval_tags=["trust", "skeptic", "counterargument", "risk"],
        visibility_scope=VisibilityScope.PUBLIC_VERIFIED,
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
    specs.extend([verifier, skeptic, aggregator])
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

    async def persist_planned_tasks(items: list[dict[str, Any]]) -> int:
        created_count = 0
        for item in items:
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
            created_count += 1
        return created_count

    created = await persist_planned_tasks(task_items)
    if created == 0 and not fallback_used:
        fallback_used = True
        task_items = deterministic_task_items(run.question, organization)
        logger.warning(
            "planner returned no valid tasks run_id=%s using deterministic fallback",
            run.id,
        )
        created = await persist_planned_tasks(task_items)

    if created == 0:
        await stop_run(
            runtime,
            run.id,
            planner_failure_reason(last_error, retry_used=retry_used, fallback_used=fallback_used),
        )
    await evaluate_principal_policy_safely(runtime, run.id, trigger="plan.finished")


async def request_followup_wave(
    runtime: Runtime,
    run: Run,
    final: FinalReport,
    existing_tasks: list[ResearchTask],
) -> int:
    wave_number = min(max_task_wave(existing_tasks) + 1, MAX_FOLLOWUP_WAVES)
    score = final.judge_score if final.judge_score is not None else 0
    reason = (
        f"Judge score {score:.2f} below {FOLLOWUP_JUDGE_SCORE_THRESHOLD:.2f}; "
        f"requesting targeted follow-up wave {wave_number}."
    )
    await persist_action(
        runtime,
        run.id,
        PrincipalActionType.REQUEST_FOLLOWUP,
        reason,
        required_role="principal_policy",
        target_branch="synthesis/judge",
        expected_information_gain=InformationGain.MEDIUM,
        priority=8,
        producer="judge-agent",
    )
    emit(
        runtime,
        EventType.FOLLOWUP_REQUESTED,
        run.id,
        "judge-agent",
        wave_number=wave_number,
        judge_score=score,
    )

    created = 0
    for item in followup_task_items(run, final, wave_number):
        task = ResearchTask(run_id=run.id, **item)
        await runtime.blackboard.put_task(task)
        await persist_action(
            runtime,
            run.id,
            PrincipalActionType.ASSIGN_TASK,
            f"Follow-up wave {wave_number}: {task.reason or task.title}",
            required_role="research_agent",
            target_branch=branch_for_task(task),
            expected_information_gain=InformationGain.HIGH,
            priority=8,
            producer="judge-agent",
        )
        emit(
            runtime,
            EventType.TASK_CREATED,
            run.id,
            "judge-agent",
            task_id=str(task.id),
            wave_number=wave_number,
            followup=True,
        )
        created += 1
    return created


def _has_zero_verified_recovery_capacity(run: Run) -> bool:
    budget_remaining = run.budget.limit_usd - run.budget.spent_usd - run.budget.reserved_usd
    tavily_remaining = run.budget.tools.tavily_max_credits - run.budget.tools.tavily_credits_used
    return budget_remaining > 0.05 and tavily_remaining > 0


def _disputed_claim_texts(claims: list[Claim], verifications: list[Verification]) -> list[str]:
    disputed_ids = {
        verification.claim_id for verification in verifications if verification.verdict != "verified"
    }
    return [claim.statement for claim in claims if claim.id in disputed_ids][:5]


def _failed_claim_generation_tasks(tasks: list[ResearchTask]) -> list[ResearchTask]:
    return [
        task
        for task in tasks
        if task.status == "failed"
        and task.reason
        and "claim_generation" in task.reason.lower()
    ]


def _claim_generation_recovery_candidates(tasks: list[ResearchTask]) -> list[ResearchTask]:
    failed_claim_tasks = _failed_claim_generation_tasks(tasks)
    if failed_claim_tasks:
        return failed_claim_tasks
    return [task for task in tasks if task.status == "failed"]


def zero_verified_followup_task_items(
    run: Run,
    tasks: list[ResearchTask],
    claims: list[Claim],
    verifications: list[Verification],
    wave_number: int,
) -> list[dict[str, Any]]:
    disputed_text = "\n".join(f"- {text}" for text in _disputed_claim_texts(claims, verifications))
    if not disputed_text:
        disputed_text = "- No disputed claim text was available; use observations and the original question."
    base_context = f"Original question: {run.question}\nDisputed candidate claims:\n{disputed_text}"
    items = [
        {
            "title": "Follow-up: atomic disputed claims",
            "question": (
                "Split the broad disputed claims into narrow, source-verifiable atomic "
                "claims. Avoid portfolio advice and causal overreach. "
                f"{base_context}"
            )[:1000].rstrip(),
            "tool": "web_search",
            "wave_number": wave_number,
            "reason": (
                "Zero claims passed verification; split broad disputed claims into "
                "atomic claims before retrying verification."
            ),
        },
        {
            "title": "Follow-up: primary dated sources",
            "question": (
                "Find primary or clearly date-bearing sources for Fed policy expectations, "
                "CME/Fed/FRED/Treasury/rates evidence, USD, gold, U.S. equities, and "
                "long-duration bonds. Prefer official, exchange, central-bank, filing, "
                f"or dated major-news sources. {base_context}"
            )[:1000].rstrip(),
            "tool": "web_search",
            "wave_number": wave_number,
            "reason": (
                "Zero claims passed verification; search primary and date-bearing sources "
                "to resolve temporal ambiguity."
            ),
        },
        {
            "title": "Follow-up: contradiction search",
            "question": (
                "Search for contradiction and counterargument evidence against the disputed "
                "claims, including recession-vs-disinflation regimes, conflicting USD "
                "signals, real-yield/gold caveats, and long-duration bond risks. "
                f"{base_context}"
            )[:1000].rstrip(),
            "tool": "web_search",
            "wave_number": wave_number,
            "reason": (
                "Zero claims passed verification; run targeted contradiction search before "
                "any terminal synthesis."
            ),
        },
    ]
    retry_targets = _claim_generation_recovery_candidates(tasks)
    if retry_targets:
        target_text = "; ".join(f"{task.title}: {task.question}" for task in retry_targets[:3])
        items.append(
            {
                "title": "Follow-up: retry failed claim generation",
                "question": (
                    "Retry failed claim-generation coverage with conservative, narrow, "
                    "source-grounded claims. If source evidence remains weak, state only "
                    f"the factual observation and caveats. Failed task context: {target_text}. "
                    f"Original question: {run.question}"
                )[:1000].rstrip(),
                "tool": "web_search",
                "wave_number": wave_number,
                "reason": (
                    "Zero claims passed verification and at least one task failed claim "
                    "generation; retry with deterministic conservative fallback behavior."
                ),
            }
        )
    return items


async def request_zero_verified_followup_wave(
    runtime: Runtime,
    run: Run,
    *,
    tasks: list[ResearchTask],
    claims: list[Claim],
    verifications: list[Verification],
) -> int:
    if max_task_wave(tasks) >= MAX_FOLLOWUP_WAVES:
        return 0
    if not _has_zero_verified_recovery_capacity(run):
        return 0
    if any(task.status == "created" for task in tasks):
        return 0
    if any(
        action.action_type == PrincipalActionType.REQUEST_FOLLOWUP
        and action.status == ActionStatus.EXECUTED
        for action in await runtime.blackboard.list_models(run.id, "principal_actions", PrincipalAction)
    ):
        return 0

    wave_number = min(max_task_wave(tasks) + 1, MAX_FOLLOWUP_WAVES)
    await persist_action(
        runtime,
        run.id,
        PrincipalActionType.REQUEST_FOLLOWUP,
        (
            "Zero claims passed verification while budget/tool capacity remains; "
            f"requesting targeted evidence recovery wave {wave_number} before terminal synthesis."
        ),
        required_role="principal_policy",
        target_branch="research/recovery",
        expected_information_gain=InformationGain.HIGH,
        priority=9,
        producer="aggregator-agent",
    )
    emit(
        runtime,
        EventType.FOLLOWUP_REQUESTED,
        run.id,
        "aggregator-agent",
        wave_number=wave_number,
        zero_verified_recovery=True,
    )

    created = 0
    for item in zero_verified_followup_task_items(run, tasks, claims, verifications, wave_number):
        task = ResearchTask(run_id=run.id, **item)
        await runtime.blackboard.put_task(task)
        await persist_action(
            runtime,
            run.id,
            PrincipalActionType.ASSIGN_TASK,
            f"Zero-verified follow-up wave {wave_number}: {task.reason or task.title}",
            required_role="research_agent",
            target_branch=branch_for_task(task),
            expected_information_gain=InformationGain.HIGH,
            priority=8,
            producer="aggregator-agent",
        )
        emit(
            runtime,
            EventType.TASK_CREATED,
            run.id,
            "aggregator-agent",
            task_id=str(task.id),
            wave_number=wave_number,
            followup=True,
            zero_verified_recovery=True,
        )
        created += 1
    if created:
        await evaluate_principal_policy_safely(
            runtime, run.id, trigger="zero_verified.followup_requested"
        )
    return created


def zero_verified_recovery_exhausted(tasks: list[ResearchTask]) -> bool:
    retry_targets = _claim_generation_recovery_candidates(tasks)
    return bool(retry_targets) and all(task.status == "failed" for task in retry_targets)


DEFAULT_EVIDENCE_MAX_SOURCES = 5
MAX_EVIDENCE_SNIPPETS = 3


class EvidenceProviderError(RuntimeError):
    def __init__(self, original: Exception) -> None:
        super().__init__(str(original))
        self.original = original


def evidence_max_sources(runtime: Runtime) -> int:
    settings = getattr(runtime, "settings", None)
    for name in ("evidence_max_sources", "max_evidence_sources"):
        value = getattr(settings, name, None)
        if isinstance(value, bool) or value is None:
            continue
        try:
            return max(0, min(25, int(value)))
        except (TypeError, ValueError):
            continue
    return DEFAULT_EVIDENCE_MAX_SOURCES


def build_evidence_request(
    runtime: Runtime, task: ResearchTask, branch: str
) -> EvidenceRequest:
    return EvidenceRequest(
        run_id=task.run_id,
        task_id=task.id,
        branch=branch,
        objective=task.question,
        search_mode=SearchMode.EXPLORATORY,
        max_sources=evidence_max_sources(runtime),
    )


async def search_evidence(runtime: Runtime, request: EvidenceRequest) -> EvidenceBundle:
    async def search_client(run_id: UUID, query: str) -> dict[str, Any]:
        try:
            return await runtime.tools.web_search(run_id, query)
        except BudgetExceeded:
            raise
        except Exception as exc:
            raise EvidenceProviderError(exc) from exc

    engine = EvidenceEngine(
        search_client,
        artifact_store=getattr(runtime, "artifacts", None),
    )
    return await engine.search(request)


def evidence_source_refs(bundle: EvidenceBundle) -> list[str]:
    source_refs: list[str] = []
    seen: set[str] = set()
    for item in bundle.items:
        if item.source_url in seen:
            continue
        seen.add(item.source_url)
        source_refs.append(item.source_url)
    return source_refs


def compact_text(value: str, max_chars: int = 1200) -> str:
    cleaned = " ".join(value.split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rsplit(" ", 1)[0] + "..."


def evidence_summary(bundle: EvidenceBundle, branch: str) -> str:
    source_count = len(bundle.items)
    strongest_items = sorted(
        bundle.items,
        key=lambda item: (item.quality_score, item.confidence),
        reverse=True,
    )[:MAX_EVIDENCE_SNIPPETS]
    snippets = [
        compact_text(
            f"{item.source_title} [{item.source_tier.value}, score "
            f"{item.quality_score}, {item.domain or 'unknown domain'}; "
            f"{item.source_quality_reason}]: "
            f"{item.snippet or 'provider returned no snippet'}",
            max_chars=320,
        )
        for item in strongest_items
    ]
    strongest = "; ".join(snippets) if snippets else "none; provider returned no sources"

    limitations: list[str] = []
    for item in strongest_items:
        for limitation in item.limitations:
            if limitation not in limitations:
                limitations.append(limitation)
    quality = bundle.source_quality_summary
    if source_count == 0:
        limitations.append(
            f"No provider results were returned for {quality.get('query_count', 0)} "
            "generated evidence query or queries."
        )
    missing_dates = quality.get("missing_published_at_count", 0)
    if missing_dates:
        limitations.append(f"{missing_dates} source(s) did not include a publication date.")
    if bundle.raw_result_artifact_uri is None:
        limitations.append("No raw provider artifact URI was returned by EvidenceEngine.")

    tier_counts = quality.get("source_tiers", {})
    tier_summary = ", ".join(
        f"{count} {tier}" for tier, count in tier_counts.items() if count
    )
    if not tier_summary:
        tier_summary = "no classified sources"
    quality_text = (
        f"{tier_summary}; average score {quality.get('average_quality_score', 0.0)}; "
        f"top score {quality.get('top_quality_score', 0)}"
    )
    limitation_text = "; ".join(limitations) if limitations else "no material limitations flagged"
    return (
        f"EvidenceEngine observation for branch {branch}: {source_count} source(s). "
        f"Strongest evidence snippets: {strongest}. "
        f"Source quality: {quality_text}. "
        f"Source limitations: {limitation_text}. "
        f"Branch: {branch}."
    )


def evidence_artifact_payload(
    bundle: EvidenceBundle, summary: str, source_refs: list[str]
) -> dict[str, Any]:
    return {
        "summary": compact_text(summary),
        "branch": bundle.request.branch,
        "source_refs": source_refs,
        "raw_result_artifact_uri": bundle.raw_result_artifact_uri,
        "evidence_bundle": bundle.model_dump(mode="json"),
    }


async def legacy_tool_observation(
    runtime: Runtime,
    task: ResearchTask,
    agent_spec: AgentSpec,
    raw: dict[str, Any],
    *,
    evidence_error: Exception | None = None,
) -> tuple[str, Any, list[str]]:
    sources = [item["url"] for item in raw.get("results", []) if item.get("url")]
    artifact_pointer = runtime.artifacts.put_json(task.run_id, task.tool, raw)
    summary_result = await runtime.llm.json(
        task.run_id,
        AgentRole.RESEARCH,
        "tool-summary",
        SYSTEM,
        f"{build_agent_instruction_block(agent_spec)}\n\n"
        f"Summarize the most decision-relevant facts from this tool output. Return "
        f'{{"summary":"..."}}. Output: {json.dumps(raw)[:30000]}',
    )
    summary = summary_result["summary"]
    if evidence_error is not None:
        summary = (
            f"EvidenceEngine failed ({concise_exception(evidence_error)}). "
            f"Legacy web_search fallback observation: {summary}"
        )
    return summary, artifact_pointer, sources


async def persist_market_data_gap(
    runtime: Runtime,
    task: ResearchTask,
    branch: str,
    gap: MarketDataGap,
) -> None:
    runtime.artifacts.put_json(task.run_id, "market_data_gap", gap.to_payload())
    await persist_artifact(
        runtime,
        Artifact(
            run_id=task.run_id,
            artifact_type=ArtifactType.DATA_GAP,
            branch=branch,
            text_or_summary=compact_text(gap.summary()),
            tags=sorted(
                set(
                    tags_for_text(f"{task.title} {task.question}")
                    + [
                        "data_gap",
                        "market_data_gap",
                        "tool:market_data",
                        branch.replace("/", ":"),
                    ]
                )
            ),
            visibility=VisibilityScope.PUBLIC_UNVERIFIED,
            status=ArtifactStatus.UNVERIFIED,
            source_refs=[],
            legacy_object_type="market_data_gap",
            legacy_object_id=task.id,
        ),
        "tool-runner",
    )
    task.status = "failed"
    await runtime.blackboard.put_task(task)
    await persist_action(
        runtime,
        task.run_id,
        PrincipalActionType.REQUEST_TOOL_CALL,
        (
            f"Recorded market data gap for task: {task.title}. "
            "No endpoint-valid candidate returned data, so no observation or claim "
            "extraction event was emitted."
        ),
        required_role="tool_runner",
        target_branch=branch,
        expected_information_gain=InformationGain.LOW,
        priority=6,
        status=ActionStatus.EXECUTED,
        producer="tool-runner",
    )
    await evaluate_principal_policy_safely(
        runtime, task.run_id, trigger="market_data.gap"
    )


async def execute_tool(runtime: Runtime, event: EventEnvelope) -> None:
    tasks = await runtime.blackboard.list_models(event.run_id, "tasks", ResearchTask)
    task = next((value for value in tasks if str(value.id) == event.payload.get("task_id")), None)
    if not task:
        raise PermanentEventError(
            f"task.created references missing task_id={event.payload.get('task_id')}"
        )
    if task.status != "created":
        return
    agent_specs = await runtime.blackboard.list_models(event.run_id, "agent_specs", AgentSpec)
    branch = branch_for_task(task, agent_specs)
    agent_spec = agent_spec_for_branch(event.run_id, branch, agent_specs, task=task)
    stage = "tool_execution"
    evidence_error: Exception | None = None
    try:
        if task.tool == "market_data":
            stage = "market_snapshot_resolution"
            run = await runtime.blackboard.get_run(event.run_id)

            async def fetch_candidate(candidate):
                return await runtime.tools.market_data(task.run_id, candidate.symbol)

            snapshot_result = await get_market_snapshot(
                f"{branch} {task.title}",
                f"{run.question if run else ''} {task.question}",
                fetch_market_data=fetch_candidate,
            )
            if isinstance(snapshot_result, MarketDataGap):
                await persist_market_data_gap(runtime, task, branch, snapshot_result)
                return
            snapshot: MarketSnapshot = snapshot_result
            raw = snapshot.to_payload()
            sources = snapshot.sources
            stage = "raw_artifact_persistence"
            artifact_pointer = runtime.artifacts.put_json(task.run_id, task.tool, raw)
            summary = snapshot.summary()
        else:
            stage = "evidence_request"
            evidence_request = build_evidence_request(runtime, task, branch)
            try:
                stage = "tool_execution"
                bundle = await search_evidence(runtime, evidence_request)
            except BudgetExceeded:
                raise
            except EvidenceProviderError as exc:
                stage = "tool_execution"
                raise exc.original
            except Exception as exc:
                if is_transient_failure(exc):
                    raise
                evidence_error = exc
                stage = "evidence_engine"
                logger.warning(
                    "EvidenceEngine failed; falling back to legacy web_search "
                    "run_id=%s task_id=%s error=%s",
                    task.run_id,
                    task.id,
                    concise_exception(exc),
                )
                try:
                    stage = "legacy_web_search_fallback"
                    raw = await runtime.tools.web_search(task.run_id, task.question)
                    stage = "raw_artifact_persistence"
                    summary, artifact_pointer, sources = await legacy_tool_observation(
                        runtime,
                        task,
                        agent_spec,
                        raw,
                        evidence_error=evidence_error,
                    )
                except BudgetExceeded:
                    raise
                except Exception as fallback_exc:
                    if is_transient_failure(fallback_exc):
                        raise
                    raise RuntimeError(
                        "EvidenceEngine failed "
                        f"({concise_exception(evidence_error)}); "
                        "legacy web_search fallback failed "
                        f"({concise_exception(fallback_exc)})"
                    ) from fallback_exc
            else:
                sources = evidence_source_refs(bundle)
                summary = evidence_summary(bundle, branch)
                stage = "evidence_bundle_artifact_persistence"
                artifact_pointer = runtime.artifacts.put_json(
                    task.run_id,
                    task.tool,
                    evidence_artifact_payload(bundle, summary, sources),
                )
    except BudgetExceeded as exc:
        await skip_task_due_to_budget(runtime, task, branch, stage, exc, producer="tool-runner")
        raise
    except Exception as exc:
        if is_transient_failure(exc):
            raise
        await fail_task(runtime, task, branch, stage, exc, producer="tool-runner")
        return
    observation = Observation(
        run_id=task.run_id,
        task_id=task.id,
        tool=task.tool,
        summary=summary,
        artifact=artifact_pointer,
        sources=sources,
    )
    await runtime.blackboard.put_observation(observation)
    artifact_tags = tags_for_text(f"{task.title} {task.question} {observation.summary}")
    if task.tool == "web_search":
        artifact_tags = sorted(
            set(
                artifact_tags
                + [
                    "evidence_engine"
                    if evidence_error is None
                    else "legacy_web_search_fallback",
                    "tool:web_search",
                    branch.replace("/", ":"),
                ]
            )
        )
    elif task.tool == "market_data":
        artifact_tags = sorted(
            set(artifact_tags + ["market_snapshot", "tool:market_data", branch.replace("/", ":")])
        )
    await persist_artifact(
        runtime,
        Artifact(
            run_id=task.run_id,
            artifact_type=ArtifactType.OBSERVATION,
            branch=branch,
            text_or_summary=compact_text(observation.summary),
            tags=artifact_tags,
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
        (
            f"Executed {task.tool} for task: {task.title}"
            if evidence_error is None
            else (
                f"Executed {task.tool} for task: {task.title} via legacy fallback "
                f"after EvidenceEngine failure: {concise_exception(evidence_error)}"
            )
        ),
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
    await evaluate_principal_policy_safely(
        runtime, task.run_id, trigger="observation.created"
    )


def is_data_gap_observation(observation: Observation, artifacts: list[Artifact]) -> bool:
    summary = observation.summary.casefold()
    if "market data gap" in summary or "tool failure" in summary:
        return True
    return any(
        artifact.legacy_object_type == "observation"
        and artifact.legacy_object_id == observation.id
        and (
            artifact.artifact_type == ArtifactType.DATA_GAP
            or "data_gap" in artifact.tags
            or "market_data_gap" in artifact.tags
            or "tool_failure" in artifact.tags
        )
        for artifact in artifacts
    )


def _market_snapshot_claim_statement(observation: Observation) -> str | None:
    summary = " ".join(observation.summary.split())
    if not summary.startswith("MarketSnapshot for "):
        return None
    selected_match = re.search(
        r"selected\s+([A-Z0-9./=-]+)\s+\(([^)]+)\)\s+via\s+([^;]+);\s+(\d+)\s+result",
        summary,
    )
    if not selected_match:
        return compact_text(
            f"Market data snapshot returned: {summary}. This is a point-in-time market "
            "data fact and does not establish Fed-cut causality.",
            max_chars=700,
        )
    symbol, asset_class, provider_endpoint, result_count = selected_match.groups()
    close_match = re.search(r"Latest available close/price:\s*([0-9.,-]+)", summary)
    volume_match = re.search(r"Latest available volume:\s*([0-9.,-]+)", summary)
    details = [
        f"Market data snapshot selected {symbol} ({asset_class}) via {provider_endpoint}",
        f"returned {result_count} result(s)",
    ]
    if close_match:
        details.append(f"latest available close/price was {close_match.group(1)}")
    if volume_match:
        details.append(f"latest available volume was {volume_match.group(1)}")
    return (
        "; ".join(details)
        + ". This is a point-in-time market data fact and does not establish Fed-cut causality."
    )


def _fallback_claim_statement(observation: Observation, branch: str) -> str:
    market_snapshot_claim = _market_snapshot_claim_statement(observation)
    if market_snapshot_claim:
        return market_snapshot_claim
    return compact_text(
        f"Evidence observation for {branch}: {observation.summary}",
        max_chars=700,
    )


def _fallback_claim(
    event: EventEnvelope,
    observation: Observation,
    branch: str,
    *,
    confidence: float = 0.35,
) -> Claim:
    if _market_snapshot_claim_statement(observation):
        confidence = max(confidence, 0.55)
    return Claim(
        run_id=event.run_id,
        task_id=observation.task_id,
        statement=_fallback_claim_statement(observation, branch),
        confidence=confidence,
        evidence_observation_ids=[observation.id],
        sources=observation.sources,
    )


async def create_claim(runtime: Runtime, event: EventEnvelope) -> None:
    observations = await runtime.blackboard.list_models(event.run_id, "observations", Observation)
    observation = next(
        (value for value in observations if str(value.id) == event.payload.get("observation_id")),
        None,
    )
    if not observation:
        raise PermanentEventError(
            f"observation.created references missing observation_id={event.payload.get('observation_id')}"
        )
    tasks = await runtime.blackboard.list_models(event.run_id, "tasks", ResearchTask)
    task = next((value for value in tasks if value.id == observation.task_id), None)
    branch = branch_for_task(task) if task else infer_semantic_branch(observation.summary)
    observation_artifacts = await runtime.blackboard.list_models(
        event.run_id, "artifacts", Artifact
    )
    if is_data_gap_observation(observation, observation_artifacts):
        await persist_action(
            runtime,
            event.run_id,
            PrincipalActionType.REQUEST_VERIFICATION,
            (
                "Skipped claim extraction for data-gap/tool-failure observation; "
                "provider gaps must remain caveats or open questions, not investment claims."
            ),
            required_role="research_agent",
            target_branch=branch,
            expected_information_gain=InformationGain.LOW,
            priority=4,
            producer="worker-agents",
        )
        await evaluate_principal_policy_safely(
            runtime, event.run_id, trigger="claim.skipped_data_gap"
        )
        return
    claim: Claim
    if _market_snapshot_claim_statement(observation):
        claim = _fallback_claim(event, observation, branch)
        await persist_action(
            runtime,
            event.run_id,
            PrincipalActionType.REQUEST_VERIFICATION,
            (
                "Created deterministic factual claim from market snapshot; skipped LLM "
                "claim extraction to avoid causal overreach."
            ),
            required_role="research_agent",
            target_branch=branch,
            expected_information_gain=InformationGain.LOW,
            priority=4,
            producer="worker-agents",
        )
    else:
        try:
            result = await runtime.llm.json(
                event.run_id,
                AgentRole.RESEARCH,
                "claim-extractor",
                SYSTEM,
                f"Create one narrow, verifiable claim supported only by this observation. "
                f"Do not create portfolio advice, causal claims, or claims about data "
                f"availability unless directly stated by a source. "
                f'Return {{"statement":"...","confidence":0.0}}. Observation: '
                f"{observation.summary}",
            )
            claim = Claim(
                run_id=event.run_id,
                task_id=observation.task_id,
                statement=compact_text(text_from_model_field(result["statement"]), max_chars=700),
                confidence=score_from_model_field(result["confidence"]),
                evidence_observation_ids=[observation.id],
                sources=observation.sources,
            )
        except BudgetExceeded:
            raise
        except Exception as exc:
            if is_transient_failure(exc):
                raise
            if not task:
                raise
            if not isinstance(exc, (LLMOutputError, KeyError, TypeError, ValueError)):
                await fail_task(
                    runtime, task, branch, "claim_generation", exc, producer="worker-agents"
                )
                return
            claim = _fallback_claim(event, observation, branch)
            await persist_action(
                runtime,
                event.run_id,
                PrincipalActionType.REQUEST_VERIFICATION,
                (
                    "Claim extraction model returned invalid output; created deterministic "
                    f"fallback claim instead of failing the task: {concise_exception(exc)}"
                ),
                required_role="research_agent",
                target_branch=branch,
                expected_information_gain=InformationGain.LOW,
                priority=4,
                producer="worker-agents",
            )
    await runtime.blackboard.put_claim(claim)
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
    await evaluate_principal_policy_safely(runtime, event.run_id, trigger="claim.created")


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
        "supported_parts": [],
        "unsupported_parts": ["Automated verification did not complete."],
        "contradictions": [],
        "required_caveats": ["Verification failed; do not treat this claim as verified."],
        "source_quality_summary": "No verifier-reviewed source-quality summary was available.",
    }


STRONG_VERIFICATION_SOURCE_TIERS = {
    SourceTier.PRIMARY,
    SourceTier.HIGH_QUALITY_SECONDARY,
    SourceTier.NEWS,
}


def _claim_artifacts_for_claim(artifacts: list[Artifact], claim: Claim) -> list[Artifact]:
    return [
        value
        for value in artifacts
        if value.artifact_type == ArtifactType.CLAIM
        and value.legacy_object_type == "claim"
        and value.legacy_object_id == claim.id
    ]


def _verification_branch_for_claim(claim: Claim, claim_artifacts: list[Artifact]) -> str:
    for artifact in claim_artifacts:
        if artifact.branch:
            return artifact.branch
    return infer_semantic_branch(claim.statement)


def build_claim_verification_evidence_request(
    runtime: Runtime,
    claim: Claim,
    branch: str,
    search_mode: SearchMode,
) -> EvidenceRequest:
    max_sources = evidence_max_sources(runtime)
    if search_mode == SearchMode.CONTRADICTION:
        max_sources = min(max_sources, 3)
    return EvidenceRequest(
        run_id=claim.run_id,
        task_id=claim.task_id,
        branch=branch,
        objective=claim.statement,
        claim_id=claim.id,
        claim_text=claim.statement,
        search_mode=search_mode,
        max_sources=max_sources,
    )


def _unique_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        cleaned = value.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        unique.append(cleaned)
    return unique


def _linked_observations_for_claim(
    observations: list[Observation], claim: Claim
) -> list[Observation]:
    linked_ids = set(claim.evidence_observation_ids)
    return [observation for observation in observations if observation.id in linked_ids]


def _linked_observation_payload(observations: list[Observation]) -> list[dict[str, Any]]:
    return [
        {
            "observation_id": str(observation.id),
            "summary": compact_text(observation.summary, max_chars=1200),
            "sources": observation.sources,
        }
        for observation in observations
    ]


def _evidence_item_prompt_payload(item: EvidenceItem) -> dict[str, Any]:
    return {
        "ref": item.source_url,
        "url": item.source_url,
        "title": item.source_title,
        "snippet": compact_text(item.snippet or "provider returned no snippet", max_chars=1000),
        "support_type": item.support_type,
        "source_tier": item.source_tier.value,
        "quality_score": item.quality_score,
        "source_quality_reason": item.source_quality_reason,
        "publisher": item.publisher,
        "domain": item.domain,
        "published_at": item.published_at,
        "retrieval_confidence": item.confidence,
        "limitations": item.limitations[:5],
    }


def _evidence_items_prompt_payload(bundle: EvidenceBundle | None) -> list[dict[str, Any]]:
    if bundle is None:
        return []
    return [_evidence_item_prompt_payload(item) for item in bundle.items[:10]]


def _source_quality_summary_text(
    support_bundle: EvidenceBundle | None,
    contradiction_bundle: EvidenceBundle | None,
    contradiction_error: Exception | None = None,
) -> str:
    parts: list[str] = []
    for label, bundle in (
        ("support", support_bundle),
        ("contradiction/context", contradiction_bundle),
    ):
        if bundle is None:
            continue
        summary = bundle.source_quality_summary
        tiers = summary.get("source_tiers", {})
        tier_text = ", ".join(
            f"{count} {tier}" for tier, count in tiers.items() if count
        )
        if not tier_text:
            tier_text = "no classified sources"
        parts.append(
            f"{label}: {summary.get('item_count', len(bundle.items))} item(s), "
            f"{tier_text}, average score {summary.get('average_quality_score', 0.0)}, "
            f"top score {summary.get('top_quality_score', 0)}, "
            f"{summary.get('missing_published_at_count', 0)} missing publication date(s)"
        )
    if contradiction_error is not None:
        parts.append(
            "contradiction/context search unavailable: "
            f"{concise_exception(contradiction_error)}"
        )
    if not parts:
        return "No EvidenceEngine source-quality summary was available."
    return "; ".join(parts)


def _verification_evidence_item_refs(
    support_bundle: EvidenceBundle | None,
    contradiction_bundle: EvidenceBundle | None,
) -> list[str]:
    refs: list[str] = []
    for bundle in (support_bundle, contradiction_bundle):
        if bundle is None:
            continue
        refs.extend(item.source_url for item in bundle.items)
    return _unique_strings(refs)


def _verification_source_refs(
    linked_observations: list[Observation],
    support_bundle: EvidenceBundle | None,
    contradiction_bundle: EvidenceBundle | None,
) -> list[str]:
    refs = _verification_evidence_item_refs(support_bundle, contradiction_bundle)
    for observation in linked_observations:
        refs.extend(observation.sources)
    return _unique_strings(refs)


def _verification_prompt(
    claim: Claim,
    linked_observations: list[Observation],
    support_bundle: EvidenceBundle,
    contradiction_bundle: EvidenceBundle | None,
    contradiction_error: Exception | None = None,
) -> str:
    payload = {
        "claim_text": claim.statement,
        "linked_observation_summaries": _linked_observation_payload(linked_observations),
        "evidence_items": _evidence_items_prompt_payload(support_bundle),
        "contradictory_or_contextual_evidence": _evidence_items_prompt_payload(
            contradiction_bundle
        ),
        "source_quality_summary": _source_quality_summary_text(
            support_bundle, contradiction_bundle, contradiction_error
        ),
    }
    if contradiction_error is not None:
        payload["contradictory_or_contextual_evidence_error"] = concise_exception(
            contradiction_error
        )
    return (
        "Verify the claim using the linked observations and EvidenceEngine evidence items. "
        "Do not treat search results as generic support: support_type is inferred from the "
        "query mode, so reason over the actual snippets, URLs, source quality, "
        "contradictions, and limitations.\n\n"
        "Rules:\n"
        "- A claim cannot be verified based only on weak or unknown sources unless it is "
        "low-stakes contextual background.\n"
        "- If evidence supports only part of the claim, return uncertain or include "
        "required caveats.\n"
        "- If sources conflict, return uncertain or rejected depending on source strength.\n"
        "- Include material limitations in required_caveats.\n\n"
        "Return strict JSON with exactly this shape and no markdown: "
        '{"verdict":"verified|rejected|uncertain","rationale":"...",'
        '"confidence":0.0,"supported_parts":["..."],"unsupported_parts":["..."],'
        '"contradictions":["..."],"required_caveats":["..."],'
        '"source_quality_summary":"..."}.\n\n'
        f"Input: {json.dumps(payload, ensure_ascii=False, sort_keys=True)[:30000]}"
    )


def _empty_list_marker(value: str) -> bool:
    normalized = value.strip().strip(".").lower()
    if normalized in {
        "",
        "n/a",
        "na",
        "none",
        "not applicable",
        "not provided",
        "no caveats",
        "no contradictions",
        "no unsupported parts",
    }:
        return True
    return normalized.startswith("no material ") and any(
        token in normalized for token in ("caveat", "contradiction", "unsupported")
    )


def _string_list_from_model_field(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        strings = [compact_text(text_from_model_field(item), max_chars=700) for item in value]
    else:
        strings = [compact_text(text_from_model_field(value), max_chars=700)]
    return _unique_strings([item for item in strings if not _empty_list_marker(item)])


def _has_strong_supporting_evidence(bundle: EvidenceBundle | None) -> bool:
    if bundle is None:
        return False
    return any(
        item.source_tier in STRONG_VERIFICATION_SOURCE_TIERS and item.quality_score >= 50
        for item in bundle.items
    )


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def normalize_verification_result(
    raw_result: dict[str, Any],
    support_bundle: EvidenceBundle | None,
    contradiction_bundle: EvidenceBundle | None,
    contradiction_error: Exception | None = None,
) -> dict[str, Any]:
    verdict = str(raw_result.get("verdict", "uncertain")).strip().lower()
    if verdict not in {"verified", "rejected", "uncertain"}:
        verdict = "uncertain"
    rationale = compact_text(
        text_from_model_field(
            raw_result.get("rationale", "Verifier did not provide a rationale.")
        ),
        max_chars=4000,
    )
    try:
        confidence = score_from_model_field(raw_result.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    source_quality_summary = raw_result.get("source_quality_summary")
    if isinstance(source_quality_summary, str) and source_quality_summary.strip():
        quality_text = compact_text(source_quality_summary, max_chars=1200)
    else:
        quality_text = _source_quality_summary_text(
            support_bundle, contradiction_bundle, contradiction_error
        )

    result = {
        "verdict": verdict,
        "rationale": rationale,
        "confidence": confidence,
        "supported_parts": _string_list_from_model_field(
            raw_result.get("supported_parts", [])
        ),
        "unsupported_parts": _string_list_from_model_field(
            raw_result.get("unsupported_parts", [])
        ),
        "contradictions": _string_list_from_model_field(
            raw_result.get("contradictions", [])
        ),
        "required_caveats": _string_list_from_model_field(
            raw_result.get("required_caveats", [])
        ),
        "source_quality_summary": quality_text,
    }

    if contradiction_error is not None:
        _append_unique(
            result["required_caveats"],
            "Contradictory/contextual evidence search was unavailable.",
        )
    if result["unsupported_parts"] and not result["required_caveats"]:
        _append_unique(
            result["required_caveats"],
            "Evidence supports only part of the claim; unsupported parts remain caveated.",
        )
    if result["verdict"] == "verified" and result["contradictions"]:
        result["verdict"] = "uncertain"
        result["confidence"] = min(result["confidence"], 0.55)
        _append_unique(
            result["required_caveats"],
            "Verifier identified contradictions, so the claim cannot be public-verified.",
        )
    if result["verdict"] == "verified" and not _has_strong_supporting_evidence(
        support_bundle
    ):
        result["verdict"] = "uncertain"
        result["confidence"] = min(result["confidence"], 0.49)
        _append_unique(
            result["required_caveats"],
            "EvidenceEngine returned only weak, unknown, or missing supporting sources.",
        )
        result["rationale"] = compact_text(
            result["rationale"]
            + " Verification downgraded because source quality was insufficient.",
            max_chars=4000,
        )
    return result


def verification_search_payload(
    claim: Claim,
    linked_observations: list[Observation],
    support_bundle: EvidenceBundle | None,
    contradiction_bundle: EvidenceBundle | None,
    result: dict[str, Any],
    *,
    search_error: Exception | None = None,
    contradiction_error: Exception | None = None,
) -> dict[str, Any]:
    return {
        "claim_text": claim.statement,
        "linked_observations": _linked_observation_payload(linked_observations),
        "support_evidence_bundle": support_bundle.model_dump(mode="json")
        if support_bundle is not None
        else None,
        "contradictory_or_contextual_evidence_bundle": (
            contradiction_bundle.model_dump(mode="json")
            if contradiction_bundle is not None
            else None
        ),
        "search_error": concise_exception(search_error) if search_error else None,
        "contradiction_error": concise_exception(contradiction_error)
        if contradiction_error
        else None,
        "verifier_result": result,
    }


def verification_artifact_summary(verification: Verification) -> str:
    parts = [verification.rationale]
    if verification.supported_parts:
        parts.append("Supported parts: " + "; ".join(verification.supported_parts))
    if verification.unsupported_parts:
        parts.append("Unsupported parts: " + "; ".join(verification.unsupported_parts))
    if verification.contradictions:
        parts.append("Contradictions: " + "; ".join(verification.contradictions))
    if verification.required_caveats:
        parts.append("Required caveats: " + "; ".join(verification.required_caveats))
    if verification.source_quality_summary:
        parts.append("Source quality: " + verification.source_quality_summary)
    return compact_text("\n\n".join(parts), max_chars=4000)


def _run_state_requests_skeptic_review(run_state) -> bool:
    return any(
        action.action_type == PrincipalActionType.REQUEST_SKEPTIC_REVIEW
        for action in run_state.next_action_candidates
    )


async def request_skeptic_review(
    runtime: Runtime,
    run_id: UUID,
    *,
    reason: str,
    producer: str,
) -> None:
    await persist_action(
        runtime,
        run_id,
        PrincipalActionType.REQUEST_SKEPTIC_REVIEW,
        reason,
        required_role="skeptic_agent",
        target_branch="trust/skeptic",
        expected_information_gain=InformationGain.MEDIUM,
        priority=6,
        producer=producer,
    )
    emit(runtime, EventType.SKEPTIC_REVIEW_REQUESTED, run_id, producer)


def _skeptic_agent_spec(run_id: UUID, agent_specs: list[AgentSpec]) -> AgentSpec:
    return next(
        (
            spec
            for spec in agent_specs
            if spec.role_template == "skeptic_agent" or spec.branch == "trust/skeptic"
        ),
        None,
    ) or AgentSpec(
        run_id=run_id,
        name="skeptic",
        role_template="skeptic_agent",
        branch="trust/skeptic",
        domain="trust",
        objective="Challenge the emerging thesis with counterarguments, risks, and calibration checks.",
        retrieval_tags=["trust", "skeptic", "counterargument", "risk"],
        visibility_scope=VisibilityScope.PUBLIC_VERIFIED,
        status=AgentStatus.ACTIVE,
    )


def _counterargument_artifacts(artifacts: list[Artifact]) -> list[Artifact]:
    return [
        artifact
        for artifact in artifacts
        if artifact.artifact_type == ArtifactType.COUNTERARGUMENT
    ]


def _skeptic_feedback_text(result: dict[str, Any]) -> str:
    fields = [
        ("Strongest counterargument", "strongest_counterargument"),
        ("Evidence that would contradict the main thesis", "contradicting_evidence"),
        ("Risks / regime changes", "risks_regime_changes"),
        ("What would change conclusion", "what_would_change_conclusion"),
        ("Confidence / calibration comments", "confidence_calibration_comments"),
    ]
    return "\n\n".join(
        f"{label}: {text_from_model_field(result.get(key, 'Not provided.'))}"
        for label, key in fields
    )


async def _persist_skeptic_failure_artifact(
    runtime: Runtime,
    run_id: UUID,
    exc: Exception,
) -> Artifact:
    artifact = Artifact(
        run_id=run_id,
        artifact_type=ArtifactType.COUNTERARGUMENT,
        branch="trust/skeptic",
        text_or_summary=(
            "Skeptic review failed; aggregation may proceed without a completed "
            f"counterargument. Failure: {concise_exception(exc)}"
        ),
        tags=["trust", "skeptic", "counterargument", "failure", "synthesis"],
        visibility=VisibilityScope.PUBLIC_VERIFIED,
        status=ArtifactStatus.DISPUTED,
    )
    await persist_artifact(runtime, artifact, "skeptic-agent")
    await persist_action(
        runtime,
        run_id,
        PrincipalActionType.REQUEST_SKEPTIC_REVIEW,
        f"Skeptic review failed; allowing aggregation to proceed: {concise_exception(exc)}",
        required_role="skeptic_agent",
        target_branch="trust/skeptic",
        expected_information_gain=InformationGain.LOW,
        priority=6,
        status=ActionStatus.FAILED,
        producer="skeptic-agent",
    )
    return artifact


def _verified_claims_and_artifacts(
    claims: list[Claim],
    verifications: list[Verification],
    artifacts: list[Artifact],
) -> tuple[list[Claim], list[Artifact]]:
    verified_ids = {value.claim_id for value in verifications if value.verdict == "verified"}
    verified_claims = [claim for claim in claims if claim.id in verified_ids]
    verified_artifacts = [
        artifact
        for artifact in artifacts
        if artifact.status == ArtifactStatus.VERIFIED
        and artifact.artifact_type in {ArtifactType.CLAIM, ArtifactType.FORECAST}
    ]
    return verified_claims, verified_artifacts


async def verify_claim(runtime: Runtime, event: EventEnvelope) -> None:
    claims = await runtime.blackboard.list_models(event.run_id, "claims", Claim)
    claim = next(
        (value for value in claims if str(value.id) == event.payload.get("claim_id")), None
    )
    if not claim:
        raise PermanentEventError(
            f"claim.created references missing claim_id={event.payload.get('claim_id')}"
        )
    observations = await runtime.blackboard.list_models(event.run_id, "observations", Observation)
    linked_observations = _linked_observations_for_claim(observations, claim)
    artifacts = await runtime.blackboard.list_models(event.run_id, "artifacts", Artifact)
    claim_artifacts = _claim_artifacts_for_claim(artifacts, claim)
    branch = _verification_branch_for_claim(claim, claim_artifacts)
    support_bundle: EvidenceBundle | None = None
    contradiction_bundle: EvidenceBundle | None = None
    search_error: Exception | None = None
    contradiction_error: Exception | None = None
    try:
        support_bundle = await search_evidence(
            runtime,
            build_claim_verification_evidence_request(
                runtime, claim, branch, SearchMode.VERIFICATION
            ),
        )
    except BudgetExceeded:
        raise
    except EvidenceProviderError as exc:
        if is_transient_failure(exc.original):
            raise exc.original
        logger.warning(
            "verification evidence search failed run_id=%s claim_id=%s error=%s",
            event.run_id,
            claim.id,
            concise_exception(exc.original),
        )
        search_error = exc.original
        raw_result = failed_verification_result(exc.original)
    except Exception as exc:
        if is_transient_failure(exc):
            raise
        logger.warning(
            "verification evidence search failed run_id=%s claim_id=%s error=%s",
            event.run_id,
            claim.id,
            concise_exception(exc),
        )
        search_error = exc
        raw_result = failed_verification_result(exc)
    else:
        try:
            contradiction_bundle = await search_evidence(
                runtime,
                build_claim_verification_evidence_request(
                    runtime, claim, branch, SearchMode.CONTRADICTION
                ),
            )
        except BudgetExceeded:
            raise
        except EvidenceProviderError as exc:
            if is_transient_failure(exc.original):
                raise exc.original
            logger.warning(
                "verification contradiction search failed run_id=%s claim_id=%s error=%s",
                event.run_id,
                claim.id,
                concise_exception(exc.original),
            )
            contradiction_error = exc.original
        except Exception as exc:
            if is_transient_failure(exc):
                raise
            logger.warning(
                "verification contradiction search failed run_id=%s claim_id=%s error=%s",
                event.run_id,
                claim.id,
                concise_exception(exc),
            )
            contradiction_error = exc
        try:
            raw_result = await runtime.llm.json(
                event.run_id,
                AgentRole.VERIFIER,
                "claim-verifier",
                SYSTEM,
                _verification_prompt(
                    claim,
                    linked_observations,
                    support_bundle,
                    contradiction_bundle,
                    contradiction_error,
                ),
            )
        except BudgetExceeded:
            raise
        except Exception as exc:
            if is_transient_failure(exc):
                raise
            logger.warning(
                "verification model failed run_id=%s claim_id=%s error=%s",
                event.run_id,
                claim.id,
                concise_exception(exc),
            )
            raw_result = failed_verification_result(exc)
    result = normalize_verification_result(
        raw_result, support_bundle, contradiction_bundle, contradiction_error
    )
    evidence_item_refs = _verification_evidence_item_refs(support_bundle, contradiction_bundle)
    result["evidence_item_refs"] = evidence_item_refs
    sources = _verification_source_refs(linked_observations, support_bundle, contradiction_bundle)
    artifact_pointer = runtime.artifacts.put_json(
        event.run_id,
        "verification-search",
        verification_search_payload(
            claim,
            linked_observations,
            support_bundle,
            contradiction_bundle,
            result,
            search_error=search_error,
            contradiction_error=contradiction_error,
        ),
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
    status = {
        "verified": ArtifactStatus.VERIFIED,
        "rejected": ArtifactStatus.REJECTED,
        "uncertain": ArtifactStatus.DISPUTED,
    }[verification.verdict]
    for claim_artifact in claim_artifacts:
        claim_artifact.status = status
        if verification.verdict == "verified":
            claim_artifact.visibility = VisibilityScope.PUBLIC_VERIFIED
        elif claim_artifact.visibility == VisibilityScope.PUBLIC_VERIFIED:
            claim_artifact.visibility = VisibilityScope.PUBLIC_UNVERIFIED
        await runtime.blackboard.put_artifact(claim_artifact)
    claim_artifact_ids = [value.id for value in claim_artifacts]
    verification_summary = verification_artifact_summary(verification)
    await persist_artifact(
        runtime,
        Artifact(
            run_id=event.run_id,
            artifact_type=ArtifactType.VERIFICATION,
            branch="trust/source_verifier",
            text_or_summary=verification_summary,
            tags=tags_for_text(verification_summary),
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
    run = await runtime.blackboard.get_run(event.run_id)
    if run:
        tasks = await runtime.blackboard.list_models(event.run_id, "tasks", ResearchTask)
        latest_claims = await runtime.blackboard.list_models(event.run_id, "claims", Claim)
        latest_verifications = await runtime.blackboard.list_models(
            event.run_id, "verifications", Verification
        )
        latest_artifacts = await runtime.blackboard.list_models(
            event.run_id, "artifacts", Artifact
        )
        agent_specs = await runtime.blackboard.list_models(event.run_id, "agent_specs", AgentSpec)
        run_state = build_run_state(
            run,
            tasks,
            latest_claims,
            latest_verifications,
            agent_specs=agent_specs,
            artifacts=latest_artifacts,
        )
        if _run_state_requests_skeptic_review(run_state):
            await request_skeptic_review(
                runtime,
                event.run_id,
                reason=(
                    "Verified evidence is sufficient for a preliminary thesis; request an "
                    "adversarial counterargument before final synthesis."
                ),
                producer="verifier-agent",
            )
            await evaluate_principal_policy_safely(
                runtime, event.run_id, trigger=f"claim.{verification.verdict}"
            )
            return
    emit(
        runtime,
        EventType.CLAIM_VERIFIED,
        event.run_id,
        "verifier-agent",
        verification_id=str(verification.id),
    )
    await evaluate_principal_policy_safely(
        runtime, event.run_id, trigger=f"claim.{verification.verdict}"
    )


async def skeptic_review(runtime: Runtime, event: EventEnvelope) -> None:
    run = await runtime.blackboard.get_run(event.run_id)
    tasks = await runtime.blackboard.list_models(event.run_id, "tasks", ResearchTask)
    existing_final = await runtime.blackboard.get_final(event.run_id)
    if not run or (
        existing_final and not should_reaggregate_after_followup(existing_final, tasks)
    ):
        return
    claims = await runtime.blackboard.list_models(event.run_id, "claims", Claim)
    verifications = await runtime.blackboard.list_models(
        event.run_id, "verifications", Verification
    )
    artifacts = await runtime.blackboard.list_models(event.run_id, "artifacts", Artifact)
    if _counterargument_artifacts(artifacts):
        emit(runtime, EventType.CLAIM_VERIFIED, event.run_id, "skeptic-agent")
        return

    verified_claims, verified_artifacts = _verified_claims_and_artifacts(
        claims, verifications, artifacts
    )
    verified_knowledge_ids = {claim.id for claim in verified_claims} | {
        artifact.legacy_object_id or artifact.id for artifact in verified_artifacts
    }
    if len(verified_knowledge_ids) < 2:
        emit(runtime, EventType.CLAIM_VERIFIED, event.run_id, "skeptic-agent")
        return

    agent_specs = await runtime.blackboard.list_models(event.run_id, "agent_specs", AgentSpec)
    skeptic_spec = _skeptic_agent_spec(event.run_id, agent_specs)
    verified_claim_payload = [claim.model_dump(mode="json") for claim in verified_claims]
    verified_artifact_payload = [
        {
            "id": str(artifact.id),
            "type": artifact.artifact_type.value,
            "branch": artifact.branch,
            "summary": artifact.text_or_summary,
            "confidence": artifact.confidence,
            "sources": artifact.source_refs,
        }
        for artifact in verified_artifacts
    ]
    input_artifact_ids = [artifact.id for artifact in verified_artifacts]
    try:
        result = await runtime.llm.json(
            event.run_id,
            AgentRole.VERIFIER,
            "skeptic-counterargument",
            SYSTEM,
            f"{build_agent_instruction_block(skeptic_spec)}\n\n"
            "Challenge the emerging investment thesis before final aggregation. Use only "
            "the verified evidence supplied here; do not invent sources. Return "
            '{"strongest_counterargument":"...",'
            '"contradicting_evidence":"...",'
            '"risks_regime_changes":"...",'
            '"what_would_change_conclusion":"...",'
            '"confidence_calibration_comments":"..."}. '
            f"Question: {run.question}. Verified claims: "
            f"{json.dumps(verified_claim_payload)[:12000]}. Verified artifacts: "
            f"{json.dumps(verified_artifact_payload)[:12000]}",
        )
    except Exception as exc:
        await _persist_skeptic_failure_artifact(runtime, event.run_id, exc)
        emit(runtime, EventType.CLAIM_VERIFIED, event.run_id, "skeptic-agent")
        await evaluate_principal_policy_safely(
            runtime, event.run_id, trigger="skeptic.artifact.failed"
        )
        return

    artifact = Artifact(
        run_id=event.run_id,
        artifact_type=ArtifactType.COUNTERARGUMENT,
        branch="trust/skeptic",
        text_or_summary=_skeptic_feedback_text(result),
        tags=["trust", "skeptic", "counterargument", "risk", "synthesis"],
        visibility=VisibilityScope.PUBLIC_VERIFIED,
        status=ArtifactStatus.VERIFIED,
        source_refs=sorted(
            {
                source
                for claim in verified_claims
                for source in claim.sources
            }
            | {
                source
                for verified_artifact in verified_artifacts
                for source in verified_artifact.source_refs
            }
        ),
        depends_on_artifact_ids=input_artifact_ids[:12],
        contradicts_artifact_ids=input_artifact_ids[:12],
    )
    await persist_artifact(runtime, artifact, "skeptic-agent")
    await persist_action(
        runtime,
        event.run_id,
        PrincipalActionType.REQUEST_SKEPTIC_REVIEW,
        "Created adversarial counterargument artifact for final aggregation.",
        required_role="skeptic_agent",
        target_branch="trust/skeptic",
        expected_information_gain=InformationGain.MEDIUM,
        priority=6,
        producer="skeptic-agent",
    )
    emit(
        runtime,
        EventType.CLAIM_VERIFIED,
        event.run_id,
        "skeptic-agent",
        skeptic_artifact_id=str(artifact.id),
    )
    await evaluate_principal_policy_safely(
        runtime, event.run_id, trigger="skeptic.artifact.created"
    )


async def aggregate(runtime: Runtime, event: EventEnvelope) -> None:
    tasks = await runtime.blackboard.list_models(event.run_id, "tasks", ResearchTask)
    verifications = await runtime.blackboard.list_models(
        event.run_id, "verifications", Verification
    )
    pending_tasks = [task for task in tasks if task.status == "created"]
    existing_final = await runtime.blackboard.get_final(event.run_id)
    claims = await runtime.blackboard.list_models(event.run_id, "claims", Claim)
    checked_claim_ids = {verification.claim_id for verification in verifications}
    unchecked_claims = [claim for claim in claims if claim.id not in checked_claim_ids]
    if (
        not tasks
        or (not event.payload.get("force") and (pending_tasks or unchecked_claims))
        or (existing_final and not should_reaggregate_after_followup(existing_final, tasks))
    ):
        return
    verified_ids = {value.claim_id for value in verifications if value.verdict == "verified"}
    verified = [claim for claim in claims if claim.id in verified_ids]
    run = await runtime.blackboard.get_run(event.run_id)
    if not run:
        return
    agent_specs = await runtime.blackboard.list_models(event.run_id, "agent_specs", AgentSpec)
    aggregator_spec = next(
        (
            spec
            for spec in agent_specs
            if spec.role_template == "aggregator_agent" or spec.branch == "synthesis/aggregator"
        ),
        None,
    ) or AgentSpec(
        run_id=event.run_id,
        name="aggregator",
        role_template="aggregator_agent",
        branch="synthesis/aggregator",
        objective="Synthesize trusted claims into a decision-grade final report.",
    )
    artifacts = await runtime.blackboard.list_models(event.run_id, "artifacts", Artifact)
    run_state = build_run_state(
        run,
        tasks,
        claims,
        verifications,
        agent_specs=agent_specs or [aggregator_spec],
        artifacts=artifacts,
    )
    if not verified:
        if await request_zero_verified_followup_wave(
            runtime,
            run,
            tasks=tasks,
            claims=claims,
            verifications=verifications,
        ):
            return
        await _write_no_verified_evidence_final(
            runtime,
            run,
            tasks=tasks,
            claims=claims,
            verifications=verifications,
            artifacts=artifacts,
        )
        return
    if (
        _run_state_requests_skeptic_review(run_state)
        and not event.payload.get("force")
        and event.producer != "skeptic-agent"
    ):
        await request_skeptic_review(
            runtime,
            event.run_id,
            reason=(
                "Aggregation found enough verified evidence for a preliminary thesis but "
                "no counterargument artifact; request skeptic review first."
            ),
            producer="aggregator-agent",
        )
        return
    routed_artifacts = select_context_for_agent(
        agent_spec=aggregator_spec,
        run_state=run_state,
        artifacts=artifacts,
        tasks=tasks,
    )
    routed_context = [
        {
            "id": str(artifact.id),
            "type": artifact.artifact_type.value,
            "branch": artifact.branch,
            "status": artifact.status.value,
            "confidence": artifact.confidence,
            "summary": artifact.text_or_summary,
            "sources": artifact.source_refs,
        }
        for artifact in routed_artifacts
    ]
    skeptic_context = [
        {
            "id": str(artifact.id),
            "status": artifact.status.value,
            "confidence": artifact.confidence,
            "summary": artifact.text_or_summary,
            "sources": artifact.source_refs,
        }
        for artifact in _counterargument_artifacts(artifacts)
    ]
    result = await runtime.llm.json(
        event.run_id,
        AgentRole.AGGREGATOR,
        "aggregator",
        SYSTEM,
        f"{build_agent_instruction_block(aggregator_spec)}\n\n"
        f"Answer the investment question using verified claims as support. Explicitly "
        f"consider skeptic/counterargument context when discussing risks, regime changes, "
        f'calibration, and limitations. Return {{"answer":"..."}}. Question: {run.question}. Claims: '
        f"{json.dumps([value.model_dump(mode='json') for value in verified])}. "
        f"Routed artifact context: {json.dumps(routed_context)[:12000]}. "
        f"Skeptic/counterargument context: {json.dumps(skeptic_context)[:12000]}",
    )
    final = FinalReport(
        run_id=event.run_id,
        answer=text_from_model_field(result["answer"]),
        verified_claim_ids=[value.id for value in verified],
        sources=sorted({source for value in verified for source in value.sources}),
        wave_number=max_task_wave(tasks),
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
    await evaluate_principal_policy_safely(runtime, event.run_id, trigger="final.created")


async def judge(runtime: Runtime, event: EventEnvelope) -> None:
    final = await runtime.blackboard.get_final(event.run_id)
    run = await runtime.blackboard.get_run(event.run_id)
    if not final or not run:
        return
    tasks = await runtime.blackboard.list_models(event.run_id, "tasks", ResearchTask)
    actions = await runtime.blackboard.list_models(
        event.run_id, "principal_actions", PrincipalAction
    )
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
    run.final_answer = final.answer
    if final.judge_score < FOLLOWUP_JUDGE_SCORE_THRESHOLD:
        if not followup_already_requested(tasks, actions):
            run.status = RunStatus.RUNNING
            await runtime.blackboard.put_run(run)
            await request_followup_wave(runtime, run, final, tasks)
            await evaluate_principal_policy_safely(
                runtime, event.run_id, trigger="judge.completed"
            )
            return
        if should_reaggregate_after_followup(final, tasks):
            run.status = RunStatus.RUNNING
            await runtime.blackboard.put_run(run)
            await evaluate_principal_policy_safely(
                runtime, event.run_id, trigger="judge.completed"
            )
            return

    run.status = RunStatus.COMPLETED
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
    await evaluate_principal_policy_safely(runtime, event.run_id, trigger="judge.completed")


async def _write_no_verified_evidence_final(
    runtime: Runtime,
    run: Run,
    *,
    tasks: list[ResearchTask],
    claims: list[Claim],
    verifications: list[Verification],
    artifacts: list[Artifact],
) -> None:
    """Create a deterministic final when research produced no synthesis-safe claims."""
    answer = _no_verified_evidence_answer(run, tasks, claims, verifications, artifacts)
    caveated_sources = _caveated_evidence_sources(verifications, artifacts)
    caveated_dependencies = _caveated_evidence_artifact_ids(verifications, artifacts)
    final = FinalReport(
        run_id=run.id,
        answer=answer,
        verified_claim_ids=[],
        sources=caveated_sources,
        partial=True,
        wave_number=max_task_wave(tasks),
    )
    await runtime.blackboard.put_final(final)
    await persist_artifact(
        runtime,
        Artifact(
            run_id=run.id,
            artifact_type=ArtifactType.FINAL_REPORT,
            branch="synthesis/aggregator",
            text_or_summary=answer,
            tags=["synthesis", "final", "insufficient_verified_evidence", "caveated_evidence"],
            visibility=VisibilityScope.PUBLIC_VERIFIED,
            status=ArtifactStatus.VERIFIED,
            source_refs=final.sources,
            legacy_object_type="final_report",
            legacy_object_id=run.id,
            depends_on_artifact_ids=caveated_dependencies[:24],
        ),
        "aggregator-agent",
    )
    await persist_action(
        runtime,
        run.id,
        PrincipalActionType.REQUEST_AGGREGATION,
        "Produced deterministic final because no claim passed source verification.",
        required_role="aggregator_agent",
        target_branch="synthesis/aggregator",
        expected_information_gain=InformationGain.LOW,
        priority=5,
        producer="aggregator-agent",
    )
    await persist_action(
        runtime,
        run.id,
        PrincipalActionType.STOP_RUN,
        "Completed with an evidence-limited final: no synthesis-safe verified claims were available.",
        required_role="principal_policy",
        target_branch="synthesis/aggregator",
        expected_information_gain=InformationGain.LOW,
        priority=4,
        producer="aggregator-agent",
    )
    run.final_answer = answer
    run.status = RunStatus.COMPLETED
    run.failure_reason = None
    await runtime.blackboard.put_run(run)
    await evaluate_principal_policy_safely(runtime, run.id, trigger="final.no_verified")


def _unique_nonempty_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _caveated_evidence_sources(
    verifications: list[Verification],
    artifacts: list[Artifact],
) -> list[str]:
    non_verified_claim_ids = {
        verification.claim_id for verification in verifications if verification.verdict != "verified"
    }
    source_refs: list[str] = []
    for verification in verifications:
        if verification.verdict == "verified":
            continue
        source_refs.extend(verification.sources)
        source_refs.extend(verification.evidence_item_refs)
    for artifact in artifacts:
        if _is_caveated_evidence_artifact(artifact, non_verified_claim_ids):
            source_refs.extend(artifact.source_refs)
    return _unique_nonempty_strings(source_refs)


def _caveated_evidence_artifact_ids(
    verifications: list[Verification],
    artifacts: list[Artifact],
) -> list[UUID]:
    non_verified_claim_ids = {
        verification.claim_id for verification in verifications if verification.verdict != "verified"
    }
    ids: list[UUID] = []
    for artifact in artifacts:
        if _is_caveated_evidence_artifact(artifact, non_verified_claim_ids):
            ids.append(artifact.id)
    seen: set[UUID] = set()
    result: list[UUID] = []
    for artifact_id in ids:
        if artifact_id in seen:
            continue
        seen.add(artifact_id)
        result.append(artifact_id)
    return result


def _is_caveated_evidence_artifact(
    artifact: Artifact,
    non_verified_claim_ids: set[UUID],
) -> bool:
    if artifact.artifact_type not in {ArtifactType.CLAIM, ArtifactType.VERIFICATION}:
        return False
    if artifact.status in {ArtifactStatus.DISPUTED, ArtifactStatus.REJECTED}:
        return True
    return bool(
        artifact.legacy_object_type == "claim"
        and artifact.legacy_object_id in non_verified_claim_ids
    )


def _no_verified_evidence_answer(
    run: Run,
    tasks: list[ResearchTask],
    claims: list[Claim],
    verifications: list[Verification],
    artifacts: list[Artifact],
) -> str:
    verification_by_claim_id = {
        verification.claim_id: verification for verification in verifications
    }
    rejected_count = sum(1 for value in verifications if value.verdict == "rejected")
    uncertain_count = sum(1 for value in verifications if value.verdict == "uncertain")
    completed_count = sum(1 for task in tasks if task.status == "completed")
    failed_count = sum(1 for task in tasks if task.status == "failed")
    source_count = len(
        {
            source
            for verification in verifications
            for source in verification.sources + verification.evidence_item_refs
        }
        | {source for artifact in artifacts for source in artifact.source_refs}
    )

    claim_lines = []
    for claim in claims[:6]:
        verification = verification_by_claim_id.get(claim.id)
        if verification:
            reason = verification.rationale or "No verifier rationale recorded."
            claim_lines.append(
                "- "
                f"{_compact_text(claim.statement, 220)} "
                f"(verdict: {verification.verdict}, "
                f"confidence: {verification.confidence:.0%}; "
                f"{_compact_text(reason, 180)})"
            )
        else:
            claim_lines.append(
                f"- {_compact_text(claim.statement, 220)} (verdict: not verified yet)"
            )
    if len(claims) > 6:
        claim_lines.append(f"- {len(claims) - 6} additional candidate claim(s) omitted.")
    if not claim_lines:
        claim_lines.append("- No candidate claims were produced.")

    partially_supported_lines = _partially_supported_fact_lines(
        claims, verifications, artifacts
    )
    caveats = _verification_caveats(verifications)
    if not caveats:
        caveats = [
            "No claim reached verified status, so disputed or unverified material cannot be used as final support.",
            "A follow-up run should gather dated primary sources and re-check any missing market-data feeds.",
        ]

    return "\n".join(
        [
            "# Evidence-limited research result",
            "",
            f'Question: "{run.question}"',
            "",
            "No decision-grade final forecast was produced because no candidate claim passed "
            "source verification. This is a completed partial result, not a supported "
            "investment recommendation.",
            "",
            "## Run evidence status",
            "",
            f"- Research tasks completed: {completed_count}",
            f"- Research tasks failed: {failed_count}",
            f"- Candidate claims assessed: {len(claims)}",
            "- Verified claims available for synthesis: 0",
            f"- Rejected claims: {rejected_count}",
            f"- Disputed / uncertain claims: {uncertain_count}",
            f"- Distinct source references inspected: {source_count}",
            "",
            "## Candidate claims not accepted as final support",
            "",
            *claim_lines,
            "",
            "## Verifier-supported facts below public-verified threshold",
            "",
            *(
                partially_supported_lines
                or [
                    "- None recorded. The verifier did not identify supported sub-claims "
                    "that could be safely separated from disputed material."
                ]
            ),
            "",
            "## Main evidence gaps",
            "",
            *[f"- {_compact_text(caveat, 240)}" for caveat in caveats[:8]],
            "",
            "## Minimum next steps for a decision-grade answer",
            "",
            "- Re-run failed or zero-result market-data lookups with a verified fallback source.",
            "- Prefer dated primary sources for Fed policy expectations, rates, FX, gold, and duration evidence.",
            "- Split broad portfolio-positioning statements into narrower asset-class claims before verification.",
            "- Re-aggregate only after at least one material claim is verified.",
            "",
            "No personalized investment advice.",
        ]
    )


def _partially_supported_fact_lines(
    claims: list[Claim],
    verifications: list[Verification],
    artifacts: list[Artifact],
) -> list[str]:
    claims_by_id = {claim.id: claim for claim in claims}
    branch_by_claim_id: dict[UUID, str] = {}
    for artifact in artifacts:
        if artifact.artifact_type == ArtifactType.CLAIM and artifact.legacy_object_id:
            branch_by_claim_id[artifact.legacy_object_id] = artifact.branch or "unknown"

    lines: list[str] = []
    seen: set[str] = set()
    for verification in verifications:
        if verification.verdict == "verified":
            continue
        claim = claims_by_id.get(verification.claim_id)
        branch = branch_by_claim_id.get(
            verification.claim_id,
            infer_semantic_branch(claim.statement if claim else verification.rationale),
        )
        for supported_part in verification.supported_parts[:3]:
            compact = _compact_text(supported_part, 220)
            key = f"{branch}:{compact}"
            if key in seen:
                continue
            seen.add(key)
            lines.append(
                f"- [{branch}] {compact} "
                f"(not public-verified; verifier confidence {verification.confidence:.0%})"
            )
    return lines[:10]


def _verification_caveats(verifications: list[Verification]) -> list[str]:
    caveats: list[str] = []
    for verification in verifications:
        caveats.extend(verification.required_caveats)
        caveats.extend(verification.unsupported_parts)
        if verification.verdict != "verified" and verification.rationale:
            caveats.append(verification.rationale)
    deduped: list[str] = []
    seen: set[str] = set()
    for caveat in caveats:
        compact = " ".join(caveat.split())
        if compact and compact not in seen:
            deduped.append(compact)
            seen.add(compact)
    return deduped


def _compact_text(value: str, limit: int) -> str:
    text = " ".join(value.split())
    if len(text) <= limit:
        return text
    return f"{text[: max(0, limit - 1)].rstrip()}…"


HANDLERS = {
    "planner-agent": {EventType.RUN_CREATED: plan},
    "tool-runner": {EventType.TASK_CREATED: execute_tool},
    "worker-agents": {EventType.OBSERVATION_CREATED: create_claim},
    "verifier-agent": {EventType.CLAIM_CREATED: verify_claim},
    "skeptic-agent": {EventType.SKEPTIC_REVIEW_REQUESTED: skeptic_review},
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
    run.failure_reason = reason
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
    await persist_action(
        runtime,
        run_id,
        PrincipalActionType.STOP_RUN,
        f"Stopped after budget block: {reason}",
        required_role="principal_policy",
        target_branch="root",
        expected_information_gain=InformationGain.LOW,
        priority=4,
        producer="aggregator-agent",
    )
    await runtime.blackboard.put_run(run)
    await evaluate_principal_policy_safely(runtime, run_id, trigger="final.partial")
