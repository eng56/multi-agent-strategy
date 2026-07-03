from __future__ import annotations

import re
from src.common.models import (
    AgentSpec,
    Artifact,
    ArtifactStatus,
    ArtifactType,
    Claim,
    FinalReport,
    InformationGain,
    OrganizationPlan,
    PrincipalAction,
    PrincipalActionType,
    ResearchTask,
    Run,
    RunPhase,
    RunState,
    RunStatus,
    Verification,
)

BRANCH_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("market/gold", ("gold", "xau")),
    ("market/fx", ("usd", "dxy", "fx", "eur/usd", "eurusd", "usdjpy", "usd/jpy", "currency")),
    ("market/oil", ("oil", "crude", "wti", "brent")),
    ("market/equities", ("equities", "equity", "stocks", "stock", "spx", "s&p", "nasdaq", "sap")),
    (
        "macro/rates",
        ("rates", "fed", "cut", "cuts", "hike", "hikes", "yield", "yields", "treasury"),
    ),
    ("macro/inflation", ("inflation", "cpi", "pce")),
    ("trust/source_verifier", ("verifier", "verification", "source")),
    ("synthesis/aggregator", ("aggregator", "aggregate", "final", "synthesis")),
    ("synthesis/judge", ("judge", "score", "payoff")),
)


def text_matches_any(text: str, needles: tuple[str, ...]) -> bool:
    for needle in needles:
        pattern = rf"(?<![A-Za-z0-9]){re.escape(needle.lower())}(?![A-Za-z0-9])"
        if re.search(pattern, text.lower()):
            return True
    return False


def infer_semantic_branch(text: str, fallback: str = "research/general") -> str:
    for branch, needles in BRANCH_PATTERNS:
        if text_matches_any(text, needles):
            return branch
    return fallback


def branch_for_task(task: ResearchTask, agent_specs: list[AgentSpec] | None = None) -> str:
    inferred = infer_semantic_branch(f"{task.title} {task.question}", fallback="")
    if inferred:
        for spec in agent_specs or []:
            if spec.branch == inferred:
                return spec.branch
        return inferred
    return task.tool


def tags_for_text(text: str) -> list[str]:
    tags = []
    for branch, needles in BRANCH_PATTERNS:
        if text_matches_any(text, needles):
            tags.append(branch.replace("/", ":"))
    return sorted(set(tags))


def build_run_state(
    run: Run,
    tasks: list[ResearchTask],
    claims: list[Claim],
    verifications: list[Verification],
    final: FinalReport | None = None,
    agent_specs: list[AgentSpec] | None = None,
    artifacts: list[Artifact] | None = None,
    organization_plan: OrganizationPlan | None = None,
    iteration: int = 0,
) -> RunState:
    """Summarize blackboard contents into the Principal's control-state view."""
    artifacts = artifacts or []
    agent_specs = agent_specs or []
    verified_ids = {value.claim_id for value in verifications if value.verdict == "verified"}
    rejected_ids = {value.claim_id for value in verifications if value.verdict == "rejected"}
    uncertain_ids = {value.claim_id for value in verifications if value.verdict == "uncertain"}
    verified_claims = [claim for claim in claims if claim.id in verified_ids]
    pending_tasks = [task for task in tasks if task.status == "created"]
    verified_artifacts = [value for value in artifacts if value.status == ArtifactStatus.VERIFIED]
    rejected_artifacts = [
        value
        for value in artifacts
        if value.status == ArtifactStatus.REJECTED
        and value.artifact_type in {ArtifactType.CLAIM, ArtifactType.FORECAST}
    ]
    disputed_artifacts = [
        value
        for value in artifacts
        if value.status == ArtifactStatus.DISPUTED
        and value.artifact_type in {ArtifactType.CLAIM, ArtifactType.FORECAST}
    ]
    verified_knowledge_artifacts = [
        value
        for value in verified_artifacts
        if value.artifact_type in {ArtifactType.CLAIM, ArtifactType.FORECAST}
    ]
    verified_artifact_ids = {
        value.legacy_object_id or value.id for value in verified_knowledge_artifacts
    }
    rejected_artifact_ids = {value.legacy_object_id or value.id for value in rejected_artifacts}
    disputed_artifact_ids = {value.legacy_object_id or value.id for value in disputed_artifacts}
    open_question_artifacts = [
        value for value in artifacts if value.artifact_type == ArtifactType.OPEN_QUESTION
    ]
    coverage_by_topic = {task.title: branch_for_task(task, agent_specs) for task in tasks}
    for artifact in artifacts:
        if artifact.branch:
            coverage_by_topic.setdefault(artifact.branch, artifact.status.value)
    phase = _phase(run, tasks, claims, verifications, final)
    active_branches = _active_branches(tasks, agent_specs, artifacts, organization_plan)
    budget_remaining = max(0, run.budget.limit_usd - run.budget.spent_usd - run.budget.reserved_usd)
    stop_reasons = []
    if run.failure_reason:
        stop_reasons.append(run.failure_reason)
    if final and final.partial:
        stop_reasons.append("partial final report produced before full synthesis")
    if run.status == RunStatus.COMPLETED:
        stop_reasons.append("run completed")

    state = RunState(
        run_id=run.id,
        iteration=iteration,
        question=run.question,
        objective=run.question,
        current_phase=phase,
        active_branches=active_branches,
        known_facts=[value.text_or_summary for value in verified_knowledge_artifacts]
        or [claim.statement for claim in verified_claims],
        open_questions=[value.text_or_summary for value in open_question_artifacts]
        + [task.question for task in pending_tasks],
        verified_claim_count=len(verified_ids | verified_artifact_ids),
        rejected_claim_count=len(rejected_ids | rejected_artifact_ids),
        disputed_claim_count=len(uncertain_ids | disputed_artifact_ids),
        coverage_by_topic=coverage_by_topic,
        budget_remaining=budget_remaining,
        tool_budget_remaining={
            "tavily_credits": max(
                0, run.budget.tools.tavily_max_credits - run.budget.tools.tavily_credits_used
            ),
            "market_data_requests": max(
                0,
                run.budget.tools.market_data_max_requests
                - run.budget.tools.market_data_requests_used,
            ),
        },
        agent_count=len(agent_specs),
        last_judge_score=final.judge_score if final else None,
        last_judge_feedback=final.judge_feedback if final else None,
        stop_reasons=stop_reasons,
    )
    state.next_action_candidates = _next_actions(
        state, run, tasks, claims, verifications, final, artifacts
    )
    return state


def _active_branches(
    tasks: list[ResearchTask],
    agent_specs: list[AgentSpec],
    artifacts: list[Artifact],
    organization_plan: OrganizationPlan | None,
) -> list[str]:
    branches: list[str] = []
    if organization_plan:
        branches.extend(organization_plan.branches)
    branches.extend(spec.branch for spec in agent_specs)
    branches.extend(artifact.branch for artifact in artifacts if artifact.branch)
    if not branches:
        branches.extend(branch_for_task(task, agent_specs) for task in tasks)
    return sorted(set(branches))


def _phase(
    run: Run,
    tasks: list[ResearchTask],
    claims: list[Claim],
    verifications: list[Verification],
    final: FinalReport | None,
) -> RunPhase:
    if run.status == RunStatus.FAILED:
        return RunPhase.FAILED
    if run.status in {RunStatus.COMPLETED, RunStatus.PARTIAL_BUDGET_EXHAUSTED}:
        return RunPhase.COMPLETED
    if final and final.judge_score is None and run.models.judge:
        return RunPhase.JUDGING
    if final:
        return RunPhase.COMPLETED
    if verifications and len(verifications) >= max(1, len(tasks)):
        return RunPhase.SYNTHESIZING
    if claims:
        return RunPhase.VERIFYING
    if tasks:
        return RunPhase.RESEARCHING
    if run.status == RunStatus.RUNNING:
        return RunPhase.PLANNING
    return RunPhase.INTAKE


def _next_actions(
    state: RunState,
    run: Run,
    tasks: list[ResearchTask],
    claims: list[Claim],
    verifications: list[Verification],
    final: FinalReport | None,
    artifacts: list[Artifact],
) -> list[PrincipalAction]:
    if run.status in {RunStatus.COMPLETED, RunStatus.PARTIAL_BUDGET_EXHAUSTED, RunStatus.FAILED}:
        return []
    actions: list[PrincipalAction] = []
    pending_tasks = [task for task in tasks if task.status == "created"]
    if not tasks:
        actions.append(
            PrincipalAction(
                run_id=run.id,
                action_type=PrincipalActionType.ASSIGN_TASK,
                reason="No research tasks exist yet; the Principal should decompose the objective.",
                expected_information_gain=InformationGain.HIGH,
                required_role="principal_policy",
                priority=9,
            )
        )
    elif pending_tasks:
        actions.append(
            PrincipalAction(
                run_id=run.id,
                action_type=PrincipalActionType.REQUEST_TOOL_CALL,
                reason=f"{len(pending_tasks)} planned task(s) still need tool evidence.",
                expected_information_gain=InformationGain.HIGH,
                target_branch=branch_for_task(pending_tasks[0]),
                required_role="tool_runner",
                priority=8,
            )
        )
    unverified_claim_ids = {claim.id for claim in claims} - {
        verification.claim_id for verification in verifications
    }
    if unverified_claim_ids:
        actions.append(
            PrincipalAction(
                run_id=run.id,
                action_type=PrincipalActionType.REQUEST_VERIFICATION,
                reason=f"{len(unverified_claim_ids)} claim(s) need trust-layer review before synthesis.",
                expected_information_gain=InformationGain.MEDIUM,
                required_role="source_verifier_agent",
                target_branch="trust/source_verifier",
                priority=7,
            )
        )
    has_counterargument = any(
        value.artifact_type == ArtifactType.COUNTERARGUMENT for value in artifacts
    )
    if state.verified_claim_count >= 2 and not has_counterargument and state.budget_remaining > 0:
        actions.append(
            PrincipalAction(
                run_id=run.id,
                action_type=PrincipalActionType.REQUEST_SKEPTIC_REVIEW,
                reason="Multiple claims have been verified but no adversarial counterargument has been produced yet.",
                expected_information_gain=InformationGain.MEDIUM,
                required_role="skeptic_agent",
                target_branch="trust/skeptic",
                priority=6,
            )
        )
    if tasks and not pending_tasks and not final:
        actions.append(
            PrincipalAction(
                run_id=run.id,
                action_type=PrincipalActionType.REQUEST_AGGREGATION,
                reason="All planned tasks have finished; synthesize only trusted artifacts into an answer.",
                expected_information_gain=InformationGain.MEDIUM,
                required_role="aggregator_agent",
                target_branch="synthesis/aggregator",
                priority=5,
            )
        )
    if final and final.judge_score is not None:
        actions.append(
            PrincipalAction(
                run_id=run.id,
                action_type=PrincipalActionType.STOP_RUN,
                reason="The payoff layer scored the final report; stop unless a follow-up threshold is configured.",
                expected_information_gain=InformationGain.LOW,
                target_branch="synthesis/judge",
                priority=3,
            )
        )
    return actions


def title_from_branch(branch: str) -> str:
    return re.sub(r"[_/]+", " ", branch).title()
