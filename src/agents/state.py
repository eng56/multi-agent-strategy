from __future__ import annotations

import re
from src.common.models import (
    AgentRole,
    AgentSpec,
    Artifact,
    ArtifactStatus,
    ArtifactType,
    BudgetSummary,
    Claim,
    DeadLetterRecord,
    FinalReport,
    OrganizationPlan,
    Observation,
    PrincipalAction,
    ResearchTask,
    Run,
    RunPhase,
    RunState,
    RunStatus,
    ToolUsageSummary,
    Verification,
)

BRANCH_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("market/gold", ("gold", "xau")),
    (
        "market/fx",
        (
            "usd",
            "dxy",
            "fx",
            "eur/usd",
            "eurusd",
            "usdjpy",
            "usd/jpy",
            "currency",
            "u.s. dollar",
            "us dollar",
        ),
    ),
    ("market/oil", ("oil", "crude", "wti", "brent")),
    ("market/equities", ("equities", "equity", "stocks", "stock", "spx", "s&p", "nasdaq", "sap")),
    ("market/bonds", ("bond", "bonds", "duration", "long-duration", "long duration", "tlt")),
    (
        "macro/rates",
        ("rates", "fed", "cut", "cuts", "hike", "hikes", "yield", "yields", "treasury"),
    ),
    ("macro/inflation", ("inflation", "cpi", "pce")),
    ("trust/source_verifier", ("verifier", "verification", "source")),
    ("trust/skeptic", ("skeptic", "counterargument", "contradict", "risk")),
    ("synthesis/aggregator", ("aggregator", "aggregate", "final", "synthesis")),
    ("synthesis/judge", ("judge", "score", "payoff")),
)
FOLLOWUP_JUDGE_SCORE_THRESHOLD = 0.75
MAX_FOLLOWUP_WAVES = 1
BUDGET_SUMMARY_ROLES: tuple[AgentRole, ...] = (
    AgentRole.PLANNER,
    AgentRole.RESEARCH,
    AgentRole.VERIFIER,
    AgentRole.AGGREGATOR,
    AgentRole.JUDGE,
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
    dead_letters: list[DeadLetterRecord] | None = None,
    iteration: int = 0,
    principal_actions: list[PrincipalAction] | None = None,
    observations: list[Observation] | None = None,
) -> RunState:
    """Summarize blackboard contents into the Principal's control-state view."""
    artifacts = artifacts or []
    agent_specs = agent_specs or []
    dead_letters = dead_letters or []
    principal_actions = principal_actions or []
    observations = observations or []
    verified_ids = {value.claim_id for value in verifications if value.verdict == "verified"}
    rejected_ids = {value.claim_id for value in verifications if value.verdict == "rejected"}
    uncertain_ids = {value.claim_id for value in verifications if value.verdict == "uncertain"}
    verified_claims = [claim for claim in claims if claim.id in verified_ids]
    pending_tasks = [task for task in tasks if task.status == "created"]
    failed_tasks = [task for task in tasks if task.status == "failed"]
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
        value
        for value in artifacts
        if value.artifact_type in {ArtifactType.OPEN_QUESTION, ArtifactType.DATA_GAP}
    ]
    coverage_by_topic = {task.title: branch_for_task(task, agent_specs) for task in tasks}
    for artifact in artifacts:
        if artifact.branch:
            coverage_by_topic.setdefault(artifact.branch, artifact.status.value)
    phase = _phase(run, tasks, claims, verifications, final)
    active_branches = _active_branches(tasks, agent_specs, artifacts, organization_plan)
    budget_summary = build_budget_summary(run, tasks, final)
    budget_remaining = budget_summary.remaining_usd
    stop_reasons = []
    if run.failure_reason:
        stop_reasons.append(run.failure_reason)
    for reason in budget_summary.stop_reasons:
        if reason not in stop_reasons:
            stop_reasons.append(reason)
    if final and final.partial:
        stop_reasons.append("partial final report produced before full synthesis")
    if run.status == RunStatus.COMPLETED:
        stop_reasons.append("run completed")
    if dead_letters:
        stop_reasons.append(f"{len(dead_letters)} dead-lettered event(s)")

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
        failed_tasks=[task.title for task in failed_tasks],
        verified_claim_count=len(verified_ids | verified_artifact_ids),
        rejected_claim_count=len(rejected_ids | rejected_artifact_ids),
        disputed_claim_count=len(uncertain_ids | disputed_artifact_ids),
        coverage_by_topic=coverage_by_topic,
        budget_summary=budget_summary,
        budget_remaining=budget_remaining,
        tool_budget_remaining={
            key: value.remaining for key, value in budget_summary.tool_usage.items()
        },
        agent_count=len(agent_specs),
        last_judge_score=final.judge_score if final else None,
        last_judge_feedback=final.judge_feedback if final else None,
        dead_letter_count=len(dead_letters),
        stop_reasons=stop_reasons,
    )
    state.next_action_candidates = _next_actions(
        state,
        run,
        tasks,
        claims,
        verifications,
        final,
        artifacts,
        agent_specs,
        organization_plan,
        dead_letters,
        principal_actions,
        observations,
    )
    return state


def build_budget_summary(
    run: Run,
    tasks: list[ResearchTask],
    final: FinalReport | None = None,
) -> BudgetSummary:
    """Build a compact budget view for API serialization and Principal control state."""
    budget = run.budget
    role_spent = _role_amounts(budget.role_spent_usd)
    role_reserved = _role_amounts(budget.role_reserved_usd)
    role_protected = _role_protected_amounts(run)
    protected_remaining = 0.0
    for role_name, protected in role_protected.items():
        protected_remaining += max(
            0,
            protected - role_spent.get(role_name, 0) - role_reserved.get(role_name, 0),
        )
    tool_usage = {
        "tavily_credits": ToolUsageSummary(
            used=budget.tools.tavily_credits_used,
            max=budget.tools.tavily_max_credits,
            remaining=max(0, budget.tools.tavily_max_credits - budget.tools.tavily_credits_used),
        ),
        "market_data_requests": ToolUsageSummary(
            used=budget.tools.market_data_requests_used,
            max=budget.tools.market_data_max_requests,
            remaining=max(
                0,
                budget.tools.market_data_max_requests - budget.tools.market_data_requests_used,
            ),
        ),
    }
    return BudgetSummary(
        total_limit_usd=budget.limit_usd,
        spent_usd=budget.spent_usd,
        reserved_usd=budget.reserved_usd,
        remaining_usd=max(0, budget.limit_usd - budget.spent_usd - budget.reserved_usd),
        role_spent_usd=role_spent,
        role_reserved_usd=role_reserved,
        role_protected_usd=role_protected,
        protected_usd=sum(role_protected.values()),
        protected_remaining_usd=protected_remaining,
        tool_usage=tool_usage,
        stop_reasons=_budget_stop_reasons(run, tasks, final),
    )


def _role_amounts(values: dict[AgentRole, float]) -> dict[str, float]:
    amounts = {role.value: float(values.get(role, values.get(role.value, 0))) for role in BUDGET_SUMMARY_ROLES}
    for key, value in values.items():
        role_name = key.value if isinstance(key, AgentRole) else str(key)
        amounts.setdefault(role_name, float(value))
    return amounts


def _role_protected_amounts(run: Run) -> dict[str, float]:
    amounts: dict[str, float] = {}
    for role in BUDGET_SUMMARY_ROLES:
        policy = getattr(run.models, role.value, None)
        if policy and policy.protected_usd > 0:
            amounts[role.value] = policy.protected_usd
    return amounts


def _budget_stop_reasons(
    run: Run,
    tasks: list[ResearchTask],
    final: FinalReport | None,
) -> list[str]:
    reasons: list[str] = []
    if run.status == RunStatus.PARTIAL_BUDGET_EXHAUSTED:
        if run.failure_reason:
            reasons.append(run.failure_reason)
        elif final and final.partial:
            reasons.append("budget exhausted before full synthesis")
        else:
            reasons.append("budget exhausted")
    skipped_tasks = [task for task in tasks if task.status == "skipped_budget"]
    if skipped_tasks:
        details = []
        for task in skipped_tasks[:3]:
            details.append(task.reason or f"Budget skipped task: {task.title}")
        suffix = "; ".join(details)
        if len(skipped_tasks) > 3:
            suffix += f"; {len(skipped_tasks) - 3} more"
        reasons.append(f"{len(skipped_tasks)} task(s) skipped due to budget: {suffix}")
    return reasons


def max_task_wave(tasks: list[ResearchTask]) -> int:
    return max((task.wave_number for task in tasks), default=0)


def has_followup_wave(tasks: list[ResearchTask]) -> bool:
    return max_task_wave(tasks) > 0


def judge_score_needs_followup(final: FinalReport | None) -> bool:
    return bool(
        final
        and final.judge_score is not None
        and final.judge_score < FOLLOWUP_JUDGE_SCORE_THRESHOLD
    )


def can_request_followup_wave(final: FinalReport | None, tasks: list[ResearchTask]) -> bool:
    return judge_score_needs_followup(final) and max_task_wave(tasks) < MAX_FOLLOWUP_WAVES


def should_reaggregate_after_followup(
    final: FinalReport | None, tasks: list[ResearchTask]
) -> bool:
    return judge_score_needs_followup(final) and max_task_wave(tasks) > (
        final.wave_number if final else 0
    )


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
    pending_tasks = [task for task in tasks if task.status == "created"]
    completed_tasks = [task for task in tasks if task.status == "completed"]
    if final and final.partial:
        return RunPhase.COMPLETED
    if final and judge_score_needs_followup(final):
        if can_request_followup_wave(final, tasks):
            return RunPhase.JUDGING
        if pending_tasks:
            return RunPhase.RESEARCHING
        if should_reaggregate_after_followup(final, tasks) and completed_tasks:
            return RunPhase.SYNTHESIZING
    if final and final.judge_score is None and run.models.judge:
        return RunPhase.JUDGING
    if final:
        return RunPhase.COMPLETED
    if tasks and not pending_tasks and max_task_wave(tasks) >= MAX_FOLLOWUP_WAVES:
        checked_claim_ids = {verification.claim_id for verification in verifications}
        claim_ids = {claim.id for claim in claims}
        if claim_ids <= checked_claim_ids:
            return RunPhase.SYNTHESIZING
    if tasks and all(task.status == "failed" for task in tasks):
        return RunPhase.FAILED
    if verifications and not pending_tasks and len(verifications) >= max(1, len(completed_tasks)):
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
    agent_specs: list[AgentSpec] | None = None,
    organization_plan: OrganizationPlan | None = None,
    dead_letters: list[DeadLetterRecord] | None = None,
    principal_actions: list[PrincipalAction] | None = None,
    observations: list[Observation] | None = None,
) -> list[PrincipalAction]:
    from src.agents.principal_policy import build_principal_snapshot, propose_principal_actions

    snapshot = build_principal_snapshot(
        run,
        state,
        organization_plan,
        agent_specs=agent_specs,
        tasks=tasks,
        observations=observations,
        claims=claims,
        verifications=verifications,
        artifacts=artifacts,
        principal_actions=principal_actions,
        dead_letters=dead_letters,
        final=final,
    )
    return propose_principal_actions(snapshot)


def title_from_branch(branch: str) -> str:
    return re.sub(r"[_/]+", " ", branch).title()
