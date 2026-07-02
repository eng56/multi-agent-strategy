from __future__ import annotations

from collections import Counter

from src.common.models import (
    AgentSpec,
    Claim,
    FinalReport,
    InformationGain,
    PrincipalAction,
    PrincipalActionType,
    ResearchTask,
    Run,
    RunPhase,
    RunState,
    RunStatus,
    Verification,
)


def build_run_state(
    run: Run,
    tasks: list[ResearchTask],
    claims: list[Claim],
    verifications: list[Verification],
    final: FinalReport | None = None,
    agent_specs: list[AgentSpec] | None = None,
    iteration: int = 0,
) -> RunState:
    """Summarize blackboard contents into the Principal's control-state view."""
    verified_ids = {value.claim_id for value in verifications if value.verdict == "verified"}
    rejected_ids = {value.claim_id for value in verifications if value.verdict == "rejected"}
    uncertain_ids = {value.claim_id for value in verifications if value.verdict == "uncertain"}
    verified_claims = [claim for claim in claims if claim.id in verified_ids]
    pending_tasks = [task for task in tasks if task.status == "created"]
    coverage_by_topic = {task.title: task.status for task in tasks}
    phase = _phase(run, tasks, claims, verifications, final)
    branch_counter = Counter(task.tool for task in tasks)
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
        active_branches=sorted(branch_counter),
        known_facts=[claim.statement for claim in verified_claims],
        open_questions=[task.question for task in pending_tasks],
        verified_claim_count=len(verified_ids),
        rejected_claim_count=len(rejected_ids),
        disputed_claim_count=len(uncertain_ids),
        coverage_by_topic=coverage_by_topic,
        budget_remaining=budget_remaining,
        tool_budget_remaining={
            "tavily_credits": max(0, run.budget.tools.tavily_max_credits - run.budget.tools.tavily_credits_used),
            "market_data_requests": max(
                0, run.budget.tools.market_data_max_requests - run.budget.tools.market_data_requests_used
            ),
        },
        agent_count=len(agent_specs or []),
        last_judge_score=final.judge_score if final else None,
        last_judge_feedback=final.judge_feedback if final else None,
        stop_reasons=stop_reasons,
    )
    state.next_action_candidates = _next_actions(state, run, tasks, claims, verifications, final)
    return state


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
                target_branch=pending_tasks[0].tool,
                required_role="tool_runner",
                priority=8,
            )
        )
    unverified_claim_ids = {claim.id for claim in claims} - {verification.claim_id for verification in verifications}
    if unverified_claim_ids:
        actions.append(
            PrincipalAction(
                run_id=run.id,
                action_type=PrincipalActionType.REQUEST_VERIFICATION,
                reason=f"{len(unverified_claim_ids)} claim(s) need trust-layer review before synthesis.",
                expected_information_gain=InformationGain.MEDIUM,
                required_role="source_verifier_agent",
                priority=7,
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
                priority=6,
            )
        )
    if final and final.judge_score is not None:
        actions.append(
            PrincipalAction(
                run_id=run.id,
                action_type=PrincipalActionType.STOP_RUN,
                reason="The payoff layer scored the final report; stop unless a follow-up threshold is configured.",
                expected_information_gain=InformationGain.LOW,
                priority=3,
            )
        )
    return actions
