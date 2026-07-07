from uuid import uuid4

from src.agents.principal_policy import (
    PrincipalSnapshot,
    build_principal_snapshot,
    propose_principal_actions,
    select_principal_action,
)
from src.agents.state import build_run_state
from src.common.models import (
    ActionStatus,
    Artifact,
    ArtifactStatus,
    ArtifactType,
    Budget,
    Claim,
    FinalReport,
    InformationGain,
    ModelPolicy,
    OrganizationPlan,
    PrincipalAction,
    PrincipalActionType,
    ResearchTask,
    RoleModelPolicy,
    Run,
    RunState,
    RunStatus,
    ToolBudget,
    Verification,
)


def role_policy(protected: float = 0) -> RoleModelPolicy:
    return RoleModelPolicy(
        model="provider/model",
        cap_usd=1,
        max_call_cost_usd=0.1,
        protected_usd=protected,
    )


def model_policy(judge_enabled: bool = True) -> ModelPolicy:
    return ModelPolicy(
        planner=role_policy(),
        utility=role_policy(),
        research=role_policy(),
        verifier=role_policy(),
        aggregator=role_policy(0.1),
        judge=role_policy(0.1) if judge_enabled else None,
    )


def run(status: RunStatus = RunStatus.RUNNING, remaining: float = 3) -> Run:
    spent = 3 - remaining
    return Run(
        question="Should I buy SAP for a two-month horizon?",
        status=status,
        models=model_policy(),
        budget=Budget(limit_usd=3, spent_usd=spent, reserved_usd=0, tools=ToolBudget()),
    )


def organization(current_run: Run) -> OrganizationPlan:
    return OrganizationPlan(
        run_id=current_run.id,
        root_agent_id=uuid4(),
        branches=["research/general", "trust/source_verifier", "synthesis/aggregator"],
    )


def snapshot(
    current_run: Run,
    *,
    tasks=None,
    claims=None,
    verifications=None,
    artifacts=None,
    final=None,
    principal_actions=None,
    organization_plan=None,
) -> tuple[RunState, PrincipalSnapshot]:
    tasks = tasks or []
    claims = claims or []
    verifications = verifications or []
    artifacts = artifacts or []
    principal_actions = principal_actions or []
    current_organization = organization_plan or organization(current_run)
    state = build_run_state(
        current_run,
        tasks,
        claims,
        verifications,
        final,
        artifacts=artifacts,
        organization_plan=current_organization,
        principal_actions=principal_actions,
    )
    return state, build_principal_snapshot(
        current_run,
        state,
        current_organization,
        tasks=tasks,
        claims=claims,
        verifications=verifications,
        artifacts=artifacts,
        principal_actions=principal_actions,
        final=final,
    )


def task(current_run: Run, status: str = "created", wave_number: int = 0) -> ResearchTask:
    return ResearchTask(
        run_id=current_run.id,
        title="SAP catalysts",
        question="Find SAP catalysts",
        tool="web_search",
        status=status,
        wave_number=wave_number,
    )


def verified_claim(current_run: Run, current_task: ResearchTask) -> tuple[Claim, Verification]:
    claim = Claim(
        run_id=current_run.id,
        task_id=current_task.id,
        statement=f"SAP claim {uuid4()}",
        evidence_observation_ids=[uuid4()],
        confidence=0.8,
    )
    verification = Verification(
        run_id=current_run.id,
        claim_id=claim.id,
        verdict="verified",
        rationale="Supported.",
        confidence=0.75,
    )
    return claim, verification


def action_types(actions: list[PrincipalAction]) -> set[PrincipalActionType]:
    return {action.action_type for action in actions}


def test_no_tasks_selects_assign_task() -> None:
    current_run = run()
    _, policy_snapshot = snapshot(current_run)

    selected = select_principal_action(policy_snapshot)

    assert selected is not None
    assert selected.action_type == PrincipalActionType.ASSIGN_TASK


def test_pending_tasks_selects_request_tool_call() -> None:
    current_run = run()
    current_task = task(current_run)
    _, policy_snapshot = snapshot(current_run, tasks=[current_task])

    selected = select_principal_action(policy_snapshot)

    assert selected is not None
    assert selected.action_type == PrincipalActionType.REQUEST_TOOL_CALL
    assert selected.target_branch == "market/equities"


def test_unverified_claims_request_verification() -> None:
    current_run = run()
    current_task = task(current_run, status="completed")
    claim = Claim(
        run_id=current_run.id,
        task_id=current_task.id,
        statement="SAP has cloud-growth momentum.",
        evidence_observation_ids=[uuid4()],
        confidence=0.8,
    )
    _, policy_snapshot = snapshot(current_run, tasks=[current_task], claims=[claim])

    actions = propose_principal_actions(policy_snapshot)

    assert PrincipalActionType.REQUEST_VERIFICATION in action_types(actions)


def test_verified_claims_without_skeptic_request_skeptic_review() -> None:
    current_run = run()
    current_task = task(current_run, status="completed")
    claim_a, verification_a = verified_claim(current_run, current_task)
    claim_b, verification_b = verified_claim(current_run, current_task)
    _, policy_snapshot = snapshot(
        current_run,
        tasks=[current_task],
        claims=[claim_a, claim_b],
        verifications=[verification_a, verification_b],
    )

    actions = propose_principal_actions(policy_snapshot)

    assert PrincipalActionType.REQUEST_SKEPTIC_REVIEW in action_types(actions)


def test_verified_claims_with_skeptic_request_aggregation() -> None:
    current_run = run()
    current_task = task(current_run, status="completed")
    claim, verification = verified_claim(current_run, current_task)
    counterargument = Artifact(
        run_id=current_run.id,
        artifact_type=ArtifactType.COUNTERARGUMENT,
        branch="trust/skeptic",
        text_or_summary="Valuation may already price the upside.",
        status=ArtifactStatus.VERIFIED,
    )
    _, policy_snapshot = snapshot(
        current_run,
        tasks=[current_task],
        claims=[claim],
        verifications=[verification],
        artifacts=[counterargument],
    )

    selected = select_principal_action(policy_snapshot)

    assert selected is not None
    assert selected.action_type == PrincipalActionType.REQUEST_AGGREGATION


def test_low_judge_score_without_followup_requests_followup() -> None:
    current_run = run()
    final = FinalReport(
        run_id=current_run.id,
        answer="Buy with caution.",
        verified_claim_ids=[],
        sources=[],
        judge_score=0.62,
        judge_feedback="Needs stronger valuation evidence.",
    )
    _, policy_snapshot = snapshot(current_run, final=final)

    selected = select_principal_action(policy_snapshot)

    assert selected is not None
    assert selected.action_type == PrincipalActionType.REQUEST_FOLLOWUP


def test_terminal_run_produces_no_actions() -> None:
    current_run = run(RunStatus.COMPLETED)
    _, policy_snapshot = snapshot(current_run)

    assert propose_principal_actions(policy_snapshot) == []


def test_duplicate_recent_action_is_filtered() -> None:
    current_run = run()
    current_task = task(current_run, status="completed")
    claim, verification = verified_claim(current_run, current_task)
    duplicate = PrincipalAction(
        run_id=current_run.id,
        action_type=PrincipalActionType.REQUEST_AGGREGATION,
        reason="Already synthesized.",
        expected_information_gain=InformationGain.MEDIUM,
        target_branch="synthesis/aggregator",
        required_role="aggregator_agent",
        status=ActionStatus.EXECUTED,
    )
    _, policy_snapshot = snapshot(
        current_run,
        tasks=[current_task],
        claims=[claim],
        verifications=[verification],
        principal_actions=[duplicate],
    )

    actions = propose_principal_actions(policy_snapshot)

    assert PrincipalActionType.REQUEST_AGGREGATION not in action_types(actions)


def test_budget_exhausted_filters_expensive_actions() -> None:
    current_run = run(remaining=0)
    current_task = task(current_run)
    _, policy_snapshot = snapshot(current_run, tasks=[current_task])

    actions = propose_principal_actions(policy_snapshot)

    assert PrincipalActionType.REQUEST_TOOL_CALL not in action_types(actions)
    assert {action.action_type for action in actions} == {PrincipalActionType.STOP_RUN}
    assert all(action.estimated_cost == 0 for action in actions)
