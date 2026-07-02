from uuid import uuid4

from src.agents.state import build_run_state
from src.common.models import (
    AgentSpec,
    Budget,
    Claim,
    FinalReport,
    ModelPolicy,
    PrincipalActionType,
    ResearchTask,
    RoleModelPolicy,
    Run,
    RunPhase,
    RunStatus,
    ToolBudget,
    Verification,
)


def role_policy(protected: float = 0) -> RoleModelPolicy:
    return RoleModelPolicy(model="provider/model", cap_usd=1, max_call_cost_usd=0.1, protected_usd=protected)


def model_policy() -> ModelPolicy:
    return ModelPolicy(
        planner=role_policy(),
        utility=role_policy(),
        research=role_policy(),
        verifier=role_policy(),
        aggregator=role_policy(0.1),
        judge=role_policy(0.1),
    )


def run(status: RunStatus = RunStatus.RUNNING) -> Run:
    return Run(
        question="Should I buy SAP for a two-month horizon?",
        status=status,
        models=model_policy(),
        budget=Budget(limit_usd=4, spent_usd=0.5, reserved_usd=0.25, tools=ToolBudget()),
    )


def test_run_state_summarizes_pending_research_and_next_tool_action() -> None:
    current_run = run()
    task = ResearchTask(
        run_id=current_run.id,
        title="SAP catalysts",
        question="Find SAP catalysts",
        tool="web_search",
    )
    agent = AgentSpec(
        run_id=current_run.id,
        name="SAP catalyst researcher",
        role_template="asset_research_agent",
        branch="equities/sap",
        objective="Find two-month SAP catalysts",
    )

    state = build_run_state(current_run, [task], [], [], agent_specs=[agent])

    assert state.current_phase == RunPhase.RESEARCHING
    assert state.open_questions == ["Find SAP catalysts"]
    assert state.agent_count == 1
    assert state.budget_remaining == 3.25
    assert state.tool_budget_remaining == {"tavily_credits": 2, "market_data_requests": 1}
    assert state.next_action_candidates[0].action_type == PrincipalActionType.REQUEST_TOOL_CALL


def test_run_state_promotes_verified_claims_and_requests_aggregation() -> None:
    current_run = run()
    task = ResearchTask(
        run_id=current_run.id,
        title="SAP catalysts",
        question="Find SAP catalysts",
        tool="web_search",
        status="completed",
    )
    claim = Claim(
        run_id=current_run.id,
        task_id=task.id,
        statement="SAP has cloud-growth momentum.",
        evidence_observation_ids=[uuid4()],
        confidence=0.8,
    )
    verification = Verification(
        run_id=current_run.id,
        claim_id=claim.id,
        verdict="verified",
        rationale="Supported by sources.",
        confidence=0.75,
    )

    state = build_run_state(current_run, [task], [claim], [verification])

    assert state.current_phase == RunPhase.SYNTHESIZING
    assert state.known_facts == ["SAP has cloud-growth momentum."]
    assert state.verified_claim_count == 1
    assert state.next_action_candidates[0].action_type == PrincipalActionType.REQUEST_AGGREGATION


def test_completed_run_state_exposes_stop_reason_and_judge_payoff() -> None:
    current_run = run(RunStatus.COMPLETED)
    final = FinalReport(
        run_id=current_run.id,
        answer="Hold until risk/reward improves.",
        verified_claim_ids=[],
        sources=[],
        judge_score=0.82,
        judge_feedback="Well evidenced.",
    )

    state = build_run_state(current_run, [], [], [], final)

    assert state.current_phase == RunPhase.COMPLETED
    assert state.last_judge_score == 0.82
    assert state.last_judge_feedback == "Well evidenced."
    assert state.stop_reasons == ["run completed"]
    assert state.next_action_candidates == []
