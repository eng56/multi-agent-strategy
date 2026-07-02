from uuid import uuid4

from src.agents.state import build_run_state
from src.common.models import (
    AgentSpec,
    Artifact,
    ArtifactStatus,
    ArtifactType,
    Budget,
    Claim,
    FinalReport,
    ModelPolicy,
    OrganizationPlan,
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
    return RoleModelPolicy(
        model="provider/model", cap_usd=1, max_call_cost_usd=0.1, protected_usd=protected
    )


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


def test_run_state_prefers_agent_spec_and_artifact_branches_before_task_tool() -> None:
    current_run = run()
    task = ResearchTask(
        run_id=current_run.id,
        title="Gold and Fed rates",
        question="Will gold react to Fed cuts?",
        tool="web_search",
    )
    agent = AgentSpec(
        run_id=current_run.id,
        name="gold agent",
        role_template="asset_research_agent",
        branch="market/gold",
        objective="Research gold",
    )
    artifact = Artifact(
        run_id=current_run.id,
        artifact_type=ArtifactType.OBSERVATION,
        branch="macro/rates",
        text_or_summary="Fed cuts matter for real yields.",
    )
    organization = OrganizationPlan(
        run_id=current_run.id,
        root_agent_id=agent.id,
        agent_specs=[agent],
        branches=["market/gold", "macro/rates"],
    )

    state = build_run_state(
        current_run,
        [task],
        [],
        [],
        agent_specs=[agent],
        artifacts=[artifact],
        organization_plan=organization,
    )

    assert "market/gold" in state.active_branches
    assert "macro/rates" in state.active_branches
    assert "web_search" not in state.active_branches
    assert state.coverage_by_topic["Gold and Fed rates"] == "market/gold"


def test_run_state_derives_counts_and_facts_from_artifacts() -> None:
    current_run = run()
    verified = Artifact(
        run_id=current_run.id,
        artifact_type=ArtifactType.CLAIM,
        branch="market/gold",
        text_or_summary="Gold has upside if real yields fall.",
        status=ArtifactStatus.VERIFIED,
    )
    rejected = Artifact(
        run_id=current_run.id,
        artifact_type=ArtifactType.CLAIM,
        branch="market/gold",
        text_or_summary="Gold always rises on cuts.",
        status=ArtifactStatus.REJECTED,
    )
    disputed = Artifact(
        run_id=current_run.id,
        artifact_type=ArtifactType.CLAIM,
        branch="market/gold",
        text_or_summary="ETF flows are strongly positive.",
        status=ArtifactStatus.DISPUTED,
    )
    question = Artifact(
        run_id=current_run.id,
        artifact_type=ArtifactType.OPEN_QUESTION,
        branch="market/gold",
        text_or_summary="Are ETF flows confirming the move?",
    )

    state = build_run_state(
        current_run, [], [], [], artifacts=[verified, rejected, disputed, question]
    )

    assert state.known_facts == ["Gold has upside if real yields fall."]
    assert state.open_questions == ["Are ETF flows confirming the move?"]
    assert state.verified_claim_count == 1
    assert state.rejected_claim_count == 1
    assert state.disputed_claim_count == 1


def test_run_state_suggests_skeptic_candidate_until_counterargument_exists() -> None:
    current_run = run()
    artifacts = [
        Artifact(
            run_id=current_run.id,
            artifact_type=ArtifactType.CLAIM,
            branch="market/equities",
            text_or_summary="Claim A",
            status=ArtifactStatus.VERIFIED,
        ),
        Artifact(
            run_id=current_run.id,
            artifact_type=ArtifactType.CLAIM,
            branch="market/equities",
            text_or_summary="Claim B",
            status=ArtifactStatus.VERIFIED,
        ),
    ]

    state = build_run_state(current_run, [], [], [], artifacts=artifacts)

    assert PrincipalActionType.REQUEST_SKEPTIC_REVIEW in {
        action.action_type for action in state.next_action_candidates
    }

    artifacts.append(
        Artifact(
            run_id=current_run.id,
            artifact_type=ArtifactType.COUNTERARGUMENT,
            branch="trust/skeptic",
            text_or_summary="Counterargument exists.",
        )
    )
    state = build_run_state(current_run, [], [], [], artifacts=artifacts)

    assert PrincipalActionType.REQUEST_SKEPTIC_REVIEW not in {
        action.action_type for action in state.next_action_candidates
    }


def test_run_state_rejected_and_disputed_counts_ignore_verification_artifacts() -> None:
    current_run = run()
    rejected_verification = Artifact(
        run_id=current_run.id,
        artifact_type=ArtifactType.VERIFICATION,
        branch="trust/source_verifier",
        text_or_summary="Verification rejected a claim.",
        status=ArtifactStatus.REJECTED,
    )
    disputed_claim = Artifact(
        run_id=current_run.id,
        artifact_type=ArtifactType.CLAIM,
        branch="market/equities",
        text_or_summary="Claim is disputed.",
        status=ArtifactStatus.DISPUTED,
    )

    state = build_run_state(
        current_run, [], [], [], artifacts=[rejected_verification, disputed_claim]
    )

    assert state.rejected_claim_count == 0
    assert state.disputed_claim_count == 1


def test_run_state_deduplicates_legacy_and_claim_artifact_counts() -> None:
    current_run = run()
    claim = Claim(
        run_id=current_run.id,
        task_id=uuid4(),
        statement="Claim rejected twice by legacy and artifact views.",
        evidence_observation_ids=[uuid4()],
        confidence=0.4,
    )
    verification = Verification(
        run_id=current_run.id,
        claim_id=claim.id,
        verdict="rejected",
        rationale="Unsupported.",
        confidence=0.8,
    )
    claim_artifact = Artifact(
        run_id=current_run.id,
        artifact_type=ArtifactType.CLAIM,
        text_or_summary=claim.statement,
        status=ArtifactStatus.REJECTED,
        legacy_object_type="claim",
        legacy_object_id=claim.id,
    )

    state = build_run_state(current_run, [], [claim], [verification], artifacts=[claim_artifact])

    assert state.rejected_claim_count == 1


def test_run_state_verified_count_ignores_non_knowledge_artifacts() -> None:
    current_run = run()
    verified_verification = Artifact(
        run_id=current_run.id,
        artifact_type=ArtifactType.VERIFICATION,
        branch="trust/source_verifier",
        text_or_summary="Verification provenance is verified.",
        status=ArtifactStatus.VERIFIED,
    )
    verified_final = Artifact(
        run_id=current_run.id,
        artifact_type=ArtifactType.FINAL_REPORT,
        branch="synthesis/aggregator",
        text_or_summary="Final report is verified.",
        status=ArtifactStatus.VERIFIED,
    )

    state = build_run_state(
        current_run, [], [], [], artifacts=[verified_verification, verified_final]
    )

    assert state.verified_claim_count == 0


def test_run_state_verified_count_includes_claim_artifacts() -> None:
    current_run = run()
    verified_claim = Artifact(
        run_id=current_run.id,
        artifact_type=ArtifactType.CLAIM,
        branch="market/equities",
        text_or_summary="SAP cloud backlog is growing.",
        status=ArtifactStatus.VERIFIED,
    )

    state = build_run_state(current_run, [], [], [], artifacts=[verified_claim])

    assert state.verified_claim_count == 1
