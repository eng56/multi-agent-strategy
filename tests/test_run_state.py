from uuid import uuid4

from src.agents.state import build_run_state
from src.common.models import (
    AgentRole,
    AgentSpec,
    Artifact,
    ArtifactPointer,
    ArtifactStatus,
    ArtifactType,
    Budget,
    Claim,
    DeadLetterRecord,
    EventType,
    FinalReport,
    ModelPolicy,
    Observation,
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
    assert state.tool_budget_remaining == {"tavily_credits": 50, "market_data_requests": 20}
    assert state.next_action_candidates[0].action_type == PrincipalActionType.REQUEST_TOOL_CALL


def test_run_state_exposes_budget_summary() -> None:
    current_run = run()
    current_run.budget.tools = ToolBudget(
        tavily_max_credits=5,
        market_data_max_requests=3,
        tavily_credits_used=2,
        market_data_requests_used=1,
    )
    current_run.budget.role_spent_usd = {
        AgentRole.PLANNER: 0.11,
        AgentRole.RESEARCH: 0.22,
        AgentRole.VERIFIER: 0.33,
        AgentRole.AGGREGATOR: 0.04,
        AgentRole.JUDGE: 0.01,
    }
    current_run.budget.role_reserved_usd = {
        AgentRole.RESEARCH: 0.05,
        AgentRole.AGGREGATOR: 0.02,
    }

    state = build_run_state(current_run, [], [], [])

    assert state.budget_summary is not None
    assert state.budget_summary.total_limit_usd == 4
    assert state.budget_summary.spent_usd == 0.5
    assert state.budget_summary.reserved_usd == 0.25
    assert state.budget_summary.remaining_usd == 3.25
    assert state.budget_summary.role_spent_usd["planner"] == 0.11
    assert state.budget_summary.role_spent_usd["research"] == 0.22
    assert state.budget_summary.role_spent_usd["verifier"] == 0.33
    assert state.budget_summary.role_spent_usd["aggregator"] == 0.04
    assert state.budget_summary.role_spent_usd["judge"] == 0.01
    assert state.budget_summary.role_reserved_usd["research"] == 0.05
    assert state.budget_summary.role_reserved_usd["aggregator"] == 0.02
    assert state.budget_summary.role_protected_usd == {"aggregator": 0.1, "judge": 0.1}
    assert round(state.budget_summary.protected_remaining_usd, 2) == 0.13
    assert state.budget_summary.tool_usage["tavily_credits"].used == 2
    assert state.budget_summary.tool_usage["tavily_credits"].max == 5
    assert state.budget_summary.tool_usage["tavily_credits"].remaining == 3
    assert state.budget_summary.tool_usage["market_data_requests"].used == 1
    assert state.budget_summary.tool_usage["market_data_requests"].max == 3
    assert state.budget_summary.tool_usage["market_data_requests"].remaining == 2
    assert state.tool_budget_remaining == {"tavily_credits": 3, "market_data_requests": 2}
    assert state.capacity is not None
    assert state.capacity.llm_remaining_usd == 3.25
    assert state.capacity.tavily_remaining == 3
    assert state.capacity.market_data_remaining == 2
    assert round(state.capacity.verifier_budget_remaining, 2) == 0.67
    assert round(state.capacity.aggregator_budget_protected_remaining, 2) == 0.04
    assert round(state.capacity.judge_budget_protected_remaining, 2) == 0.09
    assert state.capacity.search_exhausted is False
    assert state.capacity.market_data_exhausted is False


def test_run_state_tracks_conversion_quality_metrics() -> None:
    current_run = run()
    current_run.budget.tools = ToolBudget(tavily_max_credits=100, tavily_credits_used=100)
    task = ResearchTask(
        run_id=current_run.id,
        title="Fed cuts and rates",
        question="Find rates evidence",
        tool="web_search",
        status="completed",
    )
    verified_claim = Claim(
        run_id=current_run.id,
        task_id=task.id,
        statement="Lower expected Fed policy rates tend to reduce front-end Treasury yields.",
        evidence_observation_ids=[uuid4()],
        confidence=0.8,
    )
    disputed_claim = Claim(
        run_id=current_run.id,
        task_id=task.id,
        statement="Fed cuts guarantee a bullish cross-asset outcome.",
        evidence_observation_ids=[uuid4()],
        confidence=0.8,
        derived_from_verification_id=uuid4(),
    )
    verifications = [
        Verification(
            run_id=current_run.id,
            claim_id=verified_claim.id,
            verdict="verified",
            rationale="Supported.",
            confidence=0.82,
            source_quality_summary="1 primary and 2 high_quality_secondary sources.",
        ),
        Verification(
            run_id=current_run.id,
            claim_id=disputed_claim.id,
            verdict="uncertain",
            rationale="Too broad.",
            confidence=0.42,
            source_quality_summary="1 unknown source.",
        ),
    ]
    artifacts = [
        Artifact(
            run_id=current_run.id,
            artifact_type=ArtifactType.CLAIM,
            text_or_summary=verified_claim.statement,
            status=ArtifactStatus.VERIFIED,
            legacy_object_type="claim",
            legacy_object_id=verified_claim.id,
        ),
        Artifact(
            run_id=current_run.id,
            artifact_type=ArtifactType.CLAIM,
            text_or_summary=disputed_claim.statement,
            status=ArtifactStatus.DISPUTED,
            tags=["supported_part_subclaim"],
            legacy_object_type="claim",
            legacy_object_id=disputed_claim.id,
        ),
        Artifact(
            run_id=current_run.id,
            artifact_type=ArtifactType.JUDGE_FEEDBACK,
            text_or_summary="Clear caveats.",
            status=ArtifactStatus.VERIFIED,
        ),
    ]
    final = FinalReport(
        run_id=current_run.id,
        answer="Caveated final.",
        verified_claim_ids=[verified_claim.id],
        sources=[],
        judge_score=0.8,
    )

    state = build_run_state(
        current_run,
        [task],
        [verified_claim, disputed_claim],
        verifications,
        final,
        artifacts=artifacts,
    )

    assert state.tavily_used == 100
    assert state.verified_claims == 1
    assert state.verified_claims_per_10_tavily == 0.1
    assert state.disputed_claims == 1
    assert state.source_quality_distribution["primary"] == 1
    assert state.source_quality_distribution["high_quality_secondary"] == 2
    assert state.source_quality_distribution["unknown"] == 1
    assert state.claims_created_from_supported_parts == 1
    assert state.aggregator_used is True
    assert state.judge_used is True


def test_run_state_exposes_claim_explosion_and_suppresses_more_extraction() -> None:
    current_run = run()
    tasks = [
        ResearchTask(
            run_id=current_run.id,
            title=f"Fed branch {index}",
            question="Find Fed cut evidence",
            tool="web_search",
            branch="macro/rates" if index < 2 else "market/equities",
            status="completed",
        )
        for index in range(4)
    ]
    observations = [
        Observation(
            run_id=current_run.id,
            task_id=task.id,
            tool="web_search",
            summary=f"Research observation {index}",
            artifact=ArtifactPointer(uri=f"gs://bucket/{index}.json", size_bytes=2, sha256="0" * 64),
            sources=["https://example.com/source"],
        )
        for index, task in enumerate(tasks)
    ]
    claims = [
        Claim(
            run_id=current_run.id,
            task_id=tasks[index % len(tasks)].id,
            statement=f"Candidate claim {index}.",
            evidence_observation_ids=[observations[index % len(observations)].id],
            confidence=0.4,
        )
        for index in range(58)
    ]
    artifacts = [
        Artifact(
            run_id=current_run.id,
            artifact_type=ArtifactType.OBSERVATION,
            branch="macro/rates" if index < 2 else "market/equities",
            text_or_summary=observation.summary,
            legacy_object_type="observation",
            legacy_object_id=observation.id,
        )
        for index, observation in enumerate(observations)
    ]
    artifacts.extend(
        Artifact(
            run_id=current_run.id,
            artifact_type=ArtifactType.CLAIM,
            branch="macro/rates" if index < 29 else "market/equities",
            text_or_summary=claim.statement,
            legacy_object_type="claim",
            legacy_object_id=claim.id,
        )
        for index, claim in enumerate(claims)
    )

    state = build_run_state(
        current_run,
        tasks,
        claims,
        [],
        artifacts=artifacts,
        observations=observations,
    )

    assert state.claims_per_research_observation == 14.5
    assert any("Claim generation cap reached" in reason for reason in state.stop_reasons)
    assert not any(
        action.action_type == PrincipalActionType.ASSIGN_TASK
        and "observation(s) have no extracted claim" in action.reason
        for action in state.next_action_candidates
    )


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


def test_run_state_exposes_failed_tasks_and_allows_partial_aggregation_candidate() -> None:
    current_run = run()
    completed_task = ResearchTask(
        run_id=current_run.id,
        title="SAP catalysts",
        question="Find SAP catalysts",
        tool="web_search",
        status="completed",
    )
    failed_task = ResearchTask(
        run_id=current_run.id,
        title="SAP valuation",
        question="Find SAP valuation evidence",
        tool="web_search",
        status="failed",
    )
    claim = Claim(
        run_id=current_run.id,
        task_id=completed_task.id,
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

    state = build_run_state(
        current_run, [completed_task, failed_task], [claim], [verification]
    )

    assert state.current_phase == RunPhase.SYNTHESIZING
    assert state.failed_tasks == ["SAP valuation"]
    assert state.open_questions == []
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


def test_budget_exhausted_run_state_exposes_stop_reasons() -> None:
    current_run = run(RunStatus.PARTIAL_BUDGET_EXHAUSTED)
    current_run.failure_reason = (
        "insufficient shared budget for research: requested=$0.0500, "
        "available=$0.0100, protected=$0.0500"
    )
    task = ResearchTask(
        run_id=current_run.id,
        title="SAP catalysts",
        question="Find SAP catalysts",
        tool="web_search",
        status="skipped_budget",
        reason="Budget blocked tool_execution for task: SAP catalysts (Tavily credit budget reached)",
    )
    final = FinalReport(
        run_id=current_run.id,
        answer="Partial answer",
        verified_claim_ids=[],
        sources=[],
        partial=True,
    )

    state = build_run_state(current_run, [task], [], [], final)

    assert state.current_phase == RunPhase.COMPLETED
    assert state.budget_summary is not None
    assert state.budget_summary.stop_reasons == [
        current_run.failure_reason,
        "1 task(s) skipped due to budget: Budget blocked tool_execution for task: "
        "SAP catalysts (Tavily credit budget reached)",
    ]
    assert current_run.failure_reason in state.stop_reasons
    assert any("skipped due to budget" in reason for reason in state.stop_reasons)
    assert state.next_action_candidates == []


def test_low_judge_score_run_state_requests_first_followup_wave() -> None:
    current_run = run()
    final = FinalReport(
        run_id=current_run.id,
        answer="Buy with caution.",
        verified_claim_ids=[],
        sources=[],
        judge_score=0.62,
        judge_feedback="Needs stronger valuation evidence.",
    )

    state = build_run_state(current_run, [], [], [], final)

    assert state.current_phase == RunPhase.JUDGING
    assert state.last_judge_score == 0.62
    assert PrincipalActionType.REQUEST_FOLLOWUP in {
        action.action_type for action in state.next_action_candidates
    }
    assert PrincipalActionType.STOP_RUN not in {
        action.action_type for action in state.next_action_candidates
    }


def test_partial_evidence_limited_final_run_state_requests_judge() -> None:
    current_run = run()
    final = FinalReport(
        run_id=current_run.id,
        answer="Evidence-limited result.",
        verified_claim_ids=[],
        sources=["https://example.com/caveated-source"],
        partial=True,
    )

    state = build_run_state(current_run, [], [], [], final)

    assert state.current_phase == RunPhase.JUDGING
    judge_actions = [
        action
        for action in state.next_action_candidates
        if action.action_type == PrincipalActionType.REQUEST_TOOL_CALL
        and action.required_role == "judge_agent"
    ]
    assert judge_actions
    assert state.capacity is not None
    assert state.capacity.useful_action_available is True


def test_followup_run_state_requests_aggregation_then_stop_after_second_low_score() -> None:
    current_run = run()
    task = ResearchTask(
        run_id=current_run.id,
        title="Follow-up: valuation",
        question="Find valuation evidence.",
        tool="web_search",
        status="completed",
        wave_number=1,
        reason="Judge score 0.62 below 0.75: Needs valuation evidence.",
    )
    claim = Claim(
        run_id=current_run.id,
        task_id=task.id,
        statement="SAP valuation has improved.",
        evidence_observation_ids=[uuid4()],
        confidence=0.8,
    )
    verification = Verification(
        run_id=current_run.id,
        claim_id=claim.id,
        verdict="verified",
        rationale="Supported.",
        confidence=0.7,
    )
    stale_final = FinalReport(
        run_id=current_run.id,
        answer="Buy with caution.",
        verified_claim_ids=[],
        sources=[],
        wave_number=0,
        judge_score=0.62,
        judge_feedback="Needs stronger valuation evidence.",
    )

    state = build_run_state(current_run, [task], [claim], [verification], stale_final)

    assert state.current_phase == RunPhase.SYNTHESIZING
    assert PrincipalActionType.REQUEST_AGGREGATION in {
        action.action_type for action in state.next_action_candidates
    }
    assert PrincipalActionType.STOP_RUN not in {
        action.action_type for action in state.next_action_candidates
    }

    second_low_final = FinalReport(
        run_id=current_run.id,
        answer="Still weak.",
        verified_claim_ids=[claim.id],
        sources=[],
        wave_number=1,
        judge_score=0.51,
        judge_feedback="Still too thin.",
    )
    state = build_run_state(current_run, [task], [claim], [verification], second_low_final)

    assert state.current_phase == RunPhase.COMPLETED
    assert PrincipalActionType.REQUEST_FOLLOWUP not in {
        action.action_type for action in state.next_action_candidates
    }
    assert PrincipalActionType.STOP_RUN in {
        action.action_type for action in state.next_action_candidates
    }


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
    data_gap = Artifact(
        run_id=current_run.id,
        artifact_type=ArtifactType.DATA_GAP,
        branch="market/gold",
        text_or_summary="Gold spot data requires a configured spot or futures provider.",
    )

    state = build_run_state(
        current_run, [], [], [], artifacts=[verified, rejected, disputed, question, data_gap]
    )

    assert state.known_facts == ["Gold has upside if real yields fall."]
    assert state.open_questions == [
        "Are ETF flows confirming the move?",
        "Gold spot data requires a configured spot or futures provider.",
    ]
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


def test_run_state_does_not_suggest_skeptic_with_insufficient_verified_evidence() -> None:
    current_run = run()
    artifact = Artifact(
        run_id=current_run.id,
        artifact_type=ArtifactType.CLAIM,
        branch="market/equities",
        text_or_summary="Only one verified claim exists.",
        status=ArtifactStatus.VERIFIED,
    )

    state = build_run_state(current_run, [], [], [], artifacts=[artifact])

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


def test_market_snapshot_claim_is_context_only_not_normal_verification_work() -> None:
    current_run = run()
    task = ResearchTask(
        run_id=current_run.id,
        title="Market data snapshot",
        question="Fetch market snapshot.",
        tool="market_data",
        status="completed",
        branch="market/bonds",
    )
    claim = Claim(
        run_id=current_run.id,
        task_id=task.id,
        statement="Market data snapshot selected TLT (etf) via massive/stocks.",
        evidence_observation_ids=[uuid4()],
        confidence=0.6,
    )
    artifact = Artifact(
        run_id=current_run.id,
        artifact_type=ArtifactType.CLAIM,
        branch="market/bonds",
        text_or_summary=claim.statement,
        tags=["market_snapshot", "context_only", "tool:market_data"],
        legacy_object_type="claim",
        legacy_object_id=claim.id,
    )

    state = build_run_state(current_run, [task], [claim], [], artifacts=[artifact])

    assert state.verified_claim_count == 0
    assert PrincipalActionType.REQUEST_VERIFICATION not in {
        action.action_type for action in state.next_action_candidates
    }


def test_run_state_reports_search_exhausted_when_tavily_zero_despite_llm_budget() -> None:
    current_run = run()
    current_run.budget.limit_usd = 20
    current_run.budget.spent_usd = 0.1
    current_run.budget.reserved_usd = 0
    current_run.budget.tools = ToolBudget(
        tavily_max_credits=0,
        tavily_credits_used=0,
        market_data_max_requests=10,
        market_data_requests_used=0,
    )
    task = ResearchTask(
        run_id=current_run.id,
        title="Equity evidence",
        question="Find equity evidence.",
        tool="web_search",
        status="completed",
        branch="market/equities",
    )
    claim = Claim(
        run_id=current_run.id,
        task_id=task.id,
        statement="Equities rise on faster Fed cuts.",
        evidence_observation_ids=[uuid4()],
        confidence=0.6,
    )

    state = build_run_state(current_run, [task], [claim], [])

    assert state.capacity is not None
    assert state.capacity.llm_remaining_usd == 19.9
    assert state.capacity.search_exhausted is True
    assert state.capacity.market_data_exhausted is False
    assert (
        "Search/tool budget exhausted before enough claims passed verification."
        in state.stop_reasons
    )


def test_latest_run_like_underutilized_zero_verified_state_has_useful_capacity() -> None:
    current_run = Run(
        question=(
            "If the Fed signals faster rate cuts, compare macro rates, U.S. equities, "
            "the U.S. dollar, gold, and long-duration bonds."
        ),
        status=RunStatus.RUNNING,
        models=model_policy(),
        budget=Budget(
            limit_usd=5,
            spent_usd=0.06,
            tools=ToolBudget(
                tavily_max_credits=100,
                tavily_credits_used=12,
                market_data_max_requests=50,
                market_data_requests_used=1,
            ),
        ),
    )
    task = ResearchTask(
        run_id=current_run.id,
        title="Rates repair",
        question="Find dated evidence for Fed cuts.",
        tool="web_search",
        status="completed",
        branch="macro/rates",
        wave_number=1,
    )
    claim = Claim(
        run_id=current_run.id,
        task_id=task.id,
        statement="A faster Fed cutting path guarantees a bullish cross-asset outcome.",
        evidence_observation_ids=[uuid4()],
        confidence=0.7,
    )
    verification = Verification(
        run_id=current_run.id,
        claim_id=claim.id,
        verdict="uncertain",
        rationale="Too broad for verification.",
        confidence=0.4,
        unsupported_parts=["The guaranteed bullish outcome is not supported."],
        source_quality_summary="No dated primary source.",
    )

    state = build_run_state(current_run, [task], [claim], [verification])

    assert state.verified_claim_count == 0
    assert state.capacity is not None
    assert state.capacity.tavily_utilization_ratio == 0.12
    assert round(state.capacity.llm_utilization_ratio, 3) == 0.012
    assert state.capacity.terminal_final_allowed is False
    assert state.capacity.useful_action_available is True
    assert "Tavily utilization is below partial-final threshold" in (
        state.capacity.terminal_final_blockers
    )
    assert PrincipalActionType.REQUEST_FOLLOWUP in {
        action.action_type for action in state.next_action_candidates
    }


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


def test_run_state_exposes_dead_letter_count_and_stop_reason() -> None:
    current_run = run()
    record = DeadLetterRecord(
        run_id=current_run.id,
        event_id=uuid4(),
        event_type=EventType.TASK_CREATED,
        event_producer="planner-agent",
        worker_role="tool-runner",
        retry_count=2,
        max_retries=2,
        classification="transient",
        error_type="TimeoutError",
        error_message="temporary tool timeout",
        event_payload={"task_id": str(uuid4())},
    )

    state = build_run_state(current_run, [], [], [], dead_letters=[record])

    assert state.dead_letter_count == 1
    assert state.stop_reasons == ["1 dead-lettered event(s)"]
