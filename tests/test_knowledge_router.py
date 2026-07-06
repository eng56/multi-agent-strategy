from uuid import uuid4

from src.agents.knowledge_router import select_context_for_agent
from src.common.models import (
    AgentSpec,
    Artifact,
    ArtifactStatus,
    ArtifactType,
    ResearchTask,
    RunPhase,
    RunState,
    VisibilityScope,
)


def agent(
    run_id,
    *,
    branch: str = "market/gold",
    role_template: str = "asset_research_agent",
    retrieval_tags: list[str] | None = None,
) -> AgentSpec:
    return AgentSpec(
        run_id=run_id,
        name=role_template.replace("_", " "),
        role_template=role_template,
        branch=branch,
        objective=f"Research {branch}",
        retrieval_tags=retrieval_tags or [],
    )


def state(run_id) -> RunState:
    return RunState(
        run_id=run_id,
        question="Will gold rise if Fed cuts rates?",
        objective="Will gold rise if Fed cuts rates?",
        current_phase=RunPhase.RESEARCHING,
        active_branches=["market/gold", "macro/rates"],
    )


def task(run_id) -> ResearchTask:
    return ResearchTask(
        run_id=run_id,
        title="Gold and rates",
        question="Find evidence on gold and rates.",
        tool="web_search",
    )


def artifact(
    run_id,
    *,
    branch: str = "market/gold",
    visibility: VisibilityScope = VisibilityScope.TEAM,
    status: ArtifactStatus = ArtifactStatus.UNVERIFIED,
    artifact_type: ArtifactType = ArtifactType.OBSERVATION,
    tags: list[str] | None = None,
    confidence: float | None = None,
    summary: str = "Branch-local evidence.",
) -> Artifact:
    return Artifact(
        run_id=run_id,
        artifact_type=artifact_type,
        branch=branch,
        text_or_summary=summary,
        tags=tags or [],
        visibility=visibility,
        status=status,
        confidence=confidence,
    )


def route(current_agent: AgentSpec, artifacts: list[Artifact], *, max_items: int = 12):
    return select_context_for_agent(
        agent_spec=current_agent,
        run_state=state(current_agent.run_id),
        artifacts=artifacts,
        tasks=[task(current_agent.run_id)],
        max_items=max_items,
    )


def test_same_branch_artifacts_are_selected() -> None:
    run_id = uuid4()
    current_agent = agent(run_id, branch="market/gold")
    same_branch = artifact(run_id, branch="market/gold")

    assert route(current_agent, [same_branch]) == [same_branch]


def test_unrelated_branch_artifacts_are_not_selected_by_default() -> None:
    run_id = uuid4()
    current_agent = agent(run_id, branch="market/gold")
    unrelated = artifact(run_id, branch="market/oil")

    assert route(current_agent, [unrelated]) == []


def test_public_verified_artifacts_can_cross_branch_when_relevant() -> None:
    run_id = uuid4()
    current_agent = agent(run_id, branch="market/gold")
    cross_branch = artifact(
        run_id,
        branch="macro/rates",
        visibility=VisibilityScope.PUBLIC_VERIFIED,
        status=ArtifactStatus.VERIFIED,
        tags=["market:gold"],
        summary="Falling real yields can support gold.",
    )

    assert route(current_agent, [cross_branch]) == [cross_branch]


def test_rejected_artifacts_are_excluded_for_normal_research_agents() -> None:
    run_id = uuid4()
    current_agent = agent(run_id, branch="market/gold")
    rejected = artifact(
        run_id,
        branch="market/gold",
        status=ArtifactStatus.REJECTED,
        summary="Unsupported gold claim.",
    )

    assert route(current_agent, [rejected]) == []


def test_verifier_and_judge_can_see_rejected_and_disputed_artifacts() -> None:
    run_id = uuid4()
    rejected = artifact(
        run_id,
        branch="trust/source_verifier",
        status=ArtifactStatus.REJECTED,
        summary="Claim contradicted by sources.",
    )
    disputed = artifact(
        run_id,
        branch="trust/source_verifier",
        status=ArtifactStatus.DISPUTED,
        summary="Claim needs further review.",
    )
    verifier = agent(
        run_id,
        branch="trust/source_verifier",
        role_template="source_verifier_agent",
    )
    judge = agent(run_id, branch="trust/source_verifier", role_template="judge_agent")

    assert route(verifier, [rejected, disputed]) == [rejected, disputed]
    assert route(judge, [rejected, disputed]) == [rejected, disputed]


def test_max_items_is_respected() -> None:
    run_id = uuid4()
    current_agent = agent(run_id, branch="market/gold")
    artifacts = [
        artifact(run_id, branch="market/gold", summary=f"Evidence {index}")
        for index in range(5)
    ]

    selected = route(current_agent, artifacts, max_items=3)

    assert selected == artifacts[:3]
